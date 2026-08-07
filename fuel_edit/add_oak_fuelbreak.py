"""
Add an oak woodland fuel break to wrfinput_d04.

The fuel break is defined as a buffered line (rotated rectangle):
  - Two endpoints (lat/lon) define the centerline, oriented perpendicular
    to the mean wind direction.
  - `fuelbreak_width_m` sets the along-flow extent of the break.

Cells inside the fuel break polygon are replaced with:
  - NFUEL_CAT → 9  (Anderson 13: Hardwood litter / oak woodland)
  - LU_INDEX  → 4  (IGBP MODIS Noah: Deciduous Broadleaf Forest / oak)

Urban and non-burnable cells are never modified in either variable.
"""
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
from netCDF4 import Dataset
from pathlib import Path
from shapely.geometry import LineString

# ── Configuration ─────────────────────────────────────────────────────────────
wrfinput_path = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/ifire2_oak_break/wrfinput_d04_noroads")
out_path      = wrfinput_path.parent / "wrfinput_d04_oakbreak"

# Any existing wrfout_d04 from a completed run — needed for FXLAT/FXLONG
# (fire-grid coordinates not stored in wrfinput, computed at runtime by WRF)
wrfout_ref = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/ifire2_noroad/ref/wrfout_d04_2024-11-06_17:00:00")

# Fuel break centerline: two endpoints in (lat, lon).
# Orient the line PERPENDICULAR to the mean wind direction.
# Ignition (START) is at (34.318, -118.968); fire moves SW with Santa Ana winds.
# → line runs roughly NW–SE, placed downwind (SW) of ignition.
# Adjust these to your desired location:
endpoint1 = (34.319401, -118.993324)   # NW end of the fuel break
endpoint2 = (34.303774, -118.964856)   # SE end of the fuel break

# Along-flow width of the fuel break in metres
fuelbreak_width_m = 300   # 300 m ≈ 3 cells at 111 m resolution

# Replacement values
oak_fuel_cat = 9    # Anderson 13: Hardwood litter (oak woodland)
# oak_lu_index = 4    # IGBP MODIS Noah: Deciduous Broadleaf Forest
oak_lu_index = 5    # IGBP MODIS Noah: Mixed Forest

# Non-burnable / urban categories — never overwritten
nonfuel_cats = {91, 92, 93, 98, 99}   # NFUEL_CAT
urban_lu     = {13}                    # LU_INDEX

# ── Load ──────────────────────────────────────────────────────────────────────
shutil.copy(wrfinput_path, out_path)
print(f"Working copy: {out_path}")

ds_orig = Dataset(wrfinput_path, 'r')
ds_out  = Dataset(out_path, 'r+')

fuelvar  = "FUEL_CAT" if "FUEL_CAT" in ds_orig.variables else "NFUEL_CAT"
nfuel    = np.array(ds_orig.variables[fuelvar][0, :, :])     # fire grid (ny_f, nx_f)
lu       = np.array(ds_orig.variables["LU_INDEX"][0, :, :])  # met  grid (ny_m, nx_m)
xlat     = np.array(ds_orig.variables["XLAT"][0, :, :])      # met  grid
xlon     = np.array(ds_orig.variables["XLONG"][0, :, :])     # met  grid
ds_orig.close()

# Fire-grid coordinates live in wrfout (WRF computes them at runtime)
ds_ref = Dataset(wrfout_ref, 'r')
fxlat  = np.array(ds_ref.variables["FXLAT"][0, :, :])        # fire grid
fxlon  = np.array(ds_ref.variables["FXLONG"][0, :, :])       # fire grid
ds_ref.close()

print(f"NFUEL_CAT  (fire grid): {nfuel.shape}")
print(f"LU_INDEX   (met  grid): {lu.shape}")
print(f"FXLAT/FXLONG          : {fxlat.shape}")

# ── Build fuel break polygon ───────────────────────────────────────────────────
# Buffer the centerline by half the width in degrees (lat/lon approximation).
lat_c = (endpoint1[0] + endpoint2[0]) / 2
m_per_deg_lat = 111_000
m_per_deg_lon = 111_000 * np.cos(np.radians(lat_c))

# Buffer in degrees (use lat scale as conservative approximation)
buffer_deg = (fuelbreak_width_m / 2) / m_per_deg_lat

line = LineString([(endpoint1[1], endpoint1[0]),   # shapely uses (lon, lat)
                   (endpoint2[1], endpoint2[0])])
poly = line.buffer(buffer_deg, cap_style=2)        # cap_style=2 → flat ends

# Extract exterior ring as numpy array (lon, lat)
poly_coords = np.array(poly.exterior.coords)       # (N, 2): lon, lat

# ── Vectorised point-in-polygon test — separate masks for each grid ───────────
mpl_path = MplPath(poly_coords)

# Fire grid mask → for NFUEL_CAT
fire_pts       = np.column_stack([fxlon.ravel(), fxlat.ravel()])
in_poly_fire   = mpl_path.contains_points(fire_pts).reshape(nfuel.shape)

# Met grid mask → for LU_INDEX
met_pts        = np.column_stack([xlon.ravel(), xlat.ravel()])
in_poly_met    = mpl_path.contains_points(met_pts).reshape(lu.shape)

# ── Apply urban / non-burnable exclusions ─────────────────────────────────────
urban_fuel_mask    = np.isin(nfuel, list(nonfuel_cats))   # fire grid
urban_lu_mask      = np.isin(lu,   list(urban_lu))         # met  grid

fuelbreak_mask_fire = in_poly_fire & ~urban_fuel_mask
fuelbreak_mask_met  = in_poly_met  & ~urban_lu_mask

print(f"\nFuel break geometry:")
print(f"  Endpoint 1 : {endpoint1}")
print(f"  Endpoint 2 : {endpoint2}")
print(f"  Width      : {fuelbreak_width_m} m")
print(f"\nFire grid — cells in polygon : {in_poly_fire.sum()}")
print(f"  Urban/non-burn (skipped)   : {(in_poly_fire & urban_fuel_mask).sum()}")
print(f"  NFUEL_CAT modified         : {fuelbreak_mask_fire.sum()}")
print(f"\nMet grid  — cells in polygon : {in_poly_met.sum()}")
print(f"  Urban LU (skipped)         : {(in_poly_met & urban_lu_mask).sum()}")
print(f"  LU_INDEX  modified         : {fuelbreak_mask_met.sum()}")

# ── Replace fuel and land use ─────────────────────────────────────────────────
nfuel_mod = nfuel.copy()
lu_mod    = lu.copy()

nfuel_mod[fuelbreak_mask_fire] = oak_fuel_cat
lu_mod[fuelbreak_mask_met]     = oak_lu_index

# ── Write back ────────────────────────────────────────────────────────────────
ds_out.variables[fuelvar][0, :, :]    = nfuel_mod
ds_out.variables["LU_INDEX"][0, :, :] = lu_mod
ds_out.close()
print(f"\nSaved → {out_path}")

# ── Colormaps and labels ──────────────────────────────────────────────────────
fuel_colors = {
    -9999:"#000000", 1:"#ffffbe", 2:"#ffff00", 3:"#e6c50b", 4:"#ffd37f",
    5:"#ffaa66",  6:"#cdaa66",  7:"#897044",  8:"#d3ffbe",  9:"#70a800",
    10:"#267300", 11:"#e8beff", 12:"#7a8ef5", 13:"#c500ff",
    91:"#8400a5", 92:"#9ea1f0", 93:"#e974ff", 98:"#0000ff", 99:"#bfbfbf",
}
fuel_names = {
    -9999:"Non-fuel",     1:"Short grass",      2:"Timber/grass",
    3:"Tall grass",       4:"Chaparral",         5:"Brush",
    6:"Dormant brush",    7:"Southern rough",    8:"Closed timber",
    9:"Hardwood litter",  10:"Heavy timber",     11:"Light slash",
    12:"Medium slash",    13:"Heavy slash",
    91:"Urban/developed", 92:"Snow/ice",          93:"Agriculture (NB)",
    98:"Open water",      99:"Bare ground",
}
fuel_ids    = sorted(fuel_colors.keys())
fuel_cmap   = mcolors.ListedColormap([fuel_colors[i] for i in fuel_ids])
fuel_to_idx = {fid: i for i, fid in enumerate(fuel_ids)}

lu_colors_map = {
    1:"#1a6b1a", 2:"#2ca02c", 3:"#8fbc8f", 4:"#52b052", 5:"#76b876",
    6:"#b5924c", 7:"#d4a96a", 8:"#9acd32", 9:"#c8b400", 10:"#f5e642",
    11:"#4db8ff", 12:"#f0a500", 13:"#e03030", 14:"#e8c840", 15:"#ffffff",
    16:"#c8c0a0", 17:"#0055cc", 18:"#a0d0a0", 19:"#b0c8b0", 20:"#d0d0b0",
}
lu_labels = {
    1:"Evergreen Needleleaf", 2:"Evergreen Broadleaf", 3:"Deciduous Needleleaf",
    4:"Deciduous Broadleaf",  5:"Mixed Forest",         6:"Closed Shrublands",
    7:"Open Shrublands",      8:"Woody Savannas",       9:"Savannas",
    10:"Grasslands",          11:"Permanent Wetlands",  12:"Croplands",
    13:"Urban",               14:"Cropland/Veg Mosaic", 15:"Snow/Ice",
    16:"Barren",              17:"Water",               18:"Wooded Tundra",
    19:"Mixed Tundra",        20:"Barren Tundra",
}
lu_ids  = sorted(lu_colors_map.keys())
lu_cmap = mcolors.ListedColormap([lu_colors_map[i] for i in lu_ids])
lu_norm = mcolors.BoundaryNorm(np.arange(0.5, 21.5), lu_cmap.N)

def to_fuel_idx(arr):
    out = np.zeros_like(arr, dtype=int)
    for fid, idx in fuel_to_idx.items():
        out[arr == fid] = idx
    return out

# ── Zoom bounds ───────────────────────────────────────────────────────────────
# Fire grid zoom
rows_f, cols_f = np.where(in_poly_fire)
r0f = max(rows_f.min() - 30, 0);  r1f = min(rows_f.max() + 30, nfuel.shape[0])
c0f = max(cols_f.min() - 30, 0);  c1f = min(cols_f.max() + 30, nfuel.shape[1])

# Met grid zoom
rows_m, cols_m = np.where(in_poly_met)
r0m = max(rows_m.min() - 5, 0);   r1m = min(rows_m.max() + 5, lu.shape[0])
c0m = max(cols_m.min() - 5, 0);   c1m = min(cols_m.max() + 5, lu.shape[1])

# ── Figure 1: NFUEL_CAT ───────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))

kw_fuel = dict(cmap=fuel_cmap, vmin=0, vmax=len(fuel_ids)-1, origin='lower')

axes1[0,0].imshow(to_fuel_idx(nfuel),     **kw_fuel)
axes1[0,0].set_title("NFUEL_CAT — original (full fire grid)")

axes1[0,1].imshow(to_fuel_idx(nfuel_mod), **kw_fuel)
axes1[0,1].set_title("NFUEL_CAT — oak fuel break applied (full fire grid)")

axes1[1,0].imshow(to_fuel_idx(nfuel[r0f:r1f, c0f:c1f]),     **kw_fuel,
                  extent=[c0f, c1f, r0f, r1f])
axes1[1,0].set_title("Zoomed — original\n(colors = fuel categories, see legend)")

axes1[1,1].imshow(to_fuel_idx(nfuel_mod[r0f:r1f, c0f:c1f]), **kw_fuel,
                  extent=[c0f, c1f, r0f, r1f])
axes1[1,1].set_title("Zoomed — after oak fuel break\n"
                     f"(cat {oak_fuel_cat} = Hardwood litter replaces original fuels in band)")

# Fuel break outline on zoomed panels (outline only, no fill — keeps colors readable)
for ax in axes1[1]:
    mask_zoom = fuelbreak_mask_fire[r0f:r1f, c0f:c1f].astype(float)
    ax.contour(mask_zoom, levels=[0.5], colors='white', linewidths=2,
               extent=[c0f, c1f, r0f, r1f], origin='lower')

for ax in axes1.ravel():
    ax.set_xlabel("x (fire-grid index)")
    ax.set_ylabel("y (fire-grid index)")

# Fuel category legend — only categories present in zoomed area (before + after)
zoom_cats = np.unique(np.concatenate([
    nfuel[r0f:r1f, c0f:c1f].ravel(),
    nfuel_mod[r0f:r1f, c0f:c1f].ravel()
])).astype(int)
legend_patches = [
    mpatches.Patch(color=fuel_colors.get(c, "#888888"),
                   label=f"{c} — {fuel_names.get(c, 'Unknown')}"
                         + (" ← NEW" if c == oak_fuel_cat else ""))
    for c in zoom_cats if c in fuel_colors
]
axes1[1,0].legend(handles=legend_patches, fontsize=8, loc='lower left',
                  framealpha=0.9, title="Fuel categories")

fig1.suptitle(f"Oak fuel break — NFUEL_CAT  |  width={fuelbreak_width_m} m  |  "
              f"{fuelbreak_mask_fire.sum()} cells modified\n"
              f"White outline = fuel break boundary", fontsize=12)
plt.tight_layout()
diag1_path = out_path.parent / "oakbreak_diagnostic_fuel.png"
fig1.savefig(diag1_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"NFUEL_CAT diagnostic → {diag1_path}")

# ── Figure 2: LU_INDEX ────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

kw_lu = dict(cmap=lu_cmap, norm=lu_norm, origin='lower')

# Full domain
axes2[0,0].imshow(lu,     **kw_lu)
axes2[0,0].set_title("LU_INDEX — original (full domain)")

axes2[0,1].imshow(lu_mod, **kw_lu)
axes2[0,1].set_title("LU_INDEX — after oak fuel break (full domain)")

# Zoomed
axes2[1,0].imshow(lu[r0m:r1m, c0m:c1m],     **kw_lu,
                  extent=[c0m, c1m, r0m, r1m])
axes2[1,0].set_title("LU_INDEX — original (zoomed)")

axes2[1,1].imshow(lu_mod[r0m:r1m, c0m:c1m], **kw_lu,
                  extent=[c0m, c1m, r0m, r1m])
axes2[1,1].set_title("LU_INDEX — after oak fuel break (zoomed)")

# White contour outline on zoomed panels only
for ax in axes2[1]:
    mask_zoom_m = fuelbreak_mask_met[r0m:r1m, c0m:c1m].astype(float)
    ax.contour(mask_zoom_m, levels=[0.5], colors='white', linewidths=2,
               extent=[c0m, c1m, r0m, r1m], origin='lower')

for ax in axes2.ravel():
    ax.set_xlabel("x (met-grid index)")
    ax.set_ylabel("y (met-grid index)")

# LU legend — categories present in domain (before + after)
all_lu_cats = np.unique(np.concatenate([lu.ravel(), lu_mod.ravel()])).astype(int)
lu_legend_patches = [
    mpatches.Patch(color=lu_colors_map.get(c, "#888888"),
                   label=f"{c} — {lu_labels.get(c, 'Unknown')}"
                         + (" ← NEW" if c == oak_lu_index else ""))
    for c in all_lu_cats if c in lu_colors_map
]
axes2[0,0].legend(handles=lu_legend_patches, fontsize=8, loc='upper left',
                  framealpha=0.9, title="IGBP land use")

fig2.suptitle(f"Oak fuel break — LU_INDEX  |  {fuelbreak_mask_met.sum()} cells modified\n"
              f"White outline = fuel break boundary", fontsize=12)
plt.tight_layout()
diag2_path = out_path.parent / "oakbreak_diagnostic_LU.png"
fig2.savefig(diag2_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"LU_INDEX diagnostic    → {diag2_path}")
