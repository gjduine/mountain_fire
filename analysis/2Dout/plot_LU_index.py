import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from wrf import getvar, latlon_coords, get_cartopy, to_np
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# ── IGBP MODIS Noah land use categories ───────────────────────────────────────
# Standard 20-category IGBP MODIS Noah as used in WRF
lu_labels = {
    1:  "Evergreen Needleleaf Forest",
    2:  "Evergreen Broadleaf Forest",
    3:  "Deciduous Needleleaf Forest",
    4:  "Deciduous Broadleaf Forest",
    5:  "Mixed Forest",
    6:  "Closed Shrublands",
    7:  "Open Shrublands",
    8:  "Woody Savannas",
    9:  "Savannas",
    10: "Grasslands",
    11: "Permanent Wetlands",
    12: "Croplands",
    13: "Urban and Built-Up",
    14: "Cropland / Natural Veg. Mosaic",
    15: "Snow and Ice",
    16: "Barren / Sparsely Vegetated",
    17: "Water",
    18: "Wooded Tundra",
    19: "Mixed Tundra",
    20: "Barren Tundra",
}

lu_colors = {
    1:  "#1a6b1a",   # dark green
    2:  "#2ca02c",   # green
    3:  "#8fbc8f",   # light green
    4:  "#52b052",   # medium green
    5:  "#76b876",   # mixed green
    6:  "#b5924c",   # brown-shrub
    7:  "#d4a96a",   # light brown (open shrub — likely dominant here)
    8:  "#9acd32",   # yellow-green
    9:  "#c8b400",   # savanna yellow
    10: "#f5e642",   # grassland yellow
    11: "#4db8ff",   # wetland blue
    12: "#f0a500",   # cropland orange
    13: "#e03030",   # urban red
    14: "#e8c840",   # mosaic
    15: "#ffffff",   # snow/ice white
    16: "#c8c0a0",   # barren tan
    17: "#0055cc",   # water blue
    18: "#a0d0a0",   # wooded tundra
    19: "#b0c8b0",   # mixed tundra
    20: "#d0d0b0",   # barren tundra
}

# ── Settings ───────────────────────────────────────────────────────────────────
wrf_dir = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/ifire0/ref/")
domain  = "d04"

lat_min, lat_max =  34.25,  34.35
lon_min, lon_max = -119.05, -118.95
dy = np.arange(34.25,  34.35,  0.02)
dx = np.arange(-119.05, -118.95, 0.03)

stations = {
    "START":       (34.318,    -118.968,   "black"),
    "SPOT":        (34.2528,   -119.0284,  "red"),
    "Spot Valley": (34.271191, -119.015999,"purple"),
}

out_dir = Path(".")

# ── Load LU_INDEX from first available wrfout ──────────────────────────────────
wrf_files = sorted(wrf_dir.glob(f"wrfout_{domain}_*"))
if not wrf_files:
    raise RuntimeError(f"No wrfout files found in {wrf_dir}")

ds = Dataset(wrf_files[0])
luvar  = "LU_INDEX" if "LU_INDEX" in ds.variables else "IVGTYP"
lu     = ds.variables[luvar][0, :, :]   # (ny, nx)
hgt    = getvar(ds, "HGT", timeidx=0)
cart_proj = get_cartopy(hgt)
lats, lons = latlon_coords(hgt)
ds.close()

lu_np   = np.array(lu)
lats_np = to_np(lats)
lons_np = to_np(lons)

# ── Find which LU categories actually exist in the subdomain ──────────────────
sub_mask    = ((lats_np >= lat_min) & (lats_np <= lat_max) &
               (lons_np >= lon_min) & (lons_np <= lon_max))
present_ids = sorted(set(lu_np[sub_mask].astype(int).ravel()))
print(f"LU categories present in subdomain: {present_ids}")

# Build colormap only for present categories
cmap_colors = [lu_colors.get(i, "#888888") for i in range(1, 21)]
full_cmap   = mcolors.ListedColormap(cmap_colors)
bounds      = np.arange(0.5, 21.5, 1)
norm        = mcolors.BoundaryNorm(bounds, full_cmap.N)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                        subplot_kw={'projection': cart_proj})

ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE, linewidth=2, zorder=3)
ax.set_xticks(dx, crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.set_yticks(dy, crs=ccrs.PlateCarree())
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(direction='out', labelsize=10, length=5, pad=2)

# LU pcolormesh
p = ax.pcolormesh(
    lons_np, lats_np, lu_np,
    cmap=full_cmap, norm=norm,
    shading='auto',
    transform=ccrs.PlateCarree(), zorder=1
)

# Terrain contours
ax.contour(
    lons_np, lats_np, to_np(hgt),
    levels=np.arange(100, 3000, 100),
    colors='k', linewidths=0.5, alpha=0.5,
    transform=ccrs.PlateCarree(), zorder=2
)

# Station markers
for name, (lat_s, lon_s, color) in stations.items():
    ax.plot(lon_s, lat_s, marker="*",
            markerfacecolor="none", markeredgecolor=color,
            markersize=18, markeredgewidth=2,
            linestyle="none",
            transform=ccrs.PlateCarree(), zorder=5)
    ax.text(lon_s + 0.002, lat_s + 0.002, name, fontsize=9,
            color=color, fontweight='bold',
            transform=ccrs.PlateCarree(), zorder=6)

# Legend: only categories present in subdomain
legend_patches = [
    mpatches.Patch(color=lu_colors.get(i, "#888888"),
                   label=f"{i} — {lu_labels.get(i, 'Unknown')}")
    for i in present_ids
]
ax.legend(handles=legend_patches, loc='upper left',
          fontsize=8.5, framealpha=0.9,
          title="IGBP MODIS Noah", title_fontsize=9)

ax.set_title(f"Land Use Index ({luvar}) — 111 m domain", fontsize=14)
ax.set_xlabel('Longitude °', fontsize=12)
ax.set_ylabel('Latitude °',  fontsize=12)

plt.tight_layout()
plt.savefig(out_dir / "LU_index_111m_zoom.png", dpi=150, bbox_inches='tight')
plt.savefig(out_dir / "LU_index_111m_zoom.pdf", dpi=150, bbox_inches='tight')
print("Saved LU_index_111m_zoom.png / .pdf")
plt.show()
