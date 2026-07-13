import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from wrf import getvar, latlon_coords, get_cartopy, to_np
import cartopy.feature as cfeature
import cartopy.crs as ccrs

# -----------------------------
# Font settings
# -----------------------------
plt.rcParams.update({
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 12,
    'figure.titlesize': 20
})

# -----------------------------
# Fuel category colors and labels (Anderson 13 + LANDFIRE non-burnable)
# -----------------------------
fuel_colors = {
    -9999: "#000000",
    1:  "#ffffbe",
    2:  "#ffff00",
    3:  "#e6c50b",
    4:  "#ffd37f",
    5:  "#ffaa66",
    6:  "#cdaa66",
    7:  "#897044",
    8:  "#d3ffbe",
    9:  "#70a800",
    10: "#267300",
    11: "#e8beff",
    12: "#7a8ef5",
    13: "#c500ff",
    91: "#8400a5",
    92: "#9ea1f0",
    93: "#e974ff",
    98: "#0000ff",
    99: "#bfbfbf",
}
fuel_labels = {
    1:  "Short grass",
    2:  "Timber (grass & understory)",
    3:  "Tall grass",
    4:  "Chaparral",
    5:  "Brush",
    6:  "Dormant brush / hardwood slash",
    7:  "Southern rough",
    8:  "Closed timber litter",
    9:  "Hardwood litter",
    10: "Timber (litter & understory)",
    11: "Light logging slash",
    12: "Medium logging slash",
    13: "Heavy logging slash",
    91: "Urban",
    92: "Snow / Ice",
    93: "Agriculture",
    98: "Water",
    99: "Barren",
}

fuel_ids    = sorted(fuel_colors.keys())
colors_list = [fuel_colors[i] for i in fuel_ids]
fuel_cmap   = mcolors.ListedColormap(colors_list)
fuel_to_idx = {fid: i for i, fid in enumerate(fuel_ids)}

# Colorbar: only ticks for categories that have a label
ticks_indices = [fuel_to_idx[k] for k in fuel_ids if k in fuel_labels]
tick_labels   = [fuel_labels[fuel_ids[i]] for i in ticks_indices]

# -----------------------------
# Settings
# -----------------------------
start_time = pd.Timestamp("2024-11-06 17:00") # 06:00 start of simulation
end_time   = pd.Timestamp("2024-11-07 18:00")

wrf_dir = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/ifire2/")
domain  = "d04"

out_dir = Path("./fuels_fire_winds_111m")
out_dir.mkdir(exist_ok=True)

wind_arrow_skip  = 10
wind_arrow_scale = 200

dy = np.arange(34.2,  34.4,  0.05)
dx = np.arange(-119.20, -118.89, 0.1)

lat_min, lat_max =  34.19161,  34.39659
lon_min, lon_max = -119.2194, -118.8461

OUTPUT_DPI = 150

stations = {
    "START": (34.318,  -118.968,  "black"),
    "SPOT":  (34.2528, -119.0284, "red"),
}


# -----------------------------
# Helper
# -----------------------------
def wrf_time_to_datetime(ds, itime):
    tchar = ds.variables["Times"][itime]
    tstr  = b"".join(tchar).decode("utf-8")
    return pd.to_datetime(tstr, format="%Y-%m-%d_%H:%M:%S")


# ============================================================
# STATIC: load fuel categories from the first available wrfout
# Fuel categories (NFUEL_CAT / FUEL_CAT) are static fields
# written into every wrfout; we only need to read them once.
# ============================================================
hours = pd.date_range(start_time, end_time, freq="1H")

fuel_indexed = None
flat_fuel    = None
flon_fuel    = None

print("Looking for first wrfout to load static fuel categories …")
for t in hours:
    fname0 = wrf_dir / f"wrfout_{domain}_{t.strftime('%Y-%m-%d_%H:00:00')}"
    if not fname0.exists():
        continue

    ds0 = Dataset(fname0)

    # Identify fuel variable
    fuelvar = "FUEL_CAT" if "FUEL_CAT" in ds0.variables else "NFUEL_CAT"
    fuel_vals = ds0.variables[fuelvar][0, :, :]   # shape: (fire_ny, fire_nx)

    # Fire-grid coordinates
    flat_fuel = ds0.variables["FXLAT"][0, :, :]
    flon_fuel = ds0.variables["FXLONG"][0, :, :]

    ds0.close()

    # Remap fuel IDs → colormap indices
    fuel_indexed = np.zeros_like(fuel_vals, dtype=int)
    for fid, idx in fuel_to_idx.items():
        fuel_indexed[fuel_vals == fid] = idx

    # Remove zero-coordinate padding rows/cols (WRF fire-grid artefact)
    valid_rows = np.any((flon_fuel != 0) & (flat_fuel != 0), axis=1)
    valid_cols = np.any((flon_fuel != 0) & (flat_fuel != 0), axis=0)
    flat_fuel    = flat_fuel[np.ix_(valid_rows, valid_cols)]
    flon_fuel    = flon_fuel[np.ix_(valid_rows, valid_cols)]
    fuel_indexed = fuel_indexed[np.ix_(valid_rows, valid_cols)]

    # Trim to subdomain for plotting performance
    row_ok = np.any(
        (flat_fuel >= lat_min) & (flat_fuel <= lat_max) &
        (flon_fuel >= lon_min) & (flon_fuel <= lon_max), axis=1)
    col_ok = np.any(
        (flat_fuel >= lat_min) & (flat_fuel <= lat_max) &
        (flon_fuel >= lon_min) & (flon_fuel <= lon_max), axis=0)
    flat_fuel    = flat_fuel[np.ix_(row_ok, col_ok)]
    flon_fuel    = flon_fuel[np.ix_(row_ok, col_ok)]
    fuel_indexed = fuel_indexed[np.ix_(row_ok, col_ok)]

    print(f"  Loaded {fuelvar} from {fname0.name}: "
          f"array {fuel_indexed.shape}, "
          f"lon [{flon_fuel.min():.3f}, {flon_fuel.max():.3f}], "
          f"lat [{flat_fuel.min():.3f}, {flat_fuel.max():.3f}]")
    break

if fuel_indexed is None:
    raise RuntimeError("No wrfout files found — check wrf_dir and start_time.")


# ============================================================
# TIME LOOP
# ============================================================
for t in hours:
    fname = wrf_dir / f"wrfout_{domain}_{t.strftime('%Y-%m-%d_%H:00:00')}"
    if not fname.exists():
        print(f"Missing {fname}")
        continue

    ds      = Dataset(fname)
    n_times = ds.dimensions["Time"].size

    for itime in range(1):# range(n_times):
        ts               = wrf_time_to_datetime(ds, itime)
        tsPST            = ts - pd.Timedelta(hours=8)
        tWRFstrPST       = tsPST.strftime('%Y-%m-%d %H:%M')
        tWRFstrPST_fName = tsPST.strftime('%Y-%m-%d_%H%M')

        print(f"Working on {tWRFstrPST} PST")

        # --- Met-grid variables (111 m) ---
        u10  = getvar(ds, "U10", timeidx=itime)
        v10  = getvar(ds, "V10", timeidx=itime)
        hgt  = getvar(ds, "HGT", timeidx=0)
        cart_proj = get_cartopy(hgt)
        lats, lons = latlon_coords(hgt)

        # --- Fire-grid variables ---
        hfx  = ds.variables["FGRNHFX"][itime, :, :]
        flat = ds.variables["FXLAT"][0, :, :]
        flon = ds.variables["FXLONG"][0, :, :]

        # Mask fire data outside subdomain
        fire_mask = ((flat >= lat_min) & (flat <= lat_max) &
                     (flon >= lon_min) & (flon <= lon_max))
        hfx_sub   = np.where(fire_mask, hfx, np.nan)

        # --- Figure ---
        fig, ax = plt.subplots(1, 1, figsize=(12, 8),
                               subplot_kw={'projection': cart_proj})

        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=3, zorder=3)

        ax.set_xticks(dx, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.set_yticks(dy, crs=ccrs.PlateCarree())
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.tick_params(direction='out', labelsize=8.5, length=5, pad=2, color='black')

        # --- Background: fuel categories (fire grid, static) ---
        p_fuel = ax.pcolormesh(
            flon_fuel, flat_fuel, fuel_indexed,
            cmap=fuel_cmap,
            vmin=0, vmax=len(fuel_ids) - 1,
            shading='auto',
            transform=ccrs.PlateCarree(),
            zorder=1
        )

        # --- Terrain contours (met grid) ---
        ax.contour(
            to_np(lons), to_np(lats), to_np(hgt),
            levels=np.arange(100, 3000, 100),
            colors="k", linewidths=0.6,
            transform=ccrs.PlateCarree(), zorder=2
        )

        # --- Wind vectors (met grid) ---
        ax.quiver(
            to_np(lons)[::wind_arrow_skip, ::wind_arrow_skip],
            to_np(lats)[::wind_arrow_skip, ::wind_arrow_skip],
            to_np(u10)[::wind_arrow_skip, ::wind_arrow_skip],
            to_np(v10)[::wind_arrow_skip, ::wind_arrow_skip],
            transform=ccrs.PlateCarree(),
            scale=wind_arrow_scale,
            width=0.003, headwidth=4, headlength=5,
            zorder=4
        )

        # --- Active fire front (fire grid) ---
        fire_perimeter = np.where(hfx_sub > 1000, 1, 0)
        ax.contour(
            flon, flat, fire_perimeter,
            levels=[0.5],
            colors='red', linewidths=2.5,
            transform=ccrs.PlateCarree(), zorder=5
        )

        # --- Ignition / station markers ---
        for name, (lat_s, lon_s, color) in stations.items():
            ax.plot(
                lon_s, lat_s,
                marker="*",
                markerfacecolor="none",
                markeredgecolor=color,
                markersize=20,
                markeredgewidth=2.0,
                linestyle="none",
                transform=ccrs.PlateCarree(),
                zorder=6
            )

        # --- Fuel colorbar ---
        cbar = fig.colorbar(p_fuel, ax=ax, ticks=ticks_indices,
                            orientation='vertical', pad=0.05, shrink=0.75)
        cbar.ax.set_yticklabels(tick_labels, fontsize=9)
        cbar.set_label("Fuel category", fontsize=13)

        ax.set_title(f"\n{tWRFstrPST} PST")
        ax.set_xlabel('Longitude $^\\circ$', fontsize=16)
        ax.set_ylabel('Latitude $^\\circ$',  fontsize=16)
        ax.tick_params(labelsize=14)

        plt.tight_layout()
        plt.savefig(out_dir / f"fuels_fire_winds_{tWRFstrPST_fName}_PST.png",
                    dpi=OUTPUT_DPI, bbox_inches='tight')
        plt.close()

    ds.close()

print("Processing complete!")
