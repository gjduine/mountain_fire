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

plt.rcParams.update({
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

# ── Settings ───────────────────────────────────────────────────────────────────
wrf_dir    = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/ifire2/")
domain     = "d04"
start_time = pd.Timestamp("2024-11-06 17:00")
end_time   = pd.Timestamp("2024-11-07 18:00")
hours      = pd.date_range(start_time, end_time, freq="1H")

lat_min, lat_max =  34.25,  34.35
lon_min, lon_max = -119.05, -118.95
dy = np.arange(34.25,  34.35,  0.02)
dx = np.arange(-119.05, -118.95, 0.03)

OUTPUT_DPI = 150
out_dir    = Path("./firefront_methods")
out_dir.mkdir(exist_ok=True)

stations = {
    "START": (34.318,  -118.968,  "black"),
    "SPOT":  (34.2528, -119.0284, "red"),
}

# ── Fuel colormap (same as other scripts) ─────────────────────────────────────
fuel_colors = {
    -9999:"#000000", 1:"#ffffbe", 2:"#ffff00", 3:"#e6c50b", 4:"#ffd37f",
    5:"#ffaa66", 6:"#cdaa66", 7:"#897044", 8:"#d3ffbe", 9:"#70a800",
    10:"#267300", 11:"#e8beff", 12:"#7a8ef5", 13:"#c500ff",
    91:"#8400a5", 92:"#9ea1f0", 93:"#e974ff", 98:"#0000ff", 99:"#bfbfbf",
}
fuel_ids    = sorted(fuel_colors.keys())
fuel_cmap   = mcolors.ListedColormap([fuel_colors[i] for i in fuel_ids])
fuel_to_idx = {fid: i for i, fid in enumerate(fuel_ids)}

# ── Helper ─────────────────────────────────────────────────────────────────────
def wrf_time_to_datetime(ds, itime):
    tchar = ds.variables["Times"][itime]
    return pd.to_datetime(b"".join(tchar).decode("utf-8"), format="%Y-%m-%d_%H:%M:%S")


# ── Static: load fuels + fire-grid coords from first available wrfout ──────────
fuel_indexed = flat_fuel = flon_fuel = None

for t in hours:
    fname0 = wrf_dir / f"wrfout_{domain}_{t.strftime('%Y-%m-%d_%H:00:00')}"
    if not fname0.exists():
        continue
    ds0     = Dataset(fname0)
    fuelvar = "FUEL_CAT" if "FUEL_CAT" in ds0.variables else "NFUEL_CAT"
    fuel_vals = ds0.variables[fuelvar][0, :, :]
    flat_fuel = ds0.variables["FXLAT"][0, :, :]
    flon_fuel = ds0.variables["FXLONG"][0, :, :]

    # Check which fire-front variables are available
    available_vars = set(ds0.variables.keys())
    ds0.close()

    fuel_indexed = np.zeros_like(fuel_vals, dtype=int)
    for fid, idx in fuel_to_idx.items():
        fuel_indexed[fuel_vals == fid] = idx

    valid_rows = np.any((flon_fuel != 0) & (flat_fuel != 0), axis=1)
    valid_cols = np.any((flon_fuel != 0) & (flat_fuel != 0), axis=0)
    flat_fuel    = flat_fuel[np.ix_(valid_rows, valid_cols)]
    flon_fuel    = flon_fuel[np.ix_(valid_rows, valid_cols)]
    fuel_indexed = fuel_indexed[np.ix_(valid_rows, valid_cols)]

    row_ok = np.any((flat_fuel >= lat_min) & (flat_fuel <= lat_max) &
                    (flon_fuel >= lon_min) & (flon_fuel <= lon_max), axis=1)
    col_ok = np.any((flat_fuel >= lat_min) & (flat_fuel <= lat_max) &
                    (flon_fuel >= lon_min) & (flon_fuel <= lon_max), axis=0)
    flat_fuel    = flat_fuel[np.ix_(row_ok, col_ok)]
    flon_fuel    = flon_fuel[np.ix_(row_ok, col_ok)]
    fuel_indexed = fuel_indexed[np.ix_(row_ok, col_ok)]
    break

if fuel_indexed is None:
    raise RuntimeError("No wrfout files found.")

# ── Check whether NFUEL_CAT changes over time ──────────────────────────────────
all_files = sorted(wrf_dir.glob(f"wrfout_{domain}_*"))
if len(all_files) >= 2:
    with Dataset(all_files[0]) as ds_first, Dataset(all_files[-1]) as ds_last:
        fuelvar   = "FUEL_CAT" if "FUEL_CAT" in ds_first.variables else "NFUEL_CAT"
        nfuel_t0  = ds_first.variables[fuelvar][0, :, :]
        nfuel_t1  = ds_last.variables[fuelvar][0, :, :]
        n_changed = int(np.sum(nfuel_t0 != nfuel_t1))
        if n_changed == 0:
            print(f"NFUEL_CAT is STATIC — identical in first and last file ({all_files[0].name} vs {all_files[-1].name})")
        else:
            print(f"NFUEL_CAT CHANGED in {n_changed} cells between first and last file!")
            print(f"  → Could be used as a burned-area indicator.")
else:
    print("Not enough files to compare NFUEL_CAT over time.")

# ── Define fire-front methods ──────────────────────────────────────────────────
# Each method: variable name, how to derive a 2D mask/field, contour spec
# Availability is checked at runtime against available_vars
all_methods = [
    {
        "name":    "FGRNHFX > 1000",
        "var":     "FGRNHFX",
        "label":   "Ground HFX\n(> 1000 W m⁻²)",
        "color":   "red",
        "lw":      2.5,
    },
    {
        "name":    "FIRE_AREA > 0.5",
        "var":     "FIRE_AREA",
        "label":   "Fire area\n(burned fraction > 0.5)",
        "color":   "orange",
        "lw":      2.5,
    },
    {
        "name":    "LFN = 0",
        "var":     "LFN",
        "label":   "Level set (LFN = 0)\nfire front",
        "color":   "blue",
        "lw":      2.5,
    },
]

# Keep only methods whose variable exists in the wrfout
methods = [m for m in all_methods if m["var"] in available_vars]
print(f"Available fire-front variables: {[m['var'] for m in methods]}")
if not methods:
    raise RuntimeError("None of the expected fire variables found in wrfout.")

n_methods = len(methods)


# ── Helper: derive binary mask or contour field from a method ─────────────────
def get_fire_field(ds, itime, method):
    var  = ds.variables[method["var"]][itime, :, :]
    name = method["name"]
    if "FGRNHFX" in name:
        return var, [0.5], np.where(var > 1000, 1, 0)   # binary for contour
    elif "FIRE_AREA" in name:
        return var, [0.5], var                            # contour at 0.5
    elif "LFN" in name:
        return var, [0.0], var                            # contour at 0
    return var, [0.5], var


# ── Time loop ──────────────────────────────────────────────────────────────────
for t in hours:
    fname = wrf_dir / f"wrfout_{domain}_{t.strftime('%Y-%m-%d_%H:00:00')}"
    if not fname.exists():
        print(f"Missing {fname}")
        continue

    ds      = Dataset(fname)
    n_times = ds.dimensions["Time"].size

    for itime in range(1):  # change to range(n_times) for all timesteps
        ts               = wrf_time_to_datetime(ds, itime)
        tsPST            = ts - pd.Timedelta(hours=8)
        tWRFstrPST       = tsPST.strftime('%Y-%m-%d %H:%M')
        tWRFstrPST_fName = tsPST.strftime('%Y-%m-%d_%H%M')
        print(f"Working on {tWRFstrPST} PST")

        # Met grid
        hgt  = getvar(ds, "HGT", timeidx=0)
        cart_proj = get_cartopy(hgt)
        lats, lons = latlon_coords(hgt)

        # Fire grid coordinates (full extent, for contour)
        flat_all = ds.variables["FXLAT"][0, :, :]
        flon_all = ds.variables["FXLONG"][0, :, :]

        # ── Figure: 1 row × n_methods cols ────────────────────────────────────
        fig, axes = plt.subplots(1, n_methods,
                                 figsize=(7 * n_methods, 7),
                                 subplot_kw={'projection': cart_proj})
        if n_methods == 1:
            axes = [axes]

        for ax, method in zip(axes, methods):
            _, levels, field = get_fire_field(ds, itime, method)

            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=2, zorder=3)
            ax.set_xticks(dx, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.set_yticks(dy, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(direction='out', labelsize=9, length=4, pad=2)

            # Fuel background
            ax.pcolormesh(flon_fuel, flat_fuel, fuel_indexed,
                          cmap=fuel_cmap, vmin=0, vmax=len(fuel_ids) - 1,
                          shading='auto', transform=ccrs.PlateCarree(), zorder=1)

            # Terrain contours
            ax.contour(to_np(lons), to_np(lats), to_np(hgt),
                       levels=np.arange(100, 3000, 100),
                       colors='k', linewidths=0.5, alpha=0.5,
                       transform=ccrs.PlateCarree(), zorder=2)

            # Fire front contour
            ax.contour(flon_all, flat_all, field,
                       levels=levels,
                       colors=method["color"], linewidths=method["lw"],
                       transform=ccrs.PlateCarree(), zorder=5)

            # Station markers
            for name, (lat_s, lon_s, color) in stations.items():
                ax.plot(lon_s, lat_s, marker="*",
                        markerfacecolor="none", markeredgecolor=color,
                        markersize=18, markeredgewidth=2,
                        linestyle="none",
                        transform=ccrs.PlateCarree(), zorder=6)

            ax.set_title(f"{method['label']}\n{tWRFstrPST} PST", fontsize=13)
            ax.set_xlabel('Longitude °', fontsize=11)

        axes[0].set_ylabel('Latitude °', fontsize=11)
        for ax in axes[1:]:
            ax.set_yticklabels([])

        plt.subplots_adjust(wspace=0.05, bottom=0.05)
        plt.savefig(out_dir / f"firefront_methods_{tWRFstrPST_fName}_PST.png",
                    dpi=OUTPUT_DPI, bbox_inches='tight')
        plt.close()

    ds.close()

print("Processing complete!")
