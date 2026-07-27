import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from pathlib import Path
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from wrf import getvar, latlon_coords, get_cartopy, to_np
import cartopy.feature as cfeature
import cartopy.crs as ccrs

plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

# ── Simulation definitions ─────────────────────────────────────────────────────
base = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/")
simulations = [
    {"label": "ifire2 / ref",       "dir": base / "ifire2/ref/",       "color": "red",  "lw": 2.5},
    {"label": "ifire2 / z0 double", "dir": base / "ifire2/z0_double/", "color": "blue", "lw": 2.5},
]
domain = "d04"

# ── Settings ───────────────────────────────────────────────────────────────────
start_time = pd.Timestamp("2024-11-06 17:00")
end_time   = pd.Timestamp("2024-11-07 18:00")
hours      = pd.date_range(start_time, end_time, freq="1H")

lat_min, lat_max =  34.25,  34.35
lon_min, lon_max = -119.05, -118.95
dy = np.arange(34.25,  34.35,  0.02)
dx = np.arange(-119.05, -118.95, 0.03)

OUTPUT_DPI       = 150
wind_arrow_skip  = 10
wind_arrow_scale = 200
out_dir    = Path("./LFN_compare")
out_dir.mkdir(exist_ok=True)

stations = {
    "START":       (34.318,    -118.968,   "black"),
    "SPOT":        (34.2528,   -119.0284,  "red"),
    "Spot Valley": (34.281191, -119.015999,"purple"),
}

# ── Fuel colormap ──────────────────────────────────────────────────────────────
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


# ── Static: load fuels + fire-grid coords from first sim, first file ───────────
fuel_indexed = flat_fuel = flon_fuel = None

for t in hours:
    fname0 = simulations[0]["dir"] / f"wrfout_{domain}_{t.strftime('%Y-%m-%d_%H:00:00')}"
    if not fname0.exists():
        continue
    ds0     = Dataset(fname0)
    fuelvar = "FUEL_CAT" if "FUEL_CAT" in ds0.variables else "NFUEL_CAT"
    fuel_vals = ds0.variables[fuelvar][0, :, :]
    flat_fuel = ds0.variables["FXLAT"][0, :, :]
    flon_fuel = ds0.variables["FXLONG"][0, :, :]
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
    print(f"Fuels loaded from {fname0.name}")
    break

if fuel_indexed is None:
    raise RuntimeError("No wrfout files found.")

# ── Time loop ──────────────────────────────────────────────────────────────────
for t in hours:
    fnames = {sim["label"]: sim["dir"] / f"wrfout_{domain}_{t.strftime('%Y-%m-%d_%H:00:00')}"
              for sim in simulations}

    missing = [str(f) for f in fnames.values() if not f.exists()]
    if missing:
        print(f"Skipping {t} — missing: {missing}")
        continue

    dsets = {sim["label"]: Dataset(fnames[sim["label"]]) for sim in simulations}
    n_times = dsets[simulations[0]["label"]].dimensions["Time"].size

    for itime in range(1):  # change to range(n_times) for all timesteps
        ts               = wrf_time_to_datetime(dsets[simulations[0]["label"]], itime)
        tsPST            = ts - pd.Timedelta(hours=8)
        tWRFstrPST       = tsPST.strftime('%Y-%m-%d %H:%M')
        tWRFstrPST_fName = tsPST.strftime('%Y-%m-%d_%H%M')
        print(f"Working on {tWRFstrPST} PST")

        # Met grid (from first sim — same for both)
        hgt  = getvar(dsets[simulations[0]["label"]], "HGT", timeidx=0)
        cart_proj = get_cartopy(hgt)
        lats, lons = latlon_coords(hgt)
        u10 = to_np(getvar(dsets[simulations[0]["label"]], "U10", timeidx=itime))
        v10 = to_np(getvar(dsets[simulations[0]["label"]], "V10", timeidx=itime))

        # ── Figure ────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=(10, 8),
                               subplot_kw={'projection': cart_proj})

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

        # Wind arrows (from first sim)
        ax.quiver(
            to_np(lons)[::wind_arrow_skip, ::wind_arrow_skip],
            to_np(lats)[::wind_arrow_skip, ::wind_arrow_skip],
            u10[::wind_arrow_skip, ::wind_arrow_skip],
            v10[::wind_arrow_skip, ::wind_arrow_skip],
            transform=ccrs.PlateCarree(),
            scale=wind_arrow_scale,
            width=0.003, headwidth=4, headlength=5,
            color='black', alpha=0.7, zorder=3
        )

        # LFN=0 contour for each simulation
        for sim in simulations:
            ds = dsets[sim["label"]]
            if "LFN" not in ds.variables:
                print(f"  LFN not found in {sim['label']} — skipping")
                continue
            lfn      = ds.variables["LFN"][itime, :, :]
            flat_all = ds.variables["FXLAT"][0, :, :]
            flon_all = ds.variables["FXLONG"][0, :, :]
            ax.contour(flon_all, flat_all, lfn,
                       levels=[0.0],
                       colors=sim["color"], linewidths=sim["lw"],
                       transform=ccrs.PlateCarree(), zorder=5)

        # Station markers
        for name, (lat_s, lon_s, color) in stations.items():
            ax.plot(lon_s, lat_s, marker="*",
                    markerfacecolor="none", markeredgecolor=color,
                    markersize=18, markeredgewidth=2,
                    linestyle="none",
                    transform=ccrs.PlateCarree(), zorder=6)

        # Legend for simulation lines
        legend_patches = [
            mpatches.Patch(color=sim["color"], label=sim["label"])
            for sim in simulations
        ]
        ax.legend(handles=legend_patches, loc='upper left',
                  fontsize=11, framealpha=0.9, title="LFN = 0 contour")

        ax.set_title(f"Fire front (LFN = 0) — {tWRFstrPST} PST", fontsize=14)
        ax.set_xlabel('Longitude °', fontsize=12)
        ax.set_ylabel('Latitude °',  fontsize=12)

        plt.tight_layout()
        plt.savefig(out_dir / f"LFN_compare_{tWRFstrPST_fName}_PST.png",
                    dpi=OUTPUT_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved {tWRFstrPST_fName}")

    for ds in dsets.values():
        ds.close()

print("Processing complete!")
