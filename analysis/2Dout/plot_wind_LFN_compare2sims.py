import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import json
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

# ── Simulation definitions ─────────────────────────────────────────────────────
base = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/")
simulations = [
    {"label": "ifire2 / no roads / ref",       "dir": base / "ifire2_noroad/ref/",       "color": "red",  "lw": 2.5, "ls": "-"},
    {"label": "ifire2 / no roads / oak break", "dir": base / "ifire2_oak_break/", "color": "blue", "lw": 1.5, "ls": "-"},
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

wind_arrow_skip  = 10
wind_arrow_scale = 200
OUTPUT_DPI       = 150

# Wind speed colormap (left panel)
wspd_levels = np.arange(0, 22, 2)
wspd_cmap   = "YlOrRd"

# Wind speed difference colormap (right panel)
diff_levels = np.arange(-6, 6.5, 0.5)
diff_cmap   = "RdBu_r"

# Fire perimeter GeoJSON
geojson_path = Path("/glade/work/gduine/mountain_fire/perimeter/mountain_fire_perimeter.geojson")

# Station markers
stations = {
    "START":          (34.318,    -118.968,   "magenta"),
    "Spot Valley":    (34.271033, -119.015757, "red"),
    "Spot Camarillo": (34.256678, -119.032321, "limegreen"),
}

out_dir = Path("./wind_LFN_compare")
out_dir.mkdir(exist_ok=True)

# ── Observed fire perimeter ────────────────────────────────────────────────────
fire_polygons = []
if geojson_path.exists():
    with open(geojson_path) as f:
        gj = json.load(f)
    for feat in gj["features"]:
        coords = feat["geometry"]["coordinates"][0]
        fire_polygons.append(coords)
    print(f"Loaded {len(fire_polygons)} fire perimeter polygons")
else:
    print(f"Fire perimeter GeoJSON not found: {geojson_path}")

# ── Helper ─────────────────────────────────────────────────────────────────────
def wrf_time_to_datetime(ds, itime):
    tchar = ds.variables["Times"][itime]
    return pd.to_datetime(b"".join(tchar).decode("utf-8"), format="%Y-%m-%d_%H:%M:%S")

# ── Time loop ──────────────────────────────────────────────────────────────────
for t in hours:
    fnames = {sim["label"]: sim["dir"] / f"wrfout_{domain}_{t.strftime('%Y-%m-%d_%H:00:00')}"
              for sim in simulations}

    missing = [str(f) for f in fnames.values() if not f.exists()]
    if missing:
        print(f"Skipping {t} — missing: {missing}")
        continue

    dsets   = {sim["label"]: Dataset(fnames[sim["label"]]) for sim in simulations}
    n_times = dsets[simulations[0]["label"]].dimensions["Time"].size

    for itime in range(1): # range(n_times):
        ts               = wrf_time_to_datetime(dsets[simulations[0]["label"]], itime)
        tsPST            = ts - pd.Timedelta(hours=8)
        tWRFstrPST       = tsPST.strftime('%Y-%m-%d %H:%M')
        tWRFstrPST_fName = tsPST.strftime('%Y-%m-%d_%H%M')
        print(f"Working on {tWRFstrPST} PST")

        # Met grid (same for both sims)
        hgt       = getvar(dsets[simulations[0]["label"]], "HGT", timeidx=0)
        cart_proj = get_cartopy(hgt)
        lats, lons = latlon_coords(hgt)

        # Wind fields
        u10_ref  = to_np(getvar(dsets[simulations[0]["label"]], "U10", timeidx=itime))
        v10_ref  = to_np(getvar(dsets[simulations[0]["label"]], "V10", timeidx=itime))
        u10_z0   = to_np(getvar(dsets[simulations[1]["label"]], "U10", timeidx=itime))
        v10_z0   = to_np(getvar(dsets[simulations[1]["label"]], "V10", timeidx=itime))

        wspd_ref  = np.sqrt(u10_ref**2 + v10_ref**2)
        wspd_diff = np.sqrt(u10_z0**2  + v10_z0**2) - wspd_ref

        # ── Figure ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 8),
                                 subplot_kw={'projection': cart_proj})

        for ax in axes:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=2, zorder=3)
            ax.set_xticks(dx, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.set_yticks(dy, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(direction='out', labelsize=10, length=5, pad=2)

            # Terrain contours
            ax.contour(to_np(lons), to_np(lats), to_np(hgt),
                       levels=np.arange(100, 3000, 100),
                       colors='gray', linewidths=0.5, alpha=0.7,
                       transform=ccrs.PlateCarree(), zorder=2)

        # Left: wind speed (ref)
        cf_wspd = axes[0].contourf(
            to_np(lons), to_np(lats), wspd_ref,
            levels=wspd_levels, cmap=wspd_cmap, extend='max',
            transform=ccrs.PlateCarree(), zorder=1
        )
        axes[0].set_title(f"Wind speed — ifire2/ref\n{tWRFstrPST} PST", fontsize=13)

        # Right: wind speed difference (oak break − ref)
        cf_diff = axes[1].contourf(
            to_np(lons), to_np(lats), wspd_diff,
            levels=diff_levels, cmap=diff_cmap, extend='both',
            transform=ccrs.PlateCarree(), zorder=1
        )
        axes[1].set_title(f"Wind speed difference (oak fuel break − ref)\n{tWRFstrPST} PST", fontsize=13)

        # Wind vectors (ref) on both panels
        sk = wind_arrow_skip
        for ax in axes:
            ax.quiver(
                to_np(lons)[::sk, ::sk], to_np(lats)[::sk, ::sk],
                u10_ref[::sk, ::sk],     v10_ref[::sk, ::sk],
                transform=ccrs.PlateCarree(),
                scale=wind_arrow_scale,
                width=0.003, headwidth=4, headlength=5,
                color='black', alpha=0.7, zorder=4
            )

        # LFN=0 contours from both sims on both panels
        for ax in axes:
            for sim in simulations:
                ds = dsets[sim["label"]]
                if "LFN" not in ds.variables:
                    continue
                lfn      = ds.variables["LFN"][itime, :, :]
                flat_all = ds.variables["FXLAT"][0, :, :]
                flon_all = ds.variables["FXLONG"][0, :, :]
                ax.contour(flon_all, flat_all, lfn,
                           levels=[0.0],
                           colors=sim["color"], linewidths=sim["lw"],
                           linestyles=sim["ls"],
                           transform=ccrs.PlateCarree(), zorder=5)

        # Observed fire perimeter on both panels
        for ax in axes:
            for poly in fire_polygons:
                lons_p = [c[0] for c in poly]
                lats_p = [c[1] for c in poly]
                ax.plot(lons_p, lats_p, color='limegreen', lw=2,
                        transform=ccrs.PlateCarree(), zorder=6)

        # Station markers on both panels
        for ax in axes:
            for name, (lat_s, lon_s, color) in stations.items():
                ax.plot(lon_s, lat_s, marker="*",
                        markerfacecolor="none", markeredgecolor=color,
                        markersize=18, markeredgewidth=2,
                        linestyle="none",
                        path_effects=[pe.withStroke(linewidth=4, foreground='white')],
                        transform=ccrs.PlateCarree(), zorder=7)

        # Axis labels
        axes[0].set_ylabel('Latitude °', fontsize=12)
        axes[1].set_yticklabels([])
        for ax in axes:
            ax.set_xlabel('Longitude °', fontsize=12)

        # Legend (left panel)
        legend_handles = [
            mlines.Line2D([], [], color=sim["color"], lw=sim["lw"],
                          ls=sim["ls"], label=f"LFN=0  {sim['label']}")
            for sim in simulations
        ]
        if fire_polygons:
            legend_handles.append(
                mlines.Line2D([], [], color='limegreen', lw=2, label='Observed perimeter')
            )
        axes[0].legend(handles=legend_handles, loc='upper left',
                       fontsize=10, framealpha=0.9)

        # Colorbars
        plt.subplots_adjust(wspace=0.05, bottom=0.15)

        cax_wspd = fig.add_axes([0.07, 0.06, 0.40, 0.025])
        cbar_wspd = fig.colorbar(cf_wspd, cax=cax_wspd, orientation='horizontal', extend='max')
        cbar_wspd.set_label('Wind speed [m s$^{-1}$]', fontsize=11)
        cbar_wspd.ax.tick_params(labelsize=10)

        cax_diff = fig.add_axes([0.53, 0.06, 0.40, 0.025])
        cbar_diff = fig.colorbar(cf_diff, cax=cax_diff, orientation='horizontal', extend='both')
        cbar_diff.set_label('Wind speed difference [m s$^{-1}$]', fontsize=11)
        cbar_diff.ax.tick_params(labelsize=10)

        plt.savefig(out_dir / f"wind_LFN_compare_{tWRFstrPST_fName}_PST.png",
                    dpi=OUTPUT_DPI, bbox_inches='tight')
        plt.close()
        print(f"  Saved {tWRFstrPST_fName}")

    for ds in dsets.values():
        ds.close()

print("Processing complete!")
