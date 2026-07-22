import matplotlib.pyplot as plt
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
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'figure.titlesize': 18,
})

# ── Simulations ────────────────────────────────────────────────────────────────
base = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/")
sims = {
    "ref":       base / "ifire0/ref/",
    "z0_double": base / "ifire0/z0_double/",
    "ifire2":    base / "ifire2/",
}
domain = "d04"

# Pairs: (deviation sim, reference sim, panel title)
pairs = [
    ("z0_double", "ref", "z0 double − ref"),
    ("ifire2",    "ref", "Fire − ref"),
]

# ── Settings ───────────────────────────────────────────────────────────────────
start_time = pd.Timestamp("2024-11-06 05:00")
end_time   = pd.Timestamp("2024-11-07 18:00")
hours      = pd.date_range(start_time, end_time, freq="1H")

lat_min, lat_max =  34.25,  34.35
lon_min, lon_max = -119.05, -118.95
dy = np.arange(34.25,  34.35,  0.02)
dx = np.arange(-119.05, -118.95, 0.03)

wind_arrow_skip  = 10
wind_arrow_scale = 200
OUTPUT_DPI       = 150

# Diverging levels for wind speed difference
diff_levels = np.arange(-6, 6.5, 0.5)
diff_cmap   = "RdBu_r"   # red = faster than ref, blue = slower

stations = {
    "START": (34.318,  -118.968,  "black"),
    "SPOT":  (34.2528, -119.0284, "red"),
}

out_dir = Path("./wind_diff_111m")
out_dir.mkdir(exist_ok=True)


# ── Helper ─────────────────────────────────────────────────────────────────────
def wrf_time_to_datetime(ds, itime):
    tchar = ds.variables["Times"][itime]
    tstr  = b"".join(tchar).decode("utf-8")
    return pd.to_datetime(tstr, format="%Y-%m-%d_%H:%M:%S")


# ── Time loop ──────────────────────────────────────────────────────────────────
for t in hours:
    fnames = {k: sims[k] / f"wrfout_{domain}_{t.strftime('%Y-%m-%d_%H:00:00')}"
              for k in sims}

    missing = [str(f) for f in fnames.values() if not f.exists()]
    if missing:
        print(f"Skipping {t} — missing: {missing}")
        continue

    dsets   = {k: Dataset(fnames[k]) for k in sims}
    n_times = dsets["ref"].dimensions["Time"].size

    for itime in range(1):# range(n_times):
        ts               = wrf_time_to_datetime(dsets["ref"], itime)
        tsPST            = ts - pd.Timedelta(hours=8)
        tWRFstrPST       = tsPST.strftime('%Y-%m-%d %H:%M')
        tWRFstrPST_fName = tsPST.strftime('%Y-%m-%d_%H%M')
        print(f"Working on {tWRFstrPST} PST")

        # --- Load met-grid fields for all three sims ---
        fields = {}
        for k, ds in dsets.items():
            u10  = getvar(ds, "U10", timeidx=itime)
            v10  = getvar(ds, "V10", timeidx=itime)
            fields[k] = {
                "wspd": np.sqrt(to_np(u10)**2 + to_np(v10)**2),
                "u10":  to_np(u10),
                "v10":  to_np(v10),
            }

        # Static met grid (from ref, same for all 111m sims)
        hgt  = getvar(dsets["ref"], "HGT", timeidx=0)
        cart_proj = get_cartopy(hgt)
        lats, lons = latlon_coords(hgt)

        # Fire front from ifire2
        hfx_fire  = dsets["ifire2"].variables["FGRNHFX"][itime, :, :]
        flat_fire  = dsets["ifire2"].variables["FXLAT"][0, :, :]
        flon_fire  = dsets["ifire2"].variables["FXLONG"][0, :, :]
        fire_mask  = ((flat_fire >= lat_min) & (flat_fire <= lat_max) &
                      (flon_fire >= lon_min) & (flon_fire <= lon_max))
        hfx_sub    = np.where(fire_mask, hfx_fire, np.nan)
        fire_perim = np.where(hfx_sub > 1000, 1, 0)

        # --- Figure: 1 row × 2 cols ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 8),
                                 subplot_kw={'projection': cart_proj})

        for ax, (sim_key, ref_key, title) in zip(axes, pairs):
            wspd_diff = fields[sim_key]["wspd"] - fields[ref_key]["wspd"]

            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.COASTLINE, linewidth=2, zorder=3)
            ax.set_xticks(dx, crs=ccrs.PlateCarree())
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.set_yticks(dy, crs=ccrs.PlateCarree())
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.tick_params(direction='out', labelsize=10, length=5, pad=2)

            # Wind speed difference (diverging)
            cf = ax.contourf(
                to_np(lons), to_np(lats), wspd_diff,
                levels=diff_levels, cmap=diff_cmap, extend='both',
                transform=ccrs.PlateCarree(), zorder=1
            )

            # Terrain contours
            ax.contour(
                to_np(lons), to_np(lats), to_np(hgt),
                levels=np.arange(100, 3000, 100),
                colors='gray', linewidths=0.5, alpha=0.7,
                transform=ccrs.PlateCarree(), zorder=2
            )

            # Reference wind vectors for orientation
            ax.quiver(
                to_np(lons)[::wind_arrow_skip, ::wind_arrow_skip],
                to_np(lats)[::wind_arrow_skip, ::wind_arrow_skip],
                fields[ref_key]["u10"][::wind_arrow_skip, ::wind_arrow_skip],
                fields[ref_key]["v10"][::wind_arrow_skip, ::wind_arrow_skip],
                transform=ccrs.PlateCarree(),
                scale=wind_arrow_scale,
                width=0.003, headwidth=4, headlength=5,
                color='black', alpha=0.7, zorder=4
            )

            # Fire front (ifire2 only on the right panel)
            if sim_key == "ifire2":
                ax.contour(
                    flon_fire, flat_fire, fire_perim,
                    levels=[0.5], colors='red', linewidths=2,
                    transform=ccrs.PlateCarree(), zorder=5
                )

            # Station markers
            for name, (lat_s, lon_s, color) in stations.items():
                ax.plot(lon_s, lat_s, marker="*",
                        markerfacecolor="none", markeredgecolor=color,
                        markersize=18, markeredgewidth=2,
                        linestyle="none",
                        transform=ccrs.PlateCarree(), zorder=6)

            ax.set_title(f"{title}\n{tWRFstrPST} PST", fontsize=15)
            ax.set_xlabel('Longitude °', fontsize=13)

        axes[0].set_ylabel('Latitude °', fontsize=13)
        axes[1].set_yticklabels([])

        plt.subplots_adjust(wspace=0.02, bottom=0.15)

        # Shared colorbar placed manually below both panels
        cax  = fig.add_axes([0.25, 0.06, 0.50, 0.025])
        cbar = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='both')
        cbar.set_label('Wind speed difference [m s$^{-1}$]', fontsize=13)
        cbar.ax.tick_params(labelsize=12)
        plt.savefig(out_dir / f"windspeed_diff_{tWRFstrPST_fName}_PST.png",
                    dpi=OUTPUT_DPI, bbox_inches='tight')
        plt.close()

    for ds in dsets.values():
        ds.close()

print("Processing complete!")
