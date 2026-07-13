import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from netCDF4 import Dataset
from pathlib import Path
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from wrf import getvar, latlon_coords, get_cartopy, to_np
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# -----------------------------
# Increase all fonts globally
# -----------------------------
plt.rcParams.update({
    'axes.titlesize': 18,       # subplot titles
    'axes.labelsize': 16,       # x/y axis labels
    'xtick.labelsize': 16,      # x-axis tick labels
    'ytick.labelsize': 16,      # y-axis tick labels
    'legend.fontsize': 12,      # legend text
    'figure.titlesize': 20      # overall figure title
})

def wrf_time_to_datetime(ds, itime):
    """
    Convert WRF Times char array to pandas.Timestamp
    """
    tchar = ds.variables["Times"][itime]   # shape (19,)
    tstr = b"".join(tchar).decode("utf-8")
    return pd.to_datetime(tstr, format="%Y-%m-%d_%H:%M:%S")

#polygon_df = pd.read_csv("/glade/work/gduine/lake_fire/LakeFireCoords3Days.csv")  # Replace with your actual filename
#polygon_lons = polygon_df['Longitude'].values
#polygon_lats = polygon_df['Latitude'].values

# === SETTINGS ===
start_time = pd.Timestamp("2024-11-06 06:00")
end_time = pd.Timestamp("2024-11-07 18:00")

#simName="fuel5_ign0945Z"

# Configuration for each simulation
simulations = {
    "111m": {
        "wrf_dir": Path("/glade/derecho/scratch/gduine/mountain_fire/111m/ifire2/"),
        "domain": "d04",
        "wind_arrow_skip": 10,
        "wind_arrow_scale": 200,
        "title": "111 m grid"
    }
}


out_dir = Path(f"./winds_hfx_111m")
out_dir.mkdir(exist_ok=True)

# Hourly wrfout files
hours = pd.date_range(start_time, end_time, freq="1H")

# Subdomain bounds
lat_min, lat_max =   34.19161,  34.39659 #34.65, 34.86
lon_min, lon_max = -119.2194,-118.8461 #-120.183, -119.90

dy = np.arange(34.2,34.4,0.05)
dx = np.arange(-119.20,-118.89,0.1)

# Plot settings
TERRAIN_CMAP = 'terrain'
OUTPUT_DPI = 150

# Station locations
stations = {
    "START": (34.318,  -118.968, "black"),
    "SPOT":  (34.2528, -119.0284, "purple"),
        }

# Process each timestep
for t in hours:
    # Dictionary to store data for each simulation
    sim_data = {}
    
    # Load data from all simulations
    for sim_name, sim_config in simulations.items():
        domain = sim_config["domain"]
        fname = sim_config["wrf_dir"] / f"wrfout_{domain}_{t.strftime('%Y-%m-%d_%H:00:00')}"
        
        if not fname.exists():
            print(f"Missing {fname}")
            sim_data[sim_name] = None
            continue
        
        ds = Dataset(fname)
        sim_data[sim_name] = {"ds": ds, "config": sim_config}
    
    # Skip if all simulations are missing
    if all(v is None for v in sim_data.values()):
        continue
    
    hgt = getvar(ds, "HGT", timeidx=0)
    cart_proj = get_cartopy(hgt)
    lats, lons = latlon_coords(hgt)

    # number of times

    n_times = max(
        d["ds"].dimensions["Time"].size
        for d in sim_data.values() if d is not None
    )


    for itime in range(n_times):
        fig, axes = plt.subplots(
                1, 1, figsize=(12, 8),
#            1, 2, figsize=(16, 8),
            subplot_kw={'projection': cart_proj}
        )

        # get timestamp from first available sim
        ts = None
        for data in sim_data.values():
            if data is not None:
                ts = wrf_time_to_datetime(data["ds"], itime)
                break

        if ts is None:
            continue

        tsPST = ts - pd.Timedelta(hours=8)
        tWRFstrPST = tsPST.strftime('%Y-%m-%d %H:%M')
        tWRFstrPST_fName = tsPST.strftime('%Y-%m-%d_%H%M')


        print(f"Working on plot {tWRFstrPST}")
        # Set common scale (0 to max value found)
        vmin, vmax = 0, 300000 #hfx_max
        
        # Plot each simulation
        for idx, (sim_name, data) in enumerate(sim_data.items()):
            ax = axes
            
            if data is None:
                ax.text(0.5, 0.5, f"Data not available\n({sim_name})", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
#                ax.set_title(simulations[sim_name]["title"])
                ax.set_title(f"\n{tWRFstrPST} PST")
                continue

            
            ds = data["ds"]
            config = data["config"]
            
            # Fire variables (fine mesh)
            hfx = ds.variables["FGRNHFX"][itime, :, :]
            flat = ds.variables["FXLAT"][0, :, :]
            flon = ds.variables["FXLONG"][0, :, :]
            

            # Mask points outside subdomain
            mask = (flat >= lat_min) & (flat <= lat_max) & (flon >= lon_min) & (flon <= lon_max)
            hfx_sub = np.where(mask, hfx, np.nan)
            flat_sub = np.where(mask, flat, np.nan)
            flon_sub = np.where(mask, flon, np.nan)
            

            # --- WRF variables ---
            u10 = getvar(ds, "U10", timeidx=itime)
            v10 = getvar(ds, "V10", timeidx=itime)
            hgt = getvar(ds, "HGT", timeidx=0)
            lats, lons = latlon_coords(hgt)



            # --- Base map ---
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
#            ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
#            ax.add_feature(cfeature.LAND, facecolor="0.9", zorder=0)
            ax.add_feature(cfeature.COASTLINE, linewidth=3, zorder=3)
#            ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=3)
            ax.set_xticks(dx, crs=ccrs.PlateCarree())      
            lon_formatter = LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.set_yticks(dy, crs=ccrs.PlateCarree())
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.tick_params(direction='out', labelsize=8.5, length=5, pad=2, color='black')    

            # --- Terrain ---
            ax.contour(
                to_np(lons), to_np(lats), to_np(hgt),
                levels=np.arange(100, 3000, 100),
                colors="k", linewidths=0.6,
                transform=ccrs.PlateCarree(), zorder=2
            )

            # --- Wind speed ---
            wspd = np.sqrt(u10**2 + v10**2)
            cf = ax.contourf(
                to_np(lons), to_np(lats), to_np(wspd),
                levels=np.arange(2, 20, 2),
                cmap="jet",
                extend="both",
                alpha=0.6,
                transform=ccrs.PlateCarree(),
                zorder=1
                )
            
#            if idx==1:
            cbar = fig.colorbar(cf, ax=ax, orientation='vertical',
                                    pad=0.05, shrink=0.75, label="Wind Speed (m/s)")
            cbar.set_label("Wind Speed (m/s)", fontsize=14)
            cbar.ax.tick_params(labelsize=12)


            # --- Wind vectors ---
            skip = config["wind_arrow_skip"]
            ax.quiver(
                to_np(lons)[::skip, ::skip],
                to_np(lats)[::skip, ::skip],
                to_np(u10)[::skip, ::skip],
                to_np(v10)[::skip, ::skip],
                transform=ccrs.PlateCarree(),
                scale=config["wind_arrow_scale"],
                width=0.003, headwidth=4, headlength=5,
                zorder=4
            )
            
            # Plot fire perimeter as contour line
            fire_perimeter = np.where(hfx > 1000, 1, 0)  # Adjust threshold as needed
            ax.contour(flon, flat, fire_perimeter, levels=[0.5], 
                      colors='red', linewidths=2.5, zorder=5,
                       transform=ccrs.PlateCarree())

            # Plot stations
       #     for name, (lat, lon) in stations.items():
       #         ax.plot(lon, lat, marker="^", markerfacecolor="none", 
       #                 markeredgecolor="white", markersize=12, linewidth=2.5,
       #                 transform=ccrs.PlateCarree())
       #         ax.text(lon+0.002, lat+0.002, name, fontsize=14, color="white", 
       #                 ha="left", va="bottom", weight="bold",
       #                 transform=ccrs.PlateCarree())
        
            for name, (lat, lon, color) in stations.items():
                ax.plot(lon, lat,
                        marker="o",
                        markerfacecolor=color,
                        markeredgecolor="black",
                        markersize=20,
                        markeredgewidth=2.0,
                        linestyle="none",
                        transform=ccrs.PlateCarree()
                )

            # plot lake fire perimeter after three days
#            ax.plot(polygon_lons, polygon_lats, color='white', linewidth=2, 
#                    transform=ccrs.PlateCarree(), zorder=10)

            # Set labels and limits
#            ax.set_title(config["title"])
            ax.set_title(f"\n{tWRFstrPST} PST")
            ax.set_xlabel('Longitude $^\circ$', fontsize=16)
            ax.set_ylabel('Latitude $^\circ$', fontsize=16)
            ax.tick_params(labelsize=14)

        # Overall title
        
        plt.tight_layout()
        plt.savefig(out_dir / f"wind_fire_comparison_v2_{tWRFstrPST_fName}_PST.png", 
                   dpi=OUTPUT_DPI, bbox_inches='tight')
        plt.close()
    

#        break # only first time step

    # Clean up
    for sim_name, data in sim_data.items():
        if data is not None:
            data["ds"].close()

print("Processing complete!")
