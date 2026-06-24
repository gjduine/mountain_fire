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
start_time = pd.Timestamp("2024-11-05 18:00")
end_time = pd.Timestamp("2024-11-09 00:00")

#simName="fuel5_ign0945Z"

# Configuration for each simulation
simulations = {
    "1km": {
        "wrf_dir": Path("/glade/derecho/scratch/gduine/mountain_fire/1km/"),
        "domain": "d03",
        "wind_arrow_skip": 10,
        "wind_arrow_scale": 200,
        "title": "1 km grid"
    }
}


out_dir = Path(f"./winds_1km")
out_dir.mkdir(exist_ok=True)

# Hourly wrfout files
hours = pd.date_range(start_time, end_time, freq="1H")

# Subdomain bounds
lat_min, lat_max =   33.21,  35.3 #34.65, 34.86
lon_min, lon_max = -120.3,-117.45 #-120.183, -119.90

dy = np.arange(33.3,35.3,0.2)
dx = np.arange(-120.0,-117.5,0.5)

# Plot settings
TERRAIN_CMAP = 'terrain'
OUTPUT_DPI = 150

# Station locations
stations = {
    "START": (34.318,  -118.968, "none"),
}

# Cross-section lines through START point
cs_center_lat =  34.318
cs_center_lon = -118.968
cross_sections = {
    "35deg_long":  {"angle": 35, "half_length_km": 70, "color": "black", "linestyle": "-",  "linewidth": 2.5},
    "35deg_short": {"angle": 35, "half_length_km": 20, "color": "red", "linestyle": "--", "linewidth": 2.0},
    "50deg_long":  {"angle": 50, "half_length_km": 70, "color": "black",   "linestyle": "-",  "linewidth": 2.5},
    "50deg_short": {"angle": 50, "half_length_km": 20, "color": "red",   "linestyle": "--", "linewidth": 2.0},
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

        tsPDT = ts - pd.Timedelta(hours=7)
        tWRFstrPDT = tsPDT.strftime('%Y-%m-%d %H:%M')
        tWRFstrPDT_fName = tsPDT.strftime('%Y-%m-%d_%H%M')


        print(f"Working on plot {tWRFstrPDT}")
        
        # Plot each simulation
        for idx, (sim_name, data) in enumerate(sim_data.items()):
            ax = axes
            
            if data is None:
                ax.text(0.5, 0.5, f"Data not available\n({sim_name})", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=14)
#                ax.set_title(simulations[sim_name]["title"])
                ax.set_title(f"\n{tWRFstrPDT} PDT")
                continue

            
            ds = data["ds"]
            config = data["config"]
            
            

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
                levels=np.arange(100, 3000, 200),
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
            
            for name, (lat, lon, color) in stations.items():
                 ax.plot(lon, lat,
                         marker="*",
                         markerfacecolor=color,
                         markeredgecolor="red",
                         markersize=20,
                         markeredgewidth=2.0,
                         linestyle="none",
                         transform=ccrs.PlateCarree()
                 )

            # --- Cross-section angle lines through START point ---
            for cs in cross_sections.values():
                ang_rad = np.radians(cs["angle"])
                half_km = cs["half_length_km"]
                half_km_label = half_km*2
                d_lat = half_km * np.cos(ang_rad) / 111.0
                d_lon = half_km * np.sin(ang_rad) / (111.0 * np.cos(np.radians(cs_center_lat)))
                ax.plot([cs_center_lon - d_lon, cs_center_lon + d_lon],
                        [cs_center_lat - d_lat, cs_center_lat + d_lat],
                        color=cs["color"], linewidth=cs["linewidth"], linestyle=cs["linestyle"],
                        transform=ccrs.PlateCarree(), zorder=5,
                        label=f'{cs["angle"]}° from N ({half_km_label} km)')
            ax.legend(fontsize=12, loc='lower right')

            # plot lake fire perimeter after three days
#            ax.plot(polygon_lons, polygon_lats, color='white', linewidth=2,
#                    transform=ccrs.PlateCarree(), zorder=10)

            # Set labels and limits
#            ax.set_title(config["title"])
            ax.set_title(f"\n{tWRFstrPDT} PDT")
            ax.set_xlabel('Longitude', fontsize=16)
            ax.set_ylabel('Latitude', fontsize=16)
            ax.tick_params(labelsize=14)

        # Overall title
        plt.tight_layout()
        plt.savefig(out_dir / f"wind_1km_{tWRFstrPDT_fName}_PDT.png", 
                   dpi=OUTPUT_DPI, bbox_inches='tight')
        plt.close()
    

#        break # only first time step

    # Clean up
    for sim_name, data in sim_data.items():
        if data is not None:
            data["ds"].close()

print("Processing complete!")
