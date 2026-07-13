#!/usr/bin/env python
# coding: utf-8

# In[11]:
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# directory with files
base_dir = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/ifire2/")

# define time range
start = pd.Timestamp("2024-11-06 05:00:00")
end   = pd.Timestamp("2024-11-07 07:00:00")
freq  = "1H"   # WRF files every 3 hours (adjust if needed)

# generate list of datetimes
datetimes = pd.date_range(start, end, freq=freq)

# loop over files
for dt in datetimes:
    fname = base_dir / f"wrfout_d04_{dt.strftime('%Y-%m-%d_%H:%M:%S')}"
    if not fname.exists():
        print(f"Skipping missing file: {fname}")
        continue

    print(f"Processing {fname}")
    ds = xr.open_dataset(fname)

    # find times (decoded)
    times = [t.item().decode("utf-8") for t in ds["Times"].values]

    # loop over time indices
    for itime in range(ds.dims["Time"]):
        hfx = ds["FGRNHFX"].isel(Time=itime)
        max_hfx = float(hfx.max().values)
        print(f"time {times[itime]}  max(FGRNHFX) = {max_hfx:.3f}")
        if max_hfx > 0:
            idx = np.where(hfx.values > 0)
            print("  fire present, sample coords (i,j):", list(zip(idx[0][:5], idx[1][:5])))

    # also check fire area
    fa = ds["FIRE_AREA"].isel(Time=0)
    print("max FIRE_AREA:", float(fa.max().values))

    ds.close()



import matplotlib.pyplot as plt

# ig = ds["FIRE_AREA"].isel(Time=0)
# plt.imshow(ig.values, origin="lower")
# plt.colorbar(label="FIRE_AREA")
# plt.show()

