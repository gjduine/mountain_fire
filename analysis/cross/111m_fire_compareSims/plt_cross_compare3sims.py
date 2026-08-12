#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob
import os
import wrf
from netCDF4 import Dataset
from shapely.geometry import LineString, Point


# In[8]:


# colormap definition with white in the middle
n = 39
x = 0.5
lower = plt.cm.seismic_r(np.linspace(0, x, n))
white = plt.cm.seismic_r(np.ones(80 - 2*n) * x)
upper = plt.cm.seismic_r(np.linspace(1-x, 1, n))
tmap  = LinearSegmentedColormap.from_list('map_white', np.vstack((lower, white, upper)))


# In[9]:


# ── Simulation definitions (order = top to bottom row) ────────────────────────
base = '/glade/derecho/scratch/gduine/mountain_fire/111m/'
simulations = [
    {'label': 'Fire (cntl)',     'dir': base + 'ifire2/ref/'},
    {'label': 'Fire (z0 double)','dir': base + 'ifire2/z0_double/'},
    {'label': 'Fire (oak break)','dir': base + 'ifire2_oak_break/'},
]
domain = 'd04'

# ── Cross-section geometry ─────────────────────────────────────────────────────
center_lat       = 34.318
center_lon       = -118.968
angle_from_north = 35.0
half_length_km   = 10.0

angle_rad = np.radians(angle_from_north)
delta_lat = half_length_km * np.cos(angle_rad) / 111.0
delta_lon = half_length_km * np.sin(angle_rad) / (111.0 * np.cos(np.radians(center_lat)))
start_lat, start_lon = center_lat - delta_lat, center_lon - delta_lon
end_lat,   end_lon   = center_lat + delta_lat, center_lon + delta_lon

# ── Fuel break geometry (for axvspan on lowest row) ───────────────────────────
fb_endpoint1    = (34.319401, -118.993324)   # (lat, lon) NW end
fb_endpoint2    = (34.303774, -118.964856)   # (lat, lon) SE end
fb_width_m      = 300
fb_buffer_deg   = (fb_width_m / 2) / 111_000
fb_line         = LineString([(fb_endpoint1[1], fb_endpoint1[0]),
                               (fb_endpoint2[1], fb_endpoint2[0])])
fb_poly         = fb_line.buffer(fb_buffer_deg, cap_style=2)

cos_lat  = np.cos(np.radians(center_lat))

# ── Plot / axis settings ───────────────────────────────────────────────────────
g             = 9.81
Vs_levels     = np.arange(-24, 26,  2)
W_levels      = np.arange(-5,  5.2, 0.2)
th_levels     = np.arange(260, 340, 1)
xlim1, xlim2  = -half_length_km, half_length_km
ylim1, ylim2  = 0, 3000
north_limit_km = 10.0

out_dir = '.'


# In[10]:


# ── Static setup: terrain + cross-section path (from ref) ─────────────────────
ref_files = sorted(glob.glob(os.path.join(simulations[0]['dir'], f'wrfout_{domain}_*')))
print(f'Found {len(ref_files)} reference files')

with Dataset(ref_files[0]) as nc0:
    ter         = wrf.getvar(nc0, 'ter', -1)
    cross_start = wrf.CoordPair(lat=start_lat, lon=start_lon)
    cross_end   = wrf.CoordPair(lat=end_lat,   lon=end_lon)
    ter_line    = wrf.interpline(ter, wrfin=nc0,
                                 start_point=cross_start, end_point=cross_end)
    xy          = wrf.ll_to_xy(nc0, [start_lat, end_lat], [start_lon, end_lon])

nx_cross = len(ter_line)
dist_1d  = np.linspace(xlim1, xlim2, nx_cross)
print(f'Cross-section: {nx_cross} grid points')

# Project fuel break polygon onto cross-section dist axis
fb_mask = np.array([
    fb_poly.contains(Point(
        center_lon + d * np.sin(angle_rad) / (111.0 * cos_lat),
        center_lat + d * np.cos(angle_rad) / 111.0
    )) for d in dist_1d
])
if fb_mask.any():
    fb_x_min = dist_1d[fb_mask].min()
    fb_x_max = dist_1d[fb_mask].max()
else:
    fb_x_min = fb_x_max = None
    print('Warning: fuel break polygon does not intersect cross-section')


# In[11]:


# ── Helper: extract fields + Froude for one open Dataset ──────────────────────
def process_timestep(ncfile, idt):
    z    = wrf.getvar(ncfile, 'z',      idt)
    U    = wrf.getvar(ncfile, 'ua',     idt)
    V    = wrf.getvar(ncfile, 'va',     idt)
    W    = wrf.getvar(ncfile, 'wa',     idt)
    th   = wrf.getvar(ncfile, 'th',     idt)
    pblh = wrf.getvar(ncfile, 'PBLH',   idt)
    qv   = wrf.getvar(ncfile, 'QVAPOR', idt)
    thv  = th * (1.0 + 0.61 * qv)

    xy_line    = wrf.xy(V, start_point=xy[:, 0], end_point=xy[:, 1])
    U_cross    = wrf.interp2dxy(U,   xy_line, meta=True)
    V_cross    = wrf.interp2dxy(V,   xy_line, meta=True)
    W_cross    = wrf.interp2dxy(W,   xy_line, meta=True)
    z_cross    = wrf.interp2dxy(z,   xy_line, meta=True)
    th_cross   = wrf.interp2dxy(th,  xy_line, meta=True)
    thv_cross  = wrf.interp2dxy(thv, xy_line, meta=True)
    pblh_cross = wrf.interpline(pblh, wrfin=ncfile,
                                start_point=cross_start, end_point=cross_end)

    U_np   = wrf.to_np(U_cross)
    V_np   = wrf.to_np(V_cross)
    Valong = U_np * np.sin(angle_rad) + V_np * np.cos(angle_rad)
    nz, nx = U_np.shape
    dist_2d = np.tile(dist_1d, (nz, 1))

    # Froude number
    pblh_np = wrf.to_np(pblh_cross)
    thv_np  = wrf.to_np(thv_cross)
    z_np    = wrf.to_np(z_cross)
    offset  = {54: 2, 80: 3, 106: 4}.get(nz, 2)

    fr_profile = np.full(nx, np.nan)
    for ix in range(nx):
        zi = pblh_np[ix] if pblh_np[ix] > 0 else 1.0
        above = np.where(z_np[:, ix] > zi)[0]
        if len(above) == 0:
            continue
        ind_zi   = max(above[0] - 1, 0)
        mean_thv = np.nanmean(thv_np[:ind_zi + 1, ix])
        ind_free = min(ind_zi + offset, nz - 1)
        g_prime  = g * (thv_np[ind_free, ix] - mean_thv) / mean_thv
        if g_prime <= 0:
            continue
        mean_Vs        = np.nanmean(Valong[:ind_zi + 1, ix])
        fr_profile[ix] = np.abs(mean_Vs) / np.sqrt(g_prime * zi)

    is_super    = fr_profile > 1.0
    transitions = {'sub_to_super': None, 'super_to_sub': None}
    for ix in range(nx - 2, -1, -1):
        if np.isnan(fr_profile[ix]) or np.isnan(fr_profile[ix + 1]):
            continue
        if dist_1d[ix] > north_limit_km:
            continue
        fr_a, fr_b = fr_profile[ix], fr_profile[ix + 1]
        d_cross    = dist_1d[ix] + (1.0 - fr_a) * (dist_1d[ix + 1] - dist_1d[ix]) / (fr_b - fr_a)
        if is_super[ix] and not is_super[ix + 1] and transitions['sub_to_super'] is None:
            transitions['sub_to_super'] = d_cross
        elif (not is_super[ix] and is_super[ix + 1]
              and transitions['sub_to_super'] is not None
              and transitions['super_to_sub'] is None):
            transitions['super_to_sub'] = d_cross

    return {
        'Valong':  Valong,
        'W':       wrf.to_np(W_cross),
        'z':       z_np,
        'th':      wrf.to_np(th_cross),
        'dist_2d': dist_2d,
        'nx':      nx,
        'transitions': transitions,
    }


# In[ ]:


# ── Main loop ─────────────────────────────────────────────────────────────────
for fName_ref in ref_files:
    fname_base = os.path.basename(fName_ref)
    sim_files  = [os.path.join(sim['dir'], fname_base) for sim in simulations]

    missing = [f for f in sim_files if not os.path.exists(f)]
    if missing:
        print(f'Skipping {fname_base} — missing: {missing}')
        continue

    ncfiles = [Dataset(f) for f in sim_files]
    n_times = ncfiles[0].dimensions['Time'].size

    for idt in range(1): #range(n_times): #range(1):# 
        print(f'  {fname_base}  idt={idt}')

        timeWRF           = wrf.getvar(ncfiles[0], 'Times', idt)
        ts                = pd.to_datetime(wrf.to_np(timeWRF))
        tsPST             = ts - pd.DateOffset(hours=8)
        tWRF_strPST       = tsPST.strftime('%Y%m%d %H%M')
        tWRF_strPST_fName = tsPST.strftime('%Y%m%d_%H%M')
        time_label = 'WRF 111 m ' + tWRF_strPST[:8] + ' ' + tWRF_strPST[9:13] + ' PST'

        results = [process_timestep(nc, idt) for nc in ncfiles]

        # ── Figure: 3 rows × 2 cols ───────────────────────────────────────────
        n_sims = len(simulations)
        fig, axes = plt.subplots(n_sims, 2, figsize=(22, 6 * n_sims),
                                 sharex=True, sharey=True)

        xlabel_str = f'Distance along {angle_from_north:.0f}° cross section [km]'
        cf_V_ref   = None
        cf_W_ref   = None

        for i, (sim, res) in enumerate(zip(simulations, results)):
            ax_V = axes[i, 0]
            ax_W = axes[i, 1]
            trans = res['transitions']

            # --- Left: along-section wind ---
            cf_V = ax_V.contourf(res['dist_2d'], res['z'], res['Valong'],
                                  levels=Vs_levels, cmap=tmap, extend='both')
            ct   = ax_V.contour(res['dist_2d'], res['z'], res['th'],
                                 levels=th_levels, colors='k')
            ax_V.fill_between(dist_1d, 0, wrf.to_np(ter_line), facecolor='saddlebrown', zorder=5)
            ax_V.clabel(ct, inline=True, fontsize=12, levels=th_levels[::3], fmt='%1.0f')
            ax_V.scatter([0], [50], marker='^', color='yellow',
                          edgecolor='black', linewidth=2, s=300, zorder=8)
            ax_V.set_ylabel(f'{sim["label"]}\nElevation [m]', fontsize=15)
            ax_V.tick_params(labelsize=13)
            if trans['sub_to_super'] is not None:
                ax_V.axvline(trans['sub_to_super'], color='cyan',   lw=2.5, ls='--', zorder=9, label='sub → super')
            if trans['super_to_sub'] is not None:
                ax_V.axvline(trans['super_to_sub'], color='orange', lw=2.5, ls='--', zorder=9, label='super → sub')
            if any(trans.values()):
                ax_V.legend(fontsize=13, loc='upper right')
            if i == 0:
                ax_V.set_title(f'Along-section wind [m s$^{{-1}}$]', fontsize=18)
                ax_V.text(xlim1 + 0.5, ylim2 - 200, time_label, fontsize=15,
                          bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'),
                          zorder=10)
            if i == n_sims - 1:
                ax_V.set_xlabel(xlabel_str, fontsize=15)

            # --- Right: W ---
            cf_W = ax_W.contourf(res['dist_2d'], res['z'], res['W'],
                                  levels=W_levels, cmap=tmap, extend='both')
            ct   = ax_W.contour(res['dist_2d'], res['z'], res['th'],
                                 levels=th_levels, colors='k')
            ax_W.fill_between(dist_1d, 0, wrf.to_np(ter_line), facecolor='saddlebrown', zorder=5)
            ax_W.clabel(ct, inline=True, fontsize=12, levels=th_levels[::3], fmt='%1.0f')
            ax_W.scatter([0], [50], marker='^', color='yellow',
                          edgecolor='black', linewidth=2, s=300, zorder=8)
            ax_W.tick_params(labelsize=13, labelleft=False)
            if trans['sub_to_super'] is not None:
                ax_W.axvline(trans['sub_to_super'], color='cyan',   lw=2.5, ls='--', zorder=9, label='sub → super')
            if trans['super_to_sub'] is not None:
                ax_W.axvline(trans['super_to_sub'], color='orange', lw=2.5, ls='--', zorder=9, label='super → sub')
            if any(trans.values()):
                ax_W.legend(fontsize=13, loc='upper right')
            if i == 0:
                ax_W.set_title('W-component [m s$^{-1}$]', fontsize=18)
            if i == n_sims - 1:
                ax_W.set_xlabel(xlabel_str, fontsize=15)

            # Fuel break shading on the lowest row only
            if i == n_sims - 1 and fb_x_min is not None:
                for ax in [ax_V, ax_W]:
                    ax.axvspan(fb_x_min, fb_x_max, alpha=0.30, color='limegreen',
                               zorder=2, label='Oak fuel break')
                    ax.legend(fontsize=13, loc='upper right')

            if i == 0:
                cf_V_ref = cf_V
                cf_W_ref = cf_W

        # Apply ylim once — propagates to all panels via sharex/sharey
        axes[0, 0].set_ylim(ylim1, ylim2)
        axes[0, 0].set_xlim(xlim1, xlim2)

        plt.subplots_adjust(hspace=0.05, wspace=0.06, bottom=0.10)
        
        cax_V = fig.add_axes([0.13, 0.03, 0.35, 0.02])   # [left, bottom, width, height]
        cbar_V = fig.colorbar(cf_V_ref, cax=cax_V, orientation='horizontal', extend='both')
        cbar_V.set_label('Along-section wind [m s$^{-1}$]', fontsize=15)
        cbar_V.ax.tick_params(labelsize=13)

        cax_W = fig.add_axes([0.54, 0.03, 0.35, 0.02])
        cbar_W = fig.colorbar(cf_W_ref, cax=cax_W, orientation='horizontal', extend='both')
        cbar_W.set_label('W [m s$^{-1}$]', fontsize=15)
        cbar_W.ax.tick_params(labelsize=13)

        save_path = os.path.join(
            out_dir,
            f'MtnFire_compare111m_Cross_ang{angle_from_north:.0f}N'
            f'_Valong_Wcomp_theta_{tWRF_strPST_fName}_PST.jpg'
        )
        fig.savefig(save_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        print(f'    Saved {os.path.basename(save_path)}')

    for nc in ncfiles:
        nc.close()

print('Done!')

