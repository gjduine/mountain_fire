"""
Remove road fuel categories from wrfinput_d04 NFUEL_CAT.

Roads are identified by SHAPE, not size:
  - Morphological opening (erosion + dilation) removes features narrower
    than `road_width_cells` cells, regardless of how long they are.
  - Compact features of any size (urban blocks, small house clusters) survive
    because they are "fat" in all directions.

Removed road cells are replaced with the nearest surrounding natural fuel
(nearest-neighbour from non-candidate cells), so a road through chaparral
becomes chaparral, a road through grassland becomes grassland, etc.

Workflow:
  1. Run as-is; inspect the diagnostic plot and printed statistics.
  2. Adjust `road_candidate_cats` and `road_width_cells` if needed.
  3. Verify the three-panel diagnostic looks correct.
  4. Copy wrfinput_d04_noroads into the run directories to re-run.
"""
import numpy as np
import shutil
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
from pathlib import Path
from scipy import ndimage

# ── Configuration ─────────────────────────────────────────────────────────────
wrfinput_path = Path("/glade/derecho/scratch/gduine/mountain_fire/111m/ifire2_noroad/wrfinput_d04")
out_path      = wrfinput_path.parent / "wrfinput_d04_noroads"

# Fuel categories that appear as BOTH roads and urban/developed areas.
# The morphological filter will keep compact patches (cities, house clusters)
# and remove thin linear ones (roads).
road_candidate_cats = [91, 98, 99]

# Roads narrower than (2 * road_width_cells + 1) cells are removed.
# At 111 m resolution:
#   road_width_cells = 1  →  removes features < 3 cells (~333 m) wide
#   road_width_cells = 2  →  removes features < 5 cells (~555 m) wide
# Start with 1 and increase if roads are still visible.
road_width_cells = 2

# ── Load ──────────────────────────────────────────────────────────────────────
shutil.copy(wrfinput_path, out_path)
print(f"Working copy: {out_path}")

ds_orig = Dataset(wrfinput_path, 'r')
ds_out  = Dataset(out_path, 'r+')

fuelvar = "FUEL_CAT" if "FUEL_CAT" in ds_orig.variables else "NFUEL_CAT"
nfuel   = np.array(ds_orig.variables[fuelvar][0, :, :])   # (ny, nx)
print(f"Fuel variable : {fuelvar},  shape: {nfuel.shape}")

# ── Fuel distribution ─────────────────────────────────────────────────────────
cats, counts = np.unique(nfuel.astype(int), return_counts=True)
print("\nFuel category distribution in wrfinput:")
for c, n in zip(cats, counts):
    tag = " ← candidate (road or urban)" if c in road_candidate_cats else ""
    print(f"  {c:5d}: {n:8d} cells{tag}")

# ── Morphological opening ─────────────────────────────────────────────────────
# Opening = erosion followed by dilation with the same structuring element.
# Thin features (roads) are fully eroded away; compact features survive.
candidate_mask = np.isin(nfuel, road_candidate_cats)
struct = np.ones((2 * road_width_cells + 1,
                  2 * road_width_cells + 1), dtype=bool)

opened    = ndimage.binary_opening(candidate_mask, structure=struct)
road_mask = candidate_mask & ~opened    # thin → removed
kept_mask = candidate_mask &  opened   # compact → kept

print(f"\nStructuring element: {struct.shape}  "
      f"(removes features < {2*road_width_cells+1} cells = "
      f"{(2*road_width_cells+1)*111:.0f} m wide)")
print(f"Candidate cells : {candidate_mask.sum():8d}")
print(f"Road cells removed: {road_mask.sum():8d}  (thin linear features)")
print(f"Urban cells kept  : {kept_mask.sum():8d}  (compact: cities, house clusters)")

# ── Nearest-neighbour fill ────────────────────────────────────────────────────
# Road cells take the fuel type of the closest non-candidate cell.
# → road through chaparral → chaparral; road through grassland → grassland.
non_candidate_mask = ~candidate_mask
_, nearest_idx = ndimage.distance_transform_edt(
    ~non_candidate_mask, return_indices=True
)
nfuel_mod = nfuel.copy()
nfuel_mod[road_mask] = nfuel[nearest_idx[0][road_mask],
                              nearest_idx[1][road_mask]]

# ── Write back ────────────────────────────────────────────────────────────────
ds_out.variables[fuelvar][0, :, :] = nfuel_mod
ds_orig.close()
ds_out.close()
print(f"\nSaved → {out_path}")

# ── Diagnostic plot ───────────────────────────────────────────────────────────
fuel_colors = {
    -9999:"#000000", 1:"#ffffbe", 2:"#ffff00", 3:"#e6c50b", 4:"#ffd37f",
    5:"#ffaa66",  6:"#cdaa66",  7:"#897044",  8:"#d3ffbe",  9:"#70a800",
    10:"#267300", 11:"#e8beff", 12:"#7a8ef5", 13:"#c500ff",
    91:"#8400a5", 92:"#9ea1f0", 93:"#e974ff", 98:"#0000ff", 99:"#bfbfbf",
}
fuel_ids    = sorted(fuel_colors.keys())
fuel_cmap   = mcolors.ListedColormap([fuel_colors[i] for i in fuel_ids])
fuel_to_idx = {fid: i for i, fid in enumerate(fuel_ids)}

def to_idx(arr):
    out = np.zeros_like(arr, dtype=int)
    for fid, idx in fuel_to_idx.items():
        out[arr == fid] = idx
    return out

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(to_idx(nfuel),     cmap=fuel_cmap, vmin=0, vmax=len(fuel_ids)-1, origin='lower')
axes[0].set_title("Original NFUEL_CAT")

axes[1].imshow(to_idx(nfuel_mod), cmap=fuel_cmap, vmin=0, vmax=len(fuel_ids)-1, origin='lower')
axes[1].set_title(f"Roads removed (width < {2*road_width_cells+1} cells)")

axes[2].imshow(road_mask.astype(float), cmap='Reds', origin='lower')
axes[2].set_title(f"Removed road cells ({road_mask.sum():.0f} total)\n"
                  f"[red = removed, white = kept]")

for ax in axes:
    ax.set_xlabel("x (met grid)")
axes[0].set_ylabel("y (met grid)")

plt.suptitle(f"Road removal — road_width_cells={road_width_cells} — "
             f"{wrfinput_path.name}", fontsize=12)
plt.tight_layout()
diag_path = out_path.parent / f"fuel_roads_removed_diagnostic_width_{road_width_cells}.png"
plt.savefig(diag_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Diagnostic plot → {diag_path}")
