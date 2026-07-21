"""2D heatmap of feature-embedding activation memory vs (R rows, C columns).

Memory model: the fully-materialised per-cell embedding tensor (R, C, E) in FP16,
    mem_bytes = R * C * E * 2      with E = embed_dim = 128
so at R=1e6, C=200 -> 51.2 GB (per batch element, per estimator).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---- model constants ----
E = 128          # embed_dim
BYTES = 2        # FP16
GB = 1e9

def mem_gb(R, C):
    return R * C * E * BYTES / GB

# ---- grid (log-spaced on both axes) ----
R = np.logspace(np.log10(100), np.log10(1_000_000), 400)   # 100 .. 1M rows
C = np.logspace(np.log10(1),   np.log10(200),       400)   # 1 .. 200 cols
RR, CC = np.meshgrid(R, C)
Z = mem_gb(RR, CC)

# ---- plot ----
fig, ax = plt.subplots(figsize=(9, 6.5))

pcm = ax.pcolormesh(
    RR, CC, Z,
    norm=LogNorm(vmin=Z.min(), vmax=Z.max()),
    cmap="viridis",
    shading="auto",
)

cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
cbar.set_label("Activation memory  (GB, FP16)", fontsize=11)

# ---- dashed contour marking the 30 GB threshold ----
cs = ax.contour(
    RR, CC, Z,
    levels=[30],
    colors="crimson",
    linestyles="dashed",
    linewidths=2,
)
ax.clabel(cs, fmt={30: "30 GB"}, inline=True, fontsize=10)

# a few faint reference contours for context
ref = ax.contour(
    RR, CC, Z,
    levels=[0.1, 1, 10, 51.2],
    colors="white",
    linewidths=0.6,
    alpha=0.35,
)
ax.clabel(ref, fmt=lambda v: f"{v:g} GB", inline=True, fontsize=7)

# ---- 'realistic table' reference line: C = 0.01 * R ----
Rline = np.logspace(np.log10(100), np.log10(20_000), 100)  # 0.01R <= 200 here
ax.plot(Rline, 0.01 * Rline, color="white", ls=":", lw=1.8, label="C = 0.01·R")

# ---- axes ----
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("R  (number of rows)", fontsize=11)
ax.set_ylabel("C  (number of columns)", fontsize=11)
ax.set_title(
    "Feature-embedding activation memory  —  (R × C × 128) FP16\n"
    "dashed = 30 GB threshold",
    fontsize=12,
)
ax.set_xlim(R.min(), R.max())
ax.set_ylim(C.min(), C.max())
ax.legend(loc="lower left", framealpha=0.85, fontsize=9)

fig.tight_layout()
fig.savefig("memory_heatmap.png", dpi=150, bbox_inches="tight")
print("saved memory_heatmap.png")
print(f"max memory on grid (R=1M, C=200): {mem_gb(1e6, 200):.1f} GB")
