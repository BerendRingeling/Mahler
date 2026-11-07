#!/usr/bin/env python3
"""Plot v2_lower_bound_validation.pdf/png from the derived curve CSV."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CSV_PATH = RESULTS / "lower_bound_curve.csv"
PDF_PATH = RESULTS / "v2_lower_bound_validation.pdf"
PNG_PATH = RESULTS / "v2_lower_bound_validation.png"

rows = []
with CSV_PATH.open(newline="") as f:
    rows.extend(csv.DictReader(f))

epoch = np.array([int(r["epoch"]) for r in rows], dtype=int)
mean = np.array([float(r["mean"]) for r in rows], dtype=float)
sd = np.array([float(r["sample_sd"]) for r in rows], dtype=float)

mask = epoch <= 21
x = epoch[mask]
y = mean[mask]
yerr = sd[mask]

fig, ax = plt.subplots(figsize=(9.2, 5.2))
ax.fill_between(x, y - yerr, y + yerr, color="green", alpha=0.08, linewidth=0)
ax.plot(
    x, y,
    color="green",
    linewidth=1.5,
    marker="o",
    markersize=3,
    label=r"mean proportion with $\hat{v}(k) \leq v_2(n_k)$ (model)",
)
ax.set_title(
    r"Fraction of the 3000 validation $k$ where the model output satisfies the lower bound "
    r"$\hat{v}(k) \leq v_2(n_k)$"
)
ax.set_xlabel("epoch")
ax.set_ylabel(r"proportion of validation $k$ (%)")
ax.set_xlim(-0.2, 21)
ax.set_ylim(64, 102)
ax.set_xticks(np.arange(0, 22, 3))
ax.set_yticks(np.arange(64, 101, 4))
ax.grid(True, alpha=0.25)
ax.legend(loc="lower right", framealpha=0.9)
fig.tight_layout()
fig.savefig(PDF_PATH)
fig.savefig(PNG_PATH, dpi=220)
plt.close(fig)
print(PDF_PATH)
print(PNG_PATH)
