#!/usr/bin/env python3
"""Plot the three-run mean Pearson correlation with \hat P(k)."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CSV = RESULTS / "v2_pearson_Phat_validation_curve.csv"

rows = []
with CSV.open() as f:
    rows = list(csv.DictReader(f))

# Match the original paper figure window: epochs 0--300.
rows = [r for r in rows if int(r["epoch"]) <= 300]
epoch = [int(r["epoch"]) for r in rows]
mean = [float(r["mean"]) for r in rows]
sd = [float(r["sample_sd"]) for r in rows]
lower = [m - s for m, s in zip(mean, sd)]
upper = [m + s for m, s in zip(mean, sd)]

fig, ax = plt.subplots(figsize=(9.2, 5.2))
ax.fill_between(epoch, lower, upper, color="crimson", alpha=0.10, linewidth=0)
ax.plot(
    epoch, mean,
    color="crimson", linewidth=1.25,
    marker="o", markersize=2.1, markeredgewidth=0,
    label=r"$r(\mathrm{model\ prediction},\ \hat P(k))$",
)
ax.set_title(r"Correlation of model predictions with $\hat P(k)=\mathrm{round}(\hat v(k)+s(k)+1)$ vs. epoch")
ax.set_xlabel("epoch")
ax.set_ylabel("Pearson correlation  $r$")
ax.set_xlim(-5, 305)
ax.set_ylim(0.65, 0.98)
ax.grid(True, alpha=0.45)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS / "v2_pearson_Phat_validation.pdf")
fig.savefig(RESULTS / "v2_pearson_Phat_validation.png", dpi=180)
print("Wrote figure files in", RESULTS)
