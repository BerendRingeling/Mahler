#!/usr/bin/env python3
"""Plot the pooled three-run confusion-matrix snapshots."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EPOCHS = [0, 8, 40, 100, 210, 300]
MAXV = 24

corr = {}
with (RESULTS / "correlations.csv").open() as f:
    for r in csv.DictReader(f):
        corr[int(r["epoch"])] = (float(r["pearson_pred_vs_Phat"]), float(r["pearson_pred_vs_truth"]))


def matrix(kind, epoch):
    M = np.zeros((MAXV + 1, MAXV + 1), dtype=float)
    fname = RESULTS / f"confusion_{kind}_epoch_{epoch}.csv"
    with fname.open() as f:
        rr = csv.DictReader(f)
        for row in rr:
            x = int(row["predicted_v2_n"])
            ykey = "P_hat" if kind == "Phat" else "true_v2_n"
            y = int(row[ykey])
            if 0 <= x <= MAXV and 0 <= y <= MAXV:
                M[y, x] += int(row["count"])
    return M

matsP = [matrix("Phat", e) for e in EPOCHS]
matsT = [matrix("truth", e) for e in EPOCHS]
vmaxP = max(M.max() for M in matsP)
vmaxT = max(M.max() for M in matsT)

fig = plt.figure(figsize=(8.8, 9.8))
outer = GridSpec(2, 1, figure=fig, hspace=0.28, top=0.965, bottom=0.07, left=0.065, right=0.93)


def draw_section(spec, mats, which, title, ylabel, vmax):
    gs = GridSpecFromSubplotSpec(2, 4, subplot_spec=spec, width_ratios=[1,1,1,0.06], wspace=0.28, hspace=0.30)
    ims=[]
    for j, (e, M) in enumerate(zip(EPOCHS, mats)):
        ax = fig.add_subplot(gs[j//3, j%3])
        masked = np.ma.masked_where(M <= 0, M)
        # Match the original figure: zero-count cells should appear as the
        # darkest viridis color, not as the default white axes background.
        cmap = plt.get_cmap("viridis")
        ax.set_facecolor(cmap(0.0))
        im = ax.imshow(masked, origin="lower", interpolation="nearest", cmap=cmap,
                       norm=LogNorm(vmin=1, vmax=vmax), extent=(-0.5, MAXV+0.5, -0.5, MAXV+0.5), aspect="equal")
        ims.append(im)
        ax.plot([0, MAXV], [0, MAXV], color="red", linewidth=1.0)
        r = corr[e][0 if which == "Phat" else 1]
        ax.set_title(f"epoch {e}  ($r={r:.2f}$)", fontsize=8.5)
        ax.set_xlim(-0.5, MAXV+0.5); ax.set_ylim(-0.5, MAXV+0.5)
        ax.set_xticks([0,5,10,15,20]); ax.set_yticks([0,5,10,15,20])
        ax.tick_params(labelsize=7)
        ax.set_xlabel(r"predicted $v_2(n_k)$", fontsize=7.5)
        if j % 3 == 0:
            ax.set_ylabel(ylabel, fontsize=7.5)
    cax = fig.add_subplot(gs[:,3])
    cb = fig.colorbar(ims[-1], cax=cax)
    cb.set_label("pooled count (log scale)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    # title centered over the three heatmap columns, not colorbar
    pos0 = fig.add_subplot(gs[0,0]).get_position(); plt.delaxes(fig.axes[-1])
    pos2 = fig.add_subplot(gs[0,2]).get_position(); plt.delaxes(fig.axes[-1])
    xcenter = (pos0.x0 + pos2.x1)/2
    ytop = max(a.get_position().y1 for a in fig.axes if a is not cax and a.get_position().y0 >= spec.get_position(fig).y0 - 1e-6 and a.get_position().y1 <= spec.get_position(fig).y1 + 1e-6)
    fig.text(xcenter, spec.get_position(fig).y1 + 0.014, title, ha="center", va="bottom", fontsize=9)

# Draw top and bottom sections.
draw_section(outer[0], matsP, "Phat", r"Model prediction vs. predictor $\hat P(k)$", r"$\hat P(k)$", vmaxP)
draw_section(outer[1], matsT, "truth", r"True vs. predicted $v_2(n_k)$", r"true $v_2(n_k)$", vmaxT)

outpdf = RESULTS / "v2_confusion_snapshots_validation.pdf"
outpng = RESULTS / "v2_confusion_snapshots_validation.png"
fig.savefig(outpdf, bbox_inches="tight")
fig.savefig(outpng, dpi=180, bbox_inches="tight")
plt.close(fig)
print(outpdf)
