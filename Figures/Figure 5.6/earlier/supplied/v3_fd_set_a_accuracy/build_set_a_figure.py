#!/usr/bin/env python3
"""Draw Figure 5.6 from audited conditional validation counts.

Preserves the directly inspected original Figure 5.6 style, adding
three-run means and sample-SD bands and extending the history to 500 epochs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=Path("set_a_validation_curves.csv"))
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with args.curves.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1500 and {r["paper_role"] for r in rows} == {"validation"}
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {
        "font.family": "DejaVu Sans", "font.size": 10,
        "mathtext.fontset": "dejavusans", "axes.titlesize": 12,
        "axes.labelsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10, "axes.linewidth": 0.8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(settings):
        fig, ax = plt.subplots(figsize=(9.2, 5.2))
        for group, color, label in (("1", "#2ca02c", r"$v_3=1$"),
                                    ("0", "#1f77b4", r"$v_3=0$"),
                                    (">1", "#d62728", r"$v_3>1$")):
            rr = sorted((r for r in rows if r["true_group"] == group), key=lambda r: int(r["completed_epoch"]))
            epochs = [int(r["completed_epoch"]) for r in rr]
            assert epochs == list(range(1, 501))
            mean = [float(r["mean_percent"]) for r in rr]
            lower = [float(r["display_band_low_percent"]) for r in rr]
            upper = [float(r["display_band_high_percent"]) for r in rr]
            ax.fill_between(epochs, lower, upper, facecolor=color, alpha=0.13,
                            linewidth=0, zorder=2)
            ax.plot(epochs, mean, color=color, linewidth=1.6, label=label, zorder=3)
        ax.set_title(r"Accuracy conditioned on $v_3(n_k)$, restricted to $k\in A$")
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"$P(\mathrm{model\ correct}\mid v_3)$   (%)")
        ax.set_xlim(-25, 525)
        ax.set_xticks(range(0, 501, 50))
        ax.set_ylim(-2, 103)
        ax.set_yticks(range(0, 101, 20))
        ax.set_axisbelow(True)
        ax.grid(True, color="0.9", linewidth=0.8)
        ax.legend(loc="center right", frameon=True, framealpha=0.8,
                  facecolor="white", edgecolor="0.8")
        fig.tight_layout()
        fig.savefig(args.out / "v3_fd_set_a_accuracy.pdf", metadata={
            "Title": "Conditional exact valuation accuracy on set A",
            "Creator": "build_set_a_figure.py / Matplotlib",
            "CreationDate": None, "ModDate": None})
        fig.savefig(args.out / "v3_fd_set_a_accuracy.png", dpi=200)
        plt.close(fig)
    metadata = {
        "source_style": "Original Figure 5.6, PDF page23 of Boyd-Mahler-AI.pdf, independently visually and structurally inspected",
        "canvas_inches": [9.2, 5.2], "layout": "tight_layout",
        "colors_by_true_group": {"1": "#2ca02c", "0": "#1f77b4", ">1": "#d62728"},
        "legend_group_order": ["1", "0", ">1"], "legend_location": "center right",
        "legend_frame_alpha": 0.8, "line_width_points": 1.6, "markers": False,
        "grid": {"color": "0.9", "width_points": 0.8, "directions": "both"},
        "font": "DejaVu Sans", "label_tick_legend_points": 10, "title_points": 12,
        "x_limits": [-25, 525], "y_limits": [-2, 103],
        "shown_completed_epochs": [1, 500], "smoothing": False,
        "band": "one sample SD about the three-run mean, clipped to [0,100]% for display",
        "band_alpha": 0.13, "recommended_latex_width": "0.65\\linewidth",
        "input_sha256": hashlib.sha256(args.curves.read_bytes()).hexdigest(),
        "python_version": platform.python_version(), "matplotlib_version": matplotlib.__version__,
    }
    (args.out / "plot_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print("Created v3_fd_set_a_accuracy.pdf and .png")


if __name__ == "__main__":
    main()
