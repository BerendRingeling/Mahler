#!/usr/bin/env python3
"""Reproduce Figure 5.8 from the audited three-run validation curves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "complement_validation_curves.csv")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.data.open() as stream:
        rows = list(csv.DictReader(stream))
    epoch = np.array([int(row["completed_epoch"]) for row in rows])
    mean = np.array([float(row["mean_percent"]) for row in rows])
    sd = np.array([float(row["sample_sd_percentage_points"]) for row in rows])
    baseline = np.array([float(row["constant_one_percent"]) for row in rows])
    assert np.array_equal(epoch, np.arange(1, 501))
    assert {row["paper_role"] for row in rows} == {"validation"}
    assert {int(row["examples_per_run"]) for row in rows} == {551}
    assert np.allclose(baseline, 100 * 275 / 551, rtol=0, atol=1e-12)
    assert np.all((mean >= 0) & (mean <= 100) & (sd >= 0))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.set_axisbelow(True)
    ax.grid(True, color="0.92", linewidth=0.8)
    ax.fill_between(epoch, np.maximum(0, mean - sd), np.minimum(100, mean + sd),
                    color="#8c564b", alpha=0.13, linewidth=0, zorder=1)
    ax.plot(epoch, mean, color="#8c564b", linewidth=2.0, zorder=3,
            label=r"model on complement of $C''\cup\{k\equiv0,\pm5\ (27)\}$")
    ax.axhline(baseline[0], color="0.6", linestyle="--", linewidth=1.3,
               zorder=2, label=r"baseline (guess $v_3=1$)")
    ax.set_title(r"Accuracy outside $C''$ and the mod-27 classes")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"accuracy on $v_3(n_k)\in\{0,1\}$   (%)")
    ax.set_ylim(40, 100)
    ax.set_yticks(np.arange(40, 101, 10))
    ax.set_xlim(1 - 0.05 * 499, 500 + 0.05 * 499)
    ax.set_xticks(np.arange(0, 501, 50))
    # The new declining curve reaches the original lower-right legend area.
    # Keep the original legend styling, but place it in the empty upper right.
    ax.legend(loc="upper right", framealpha=0.8, facecolor="white", edgecolor="0.8")
    fig.tight_layout()
    stem = args.out / "v3_fd_complement_accuracy"
    fig.savefig(stem.with_suffix(".pdf"), metadata={
        "Title": "Figure 5.8: accuracy on the complement of C'' and the mod-27 classes",
        "Subject": "Three-run validation mean and sample standard deviation; true valuations 0 or 1",
        "Creator": "build_complement_figure.py",
        "CreationDate": None,
        "ModDate": None,
    })
    fig.savefig(stem.with_suffix(".png"), dpi=160)
    plt.close(fig)
    audit = {
        "source_csv": args.data.name,
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
        "canvas_inches": [9.2, 5.2],
        "model_color": "#8c564b",
        "model_linewidth_pt": 2.0,
        "model_markers": None,
        "baseline_color": "#999999",
        "baseline_linewidth_pt": 1.3,
        "baseline_linestyle": "--",
        "baseline_percent": float(baseline[0]),
        "legend_location": "upper right",
        "legend_moved_from_original_lower_right_to_avoid_covering_late_decline": True,
        "grid_grey": 0.92,
        "grid_linewidth_pt": 0.8,
        "vertical_limits_percent": [40, 100],
        "vertical_axis_truncation_disclosed_in_caption": True,
        "mean_below_display_range_epochs": epoch[mean < 40].tolist(),
        "band_below_display_range_epochs": epoch[(mean - sd) < 40].tolist(),
        "all_values_preserved_in_csv": True,
        "smoothing": "none",
        "run_aggregation": "arithmetic mean; sample SD with divisor 3-1",
        "band_alpha": 0.13,
        "test_data_used_in_plot": False,
    }
    (args.out / "figure_style_and_data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"pdf": str(stem.with_suffix(".pdf")), "png": str(stem.with_suffix(".png")),
                      "validation_examples": 551, "validation_constant_one_percent": float(baseline[0])}))


if __name__ == "__main__":
    main()
