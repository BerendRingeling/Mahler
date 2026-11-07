#!/usr/bin/env python3
"""Draw Figure 5.7 and emit its native LaTeX table and replacement text."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

SETS = ("A", "A'", "A''", "B", "B'", "B''", "C", "C'", "C''")


def replacement(summary):
    test = summary["selected_accuracies"]["test"]
    assert summary["comparisons"]["test"]["descending_order_by_selected_mean"][0] == "A"
    assert test["B'"]["mean_percent"] > test["B"]["mean_percent"]
    assert test["C'"]["mean_percent"] > test["C"]["mean_percent"]
    table = []
    for i, name in enumerate(SETS):
        x = test[name]
        n = f'{x["examples_per_run"]:,}'.replace(",", r"{,}")
        table.append(rf"  ${name}$ & ${n}$ & ${x['mean_percent']:.2f}\pm{x['sample_sd_percentage_points']:.2f}$\\")
        if i in (2, 5):
            table.append(r"  \midrule")
    template = r"""Figure~\ref{fig:acc-by-set} compares greedy exact-match
accuracy on the sets in Table~\ref{tab:sets}, restricted
to examples with true $v_3(n_k)\in\{0,1\}$.
The table reports test results at the validation-selected epochs
used above, and the curves show mean validation accuracy
over the three runs.

\begin{figure}[t]
\centering
\begin{minipage}[c]{0.30\textwidth}
  \centering\small
  \renewcommand{\arraystretch}{1.15}
  \begin{tabular}{@{}l r r@{}}
  \toprule
  set & $n$ & acc. (\%)\\
  \midrule
TABLE_ROWS
  \bottomrule
  \end{tabular}
\end{minipage}\hfill
\begin{minipage}[c]{0.66\textwidth}
  \centering
  \includegraphics[width=\linewidth]{v3_fd_sets_comparison.pdf}
\end{minipage}
\caption{Greedy exact-match accuracy restricted to true
$v_3(n_k)\in\{0,1\}$ on the sets in Table~\ref{tab:sets}.
Left: mean test accuracy, with sample standard deviation
over three runs, at the validation-selected epochs used above;
$n$ is the number of test examples in each set with true
$v_3(n_k)\in\{0,1\}$.
Right: mean validation accuracy over the same three runs,
as a function of the completed training epoch.
The vertical axis starts at $40\%$.}
\label{fig:acc-by-set}
\end{figure}

In this comparison, the highest mean test accuracy is attained
on the set $A$.
Although Table~\ref{tab:sets} records no exceptions to the parity
laws on some of the larger sets, these sets admit more varied
factorizations of $k(k^2-16)$.
We speculate that this makes the relevant parity information
more difficult for the model to extract.
Excluding the values $k\equiv4\pmod8$, passing from $B$ to $B'$
and from $C$ to $C'$, increases the mean conditional test accuracy.
"""
    return template.replace("TABLE_ROWS", "\n".join(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=Path("sets_validation_curves.csv"))
    parser.add_argument("--summary", type=Path, default=Path("sets_summary.json"))
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with args.curves.open() as f:
        rows = list(csv.DictReader(f))
    summary = json.loads(args.summary.read_text())
    assert len(rows) == 4500 and {r["paper_role"] for r in rows} == {"validation"}
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {
        "font.family": "DejaVu Sans", "font.size": 10, "mathtext.fontset": "dejavusans",
        "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 9, "axes.linewidth": 0.8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out", "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(settings):
        fig, ax = plt.subplots(figsize=(9.6, 5.6))
        for i, name in enumerate(SETS):
            rr = sorted((r for r in rows if r["set"] == name), key=lambda r:int(r["completed_epoch"]))
            epochs = [int(r["completed_epoch"]) for r in rr]
            assert epochs == list(range(1, 501))
            values = [float(r["mean_percent"]) for r in rr]
            color = ("#7f7f7f", "#1f77b4", "#d62728")[i//3]
            style = ("-", "--", ":")[i%3]
            ax.plot(epochs, values, color=color, linestyle=style, linewidth=1.7,
                    label=f"${name}$", zorder=3)
        ax.set_title("Accuracy by set vs. epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel(r"accuracy on $v_3(n_k)\in\{0,1\}$   (%)")
        ax.set_xlim(-25, 525)
        ax.set_xticks(range(0, 501, 50))
        ax.set_ylim(40, 102)
        ax.set_yticks(range(40, 101, 10))
        ax.set_axisbelow(True)
        ax.grid(True, color="0.92", linewidth=0.8)
        ax.legend(loc="lower right", ncol=3, fontsize=9, frameon=True,
                  framealpha=0.8, facecolor="white", edgecolor="0.8")
        fig.tight_layout()
        fig.savefig(args.out / "v3_fd_sets_comparison.pdf", metadata={
            "Title": "Conditional exact valuation accuracy on nine arithmetic sets",
            "Creator": "build_sets_figure.py / Matplotlib", "CreationDate": None, "ModDate": None})
        fig.savefig(args.out / "v3_fd_sets_comparison_plot.png", dpi=200)
        plt.close(fig)
    (args.out / "figure_5_7_replacement.tex").write_text(replacement(summary))
    metadata = {
        "source_style": "Original Figure5.7, PDF page23 of Boyd-Mahler-AI.pdf, directly inspected",
        "canvas_inches": [9.6,5.6], "layout": "tight_layout",
        "family_colors": {"A": "#7f7f7f", "B": "#1f77b4", "C": "#d62728"},
        "within_family_styles": ["solid","dashed","dotted"], "line_width_points":1.7,
        "markers": False, "legend_location": "lower right", "legend_columns":3,
        "legend_font_points":9, "legend_frame_alpha":0.8,
        "grid_color":"0.92", "grid_width_points":0.8,
        "font":"DejaVu Sans", "label_tick_points":10, "title_points":12,
        "x_limits":[-25,525], "y_limits":[40,102], "completed_epochs":[1,500],
        "y_display_note":"Original 40% lower bound retained; all numeric observations, including early values below40%, remain in the CSV",
        "smoothing":False, "bands":False,
        "sd_policy":"Selected test sample SD in left table; complete per-epoch SD retained numerically",
        "latex_layout":{"left_minipage_textwidth":0.30,"right_minipage_textwidth":0.66,"plot_width_inside_right_minipage":"\\linewidth"},
        "curves_sha256":hashlib.sha256(args.curves.read_bytes()).hexdigest(),
        "summary_sha256":hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "python_version":platform.python_version(),"matplotlib_version":matplotlib.__version__,
    }
    (args.out / "plot_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print("Created Figure5.7 plot, native LaTeX table and replacement text.")


if __name__ == "__main__":
    main()
