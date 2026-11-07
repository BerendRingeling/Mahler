#!/usr/bin/env python3
"""Plot Figure 5.4 from the included per-epoch audit tables.

  python3 build_v3_figure.py --results . --out rebuilt

The figure displays all 500 validation epochs. Selection remains restricted
to the first 300 completed epochs in each run. No test curves are plotted.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from paper_plot_style import (
    apply_paper_style, paper_style_metadata, FIGURE_SIZE,
    BLUE, GRAY, LINE_WIDTH, BAND_ALPHA,
)

RUNS = ("1000260_9", "1000260_10", "1000260_11")
BASENAME = "v3_fd_validation_accuracy"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--no-plot",action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    source=args.results/"all_per_epoch_metrics.csv"
    with source.open(newline="") as f:
        rows=list(csv.DictReader(f))
    audit=json.loads((args.results/"v3_audit_summary.json").read_text())
    table={}
    for row in rows:
        key=(row["run_id"],row["paper_role"],int(row["completed_epoch"]))
        assert key not in table
        assert int(row["examples"])==3000
        table[key]=row
    expected={(run,role,e) for run in RUNS for role in ("validation","test") for e in range(1,501)}
    assert set(table)==expected
    aggregated=[]
    for role in ("validation","test"):
        for e in range(1,501):
            counts=[int(table[(run,role,e)]["greedy_exact_count"]) for run in RUNS]
            percentages=[100*n/3000 for n in counts]
            aggregated.append({
                "paper_role":role,"completed_epoch":e,
                **{run+"_count":n for run,n in zip(RUNS,counts)},
                "examples_per_run":3000,
                "mean_percent":statistics.mean(percentages),
                "sample_sd_percentage_points":statistics.stdev(percentages),
                "population_sd_percentage_points":statistics.pstdev(percentages),
            })
    validation=[r for r in aggregated if r["paper_role"]=="validation"]
    maximum=max(r["mean_percent"] for r in validation)
    maximum_epochs=[r["completed_epoch"] for r in validation if r["mean_percent"]==maximum]
    baseline_counts={}
    for label,role in [("test1","validation"),("valid","test")]:
        counts=[audit["runs"][run]["integrity"][label]["constant_prediction_1_count"] for run in RUNS]
        assert len(set(counts))==1
        baseline_counts[role]={"correct":counts[0],"examples":3000,"percent":100*counts[0]/3000}
    summary={
        "figure":BASENAME,
        "runs":list(RUNS),"configuration":"v3_fD","features":["k","D_Factor"],"base":3,
        "metric":"Greedy exact-match accuracy, recomputed from saved predictions",
        "plotted_pool":"validation","plotted_completed_epochs":[1,500],
        "selection_completed_epochs":[1,300],"selection":"Earliest validation maximum separately in each run",
        "sd":"Sample standard deviation across three run percentages (ddof=1)",
        "smoothing":None,
        "constant_prediction_1":baseline_counts,
        "mean_validation_curve_maximum":{"mean_percent":maximum,"completed_epochs":maximum_epochs},
        "mean_validation_at_500":validation[-1],
        "mean_validation_over_last_50_epochs":statistics.mean(r["mean_percent"] for r in validation[-50:]),
        "selected_test":audit["selected_aggregates"]["test"],
        "selected_validation":audit["selected_aggregates"]["validation"],
        "selected_epochs":{run:audit["runs"][run]["selection"]["completed_epoch"] for run in RUNS},
        "source_per_epoch_csv_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "python_version":sys.version,
    }
    write_csv(args.out/"learning_curve_metrics.csv",aggregated)
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        apply_paper_style()
        xs=np.array([r["completed_epoch"] for r in validation])
        ys=np.array([r["mean_percent"] for r in validation])
        sd=np.array([r["sample_sd_percentage_points"] for r in validation])
        fig,ax=plt.subplots(figsize=FIGURE_SIZE)
        purple="#800080"
        ax.fill_between(xs,ys-sd,ys+sd,color=purple,alpha=BAND_ALPHA,linewidth=0)
        ax.plot(xs,ys,color=purple,linewidth=1.5,marker="o",markersize=3,
                markeredgewidth=1,label="greedy exact-match")
        ax.axhline(baseline_counts["validation"]["percent"],color=GRAY,
                   linestyle="--",linewidth=1.0,label="constant prediction 1")
        ax.set(xlabel="epoch",ylabel="validation greedy accuracy (%)",
               title=r"$p=3$: validation greedy accuracy vs. epoch",
               xlim=(-25,525),ylim=(8.5,53))
        ax.set_xticks(np.arange(0,501,50))
        ax.set_yticks(np.arange(10,51,5))
        ax.set_axisbelow(True)
        ax.grid(True,color="0.85",alpha=1.0,linewidth=0.8)
        ax.legend(loc="lower right",fontsize=10,frameon=True,framealpha=1.0,
                  facecolor="white",edgecolor="#cccccc")
        fig.tight_layout()
        fig.savefig(args.out/(BASENAME+".pdf"),
                    metadata={"Creator":"build_v3_figure.py","CreationDate":None})
        fig.savefig(args.out/(BASENAME+".png"),dpi=200,facecolor="white")
        plt.close(fig)
        summary["plot_environment"]={
            "matplotlib":matplotlib.__version__,"numpy":np.__version__,
            "png_dpi":200,
            **paper_style_metadata(),
            "reference":"Original Figure 5.4, manuscript PDF page 21",
            "grid":True,
            "grid_color":"0.85",
            "grid_alpha":1.0,
            "source_legend_size_pt":10,
            "line_color":purple,
            "line_width_pt":1.5,
            "marker":"o",
            "marker_size_pt":3,
            "marker_edge_width_pt":1,
            "layout":"tight_layout",
            "x_limits":[-25,525],
            "y_limits":[8.5,53],
            "title":"p = 3: validation greedy accuracy vs. epoch",
            "mean_sd_band_and_baseline_retained":True,
        }
    (args.out/"figure_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:summary[k] for k in ["selected_epochs","selected_test","mean_validation_curve_maximum","constant_prediction_1"]},indent=2))


if __name__=="__main__":
    main()
