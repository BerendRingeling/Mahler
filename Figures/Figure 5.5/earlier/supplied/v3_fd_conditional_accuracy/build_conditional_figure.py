#!/usr/bin/env python3
"""Reproduce Figure 5.5 from independently audited exact per-class counts.

  python3 build_conditional_figure.py --inputs inputs --out rebuilt

The three conditioning groups are true v3(n_k)=0, =1 and >1.
Correctness in the last group still requires the exact integer valuation.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

RUNS = ("1000260_9", "1000260_10", "1000260_11")
ROLES = {"test1": "validation", "valid": "test"}
GROUPS = ("0", "1", ">1")
BASENAME = "v3_fd_conditional_accuracy"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def group_of(v):
    if v < 0:
        raise ValueError("Unexpected negative true paper valuation")
    return str(v) if v < 2 else ">1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    class_path = args.inputs / "all_per_epoch_class_metrics.csv"
    overall_path = args.inputs / "all_per_epoch_metrics.csv"
    audit_path = args.inputs / "v3_audit_summary.json"
    audit = json.loads(audit_path.read_text())
    assert audit["policy"]["paper_role_mapping"] == ROLES
    assert audit["policy"]["selection_completed_epoch_window"] == [1, 300]
    assert audit["policy"]["test_used_for_selection"] is False
    total = {}
    for row in read_csv(overall_path):
        key = (row["run_id"], row["paper_role"], int(row["completed_epoch"]))
        assert key not in total
        assert ROLES[row["file_label"]] == row["paper_role"]
        assert int(row["epoch_index"]) + 1 == key[2]
        total[key] = row
    expected = {(r, p, e) for r in RUNS for p in ("validation", "test")
                for e in range(1, 501)}
    assert set(total) == expected
    grouped = defaultdict(lambda: [0, 0])
    seen = set()
    for row in read_csv(class_path):
        run, role, e = row["run_id"], row["paper_role"], int(row["completed_epoch"])
        v = int(row["true_v3_n"])
        assert (run, role, e) in expected
        assert ROLES[row["file_label"]] == role
        assert int(row["epoch_index"]) + 1 == e
        class_key = (run, role, e, v)
        assert class_key not in seen
        seen.add(class_key)
        n, correct = int(row["examples"]), int(row["greedy_correct"])
        assert 0 <= correct <= n
        key = (run, role, e, group_of(v))
        grouped[key][0] += n
        grouped[key][1] += correct
    group_sizes = {}
    for role in ("validation", "test"):
        group_sizes[role] = {}
        for group in GROUPS:
            sizes = {grouped[(run, role, e, group)][0]
                     for run in RUNS for e in range(1, 501)}
            assert len(sizes) == 1 and next(iter(sizes)) > 0
            group_sizes[role][group] = next(iter(sizes))
    assert group_sizes == {"validation": {"0": 800, "1": 935, ">1": 1265},
                           "test": {"0": 810, "1": 967, ">1": 1223}}
    for key in expected:
        assert sum(grouped[(*key, g)][0] for g in GROUPS) == 3000
        assert sum(grouped[(*key, g)][1] for g in GROUPS) == int(total[key]["greedy_exact_count"])
    per_run = []
    for run in RUNS:
        for role in ("validation", "test"):
            for e in range(1, 501):
                for group in GROUPS:
                    n, correct = grouped[(run, role, e, group)]
                    per_run.append({"run_id": run, "paper_role": role,
                                    "completed_epoch": e, "true_group": group,
                                    "examples": n, "correct": correct,
                                    "accuracy_percent": 100*correct/n})
    means = []
    for role in ("validation", "test"):
        for e in range(1, 501):
            for group in GROUPS:
                n = group_sizes[role][group]
                counts = [grouped[(r, role, e, group)][1] for r in RUNS]
                values = [100*c/n for c in counts]
                mean, sd = statistics.mean(values), statistics.stdev(values)
                means.append({"paper_role": role, "completed_epoch": e,
                              "true_group": group, "examples_per_run": n,
                              **{r+"_correct": c for r,c in zip(RUNS,counts)},
                              "mean_percent": mean, "sample_sd_percentage_points": sd,
                              "display_band_lower_percent": max(0, mean-sd),
                              "display_band_upper_percent": min(100, mean+sd)})
    selected = {}
    for run in RUNS:
        eligible = [(e, int(total[(run,"validation",e)]["greedy_exact_count"]))
                    for e in range(1,301)]
        maximum = max(count for _,count in eligible)
        chosen = min(e for e,count in eligible if count == maximum)
        assert chosen == audit["runs"][run]["selection"]["completed_epoch"]
        selected[run] = chosen
    assert list(selected.values()) == [219,279,297]
    selected_rows, selected_aggregates = [], {}
    for role in ("validation", "test"):
        selected_aggregates[role] = {}
        for group in GROUPS:
            n = group_sizes[role][group]
            counts = []
            for run in RUNS:
                correct = grouped[(run, role, selected[run], group)][1]
                counts.append(correct)
                selected_rows.append({"run_id":run,"paper_role":role,
                                      "completed_epoch":selected[run],"true_group":group,
                                      "examples":n,"correct":correct,
                                      "accuracy_percent":100*correct/n})
            values = [100*c/n for c in counts]
            selected_aggregates[role][group] = {
                "examples_per_run":n,"correct_counts":counts,"percentages":values,
                "mean_percent":statistics.mean(values),
                "sample_sd_percentage_points":statistics.stdev(values)}
    constant_epochs = {
        run: [e for e in range(1,501)
              if int(total[(run,"validation",e)]["predicted_n_1"])==3000]
        for run in RUNS}
    fixed = {}
    for e in (1,50,100,200,300,400,500):
        fixed[str(e)] = {g: next(r for r in means if r["paper_role"]=="validation"
                                and r["completed_epoch"]==e and r["true_group"]==g)
                        for g in GROUPS}
    summary = {
        "figure":BASENAME,"runs":list(RUNS),"features":["k","D_Factor"],"base":3,
        "plotted_pool":"validation","plotted_completed_epochs":[1,500],
        "metric":"Exact greedy valuation accuracy conditional on the TRUE valuation group",
        "higher_group_rule":"For true v3(n_k)>1, success still requires the exact valuation; predicting any value >1 is insufficient.",
        "file_label_to_paper_role":ROLES,"paper_target":"v3(n_k)=-v3(r_k)",
        "class_rows_input":len(seen),"grouped_epoch_rows":len(per_run),
        "weighted_group_totals_match_all_overall_counts":True,
        "group_sizes":group_sizes,"selected_epochs":selected,
        "selection":"Earliest maximum of GLOBAL validation exact accuracy in completed epochs1-300; no per-group selection.",
        "selected_group_accuracies":selected_aggregates,
        "validation_epochs_predicting_one_for_all_examples":constant_epochs,
        "fixed_validation_epochs":fixed,
        "sd":"Sample standard deviation across three run percentages, ddof=1",
        "display_band":"Mean +/- sample SD, clipped to the admissible range0-100 percent for display only",
        "smoothing":None,
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (class_path,overall_path,audit_path)},
        "python_version":sys.version,
    }
    write_csv(args.out/"grouped_per_epoch_metrics.csv",per_run)
    write_csv(args.out/"conditional_curve_metrics.csv",means)
    write_csv(args.out/"selected_group_metrics.csv",selected_rows)
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({
            "font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],
            "mathtext.fontset":"dejavusans","font.size":10,
            "axes.labelsize":10,"axes.titlesize":12,"legend.fontsize":10,
            "xtick.labelsize":10,"ytick.labelsize":10,"axes.linewidth":0.8,
            "axes.facecolor":"white","figure.facecolor":"white",
            "pdf.fonttype":42,"ps.fonttype":42,
        })
        fig,ax=plt.subplots(figsize=(9.2,5.2))
        palette=[("1","#2ca02c",r"$v_3=1$"),
                 ("0","#1f77b4",r"$v_3=0$"),
                 (">1","#d62728",r"$v_3>1$")]
        for group,color,label in palette:
            series=[r for r in means if r["paper_role"]=="validation" and r["true_group"]==group]
            xs=np.array([r["completed_epoch"] for r in series])
            ys=np.array([r["mean_percent"] for r in series])
            lower=np.array([r["display_band_lower_percent"] for r in series])
            upper=np.array([r["display_band_upper_percent"] for r in series])
            ax.fill_between(xs,lower,upper,color=color,alpha=0.13,linewidth=0)
            ax.plot(xs,ys,color=color,linewidth=1.6,label=label)
        ax.set(title=r"Accuracy conditioned on $v_3(n_k)$",xlabel="epoch",
               ylabel=r"$P(\mathrm{model\ correct}\mid v_3)$   (%)",
               xlim=(-25,525),ylim=(-2,103))
        ax.set_xticks(np.arange(0,501,50))
        ax.set_yticks(np.arange(0,101,20))
        ax.set_axisbelow(True)
        ax.grid(True,color="0.9",linewidth=0.8)
        # The new zero-class curve passes through the original legend's
        # position late in training. Keep it on the right, in the clear gap.
        ax.legend(loc="center right",bbox_to_anchor=(1,0.35),frameon=True,framealpha=1,
                  facecolor="white",edgecolor="0.8")
        fig.tight_layout()
        fig.savefig(args.out/(BASENAME+".pdf"),
                    metadata={"Creator":"build_conditional_figure.py","CreationDate":None})
        fig.savefig(args.out/(BASENAME+".png"),dpi=200,facecolor="white")
        plt.close(fig)
        summary["plot_environment"]={
            "reference":"Original Figure5.5, supplied manuscript PDFpage22",
            "matplotlib":matplotlib.__version__,"numpy":np.__version__,
            "font":"DejaVu Sans","mathtext_fontset":"dejavusans",
            "figure_size_inches":[9.2,5.2],"png_dpi":200,
            "source_font_size_pt":10,"source_title_size_pt":12,
            "line_width_pt":1.6,"markers":False,"grid_gray":0.9,
            "legend_order":["1","0",">1"],"legend_location":"right, lowered into clear space",
            "legend_anchor":[1,0.35],
            "legend_colors":[p[1] for p in palette],"band_alpha":0.13,
            "y_limits":[-2,103],"x_limits":[-25,525],"layout":"tight_layout",
            "latex_width":"0.65\\linewidth"}
    write_json(args.out/"conditional_summary.json",summary)
    print(json.dumps({"selected_epochs":selected,"group_sizes":group_sizes,
                      "selected_test":selected_aggregates["test"]},indent=2))


if __name__=="__main__":
    main()
