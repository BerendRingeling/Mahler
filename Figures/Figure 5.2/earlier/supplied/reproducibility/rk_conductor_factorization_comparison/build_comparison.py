#!/usr/bin/env python3
"""Reproduce the conductor and factorization summaries and validation learning curves.

Input: audited per-run CSVs and example-index JSONs produced by analyze_rk_run.py.
Each configuration has three runs. Accuracy counts retain all 3,000 examples.
The saved raw-file labels map to current manuscript roles as follows:
test1 = validation; valid = test. This mapping is verified by example identities.
Epoch selection maximizes greedy same-sign strict factor-two accuracy on
validation in completed epochs 1--300, with earliest-epoch tie breaking.
The plot shows validation scores only; selected-epoch tables report test scores.
The optional exact-accuracy selection is calculated separately and labelled.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from fractions import Fraction
from pathlib import Path

FAMILIES = {
    "rk_wN": ("1000262_9", "1000262_10", "1000262_11"),
    "rk_wfN": ("1000262_21", "1000262_22", "1000262_23"),
    "rk_wfD": ("1000262_18", "1000262_19", "1000262_20"),
}
ROLES = {"test1": "validation", "valid": "test"}
METRICS = {"factor_two": "greedy_factor_two_count", "exact": "greedy_exact_count"}


def mean_sd(counts, denominators):
    values = [float(100 * Fraction(c, n)) for c, n in zip(counts, denominators)]
    return {"mean_percent": statistics.mean(values),
            "sample_sd_percentage_points": statistics.stdev(values),
            "population_sd_percentage_points": statistics.pstdev(values),
            "run_percentages": values}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_epoch(validation, count_column):
    # Deliberately accepts validation rows only. Test data cannot enter the rule.
    rates = {e: Fraction(int(validation[e][count_column]),
                        int(validation[e]["examples"])) for e in range(300)}
    best = max(rates.values())
    ties = [e for e in range(300) if rates[e] == best]
    return ties[0], ties


def learning_diagnostics(tables, curves):
    """Descriptive validation-only thresholds; these do not select checkpoints."""
    individual = []
    for family, run_ids in FAMILIES.items():
        for rid in run_ids:
            rows = tables[rid]["test1"]
            for threshold in (90, 95, 99, 100):
                threshold_count = 30 * threshold
                reached = [e for e in range(300)
                           if int(rows[e]["greedy_factor_two_count"]) >= threshold_count]
                below = [e for e in range(300) if e not in reached]
                sustained = max(below)+2 if below else 1
                individual.append({
                    "configuration": family, "run_id": rid,
                    "validation_threshold_percent": threshold,
                    "first_completed_epoch_reaching_threshold": reached[0]+1 if reached else None,
                    "first_completed_epoch_above_threshold_through_300":
                        sustained if sustained <= 300 else None,
                    "epochs_reaching_threshold_in_first300": len(reached)})
    fixed_checks = [r for r in curves if r["paper_role"] == "validation"
                    and r["metric"] == "factor_two"
                    and r["epoch_number"] in (1, 5, 10, 20, 50, 100, 200, 300)]
    return individual, fixed_checks


def make_plot(curves, output, epochs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from paper_plot_style import (apply_paper_style, paper_style_metadata,
                                  FIGURE_SIZE, BLUE, ORANGE, GREEN,
                                  LINE_WIDTH, BAND_ALPHA)

    apply_paper_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    ymax = 0
    for family, color, label in (("rk_wN", BLUE, r"$N_k$"),
                                  ("rk_wfD", ORANGE, r"$\mathrm{factor}(\Delta_k)$"),
                                  ("rk_wfN", GREEN, r"$\mathrm{factor}(N_k)$")):
        rows = [r for r in curves if r["configuration"] == family
                and r["paper_role"] == "validation"
                and r["metric"] == "factor_two" and r["epoch_number"] <= epochs]
        xs = np.array([r["epoch_number"] for r in rows])
        ys = np.array([r["mean_percent"] for r in rows])
        sds = np.array([r["sample_sd_percentage_points"] for r in rows])
        ax.fill_between(xs, ys-sds, ys+sds,
                        color=color, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(xs, ys, color=color, linewidth=LINE_WIDTH, label=label)
        ymax = max(ymax, float(np.max(ys+sds)))
    ax.set(xlabel="Epoch", ylabel="Validation accuracy (%)",
           xlim=(0, epochs), ylim=(0, 105))
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.legend(loc="lower right", title="Additional input")
    base = "rk_conductor_factorization_validation" + ("" if epochs == 300 else f"_{epochs}")
    fig.savefig(output / (base + ".pdf"), metadata={"Creator": "build_comparison.py", "CreationDate": None})
    fig.savefig(output / (base + ".png"), dpi=200, facecolor="white")
    plt.close(fig)
    style_metadata = {**paper_style_metadata(),
                      "plot_basename": base, "line_width_pt": LINE_WIDTH,
                      "sd_band_alpha": BAND_ALPHA,
                      "png_dpi": 200, "axes_limits": {"x": [0, epochs], "y": [0, 105]},
                      "labels_and_legend_text_changed": False}
    (output / "plot_style_metadata.json").write_text(json.dumps(style_metadata, indent=2)+"\n")
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-epochs", type=int, default=300, choices=(300, 500))
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tables = {}
    sources = {}
    reference = None
    identity_checks = {}
    for family, run_ids in FAMILIES.items():
        for rid in run_ids:
            folder = args.results / rid
            path = folder / "per_epoch_metrics.csv"
            with path.open(newline="") as handle:
                raw = list(csv.DictReader(handle))
            indexed = {label: {} for label in ROLES}
            for row in raw:
                split, e = row["split"], int(row["epoch_index"])
                assert row["paper_role"] == ROLES[split]
                assert e not in indexed[split] and int(row["examples"]) == 3000
                indexed[split][e] = row
            assert len(raw) == 1000
            assert all(set(rows) == set(range(500)) for rows in indexed.values())
            tables[rid] = indexed
            indices = json.loads((folder / "example_index.json").read_text())
            if reference is None:
                reference = indices
            assert indices == reference, f"Different example identities/answers in {rid}"
            assert not ({r["k"] for r in indices["valid"]}
                        & {r["k"] for r in indices["test1"]})
            identity_checks[rid] = {s: len(indices[s]) for s in ROLES}
            sources[rid] = {"configuration": family,
                            "per_epoch_csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    curves = []
    fixed = []
    selections = {}
    selection_rows = []
    selected_aggregates = []
    for family, run_ids in FAMILIES.items():
        selections[family] = {}
        for split, role in ROLES.items():
            for e in range(500):
                for metric, column in METRICS.items():
                    counts = [int(tables[rid][split][e][column]) for rid in run_ids]
                    ns = [int(tables[rid][split][e]["examples"]) for rid in run_ids]
                    stats = mean_sd(counts, ns)
                    row = {"configuration": family, "paper_role": role, "file_label": split,
                           "metric": metric, "epoch_index": e, "epoch_number": e+1,
                           "run_0_count": counts[0], "run_1_count": counts[1], "run_2_count": counts[2],
                           "examples_per_run": 3000,
                           **{k:v for k,v in stats.items() if k != "run_percentages"}}
                    curves.append(row)
                    if e in (299, 499):
                        fixed.append(row)
        for rid in run_ids:
            selections[family][rid] = {}
            for metric, column in METRICS.items():
                chosen, ties = select_epoch(tables[rid]["test1"], column)
                val, test = tables[rid]["test1"][chosen], tables[rid]["valid"][chosen]
                row = {"configuration": family, "run_id": rid, "selection_metric": metric,
                       "selected_epoch_index": chosen, "selected_completed_epoch": chosen+1,
                       "tie_count": len(ties), "validation_correct": int(val[column]),
                       "test_correct": int(test[column]), "examples_per_split": 3000,
                       "validation_accuracy_percent": 100*int(val[column])/3000,
                       "test_accuracy_percent": 100*int(test[column])/3000}
                selection_rows.append(row)
                selections[family][rid][metric] = {**row, "tied_epoch_indices": ties}
        for metric in METRICS:
            for role in ("validation", "test"):
                counts = [selections[family][rid][metric][role+"_correct"] for rid in run_ids]
                stats = mean_sd(counts, [3000]*3)
                selected_aggregates.append({
                    "configuration": family, "selection_metric": metric, "paper_role": role,
                    "run_0_epoch": selections[family][run_ids[0]][metric]["selected_completed_epoch"],
                    "run_1_epoch": selections[family][run_ids[1]][metric]["selected_completed_epoch"],
                    "run_2_epoch": selections[family][run_ids[2]][metric]["selected_completed_epoch"],
                    "run_0_count": counts[0], "run_1_count": counts[1], "run_2_count": counts[2],
                    "examples_per_run": 3000,
                    **{k:v for k,v in stats.items() if k != "run_percentages"}})

    selected_test = {r["configuration"]:r for r in selected_aggregates
                     if r["selection_metric"] == "factor_two" and r["paper_role"] == "test"}
    fixed_test = {e:{r["configuration"]:r for r in fixed
                    if r["metric"] == "factor_two" and r["paper_role"] == "test"
                    and r["epoch_number"] == e} for e in (300,500)}
    differences = {}
    for label, values in [("validation_selected_1_to_300", selected_test)] + [
            (f"fixed_epoch_{e}", fixed_test[e]) for e in (300,500)]:
        reference_mean = values["rk_wN"]["mean_percent"]
        differences[label] = {
            "test_mean_percent_by_configuration": {family: values[family]["mean_percent"]
                                                    for family in FAMILIES},
            "difference_from_conductor_input_percentage_points": {
                family: values[family]["mean_percent"] - reference_mean
                for family in ("rk_wfN", "rk_wfD")}}

    learning_rows, learning_fixed = learning_diagnostics(tables, curves)
    result = {
        "policy": {"paper_role_mapping": ROLES, "decoder": "greedy",
                   "factor_two_definition": "Same sign and strict |r|/2 < |prediction| < 2|r|",
                   "selection_completed_epoch_window": [1,300],
                   "selection_metric": "factor_two (exact is a separately labelled diagnostic)",
                   "tie_break": "Earliest validation maximum", "test_used_for_selection": False,
                   "epoch_index_to_number": "completed epoch = JSON index + 1",
                   "sd_convention": "Sample SD across three run percentages, divisor 2",
                   "population_sd_also_saved": True, "plot_group": "validation",
                   "plot_epochs": args.plot_epochs, "smoothing": "none",
                   "agreed_figure_latex_width": "0.65\\linewidth"},
        "input_metric_sources": sources,
        "all_nine_runs_match_ordered_k_and_canonical_answers": True,
        "identity_checks": identity_checks, "split_overlap": 0,
        "selected_runs": selections, "selected_aggregates": selected_aggregates,
        "fixed_epoch_aggregates": fixed, "comparisons": differences,
        "learning_thresholds_validation_only": learning_rows,
        "validation_accuracy_at_fixed_learning_epochs": learning_fixed,
        "interpretation": [
            "Three training runs per configuration, not matched-seed pairs.",
            "The same 3000 examples per group recur across all runs; runs do not enlarge the test set.",
            "SD measures variability across training runs and is not a confidence interval.",
            "Epoch selection is an explicit calculation from saved predictions for the revision.",
            "Both groups were evaluated throughout training; no claim is made that test predictions were computed only after selection.",
            "The rule does not establish which decisions produced an earlier spreadsheet.",
        ],
    }
    for name, rows in (("per_epoch_comparison.csv", curves), ("fixed_epoch_comparison.csv", fixed),
                       ("selected_epoch_per_run.csv", selection_rows),
                       ("selected_epoch_comparison.csv", selected_aggregates),
                       ("learning_thresholds_validation_only.csv", learning_rows),
                       ("learning_fixed_epoch_checks.csv", learning_fixed)):
        write_csv(args.out/name, rows)
    if not args.no_plot:
        result["plot_basename"] = make_plot(curves, args.out, args.plot_epochs)
    (args.out / "comparison_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"selected_aggregates": selected_aggregates, "comparisons": differences,
                      "all_nine_example_identities_match": True}, indent=2))


if __name__ == "__main__":
    main()
