#!/usr/bin/env python3
"""Compare the main v3 figure data against the independently rebuilt audit.

Run after audit_v3_independent.py:
  python3 compare_main_v3.py --main-dir PATH_TO_MAIN_OUTPUTS \
    --independent-dir PATH_TO_INDEPENDENT_OUTPUTS
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--independent-dir", type=Path, required=True)
    args = parser.parse_args()
    main_dir, indy_dir = args.main_dir, args.independent_dir
    independent = json.loads((indy_dir / "independent_v3_audit.json").read_text())
    summary = json.loads((main_dir / "v3_audit_summary.json").read_text())
    rows = read_rows(main_dir / "all_per_epoch_metrics.csv")
    class_rows = read_rows(main_dir / "all_per_epoch_class_metrics.csv")
    own_class_rows = read_rows(indy_dir / "independent_per_epoch_class_metrics.csv")
    checks = 0
    float_checks = 0
    failures = []

    def check(actual, expected, label, floating=False):
        nonlocal checks, float_checks
        checks += 1
        if floating:
            float_checks += 1
            actual, expected = float(actual), float(expected)
            good = (math.isfinite(actual) and math.isfinite(expected)
                    and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10))
        elif type(expected) is int:
            actual = int(actual)
            good = actual == expected
        else:
            good = actual == expected
        if not good:
            failures.append({"field": label, "actual": actual, "expected": expected})

    def sequence(actual, expected, label, floating=False):
        check(len(actual), len(expected), label + ".length")
        for i, (a, e) in enumerate(zip(actual, expected)):
            check(a, e, f"{label}.{i}", floating)

    def key(r):
        return r["run_id"], r["file_label"], int(r["epoch_index"])

    def ckey(r):
        return (*key(r), int(r["true_v3_n"]))

    main_by_key = {key(r): r for r in rows}
    main_classes = {ckey(r): r for r in class_rows}
    indy_classes = {ckey(r): r for r in own_class_rows}
    check(len(main_by_key), len(rows), "unique_main_epoch_rows")
    check(len(main_classes), len(class_rows), "unique_main_class_rows")
    check(len(indy_classes), len(own_class_rows), "unique_independent_class_rows")
    check(set(main_classes), set(indy_classes), "class_row_keys")
    check(len(rows), 3000, "epoch_row_count")
    check(len(class_rows), 27000, "class_row_count")
    balanced_components = defaultdict(list)
    independent_class_correct_totals = defaultdict(int)
    independent_class_example_totals = defaultdict(int)
    for k, own in indy_classes.items():
        actual = main_classes[k]
        for name, expected in own.items():
            if name == "greedy_exact_percent":
                check(actual[name], expected, f"class.{k}.{name}", floating=True)
            elif name in ("run_id", "file_label", "paper_role"):
                check(actual[name], expected, f"class.{k}.{name}")
            else:
                check(actual[name], int(expected), f"class.{k}.{name}")
        common_key = k[:3]
        balanced_components[common_key].append(int(own["greedy_correct"]) / int(own["examples"]))
        independent_class_correct_totals[common_key] += int(own["greedy_correct"])
        independent_class_example_totals[common_key] += int(own["examples"])
    balanced = {k: 100 * statistics.mean(v) for k, v in balanced_components.items()}
    own_by_key = {}
    direct_fields = {
        "epoch_index": "epoch_index", "completed_epoch": "completed_epoch", "examples": "examples",
        "greedy_exact_count": "greedy_exact_count",
        "sampled_exact_count": "sampled_exact_count",
        "either_exact_count": "either_greedy_or_sampled_exact_count",
        "raw_pair_greedy_exact_count": "greedy_raw_pair_exact_count",
        "greedy_missing": "greedy_missing", "sampled_missing": "sampled_missing",
        "greedy_noninteger_count": "greedy_nonintegral",
        "sampled_noninteger_count": "sampled_nonintegral",
        "greedy_integer_count": "greedy_integer",
        "greedy_negative_paper_valuation_count": "greedy_paper_negative_integer",
    }

    def check_common(actual, own, hist, row_key, prefix):
        for name in ("run_id", "file_label", "paper_role"):
            if name in actual:
                check(actual[name], own[name], prefix + "." + name)
        for main_name, own_name in direct_fields.items():
            check(actual[main_name], own[own_name], prefix + "." + main_name)
        for decoder in ("greedy", "sampled"):
            invalid = sum(own[f"{decoder}_{status}"] for status in
                          ("null", "bad_pair", "zero_denominator"))
            check(actual[f"{decoder}_invalid"], invalid, prefix + f".{decoder}_invalid")
            check(actual[f"{decoder}_exact_percent"], own[f"{decoder}_exact_percent"],
                  prefix + f".{decoder}_exact_percent", floating=True)
        for value in range(4):
            check(actual[f"predicted_n_{value}"], hist.get(str(value), 0),
                  prefix + f".predicted_n_{value}")
        outside = sum(count for label, count in hist.items() if int(label) < 0 or int(label) > 3)
        check(actual["predicted_n_outside_0_to_3"], outside, prefix + ".outside_0_to_3")
        check(actual["balanced_accuracy_percent"], balanced[row_key],
              prefix + ".balanced_accuracy_percent", floating=True)

    for run, audit_run in independent["runs"].items():
        for label, audit_file in audit_run["files"].items():
            check(len(audit_file["per_epoch_metrics"]), 500, f"{run}.{label}.epoch_coverage")
            for epoch, own in enumerate(audit_file["per_epoch_metrics"]):
                k = run, label, epoch
                own_by_key[k] = own
                actual = main_by_key[k]
                hist = audit_file["paper_integer_prediction_histograms"][epoch]
                check_common(actual, own, hist, k, f"epoch.{k}")
                check(independent_class_example_totals[k], 3000, f"class.{k}.example_sum")
                check(independent_class_correct_totals[k], own["greedy_exact_count"],
                      f"class.{k}.correct_sum")
    check(set(main_by_key), set(own_by_key), "epoch_row_keys")
    check(summary["policy"]["paper_role_mapping"],
          independent["matched_file_labels_to_paper_roles"], "paper_role_mapping")
    sequence(summary["policy"]["selection_completed_epoch_window"], [1, 300], "selection_window")
    check(summary["policy"]["test_used_for_selection"], False, "selection_excludes_test")
    check(summary["all_runs_share_ordered_example_and_feature_records"], True, "identical_run_inputs")
    check(summary["validation_test_k_overlap"], 0, "disjoint_pools")
    for role in ("validation", "test"):
        actual = summary["selected_aggregates"][role]
        own = independent["selected_epoch_aggregates"][role]
        sequence(actual["counts"], own["counts"], f"selected.{role}.counts")
        sequence(actual["denominators"], [3000] * 3, f"selected.{role}.denominators")
        sequence(actual["percentages"], own["percentages"], f"selected.{role}.percentages", floating=True)
        for name in ("mean_percent", "sample_sd_percentage_points", "population_sd_percentage_points"):
            check(actual[name], own[name], f"selected.{role}.{name}", floating=True)
    for run, actual_run in summary["runs"].items():
        own_run = independent["runs"][run]
        actual = actual_run["selection"]
        own = own_run["selection"]
        for a, o in (("completed_epoch", "selected_completed_epoch"),
                     ("epoch_index", "selected_epoch_index"), ("tie_count", "tie_count"),
                     ("validation_count", "maximum_count")):
            check(actual[a], own[o], f"{run}.selection.{a}")
        sequence(actual["tied_completed_epochs"], own["tied_completed_epochs"], f"{run}.selection.ties")
        check(actual["validation_percent"], own["maximum_percent"], f"{run}.selection.validation_percent", True)
        check(actual["test_count"], own["test"]["greedy_exact_count"], f"{run}.selection.test_count")
        check(actual["test_percent"], own["test"]["greedy_exact_percent"], f"{run}.selection.test_percent", True)
        test_label = own["test"]["file_label"]
        test_key = run, test_label, own["selected_epoch_index"]
        check(actual["test_balanced_accuracy_percent"], balanced[test_key],
              f"{run}.selection.test_balanced_accuracy", True)
        maximum = actual_run["global_validation_maximum_all_500_epochs"]
        own_maximum = own_run["global_validation_maximum_1_to_500"]
        check(maximum["count"], own_maximum["maximum_count"], f"{run}.global_max.count")
        check(maximum["percent"], own_maximum["maximum_percent"], f"{run}.global_max.percent", True)
        check(maximum["earliest_completed_epoch"], own_maximum["selected_completed_epoch"],
              f"{run}.global_max.earliest")
        sequence(maximum["all_tied_completed_epochs"], own_maximum["tied_completed_epochs"],
                 f"{run}.global_max.ties")
        for epoch_number, role_rows in actual_run["fixed_epochs"].items():
            for role, row in role_rows.items():
                label = row["file_label"]
                e = int(epoch_number) - 1
                k = run, label, e
                own = own_by_key[k]
                hist = own_run["files"][label]["paper_integer_prediction_histograms"][e]
                check_common(row, own, hist, k, f"{run}.fixed.{epoch_number}.{role}")
    files = {
        "main_per_epoch_metrics": main_dir / "all_per_epoch_metrics.csv",
        "main_class_metrics": main_dir / "all_per_epoch_class_metrics.csv",
        "main_summary": main_dir / "v3_audit_summary.json",
        "independent_audit": indy_dir / "independent_v3_audit.json",
        "independent_class_metrics": indy_dir / "independent_per_epoch_class_metrics.csv",
    }
    result = {
        "all_comparisons_pass": not failures,
        "scalar_comparison_count": checks,
        "floating_comparisons": float_checks,
        "exact_comparisons": checks - float_checks,
        "absolute_float_tolerance": 1e-10,
        "per_epoch_metric_rows_compared": len(rows),
        "per_class_metric_rows_compared": len(class_rows),
        "epochs_per_run_and_pool": 500, "runs": 3, "pools_per_run": 2,
        "class_counts_and_conditional_correct_counts_recomputed_from_raw_predictions": True,
        "balanced_accuracy_checked": True,
        "unchecked_main_fields": ["sampled_almost_count_including_zero"],
        "source_hashes": {name: digest(path) for name, path in files.items()},
        "failure_count": len(failures), "failures": failures[:100],
    }
    (indy_dir / "main_independent_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_hashes",)}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
