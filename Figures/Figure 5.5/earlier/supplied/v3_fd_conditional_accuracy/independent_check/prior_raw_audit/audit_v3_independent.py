#!/usr/bin/env python3
"""Independent raw-prediction audit for the three new v3_fD runs.

python3 audit_v3_independent.py --inputs PATH_TO_EXTRACTED_RUNS \
  --arithmetic-index evaluation_arithmetic_inputs.json --out audit_output

The raw targets encode v3(r_k), as verified against stored rational targets.
Both targets and predictions are negated to describe v3(n_k) in the paper.
This bijection preserves exact-match accuracy. No model is evaluated here.
"""
from __future__ import annotations

import argparse
import bz2
from collections import Counter
import csv
from fractions import Fraction
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics

RUNS = ("1000260_9", "1000260_10", "1000260_11")
LABELS = ("valid", "test1")
EPOCH_KEYS = {str(i) for i in range(500)}
MISSING = object()


def no_duplicate_keys(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Repeated JSON object key: {key!r}")
        out[key] = value
    return out


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vp_int(value, prime=3):
    value = abs(value)
    if value == 0:
        raise ValueError("Valuation of zero is not finite")
    out = 0
    while value % prime == 0:
        out += 1
        value //= prime
    return out


def raw_valuation(pair):
    value = Fraction(*pair)
    return vp_int(value.numerator) - vp_int(value.denominator)


def parse_pair(value):
    """Return status, normalized numerator, denominator, noncanonical flag."""
    if value is MISSING:
        return "missing", None, None, False
    if value is None:
        return "null", None, None, False
    if (type(value) is not list or len(value) != 2
            or type(value[0]) is not int or type(value[1]) is not int):
        return "bad_pair", None, None, False
    n, d = value
    if d == 0:
        return "zero_denominator", None, None, False
    if d < 0:
        n, d = -n, -d
    g = math.gcd(abs(n), d)
    n, d = n // g, d // g
    status = "integer" if d == 1 else "nonintegral"
    return status, n, d, value != [n, d]


def empty_row(run, label, role, epoch, n):
    row = {
        "run_id": run, "file_label": label, "paper_role": role,
        "epoch_index": epoch, "completed_epoch": epoch + 1, "examples": n,
        "greedy_exact_count": 0, "greedy_raw_pair_exact_count": 0,
        "greedy_sign_flipped_exact_count": 0,
        "sampled_exact_count": 0, "sampled_raw_pair_exact_count": 0,
        "either_greedy_or_sampled_exact_count": 0,
        "greedy_noncanonical": 0, "sampled_noncanonical": 0,
        "greedy_paper_negative_integer": 0,
        "greedy_paper_label_0_to_3_count": 0,
    }
    for decoder in ("greedy", "sampled"):
        for status in ("missing", "null", "bad_pair", "zero_denominator",
                       "nonintegral", "integer"):
            row[f"{decoder}_{status}"] = 0
    return row


def aggregate(counts):
    percentages = [100 * x / 3000 for x in counts]
    return {
        "counts": counts, "examples_per_run": 3000,
        "percentages": percentages,
        "mean_percent": statistics.mean(percentages),
        "sample_sd_percentage_points": statistics.stdev(percentages),
        "population_sd_percentage_points": statistics.pstdev(percentages),
    }


def log_records(path):
    records = {}
    for line in path.read_text().splitlines():
        if "__log__:" not in line:
            continue
        obj = json.loads(line.split("__log__:", 1)[1])
        epoch = obj["epoch"]
        if epoch in records:
            raise ValueError(f"Repeated log epoch {epoch}")
        records[epoch] = obj
    assert set(records) == set(range(500))
    return records


def select(rows, end=300):
    window = rows[:end]
    maximum = max(r["greedy_exact_count"] for r in window)
    ties = [r["completed_epoch"] for r in window if r["greedy_exact_count"] == maximum]
    return {
        "selected_completed_epoch": ties[0],
        "selected_epoch_index": ties[0] - 1,
        "maximum_count": maximum,
        "maximum_percent": maximum / 30,
        "tied_completed_epochs": ties,
        "tie_count": len(ties),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--arithmetic-index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arithmetic = json.loads(args.arithmetic_index.read_text())
    base_roles = arithmetic["file_label_to_paper_role"]
    base = arithmetic["by_file_label"]
    assert base_roles == {"test1": "validation", "valid": "test"}
    base_sets = {label: {r["k"] for r in rows} for label, rows in base.items()}
    assert all(len(s) == 3000 for s in base_sets.values())
    assert not (base_sets["valid"] & base_sets["test1"])
    base_by_k = {label: {r["k"]: r for r in rows} for label, rows in base.items()}
    reference_inputs = {}
    result = {
        "policy": {
            "decoder": "greedy",
            "metric": "Exact equality of rational decoded outputs",
            "raw_target": "v3(r_k), checked from the saved rational r_k",
            "paper_target": "v3(n_k) = -v3(r_k)",
            "paper_prediction": "Negative of the decoded raw prediction",
            "selection_window_completed_epochs": [1, 300],
            "tie_rule": "Earliest validation maximum",
            "test_used_for_selection": False,
            "invalid_and_missing_predictions": "Failures, denominator remains 3000",
            "noninteger_predictions": "Outside integer valuation label space; failures",
            "sd_convention": "Sample SD across the three run percentages (ddof=1)",
        },
        "arithmetic_index_sha256": sha256(args.arithmetic_index),
        "runs": {},
        "matched_file_labels_to_paper_roles": {},
    }
    flat = []
    class_flat = []
    metrics_by_run = {}
    log_compare_count = 0
    anomalous_examples = []
    log_mismatches = []
    normalization_only_matches = []
    for run in RUNS:
        run_result = {"files": {}, "selection": {}, "log_reconciliation": {}}
        role_metrics = {}
        logs = log_records(args.inputs / run / "train.log")
        for label in LABELS:
            path = args.inputs / run / f"eval.{label}.epochs_0_499.json.bz2"
            with bz2.open(path, "rt", encoding="utf-8") as handle:
                data = json.load(handle, object_pairs_hook=no_duplicate_keys)
            assert type(data) is dict and len(data) == 3000
            ids = list(data)
            ks = [record["problem"]["k"] for record in data.values()]
            assert all(type(k) is int for k in ks)
            assert len(set(ks)) == len(ks)
            matches = [base_label for base_label, values in base_sets.items()
                       if set(ks) == values]
            assert len(matches) == 1, (run, label, matches)
            matched_label = matches[0]
            role = base_roles[matched_label]
            if label in result["matched_file_labels_to_paper_roles"]:
                assert result["matched_file_labels_to_paper_roles"][label] == role
            else:
                result["matched_file_labels_to_paper_roles"][label] = role
            rows = [empty_row(run, label, role, e, len(data)) for e in range(500)]
            target_counts = Counter()
            class_correct = {}
            problem_keys = Counter()
            unknown_keys = {"greedy": 0, "sampled": 0}
            first_input_sequence = []
            prediction_histograms = [Counter() for _ in range(500)]
            for example_id, record in data.items():
                problem = record["problem"]
                problem_keys[",".join(sorted(problem))] += 1
                original = base_by_k[matched_label][problem["k"]]
                status, numerator, denominator, noncanonical = parse_pair(record["answer"])
                assert status == "integer", (run, label, example_id, record["answer"])
                expected_raw = raw_valuation(original["answer"])
                assert numerator == expected_raw
                assert -numerator == -raw_valuation(original["stored_answer"])
                target_counts[-numerator] += 1
                class_correct.setdefault(-numerator, [0] * 500)
                # Canonical input/target sequence equality across all three runs.
                first_input_sequence.append([example_id, problem, record["answer"]])
                gmap, smap = record["guesses_greedy"], record["guesses_sampled"]
                assert type(gmap) is dict and type(smap) is dict
                unknown_keys["greedy"] += len(set(gmap) - EPOCH_KEYS)
                unknown_keys["sampled"] += len(set(smap) - EPOCH_KEYS)
                for e in range(500):
                    row = rows[e]
                    outcomes = {}
                    for decoder, mapping in (("greedy", gmap), ("sampled", smap)):
                        raw = mapping.get(str(e), MISSING)
                        kind, n, d, is_noncanonical = parse_pair(raw)
                        row[f"{decoder}_{kind}"] += 1
                        row[f"{decoder}_noncanonical"] += is_noncanonical
                        correct = kind == "integer" and n == numerator
                        outcomes[decoder] = correct
                        row[f"{decoder}_exact_count"] += correct
                        row[f"{decoder}_raw_pair_exact_count"] += raw == record["answer"]
                        if correct and raw != record["answer"] and len(normalization_only_matches) < 100:
                            normalization_only_matches.append({
                                "run_id": run, "file_label": label, "example_id": example_id,
                                "k": problem["k"], "completed_epoch": e + 1,
                                "decoder": decoder, "raw": raw, "target": record["answer"],
                            })
                        if decoder == "greedy":
                            sign_flip_match = kind == "integer" and -n == -numerator
                            row["greedy_sign_flipped_exact_count"] += sign_flip_match
                            class_correct[-numerator][e] += correct
                            if kind == "integer":
                                paper_prediction = -n
                                row["greedy_paper_negative_integer"] += paper_prediction < 0
                                row["greedy_paper_label_0_to_3_count"] += 0 <= paper_prediction <= 3
                                prediction_histograms[e][paper_prediction] += 1
                        if kind not in ("integer", "missing") and len(anomalous_examples) < 30:
                            anomalous_examples.append({
                                "run_id": run, "file_label": label, "example_id": example_id,
                                "k": problem["k"], "completed_epoch": e + 1,
                                "decoder": decoder, "status": kind, "raw": raw,
                            })
                    row["either_greedy_or_sampled_exact_count"] += outcomes["greedy"] or outcomes["sampled"]
            assert not any(unknown_keys.values())
            if label in reference_inputs:
                assert first_input_sequence == reference_inputs[label]
            else:
                reference_inputs[label] = first_input_sequence
            for row in rows:
                assert row["greedy_exact_count"] == row["greedy_sign_flipped_exact_count"]
                row["greedy_exact_percent"] = 100 * row["greedy_exact_count"] / 3000
                row["sampled_exact_percent"] = 100 * row["sampled_exact_count"] / 3000
                row["greedy_invalid_total"] = sum(row[f"greedy_{k}"] for k in
                                                ("null", "bad_pair", "zero_denominator", "nonintegral"))
                row["sampled_invalid_total"] = sum(row[f"sampled_{k}"] for k in
                                                 ("null", "bad_pair", "zero_denominator", "nonintegral"))
                assert sum(row[f"greedy_{k}"] for k in
                           ("missing", "null", "bad_pair", "zero_denominator", "nonintegral", "integer")) == 3000
                assert sum(row[f"sampled_{k}"] for k in
                           ("missing", "null", "bad_pair", "zero_denominator", "nonintegral", "integer")) == 3000
                log = logs[row["epoch_index"]]
                for metric, actual, logged, raw_pair in (
                    ("greedy_acc", row["greedy_exact_percent"],
                     log[f"{label}_ELLIPTIC_CURVE_greedy_acc"], row["greedy_raw_pair_exact_count"]/30),
                    ("acc", row["either_greedy_or_sampled_exact_count"]/30,
                     log[f"{label}_ELLIPTIC_CURVE_acc"], None)):
                    if not math.isclose(actual, logged, abs_tol=1e-10):
                        mismatch = {"run_id": run, "file_label": label,
                                    "completed_epoch": row["completed_epoch"], "metric": metric,
                                    "rational_score_percent": actual, "logged_percent": logged,
                                    "raw_pair_score_percent": raw_pair,
                                    "row": row}
                        log_mismatches.append(mismatch)
                        print("LOG MISMATCH "+json.dumps({k:v for k,v in mismatch.items() if k!="row"}),flush=True)
                log_compare_count += 2
            run_result["files"][label] = {
                "paper_role": role, "matched_prior_file_label": matched_label,
                "sha256": sha256(path), "examples": len(data), "distinct_k": len(set(ks)),
                "ordered_k_matches_prior_index": ks == [r["k"] for r in base[matched_label]],
                "ordered_example_ids_match_prior_index": ids == [r["example_id"] for r in base[matched_label]],
                "all_raw_targets_equal_v3_of_stored_r": True,
                "raw_to_paper_label_sign": -1,
                "problem_keys_counts": dict(problem_keys),
                "paper_target_class_counts": {str(k): target_counts[k] for k in sorted(target_counts)},
                "constant_paper_label_1_correct_count": target_counts[1],
                "prediction_rows_per_decoder": len(data) * 500,
                "unknown_epoch_key_count": unknown_keys,
                "totals_over_all_epochs": {
                    key: sum(row[key] for row in rows)
                    for key in rows[0] if key.startswith(("greedy_", "sampled_"))
                    and not key.endswith("_percent")
                },
                "raw_pair_vs_rational_correct_count_differences": {
                    "greedy": sum(row["greedy_exact_count"] - row["greedy_raw_pair_exact_count"] for row in rows),
                    "sampled": sum(row["sampled_exact_count"] - row["sampled_raw_pair_exact_count"] for row in rows),
                },
                "per_epoch_metrics": rows,
                "paper_integer_prediction_histograms": [
                    {str(k): hist[k] for k in sorted(hist)} for hist in prediction_histograms
                ],
            }
            for e in range(500):
                for true_value in sorted(target_counts):
                    count = target_counts[true_value]
                    correct = class_correct[true_value][e]
                    class_flat.append({
                        "run_id": run, "file_label": label, "paper_role": role,
                        "epoch_index": e, "completed_epoch": e + 1,
                        "true_v3_n": true_value, "examples": count,
                        "greedy_correct": correct,
                        "greedy_exact_percent": 100 * correct / count,
                    })
            role_metrics[role] = rows
            flat.extend(rows)
            del data, first_input_sequence
            gc.collect()
            print(f"Checked {run} {label}: {role}, all targets and 500 epochs", flush=True)
        assert set(role_metrics) == {"validation", "test"}
        chosen = select(role_metrics["validation"], 300)
        e = chosen["selected_epoch_index"]
        run_result["selection"] = {
            **chosen, "validation": role_metrics["validation"][e],
            "test": role_metrics["test"][e],
        }
        run_result["global_validation_maximum_1_to_500"] = select(role_metrics["validation"], 500)
        run_result["validation_diagnostics"] = {}
        for lo, hi in [(1, 10), (1, 50), (51, 100), (101, 200), (201, 300),
                       (301, 400), (401, 500), (451, 500)]:
            rs = role_metrics["validation"][lo - 1:hi]
            run_result["validation_diagnostics"][f"epochs_{lo}_to_{hi}"] = {
                "mean_accuracy_percent": statistics.mean(r["greedy_exact_percent"] for r in rs),
                "minimum_accuracy_percent": min(r["greedy_exact_percent"] for r in rs),
                "maximum_accuracy_percent": max(r["greedy_exact_percent"] for r in rs),
            }
        run_result["log_reconciliation"] = {
            "all_500_epochs_each_pool_agree": not any(x["run_id"]==run for x in log_mismatches),
            "greedy_acc": "Greedy exact rational accuracy",
            "acc": "Either greedy or sampled exact rational success",
        }
        result["runs"][run] = run_result
        metrics_by_run[run] = role_metrics
    result["all_three_runs_share_ordered_inputs_and_targets_by_pool"] = True
    result["evaluation_pools_disjoint"] = True
    result["log_field_comparisons"] = log_compare_count
    result["anomalous_examples"] = anomalous_examples
    result["log_mismatches"] = log_mismatches
    result["normalization_only_matches"] = normalization_only_matches
    result["selected_epoch_aggregates"] = {
        role: aggregate([result["runs"][run]["selection"][role]["greedy_exact_count"] for run in RUNS])
        for role in ("validation", "test")
    }
    result["fixed_epoch_aggregates"] = {
        str(epoch): {
            role: aggregate([metrics_by_run[run][role][epoch - 1]["greedy_exact_count"] for run in RUNS])
            for role in ("validation", "test")
        } for epoch in (1, 5, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
    }
    mean_counts = [statistics.mean(metrics_by_run[run]["validation"][e]["greedy_exact_count"] for run in RUNS)
                   for e in range(500)]
    best = max(mean_counts)
    result["mean_validation_curve_global_maximum"] = {
        "mean_accuracy_percent": best / 30,
        "completed_epochs": [e + 1 for e, value in enumerate(mean_counts) if value == best],
    }
    result["mean_validation_curve_last_50_epochs"] = {
        "mean_accuracy_percent": statistics.mean(mean_counts[450:]) / 30,
        "min_accuracy_percent": min(mean_counts[450:]) / 30,
        "max_accuracy_percent": max(mean_counts[450:]) / 30,
    }
    with (args.out / "independent_per_epoch_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    with (args.out / "independent_per_epoch_class_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_flat[0]))
        writer.writeheader()
        writer.writerows(class_flat)
    (args.out / "independent_v3_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    small = {
        "file_roles": result["matched_file_labels_to_paper_roles"],
        "selected_epochs": {run: result["runs"][run]["selection"]["selected_completed_epoch"] for run in RUNS},
        "selected_epoch_aggregates": result["selected_epoch_aggregates"],
        "global_validation_maxima": {run: result["runs"][run]["global_validation_maximum_1_to_500"] for run in RUNS},
        "mean_validation_curve_global_maximum": result["mean_validation_curve_global_maximum"],
        "mean_validation_curve_last_50_epochs": result["mean_validation_curve_last_50_epochs"],
        "fixed_epoch_300": result["fixed_epoch_aggregates"]["300"],
        "fixed_epoch_500": result["fixed_epoch_aggregates"]["500"],
    }
    (args.out / "independent_v3_summary.json").write_text(json.dumps(small, indent=2) + "\n")
    print(json.dumps(small, indent=2), flush=True)


if __name__ == "__main__":
    main()
