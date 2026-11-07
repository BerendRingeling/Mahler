#!/usr/bin/env python3
"""Audit saved v3_fD predictions and export reproducible accuracy tables.

Raw integer targets encode v_3(r_k). Paper valuations are their negatives.
No model training, inference, or additional experiment is performed.
"""
from __future__ import annotations
import argparse
import bz2
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

RUNS = ("1000260_9", "1000260_10", "1000260_11")
ROLES = {"test1": "validation", "valid": "test"}
EPOCHS = 500
SELECTION_EPOCHS = 300


def load_json(path):
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def valuation3(value):
    value = abs(value)
    if not value:
        raise ValueError("Valuation of zero is undefined")
    result = 0
    while value % 3 == 0:
        value //= 3
        result += 1
    return result


def decode_pair(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    n, d = value
    if type(n) is not int or type(d) is not int or d == 0:
        return None
    return (n, d) if d > 0 else (-n, -d)


def exact(pair, target):
    return pair is not None and pair[0] == target * pair[1]


def almost(pair, target):
    if pair is None:
        return False
    n, d = pair
    if target == 0:
        return n == 0
    return n * target > 0 and abs(n)*2 > abs(target)*d and abs(n) < 2*abs(target)*d


def aggregate(counts, denominators):
    percentages = [100*c/n for c, n in zip(counts, denominators)]
    return {
        "counts": counts, "denominators": denominators,
        "percentages": percentages,
        "mean_percent": statistics.mean(percentages),
        "sample_sd_percentage_points": statistics.stdev(percentages),
        "population_sd_percentage_points": statistics.pstdev(percentages),
    }


def log_metrics(path):
    records = {}
    for line in path.read_text().splitlines():
        if "__log__:" not in line:
            continue
        x = json.loads(line.split("__log__:", 1)[1])
        epoch = int(x["epoch"])
        if epoch in records:
            raise ValueError("Duplicate logged epoch")
        records[epoch] = x
    assert set(records) == set(range(EPOCHS))
    return records


def analyze_file(path, label, baseline, logs):
    data = load_json(path)
    assert len(data) == len(baseline) == 3000
    expected_keys = {str(e) for e in range(EPOCHS)}
    counts = [Counter() for _ in range(EPOCHS)]
    class_correct = [Counter() for _ in range(EPOCHS)]
    truth_counts = Counter()
    index = []
    all_features = []
    malformed_examples = []
    for (example_id, item), old in zip(data.items(), baseline):
        problem = item["problem"]
        assert set(problem) == {"k", "D_Factor"}, (example_id, list(problem))
        assert example_id == old["example_id"] and problem["k"] == old["k"]
        assert type(problem["k"]) is int
        answer_pair = decode_pair(item["answer"])
        assert answer_pair is not None
        assert answer_pair[0] % answer_pair[1] == 0
        raw_target = answer_pair[0] // answer_pair[1]
        expected_raw = valuation3(old["answer"][0]) - valuation3(old["answer"][1])
        assert raw_target == expected_raw, (label, example_id, raw_target, expected_raw)
        true_n = -raw_target
        truth_counts[true_n] += 1
        index.append({"example_id": example_id, "k": problem["k"],
                      "raw_v3_r": raw_target, "paper_v3_n": true_n})
        all_features.append({"example_id": example_id, "k": problem["k"],
                             "D_Factor": problem["D_Factor"], "raw_answer": item["answer"]})
        for decoder in ("guesses_greedy", "guesses_sampled"):
            assert isinstance(item[decoder], dict)
            extra = set(item[decoder]) - expected_keys
            assert not extra, (example_id, decoder, extra)
        for e in range(EPOCHS):
            key = str(e)
            row = counts[e]
            gv = item["guesses_greedy"].get(key)
            sv = item["guesses_sampled"].get(key)
            g, s = decode_pair(gv), decode_pair(sv)
            gmissing = key not in item["guesses_greedy"]
            smissing = key not in item["guesses_sampled"]
            row["greedy_missing"] += gmissing
            row["sampled_missing"] += smissing
            row["greedy_invalid"] += not gmissing and g is None
            row["sampled_invalid"] += not smissing and s is None
            gexact, sexact = exact(g, raw_target), exact(s, raw_target)
            row["greedy_exact_count"] += gexact
            row["sampled_exact_count"] += sexact
            row["either_exact_count"] += gexact or sexact
            row["raw_pair_greedy_exact_count"] += gv == item["answer"]
            row["sampled_almost_count_including_zero"] += almost(s, raw_target)
            class_correct[e][true_n] += gexact
            if g is not None:
                if g[0] % g[1] == 0:
                    predicted_n = -(g[0] // g[1])
                    row["greedy_integer_count"] += 1
                    row["greedy_negative_paper_valuation_count"] += predicted_n < 0
                    if predicted_n in (0, 1, 2, 3):
                        row["predicted_n_"+str(predicted_n)] += 1
                    else:
                        row["predicted_n_outside_0_to_3"] += 1
                else:
                    row["greedy_noninteger_count"] += 1
            if s is not None:
                row["sampled_noninteger_count"] += s[0] % s[1] != 0
            if (gmissing or g is None or (g[0] % g[1] != 0)) and len(malformed_examples) < 20:
                malformed_examples.append({"example_id": example_id, "k": problem["k"],
                                           "epoch_index": e, "greedy": gv, "missing": gmissing})
    keys = ["greedy_exact_count", "sampled_exact_count", "either_exact_count",
            "raw_pair_greedy_exact_count", "greedy_missing", "sampled_missing",
            "greedy_invalid", "sampled_invalid", "greedy_noninteger_count",
            "sampled_noninteger_count", "greedy_integer_count",
            "greedy_negative_paper_valuation_count", "predicted_n_0", "predicted_n_1",
            "predicted_n_2", "predicted_n_3", "predicted_n_outside_0_to_3",
            "sampled_almost_count_including_zero"]
    rows, class_rows, log_checks, log_mismatches = [], [], [], []
    for e, count in enumerate(counts):
        row = {"file_label": label, "paper_role": ROLES[label], "epoch_index": e,
               "completed_epoch": e+1, "examples": len(data),
               **{k: int(count[k]) for k in keys},
               "greedy_exact_percent": 100*count["greedy_exact_count"]/len(data),
               "sampled_exact_percent": 100*count["sampled_exact_count"]/len(data),
               "balanced_accuracy_percent": statistics.mean(
                   100*class_correct[e][target]/n for target,n in truth_counts.items())}
        rows.append(row)
        for target, n in sorted(truth_counts.items()):
            class_rows.append({"file_label": label, "paper_role": ROLES[label],
                               "epoch_index": e, "completed_epoch": e+1,
                               "true_v3_n": target, "examples": n,
                               "greedy_correct": class_correct[e][target],
                               "greedy_exact_percent": 100*class_correct[e][target]/n})
        logged = logs[e]
        expected = {
            label+"_ELLIPTIC_CURVE_greedy_acc": row["greedy_exact_percent"],
            label+"_ELLIPTIC_CURVE_acc": 100*count["either_exact_count"]/len(data),
            label+"_ELLIPTIC_CURVE_avg_almost_correctness":
                count["sampled_almost_count_including_zero"]/len(data),
        }
        for key, value in expected.items():
            actual = logged[key]
            if abs(value - actual) > 1e-10:
                log_mismatches.append({
                    "epoch_index": e, "completed_epoch": e+1, "metric": key,
                    "recomputed_from_saved_predictions": value, "logged": actual,
                    "logged_minus_recomputed": actual-value})
        log_checks.append({"epoch_index": e, "fields_reconciled": list(expected)})
    audit = {
        "file_label": label, "paper_role": ROLES[label], "examples": len(data),
        "all_ordered_ids_and_k_match_prior_pool": True,
        "all_raw_targets_equal_v3_of_prior_rational": True,
        "paper_target_is_negated_raw_target": True,
        "target_class_counts": dict(sorted(truth_counts.items())),
        "constant_prediction_1_count": truth_counts[1],
        "constant_prediction_1_accuracy_percent": 100*truth_counts[1]/len(data),
        "all_500_logged_greedy_and_either_exact_and_sampled_almost_fields_reconcile": not log_mismatches,
        "log_metric_mismatches": log_mismatches,
        "raw_pair_accuracy_equals_rational_accuracy_at_all_epochs":
            all(r["raw_pair_greedy_exact_count"] == r["greedy_exact_count"] for r in rows),
        "total_across_500_epochs": {k: sum(r[k] for r in rows) for k in keys
                                    if "missing" in k or "invalid" in k or "noninteger" in k},
        "malformed_examples_first_20": malformed_examples,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return rows, class_rows, audit, index, all_features, data


def snapshot(data, epoch_index):
    records = []
    confusion = Counter()
    for example_id, item in data.items():
        target = Fraction(*item["answer"])
        pair = decode_pair(item["guesses_greedy"].get(str(epoch_index)))
        pred = None if pair is None else Fraction(*pair)
        paper_pred = None if pred is None or pred.denominator != 1 else -pred.numerator
        truth = -int(target)
        confusion[(truth, "invalid" if paper_pred is None else str(paper_pred))] += 1
        records.append({"example_id": example_id, "k": item["problem"]["k"],
                        "paper_v3_n": truth,
                        "raw_target": item["answer"],
                        "raw_greedy": item["guesses_greedy"].get(str(epoch_index)),
                        "paper_greedy_v3_n": paper_pred,
                        "greedy_exact": pred == target})
    return {"epoch_index": epoch_index, "completed_epoch": epoch_index+1,
            "records": records,
            "confusion": [{"true_v3_n": t, "predicted_v3_n": p, "count": c}
                          for (t,p),c in sorted(confusion.items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--arithmetic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arithmetic = load_json(args.arithmetic)
    assert arithmetic["file_label_to_paper_role"] == ROLES
    baseline = arithmetic["by_file_label"]
    assert not ({x["k"] for x in baseline["test1"]} & {x["k"] for x in baseline["valid"]})
    all_rows, class_rows, run_summaries, snapshots = [], [], {}, {}
    shared_index, shared_features = {}, {}
    for run in RUNS:
        folder = args.inputs/run
        out = args.out/"per_run"/run
        out.mkdir(parents=True, exist_ok=True)
        logs = log_metrics(folder/"train.log")
        data_tables, run_class_rows, audits = {}, [], {}
        selected = None
        snapshots[run] = {}
        for label in ROLES:
            path = folder/f"eval.{label}.epochs_0_499.json.bz2"
            rows, classes, audit, index, features, data = analyze_file(
                path, label, baseline[label], logs)
            if label not in shared_index:
                shared_index[label], shared_features[label] = index, features
            else:
                assert index == shared_index[label]
                assert features == shared_features[label]
            data_tables[label] = rows
            run_class_rows.extend(classes)
            audits[label] = audit
            if label == "test1":
                eligible = rows[:SELECTION_EPOCHS]
                maximum = max(r["greedy_exact_count"] for r in eligible)
                ties = [r["completed_epoch"] for r in eligible if r["greedy_exact_count"] == maximum]
                chosen = ties[0]
                selected = {"completed_epoch": chosen, "epoch_index": chosen-1,
                            "validation_count": maximum, "tied_completed_epochs": ties,
                            "tie_count": len(ties)}
            snapshots[run][label] = snapshot(data, selected["epoch_index"])
            del data
        tables = data_tables["test1"] + data_tables["valid"]
        write_csv(out/"per_epoch_metrics.csv", tables)
        write_csv(out/"per_epoch_class_metrics.csv", run_class_rows)
        write_json(out/"audit_summary.json", audits)
        write_json(out/"example_index.json", shared_index)
        for row in tables:
            all_rows.append({"run_id": run, **row})
        for row in run_class_rows:
            class_rows.append({"run_id": run, **row})
        chosen = selected["epoch_index"]
        global_max = max(r["greedy_exact_count"] for r in data_tables["test1"])
        global_ties = [r["completed_epoch"] for r in data_tables["test1"]
                       if r["greedy_exact_count"] == global_max]
        run_summaries[run] = {
            "selection": {
                **selected,
                "validation_percent": data_tables["test1"][chosen]["greedy_exact_percent"],
                "test_count": data_tables["valid"][chosen]["greedy_exact_count"],
                "test_percent": data_tables["valid"][chosen]["greedy_exact_percent"],
                "test_balanced_accuracy_percent": data_tables["valid"][chosen]["balanced_accuracy_percent"],
            },
            "global_validation_maximum_all_500_epochs": {
                "count": global_max, "percent": 100*global_max/3000,
                "earliest_completed_epoch": global_ties[0], "all_tied_completed_epochs": global_ties},
            "fixed_epochs": {str(e): {ROLES[label]: data_tables[label][e-1] for label in ROLES}
                             for e in (1, 50, 100, 200, 300, 400, 500)},
            "integrity": audits,
        }
        print("Audited", run, "selected epoch", selected["completed_epoch"], flush=True)
    write_csv(args.out/"all_per_epoch_metrics.csv", all_rows)
    write_csv(args.out/"all_per_epoch_class_metrics.csv", class_rows)
    write_json(args.out/"selected_epoch_predictions.json", snapshots)
    write_json(args.out/"shared_example_index.json", shared_index)
    write_json(args.out/"shared_feature_index.json", shared_features)
    selected_rows = [{"run_id": run, **s["selection"]} for run,s in run_summaries.items()]
    # Store tie lists in CSV as JSON for clear portable reading.
    for row in selected_rows:
        row["tied_completed_epochs"] = json.dumps(row["tied_completed_epochs"])
    write_csv(args.out/"selected_epoch_per_run.csv", selected_rows)
    aggregates = {}
    for role in ("validation", "test"):
        counts = [run_summaries[r]["selection"][role+"_count"] for r in RUNS]
        aggregates[role] = aggregate(counts, [3000]*len(RUNS))
    summary = {
        "policy": {
            "runs": list(RUNS), "configuration": "v3_fD", "features": ["k", "D_Factor"],
            "base": 3, "paper_role_mapping": ROLES,
            "raw_target": "v3(r_k)", "paper_target": "v3(n_k)=-v3(r_k)",
            "paper_prediction": "Negate each decoded integer prediction",
            "metric": "Greedy exact rational equality; missing, invalid and noninteger predictions cannot be correct",
            "selection_completed_epoch_window": [1, 300], "tie_rule": "Earliest validation maximum",
            "test_used_for_selection": False,
            "sample_sd_convention": "Sample SD across three run percentages, ddof=1",
            "population_sd_also_saved": True,
        },
        "all_runs_share_ordered_example_and_feature_records": True,
        "validation_test_k_overlap": 0,
        "selected_aggregates": aggregates, "runs": run_summaries,
    }
    write_json(args.out/"v3_audit_summary.json", summary)
    print(json.dumps({"selected_aggregates": aggregates,
                      "selected_epochs": {r:x["selection"]["completed_epoch"] for r,x in run_summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
