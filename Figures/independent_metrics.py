#!/usr/bin/env python3
"""Independent raw-data check of the three rk_w repeats.

Uses the standard library and loads one compressed JSON file at a time.
Selection uses the current paper's validation group, established by full
example/answer identity against the already audited rk data. It uses completed
epochs 1--300 and chooses the earliest maximum. Test values do not enter the
selection. This is a specified analysis for the revision, not a reconstruction
of an undocumented historical spreadsheet rule.
"""
import argparse
import bz2
import collections
import gc
import hashlib
import json
import math
import resource
import statistics
from fractions import Fraction
from pathlib import Path

RUNS = ("1000262_3", "1000262_4", "1000262_5")
EPOCHS = tuple(range(500))
KEYS = tuple(str(e) for e in EPOCHS)
KEY_SET = set(KEYS)
ROLES = {"test1": "validation", "valid": "test"}
ABSENT = object()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_rational(value):
    return (isinstance(value, list) and len(value) == 2
            and type(value[0]) is int and type(value[1]) is int
            and value[1] != 0)


def canonical(pair):
    a, b = pair
    divisor = math.gcd(a, b)
    if b < 0:
        divisor = -divisor
    return [a // divisor, b // divisor]


def flags(answer, pred):
    # Always compare the two rational numbers by integer cross-products.
    if pred is ABSENT:
        return (0, 0, 0, 1, 0)
    if not is_rational(pred):
        return (0, 0, 1, 0, 0)
    a, b = answer
    c, d = pred
    prediction_cross = c * b
    answer_cross = a * d
    same_sign = prediction_cross * answer_cross > 0
    exact = prediction_cross == answer_cross
    factor_two = (same_sign and
                  2 * abs(prediction_cross) > abs(answer_cross) and
                  abs(prediction_cross) < 2 * abs(answer_cross))
    boundary = (2 * abs(prediction_cross) == abs(answer_cross) or
                abs(prediction_cross) == 2 * abs(answer_cross))
    return (int(exact), int(factor_two), 0, 0, int(boundary))


def inspect(data, baseline, path):
    assert isinstance(data, dict) and len(data) == 3000
    index = []
    features = []
    root_number_counts = collections.Counter()
    problem_field_sets = collections.Counter()
    coverage = {mode: {"missing_epoch_entries": 0, "extra_epoch_entries": 0,
                      "non_dict_records": 0}
                for mode in ("greedy", "sampled")}
    answer_sign_matches_root = 0
    root_number_invalid = []
    for example_id, record in data.items():
        assert isinstance(record, dict)
        answer = record["answer"]
        assert is_rational(answer)
        problem = record["problem"]
        assert isinstance(problem, dict) and type(problem["k"]) is int
        index.append({"example_id": example_id, "k": problem["k"],
                      "answer": canonical(answer)})
        problem_field_sets[tuple(sorted(problem))] += 1
        w = problem.get("w")
        features.append([problem["k"], w])
        root_number_counts[str(w)] += 1
        if type(w) is not int or w not in (-1, 1):
            root_number_invalid.append({"example_id": example_id, "value": w})
        else:
            answer_sign = (answer[0] * answer[1] > 0) - (answer[0] * answer[1] < 0)
            answer_sign_matches_root += answer_sign == w
        for mode in coverage:
            guesses = record.get("guesses_" + mode)
            if not isinstance(guesses, dict):
                coverage[mode]["non_dict_records"] += 1
                continue
            keys = set(guesses)
            coverage[mode]["missing_epoch_entries"] += len(KEY_SET - keys)
            coverage[mode]["extra_epoch_entries"] += len(keys - KEY_SET)
    assert index == baseline, "Example identities or reference rational answers differ"
    return {
        "path": str(path), "compressed_sha256": sha256(path),
        "examples": len(data),
        "ordered_example_ids_k_canonical_answers_match_rk_baseline": True,
        "all_ids_are_0_to_2999": set(data) == {str(i) for i in range(3000)},
        "distinct_k": len({x["k"] for x in index}),
        "coverage": coverage,
        "problem_field_sets": {",".join(k): v for k, v in problem_field_sets.items()},
        "root_number_value_counts": dict(root_number_counts),
        "invalid_root_number_values": root_number_invalid,
        "answer_sign_matches_root_number_count": answer_sign_matches_root,
        "root_number_feature_sequence_sha256": hashlib.sha256(
            json.dumps(features, separators=(",", ":")).encode()).hexdigest(),
    }


def count_epochs(data, indices):
    rows = {e: {"epoch_index": e, "epoch_number": e + 1,
                "examples": len(data), "exact_count": 0,
                "factor_two_count": 0, "invalid_count": 0,
                "missing_count": 0, "strict_boundary_count": 0}
            for e in indices}
    columns = ("exact_count", "factor_two_count", "invalid_count",
               "missing_count", "strict_boundary_count")
    selected_keys = [(e, KEYS[e]) for e in indices]
    for record in data.values():
        truth = record["answer"]
        guesses = record["guesses_greedy"]
        assert isinstance(guesses, dict)
        for epoch_index, key in selected_keys:
            row = rows[epoch_index]
            results = flags(truth, guesses.get(key, ABSENT))
            for column, count in zip(columns, results):
                row[column] += count
    for row in rows.values():
        row["exact_percent"] = 100 * row["exact_count"] / row["examples"]
        row["factor_two_percent"] = 100 * row["factor_two_count"] / row["examples"]
    return rows


def select(validation, metric):
    scores = {e: Fraction(validation[e][metric + "_count"],
                          validation[e]["examples"])
              for e in range(300)}
    maximum = max(scores.values())
    tied = [e for e in range(300) if scores[e] == maximum]
    return tied[0], tied


def summarize(values):
    percentages = [100 * Fraction(n, d) for n, d in values]
    floats = [float(v) for v in percentages]
    return {"correct_counts": [n for n, d in values],
            "denominators": [d for n, d in values],
            "run_percentages": floats,
            "mean_percent_exact_fraction": str(sum(percentages, Fraction(0)) / len(values)),
            "mean_percent": float(sum(percentages, Fraction(0)) / len(values)),
            "sample_sd_percentage_points_ddof1": statistics.stdev(floats),
            "population_sd_percentage_points_ddof0": statistics.pstdev(floats)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--baseline-index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    baseline = json.loads(args.baseline_index.read_text())
    baseline_overlap = len({r["k"] for r in baseline["test1"]} &
                           {r["k"] for r in baseline["valid"]})
    assert baseline_overlap == 0
    result = {
        "method": {
            "file_label_to_paper_role": ROLES,
            "role_basis": "Full ordered k/answer/example-ID match to audited rk baseline",
            "baseline_index_sha256": sha256(args.baseline_index),
            "decoder": "greedy",
            "factor_two_definition": "Same sign and strict |truth|/2 < |prediction| < 2|truth|",
            "arithmetic": "Exact integer cross-products; no floating-point comparison of rationals",
            "denominator": "All 3000 examples, including missing or invalid predictions",
            "selected_epoch_window_indices_inclusive": [0, 299],
            "selected_epoch_window_completed_epochs_inclusive": [1, 300],
            "tie_rule": "Earliest validation maximum, separately for each run and metric",
            "test_values_used_for_selection": False,
            "within_run_validation_test_k_overlap": baseline_overlap,
            "primary_sd": "Sample standard deviation across three run percentages, ddof=1",
        },
        "runs": {}, "selected_epoch_aggregates": {}, "fixed_epoch_aggregates": {},
    }
    for rid in RUNS:
        print("Reading validation", rid, flush=True)
        val_path = args.inputs / rid / "eval.test1.epochs_0_499.json.bz2"
        with bz2.open(val_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        val_integrity = inspect(data, baseline["test1"], val_path)
        val_metrics = count_epochs(data, EPOCHS)
        del data
        gc.collect()
        choices = {m: select(val_metrics, m) for m in ("factor_two", "exact")}
        selected_indices = sorted({299, 499} | {v[0] for v in choices.values()})
        print("Reading test", rid, "at indices", selected_indices, flush=True)
        test_path = args.inputs / rid / "eval.valid.epochs_0_499.json.bz2"
        with bz2.open(test_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        test_integrity = inspect(data, baseline["valid"], test_path)
        test_metrics = count_epochs(data, selected_indices)
        del data
        gc.collect()
        selection = {}
        for metric, (chosen, ties) in choices.items():
            selection[metric] = {
                "selected_epoch_index": chosen, "selected_completed_epoch": chosen + 1,
                "tied_epoch_indices": ties, "tie_count": len(ties),
                "validation": val_metrics[chosen], "test": test_metrics[chosen],
            }
        result["runs"][rid] = {
            "integrity": {"test1": val_integrity, "valid": test_integrity},
            "validation_per_epoch": list(val_metrics.values()),
            "test_evaluated_epochs": list(test_metrics.values()),
            "selection": selection,
        }
        print("Completed", rid, {m: {"epoch": s["selected_completed_epoch"],
              "validation_count": s["validation"][m + "_count"],
              "test_count": s["test"][m + "_count"]} for m, s in selection.items()}, flush=True)

    for metric in ("factor_two", "exact"):
        result["selected_epoch_aggregates"][metric] = {}
        for role in ("validation", "test"):
            selected = [result["runs"][rid]["selection"][metric][role] for rid in RUNS]
            summary = summarize([(r[metric + "_count"], r["examples"]) for r in selected])
            summary["run_ids"] = list(RUNS)
            summary["selected_completed_epochs"] = [r["epoch_number"] for r in selected]
            result["selected_epoch_aggregates"][metric][role] = summary
    for e in (299, 499):
        result["fixed_epoch_aggregates"][str(e)] = {}
        for role in ("validation", "test"):
            role_rows = []
            for rid in RUNS:
                run = result["runs"][rid]
                rows = run["validation_per_epoch"] if role == "validation" else run["test_evaluated_epochs"]
                role_rows.append(next(r for r in rows if r["epoch_index"] == e))
            result["fixed_epoch_aggregates"][str(e)][role] = {
                metric: summarize([(r[metric + "_count"], r["examples"]) for r in role_rows])
                for metric in ("factor_two", "exact")}
    result["cross_run_root_number_feature_sequences_identical"] = {
        label: len({result["runs"][rid]["integrity"][label]["root_number_feature_sequence_sha256"]
                    for rid in RUNS}) == 1 for label in ROLES}
    result["max_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_epoch_aggregates": result["selected_epoch_aggregates"],
                      "fixed_epoch_aggregates": result["fixed_epoch_aggregates"],
                      "cross_run_root_number_feature_sequences_identical": result["cross_run_root_number_feature_sequences_identical"],
                      "max_process_rss_mib": result["max_process_rss_mib"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
