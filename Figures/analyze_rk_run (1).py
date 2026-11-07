#!/usr/bin/env python3
"""Audit one exported Mahler r_k run, using Python's standard library only.

Usage:
  python3 analyze_rk_run.py --valid eval.valid.epochs_0_499.json.bz2 \
    --test eval.test1.epochs_0_499.json.bz2 --log train.log --out audit

Inputs are read only. Each compressed archive is loaded and released separately.
All accuracies retain every example in the denominator. Rational comparisons
use integer cross-products. Factor-two success requires the correct sign and
strict bounds |r|/2 < |prediction| < 2|r|. Missing/invalid outputs are failures.
No checkpoint is selected. Maxima are descriptive, identified by literal file
labels. Epoch indices 0..299 are the previously described 300-epoch window.
An optional, independently verified paper-role mapping can annotate the CSV.
"""
import argparse
import bz2
import collections
import csv
import gc
import hashlib
import json
import math
import re
from pathlib import Path

MODES = ("greedy", "sampled")
EXPECTED_EPOCHS = set(range(500))

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def rational_pair(value):
    return (isinstance(value, list) and len(value) == 2
            and all(type(x) is int for x in value) and value[1] != 0)

def success_flags(answer, guess):
    a, b = answer
    c, d = guess
    left, right = a*d, c*b
    exact = left == right
    same_sign = ((left > 0 and right > 0) or
                 (left < 0 and right < 0) or (left == right == 0))
    magnitude = abs(left) < 2*abs(right) and abs(right) < 2*abs(left)
    return exact, same_sign and magnitude, same_sign, magnitude

def canonical(pair):
    a, b = pair
    g = math.gcd(a, b)
    if b < 0:
        g = -g
    return [a//g, b//g]

def audit_archive(path, split):
    with bz2.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("The archive root must be a JSON object.")
    n = len(data)
    epochs = set(EXPECTED_EPOCHS)
    malformed_epoch_keys = []
    field_counts = collections.Counter()
    length_counts = {mode: collections.Counter() for mode in MODES}
    for example_id, rec in data.items():
        field_counts.update(rec.keys())
        for mode in MODES:
            guesses = rec.get("guesses_"+mode)
            if not isinstance(guesses, dict):
                raise ValueError(f"{split}/{example_id}: guesses_{mode} is not an object.")
            length_counts[mode][len(guesses)] += 1
            for key in guesses:
                if not isinstance(key, str) or not key.isdigit():
                    malformed_epoch_keys.append([example_id, mode, key])
                else:
                    epochs.add(int(key))
    if malformed_epoch_keys:
        raise ValueError(f"Non-numeric epoch keys: {malformed_epoch_keys[:5]}")
    epochs = sorted(epochs)
    rows = {e: {"split": split, "epoch_index": e, "epoch_number": e+1,
                "examples": n} for e in epochs}
    for row in rows.values():
        for mode in MODES:
            for field in ("present", "valid", "invalid", "missing", "zero",
                          "exact_count", "factor_two_count", "same_sign_count",
                          "magnitude_factor_two_count", "strict_boundary_count"):
                row[mode+"_"+field] = 0
        row["either_exact_count"] = 0
        row["either_factor_two_count"] = 0
    index = []
    anomalies = {"invalid_prediction_examples": [], "missing_prediction_examples": []}
    for example_id, rec in data.items():
        if not rational_pair(rec.get("answer")):
            raise ValueError(f"Invalid ground truth at {split}/{example_id}: {rec.get('answer')!r}")
        problem = rec.get("problem")
        if not isinstance(problem, dict) or type(problem.get("k")) is not int:
            raise ValueError(f"Missing integer k at {split}/{example_id}")
        answer = rec["answer"]
        index.append({"example_id": example_id, "k": problem["k"],
                      "answer": canonical(answer)})
        for e in epochs:
            row = rows[e]
            flags = {}
            for mode in MODES:
                values = rec["guesses_"+mode]
                key = str(e)
                if key not in values:
                    row[mode+"_missing"] += 1
                    flags[mode] = (False, False, False, False)
                    if len(anomalies["missing_prediction_examples"]) < 8:
                        anomalies["missing_prediction_examples"].append(
                            {"example_id": example_id, "k": problem["k"], "mode": mode, "epoch_index": e})
                    continue
                row[mode+"_present"] += 1
                pred = values[key]
                if not rational_pair(pred):
                    row[mode+"_invalid"] += 1
                    flags[mode] = (False, False, False, False)
                    if len(anomalies["invalid_prediction_examples"]) < 8:
                        anomalies["invalid_prediction_examples"].append(
                            {"example_id": example_id, "k": problem["k"], "mode": mode,
                             "epoch_index": e, "value": pred})
                    continue
                row[mode+"_valid"] += 1
                row[mode+"_zero"] += pred[0] == 0
                flags[mode] = success_flags(answer, pred)
                for label, result in zip(("exact_count", "factor_two_count", "same_sign_count",
                                          "magnitude_factor_two_count"), flags[mode]):
                    row[mode+"_"+label] += result
                left = abs(answer[0]*pred[1])
                right = abs(pred[0]*answer[1])
                row[mode+"_strict_boundary_count"] += (left == 2*right or right == 2*left)
            row["either_exact_count"] += flags["greedy"][0] or flags["sampled"][0]
            row["either_factor_two_count"] += flags["greedy"][1] or flags["sampled"][1]
    for row in rows.values():
        for mode in MODES:
            for field in ("exact", "factor_two", "same_sign", "magnitude_factor_two"):
                row[mode+"_"+field+"_pct"] = 100*row[mode+"_"+field+"_count"]/n
        for field in ("either_exact", "either_factor_two"):
            row[field+"_pct"] = 100*row[field+"_count"]/n
    ks = [x["k"] for x in index]
    sample = data.get("2026")
    sample_brief = None
    if sample:
        sample_brief = {"problem": sample.get("problem"), "answer": sample.get("answer"),
                        "greedy_at_299": sample["guesses_greedy"].get("299"),
                        "sampled_at_299": sample["guesses_sampled"].get("299")}
    summary = {
        "filename": path.name, "compressed_bytes": path.stat().st_size, "sha256": sha256(path),
        "examples": n, "unique_k": len(set(ks)), "minimum_k": min(ks), "maximum_k": max(ks),
        "duplicate_k": [k for k,c in collections.Counter(ks).items() if c > 1],
        "top_level_ids_are_0_to_n_minus_1": set(data) == {str(i) for i in range(n)},
        "record_field_counts": dict(field_counts),
        "prediction_dictionary_lengths": {m: dict(c) for m,c in length_counts.items()},
        "epoch_indices": epochs,
        "extra_epochs_outside_0_499": sorted(set(epochs)-EXPECTED_EPOCHS),
        "missing_predictions": {m: sum(r[m+"_missing"] for r in rows.values()) for m in MODES},
        "invalid_predictions": {m: sum(r[m+"_invalid"] for r in rows.values()) for m in MODES},
        "zero_predictions": {m: sum(r[m+"_zero"] for r in rows.values()) for m in MODES},
        "boundary_predictions": {m: sum(r[m+"_strict_boundary_count"] for r in rows.values()) for m in MODES},
        "example_2026": sample_brief, "anomaly_examples": anomalies}
    del data
    gc.collect()
    return list(rows.values()), index, summary

def read_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    entries = []
    config = {}
    in_header = True
    for line in text.splitlines():
        if "The experiment will be stored" in line:
            in_header = False
        if in_header:
            match = re.match(r"^\s+([A-Za-z][A-Za-z0-9_]*):\s*(.*)$", line)
            if match:
                config[match.group(1)] = match.group(2)
        if "__log__:" in line:
            entries.append(json.loads(line.split("__log__:", 1)[1]))
    by_epoch = {int(x["epoch"]): x for x in entries}
    if len(entries) != len(by_epoch):
        raise ValueError("Duplicate epoch records in train.log.")
    return config, by_epoch, {
        "filename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
        "epoch_metric_records": len(entries), "epoch_indices": sorted(by_epoch),
        "worker_seed_messages": [line for line in text.splitlines()
                                 if "Initialized random generator" in line],
        "config": config}

def compare_logs(rows, logs, split):
    comparisons = {}
    prefix = split+"_ELLIPTIC_CURVE_"
    candidates = {
        "greedy_acc": {"greedy_exact_pct": 1},
        "acc": {"sampled_exact_pct": 1, "either_exact_pct": 1, "greedy_exact_pct": 1},
        "well_formed": {"greedy_valid_pct": 1, "sampled_valid_pct": 1},
        "avg_almost_correctness": {
            "greedy_factor_two_pct": .01, "sampled_factor_two_pct": .01,
            "either_factor_two_pct": .01,
            "greedy_magnitude_factor_two_pct": .01, "sampled_magnitude_factor_two_pct": .01}
    }
    for log_suffix, choices in candidates.items():
        for column, scale in choices.items():
            diffs = []
            mismatches = []
            for r in rows:
                if r["epoch_index"] not in logs:
                    continue
                val = logs[r["epoch_index"]].get(prefix+log_suffix)
                if val is None:
                    continue
                if column.endswith("_valid_pct"):
                    countkey = column[:-4]
                    computed = 100*r[countkey]/r["examples"]
                else:
                    computed = r[column]
                diff = abs(val-computed*scale)
                diffs.append(diff)
                if diff > 1e-9 and len(mismatches) < 8:
                    mismatches.append({"epoch_index": r["epoch_index"],
                                       "logged": val, "recomputed": computed*scale})
            comparisons[log_suffix+" vs "+column] = {
                "epochs_compared": len(diffs),
                "epochs_matching": sum(d <= 1e-9 for d in diffs),
                "maximum_absolute_difference": max(diffs) if diffs else None,
                "first_mismatches": mismatches}
    return comparisons

def maxima(split_rows, count_field, last_epoch_index, file_label):
    rows = [r for r in split_rows if 0 <= r["epoch_index"] <= last_epoch_index]
    best = max(r[count_field] for r in rows)
    best_rows = [r for r in rows if r[count_field] == best]
    result = {"metric": count_field, "file_label": file_label,
              "last_epoch_index": last_epoch_index, "maximum_count": best,
              "maximum_pct": 100*best/rows[0]["examples"],
              "number_of_tied_epochs": len(best_rows),
              "epoch_indices": [r["epoch_index"] for r in best_rows],
              "epoch_numbers": [r["epoch_number"] for r in best_rows],
              "checkpoint_selected": False, "tie_break_applied": False}
    return result

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--valid", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--paper-validation-file", choices=("valid", "test1"),
                    help="Use only after independently verifying the old paper's split identities.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    indices = {}
    archive_summaries = {}
    split_rows = {}
    for split, path in (("valid", args.valid), ("test1", args.test)):
        print("Auditing "+split+"...", flush=True)
        rows, index, summary = audit_archive(path, split)
        split_rows[split] = rows
        all_rows.extend(rows)
        indices[split] = index
        archive_summaries[split] = summary
        print(split+": "+str(summary["examples"])+" examples; "+str(len(rows))+" epochs.", flush=True)
        (args.out/"example_index.json").write_text(json.dumps(indices, indent=2)+"\n")
    config, logs, log_summary = read_log(args.log)
    for r in all_rows:
        vals = logs.get(r["epoch_index"], {})
        r["paper_role"] = ("unassigned" if args.paper_validation_file is None else
                           "validation" if r["split"] == args.paper_validation_file else "test")
        for suffix in ("greedy_acc", "acc", "well_formed", "avg_almost_correctness", "xe_loss"):
            r["logged_"+suffix] = vals.get(r["split"]+"_ELLIPTIC_CURVE_"+suffix)
    fieldnames = list(all_rows[0])
    with (args.out/"per_epoch_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    overlaps = sorted(set(x["k"] for x in indices["valid"]) & set(x["k"] for x in indices["test1"]))
    descriptive_maxima = {}
    for split in split_rows:
        for field in ("greedy_exact_count", "greedy_factor_two_count", "sampled_exact_count"):
            for last in (299, 499):
                descriptive_maxima[split+"_"+field+"_0_"+str(last)] = maxima(split_rows[split], field, last, split)
    result = {
        "run_config": config, "archives": archive_summaries,
        "valid_test_overlap_count": len(overlaps), "valid_test_overlap_k": overlaps,
        "log": log_summary,
        "log_comparisons": {s: compare_logs(split_rows[s], logs, s) for s in split_rows},
        "fixed_epoch_results": [r for r in all_rows if r["epoch_index"] in (299,499)],
        "file_label_maxima_descriptive_only": descriptive_maxima,
        "paper_validation_file_argument": args.paper_validation_file,
        "metric_definition": {
            "exact": "a*d == c*b for truth a/b and prediction c/d; denominators nonzero.",
            "factor_two": "same sign and strict |truth|/2 < |prediction| < 2|truth|.",
            "denominator": "All exported examples. Missing or invalid predictions are failures.",
            "mode": "Greedy is the main paper metric; sampled is kept separate for log reconciliation.",
            "selection": "No checkpoint selection. File-wise maxima are descriptive; ties are not resolved automatically."},
        "limitations": [
            "One run cannot establish a mean and standard deviation over three seeds.",
            "Disjoint validation/test files do not by themselves establish no overlap with training.",
            "A log data-worker seed is not proof of all model/NumPy/PyTorch random seeds.",
            "Run identification and split filenames alone do not identify earlier spreadsheet results."
        ]}
    (args.out/"audit_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({
        "valid_test_overlap_count": len(overlaps),
        "coverage": {s: {"examples": v["examples"], "epochs":len(v["epoch_indices"]),
                         "missing":v["missing_predictions"],"invalid":v["invalid_predictions"]}
                     for s,v in archive_summaries.items()},
        "fixed_epoch_results": [
            {k:r[k] for k in ("split","epoch_index","epoch_number","greedy_exact_count",
                             "greedy_exact_pct","greedy_factor_two_count","greedy_factor_two_pct",
                             "sampled_exact_count","sampled_factor_two_count",
                             "logged_avg_almost_correctness")} for r in result["fixed_epoch_results"]],
        "file_maxima_first300": {name: {k:v for k,v in entry.items() if k not in ("epoch_indices","epoch_numbers")}
                               for name,entry in descriptive_maxima.items() if name.endswith("_0_299")},
        "log_match_counts": {s: {name: [v["epochs_matching"],v["epochs_compared"]]
                                  for name,v in comparisons.items()}
                             for s,comparisons in result["log_comparisons"].items()},
        "email_example": archive_summaries["test1"]["example_2026"]},
        indent=2), flush=True)

if __name__ == "__main__":
    main()

