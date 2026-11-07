#!/usr/bin/env python3
"""Compare the supplied rk and rk_w training logs without loading predictions.

Only the Python standard library is required. Input directories contain the
folders 1000262_0..2 (rk) and 1000262_3..5 (rk_w), respectively. No model is
executed, no checkpoint is selected, and input files are not modified.
"""
import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

EXPECTED_EPOCHS = list(range(500))
IDENTITY_KEYS = {"command", "dump_path", "exp_id"}
FAMILY_KEYS = {"features", "reload_data", "eval_data"}


def audit(path, family):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    config = {}
    for line in lines:
        if "The experiment will be stored" in line:
            break
        chunks = line.split(" - ", 3)
        payload = chunks[3] if line.startswith("INFO - ") and len(chunks) == 4 else line
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*):\s*(.*)$", payload)
        if match:
            config[match.group(1)] = match.group(2)
    starts = [int(x) for x in re.findall(r"Starting epoch (\d+)", text)]
    ends = [int(x) for x in re.findall(r"End of epoch (\d+)", text)]
    metrics = [json.loads(line.split("__log__:", 1)[1])
               for line in lines if "__log__:" in line]
    indices = [int(item["epoch"]) for item in metrics]
    parameters = [int(x) for x in re.findall(r"Number of parameters:\s*(\d+)", text)]
    seeds = []
    loss_records = []
    for line in lines:
        match = re.search(
            r"Initialized random generator for worker (\d+), with seed (\[[^\]]+\]) \(base seed=(\d+)\)", line)
        if match:
            seeds.append({"worker": int(match.group(1)),
                          "seed_tuple": json.loads(match.group(2)),
                          "base_seed": int(match.group(3)),
                          "source_message": line})
        if "LOSS:" in line:
            payload = line.split(" - ", 3)[3]
            loss_records.append({
                "step": int(re.match(r"\s*(\d+)\s*-", payload).group(1)),
                "loss": float(re.search(r"\bLOSS:\s*(\S+)", payload).group(1)),
                "learning_rate": float(re.search(r"\bLR:\s*(\S+)", payload).group(1))})
    finite_losses = all(math.isfinite(row["loss"]) and math.isfinite(row["learning_rate"])
                        for row in loss_records)
    nonfinite_metrics = [{"epoch": row["epoch"], "field": key, "value": str(value)}
                         for row in metrics for key, value in row.items()
                         if isinstance(value, (int, float)) and not math.isfinite(value)]
    warnings = [line for line in lines
                if re.match(r"^(WARNING|ERROR|CRITICAL)\b", line)
                or re.search(r"\b(Traceback|RuntimeError|out of memory)\b", line)]
    sample_counts = [int(x) for x in re.findall(
        r"almost_correctness: .*?\(over (\d+) samples\)", text)]
    saves = re.findall(r"Saving checkpoint to (.+?) \.\.\.", text)
    by_epoch = {int(row["epoch"]): row for row in metrics}
    fixed = {str(e): {key: value for key, value in by_epoch.get(e, {}).items()
                     if key == "epoch" or key.endswith(("_greedy_acc", "_acc",
                         "_avg_almost_correctness", "_xe_loss", "_well_formed"))}
             for e in (299, 499)}
    all_split_metrics = {}
    for label in ("valid", "test1"):
        prefix = label + "_ELLIPTIC_CURVE_"
        columns = sorted({key for row in metrics for key in row if key.startswith(prefix)})
        all_split_metrics[label] = {
            "columns": columns,
            "present_at_every_epoch": {
                key: all(key in row for row in metrics) for key in columns},
            "greedy_exact_zero_through_index_299": all(
                row.get(prefix + "greedy_acc") == 0 for row in metrics if row["epoch"] < 300)}
    return {
        "family": family, "filename": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "line_count": len(lines), "configuration": config,
        "number_of_parameters": parameters, "first_log_line": lines[0], "last_log_line": lines[-1],
        "seed_evidence": seeds,
        "epoch_coverage": {
            "starts_count": len(starts), "ends_count": len(ends), "metric_records_count": len(metrics),
            "starts_equal_0_through_499": starts == EXPECTED_EPOCHS,
            "ends_equal_0_through_499": ends == EXPECTED_EPOCHS,
            "metric_indices_equal_0_through_499": indices == EXPECTED_EPOCHS,
            "missing_metric_epochs": sorted(set(EXPECTED_EPOCHS) - set(indices)),
            "duplicate_metric_epochs": [e for e, n in collections.Counter(indices).items() if n > 1]},
        "loss_reporting": {
            "record_count": len(loss_records),
            "steps_equal_200_through_500000_by_200": [row["step"] for row in loss_records] == list(range(200, 500001, 200)),
            "first_record": loss_records[0] if loss_records else None,
            "last_record": loss_records[-1] if loss_records else None,
            "finite": finite_losses,
            "learning_rates": sorted({row["learning_rate"] for row in loss_records})},
        "evaluation_summary_count": len(sample_counts),
        "evaluation_sample_counts": dict(collections.Counter(sample_counts)),
        "split_metric_coverage": all_split_metrics,
        "checkpoint_saves": {"count": len(saves), "unique_paths": sorted(set(saves))},
        "warning_or_error_messages": warnings,
        "nonfinite_logged_metrics": nonfinite_metrics,
        "fixed_epoch_logged_values": fixed,
    }


def compare(runs):
    keys = sorted(set().union(*(set(record["configuration"]) for record in runs.values())))
    shared, differences = {}, {}
    for key in keys:
        values = {run_id: record["configuration"].get(key) for run_id, record in runs.items()}
        if len(set(values.values())) == 1:
            shared[key] = next(iter(values.values()))
        else:
            differences[key] = values
    return {"shared_configuration": shared, "configuration_differences": differences,
            "non_identity_configuration_differences": {k: v for k, v in differences.items() if k not in IDENTITY_KEYS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rk-inputs", type=Path, required=True)
    parser.add_argument("--rk-w-inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    families = {"rk": (args.rk_inputs, range(3)), "rk_w": (args.rk_w_inputs, range(3, 6))}
    runs = {}
    for family, (base, indices) in families.items():
        for index in indices:
            run_id = f"1000262_{index}"
            runs[run_id] = audit(base / run_id / "train.log", family)
    comparisons = {family: compare({rid: rec for rid, rec in runs.items() if rec["family"] == family})
                   for family in families}
    comparisons["all_six"] = compare(runs)
    unexpected_cross_family = {
        key: values for key, values in comparisons["all_six"]["configuration_differences"].items()
        if key not in IDENTITY_KEYS | FAMILY_KEYS}
    base_seeds = {rid: [seed["base_seed"] for seed in record["seed_evidence"]]
                  for rid, record in runs.items()}
    issues = []
    for rid, record in runs.items():
        coverage = record["epoch_coverage"]
        for key in ("starts_equal_0_through_499", "ends_equal_0_through_499", "metric_indices_equal_0_through_499"):
            if not coverage[key]:
                issues.append({"run_id": rid, "issue": key})
        if not record["loss_reporting"]["steps_equal_200_through_500000_by_200"]:
            issues.append({"run_id": rid, "issue": "Unexpected loss-reporting steps"})
        if not record["loss_reporting"]["finite"] or record["nonfinite_logged_metrics"]:
            issues.append({"run_id": rid, "issue": "Nonfinite numeric output"})
        if record["warning_or_error_messages"]:
            issues.append({"run_id": rid, "issue": "Warning or error messages"})
        expected_features = "k" if record["family"] == "rk" else "k,w"
        cfg = record["configuration"]
        if cfg.get("features") != expected_features or cfg.get("class_feature") != "r_k" or cfg.get("ec_output") != "r_k":
            issues.append({"run_id": rid, "issue": "Unexpected input or target configuration"})
        if record["number_of_parameters"] != [7517184]:
            issues.append({"run_id": rid, "issue": "Unexpected parameter count"})
        if record["evaluation_summary_count"] != 1000 or record["evaluation_sample_counts"] != {3000: 1000}:
            issues.append({"run_id": rid, "issue": "Unexpected evaluation sample counts"})
    for family in families:
        if comparisons[family]["non_identity_configuration_differences"]:
            issues.append({"family": family, "issue": "Within-family configuration differs beyond run identity"})
    if unexpected_cross_family:
        issues.append({"issue": "Unexpected cross-family configuration difference", "keys": sorted(unexpected_cross_family)})
    result = {
        "audit_scope": "Six supplied training logs only; prediction archives are checked separately. No checkpoint selection or model execution.",
        "runs": runs, "comparisons": comparisons,
        "unexpected_cross_family_configuration_differences": unexpected_cross_family,
        "logged_data_worker_base_seeds": base_seeds,
        "all_six_have_one_logged_data_worker_seed": all(len(values) == 1 for values in base_seeds.values()),
        "distinct_logged_data_worker_base_seeds": sorted({value for values in base_seeds.values() for value in values}),
        "issues": issues,
        "interpretation": [
            "The new family supplies input features k,w and target r_k. The baseline supplies k and the same target.",
            "Each family has three comparable full 500-epoch training runs. Recorded architecture and optimization settings agree between families.",
            "Cross-family dataset paths differ as expected because rk_w includes w; equality of actual examples must be verified from the prediction exports.",
            "The logged data-worker base seeds differ. The logs do not separately state all global, model-initialization or library seeds.",
            "The suffix of exp_id is not the configured run_id: run_id is 0 in every log.",
            "Run suffixes do not establish matched-seed pairs across families; compare summaries across the actual three runs per family.",
            "Both evaluation groups are evaluated at every epoch. Evaluation history alone does not establish whether earlier decisions used a group.",
            "The histories permit a separate explicit validation-based selection rule without executing a model again.",
            "An epoch consists of 32000 training examples/1000 batch updates, not a complete pass through the 243999-example training dataset.",
            "The logs record rolling checkpoint saves, not retained epoch-specific checkpoints. Saved predictions suffice for the present calculations."
        ]
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.out), "issues": issues,
        "within_rk_w_differing_configuration_keys": sorted(comparisons["rk_w"]["configuration_differences"]),
        "cross_family_differing_configuration_keys": sorted(comparisons["all_six"]["configuration_differences"]),
        "logged_data_worker_base_seeds": base_seeds,
        "new_run_counts": {rid: {"epochs": rec["epoch_coverage"], "loss_records": rec["loss_reporting"]["record_count"],
                                 "evaluation_summaries": rec["evaluation_summary_count"], "checkpoint_saves": rec["checkpoint_saves"]["count"]}
                           for rid, rec in runs.items() if rec["family"] == "rk_w"},
    }, indent=2))


if __name__ == "__main__":
    main()
