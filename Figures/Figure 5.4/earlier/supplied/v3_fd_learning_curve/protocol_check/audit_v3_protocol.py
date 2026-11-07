#!/usr/bin/env python3
"""Audit v3_fD configurations, inputs, epoch completeness, and log metrics.

python3 audit_v3_protocol.py --inputs PATH_TO_RUN_FOLDERS \
  --arithmetic-index evaluation_arithmetic_inputs.json --out protocol_rebuilt

The input directory contains 1000260_9, 1000260_10, and 1000260_11.
No training, checkpoint execution or new L-function evaluation is performed.
"""
from __future__ import annotations

import argparse
import bz2
from collections import Counter
import csv
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import time

RUNS = ("1000260_9", "1000260_10", "1000260_11")
FILE_LABELS = ("test1", "valid")
EPOCH_KEYS = tuple(str(i) for i in range(500))
EPOCH_SET = set(EPOCH_KEYS)


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer_valuation(n, p):
    n = abs(n)
    assert n != 0
    result = 0
    while n % p == 0:
        n //= p
        result += 1
    return result


def rational_valuation(pair, p):
    n, d = pair
    return integer_valuation(n, p) - integer_valuation(d, p)


def prediction_pair(value):
    """Return the signed numerator and positive denominator, or None."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    n, d = value
    if type(n) is not int or type(d) is not int or d == 0:
        return None
    return (n, d) if d > 0 else (-n, -d)


@lru_cache(maxsize=None)
def is_prime(n):
    if type(n) is not int or n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def audit_log(path):
    text = path.read_text()
    lines = text.splitlines()
    header_end = next(i for i, line in enumerate(lines) if "The experiment will be stored" in line)
    parameters = {}
    for line in lines[:header_end]:
        body = line.split(" - ", 3)[-1].strip() if line.startswith("INFO - ") else line.strip()
        m = re.fullmatch(r"([a-z][a-z0-9_]*):\s*(.*)", body)
        if m:
            assert m[1] not in parameters
            parameters[m[1]] = m[2]
    starts = [int(x) for x in re.findall(r"Starting epoch (\d+)", text)]
    ends = [int(x) for x in re.findall(r"End of epoch (\d+)", text)]
    metrics_list = [json.loads(line.split("__log__:", 1)[1]) for line in lines if "__log__:" in line]
    metric_epochs = [row["epoch"] for row in metrics_list]
    loss_rows = []
    for line in lines:
        m = re.search(r" -\s+(\d+) -\s+[\d.]+ examples/s .* LOSS:\s*(\S+)\s*-\s*LR:\s*(\S+)", line)
        if m:
            loss_rows.append({"step": int(m[1]), "loss": float(m[2]), "lr": float(m[3])})
    evaluated = re.findall(r"(\d+)/(\d+) \(([\d.eE+-]+)%\) equations were evaluated correctly", text)
    seeds = []
    for line in lines:
        m = re.search(r"Initialized random generator for worker (\d+), with seed (\[[^\]]+\]) \(base seed=(\d+)\)", line)
        if m:
            seeds.append({"worker": int(m[1]), "seed_vector": json.loads(m[2]), "base_seed": int(m[3])})
    num_params = [int(x) for x in re.findall(r"Number of parameters:\s*(\d+)", text)]
    loaded_counts = Counter(int(x) for x in re.findall(r"Loaded (\d+) equations from the disk", text))
    bad_markers = [line for line in lines if re.search(r"^(?:WARNING|WARN|ERROR|CRITICAL)\s*-|Traceback \(most recent call last\)", line)]
    nonfinite = [
        {"epoch": row["epoch"], "key": k, "value": str(v)}
        for row in metrics_list for k, v in row.items()
        if isinstance(v, (int, float)) and not math.isfinite(v)
    ]
    assert starts == list(range(500))
    assert ends == list(range(500))
    assert metric_epochs == list(range(500))
    assert len(loss_rows) == 2500
    assert [x["step"] for x in loss_rows] == list(range(200, 500001, 200))
    assert all(math.isfinite(x["loss"]) and math.isfinite(x["lr"]) for x in loss_rows)
    assert len(evaluated) == 1000 and all(int(x[1]) == 3000 for x in evaluated)
    assert len(seeds) == 1 and len(num_params) == 1
    assert not nonfinite and not bad_markers
    required = {
        "base": "3", "features": "k,D_Factor", "legacy_elliptic_curve_data": "True",
        "class_feature": "r_k", "ec_output": "r_k", "padic_prime": "None",
        "max_epoch": "500", "eval_size": "3000", "epoch_size": "32000",
        "batch_size": "32", "optimizer": "adam,lr=0.0001",
        "n_enc_layers": "4", "n_dec_layers": "4",
        "enc_emb_dim": "256", "dec_emb_dim": "256",
        "n_enc_heads": "8", "n_dec_heads": "8",
        "max_len": "512", "max_output_len": "512",
        "reload_checkpoint": "", "eval_only": "False",
    }
    assert all(parameters[k] == v for k, v in required.items())
    assert num_params[0] == 7759360
    assert parameters["reload_data"] == "elliptic_curve:/data/mahler/data/v3/v3_fD/tokenized_train_base_3.data"
    assert parameters["eval_data"] == (
        "/data/mahler/data/v3/v3_fD/tokenized_validate_base_3.data,"
        "/data/mahler/data/v3/v3_fD/tokenized_test_base_3.data"
    )
    return {
        "parameters": parameters,
        "log_sha256": sha256(path),
        "number_of_model_parameters": num_params[0],
        "logged_random_initializations": seeds,
        "epoch_starts": len(starts), "epoch_ends": len(ends),
        "metric_records": len(metrics_list), "evaluation_summaries": len(evaluated),
        "loss_reports": len(loss_rows), "last_training_step": loss_rows[-1]["step"],
        "training_updates_per_completed_epoch": 1000,
        "training_examples_per_completed_epoch": 32000,
        "training_rows_reported_loaded": 243999,
        "loaded_count_frequencies": dict(sorted(loaded_counts.items())),
        "learning_rates_observed": sorted(set(x["lr"] for x in loss_rows)),
        "first_loss": loss_rows[0]["loss"], "last_loss": loss_rows[-1]["loss"],
        "checkpoint_save_messages": len(re.findall(r"Saving checkpoint to ", text)),
        "reload_checkpoint": parameters["reload_checkpoint"],
        "all_recorded_numeric_metrics_finite": not nonfinite,
        "warning_error_traceback_marker_count": len(bad_markers),
        "full_ordered_epoch_coverage_0_to_499": True,
        "timestamps_raw_first_last": [lines[0], lines[-1]],
    }, {row["epoch"]: row for row in metrics_list}


def audit_evaluation(path, file_label, arithmetic, earlier_features, log_metrics):
    with bz2.open(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    assert len(data) == 3000
    ids = list(data)
    ks = [item["problem"]["k"] for item in data.values()]
    assert len(set(ks)) == 3000
    matching_labels = [
        label for label, records in arithmetic["by_file_label"].items()
        if [x["k"] for x in records] == ks
    ]
    assert len(matching_labels) == 1, matching_labels
    reference_label = matching_labels[0]
    reference = arithmetic["by_file_label"][reference_label]
    role = arithmetic["file_label_to_paper_role"][reference_label]
    signature = hashlib.sha256(json.dumps(
        [(i, row["problem"], row["answer"]) for i, row in data.items()],
        separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    if reference_label in earlier_features:
        assert earlier_features[reference_label] == signature
    else:
        earlier_features[reference_label] = signature

    rows = [{
        "epoch_index": e, "completed_epoch": e + 1,
        "greedy_exact_count": 0, "sampled_exact_count": 0,
        "either_exact_count": 0, "sampled_almost_count_zero_equal_allowed": 0,
        "greedy_raw_pair_exact_count": 0, "sampled_raw_pair_exact_count": 0,
        "either_raw_pair_exact_count": 0,
        "greedy_usable_rational_count": 0, "sampled_usable_rational_count": 0,
        "greedy_nonintegral_rational_count": 0, "sampled_nonintegral_rational_count": 0,
        "greedy_explicit_null_count": 0, "sampled_explicit_null_count": 0,
        "greedy_missing_key_count": 0, "sampled_missing_key_count": 0,
    } for e in range(500)]
    targets, negatives, base_counts = Counter(), Counter(), Counter()
    examples = []
    nonintegral_deltas = 0
    for position, (example_id, item) in enumerate(data.items()):
        assert set(item) == {"problem", "answer", "guesses_greedy", "guesses_sampled"}
        problem = item["problem"]
        assert set(problem) == {"k", "D_Factor"}
        k = problem["k"]
        assert type(k) is int and k > 4
        expected = reference[position]
        assert str(expected["example_id"]) == example_id and expected["k"] == k
        target_pair = prediction_pair(item["answer"])
        assert target_pair is not None and target_pair[0] % target_pair[1] == 0
        target = target_pair[0] // target_pair[1]
        reference_r = Fraction(*expected["answer"])
        assert reference_r == Fraction(*expected["stored_answer"])
        expected_raw_v3r = rational_valuation(expected["answer"], 3)
        assert target == expected_raw_v3r, (k, target, expected_raw_v3r)
        targets[-target] += 1
        factors = problem["D_Factor"]
        assert isinstance(factors, list) and factors
        bases, reconstructed = [], Fraction(1)
        for pair in factors:
            assert isinstance(pair, list) and len(pair) == 2
            prime, exponent = pair
            assert type(prime) is int and type(exponent) is int and exponent != 0
            assert is_prime(prime)
            bases.append(prime)
            base_counts[prime] += 1
            reconstructed *= Fraction(prime) ** exponent
            if exponent < 0:
                negatives[(prime, exponent)] += 1
        assert bases == sorted(set(bases))
        formula = Fraction(k ** 14 * (k * k - 16), 2 ** 24)
        assert reconstructed == formula
        nonintegral_deltas += reconstructed.denominator != 1
        examples.append({
            "example_id": example_id, "k": k, "D_Factor": factors,
            "reference_r": expected["answer"],
            "stored_raw_target": item["answer"],
            "raw_v3_r": target, "paper_v3_n": -target,
            "model_discriminant": [formula.numerator, formula.denominator],
        })
        greedy, sampled = item["guesses_greedy"], item["guesses_sampled"]
        assert isinstance(greedy, dict) and isinstance(sampled, dict)
        assert set(greedy) == EPOCH_SET and set(sampled) == EPOCH_SET
        for e, key in enumerate(EPOCH_KEYS):
            row = rows[e]
            gv, sv = greedy.get(key), sampled.get(key)
            g, s = prediction_pair(gv), prediction_pair(sv)
            gc = g is not None and g[0] == target * g[1]
            sc = s is not None and s[0] == target * s[1]
            gc_literal = g is not None and gv == item["answer"]
            sc_literal = s is not None and sv == item["answer"]
            row["greedy_exact_count"] += gc
            row["sampled_exact_count"] += sc
            row["either_exact_count"] += gc or sc
            row["greedy_raw_pair_exact_count"] += gc_literal
            row["sampled_raw_pair_exact_count"] += sc_literal
            row["either_raw_pair_exact_count"] += gc_literal or sc_literal
            row["greedy_usable_rational_count"] += g is not None
            row["sampled_usable_rational_count"] += s is not None
            row["greedy_nonintegral_rational_count"] += g is not None and g[0] % g[1] != 0
            row["sampled_nonintegral_rational_count"] += s is not None and s[0] % s[1] != 0
            row["greedy_explicit_null_count"] += key in greedy and gv is None
            row["sampled_explicit_null_count"] += key in sampled and sv is None
            row["greedy_missing_key_count"] += key not in greedy
            row["sampled_missing_key_count"] += key not in sampled
            almost = False
            if s is not None:
                if target == 0:
                    almost = s[0] == 0
                else:
                    almost = (target * s[0] > 0
                              and abs(target) * s[1] < 2 * abs(s[0])
                              and abs(s[0]) < 2 * abs(target) * s[1])
            row["sampled_almost_count_zero_equal_allowed"] += almost
    reconciliation = []
    all_greedy, all_union, all_almost = True, True, True
    metrics_tolerance = 1e-10
    well_formed_vs_usable_mismatches = []
    for e, row in enumerate(rows):
        logged = log_metrics[e]
        prefix = file_label + "_ELLIPTIC_CURVE_"
        comparison = {
            "file_label": file_label, "paper_role": role, **row,
            "logged_greedy_percent": logged[prefix + "greedy_acc"],
            "logged_either_percent": logged[prefix + "acc"],
            "logged_sampled_almost_fraction": logged[prefix + "avg_almost_correctness"],
            "logged_well_formed_percent": logged[prefix + "well_formed"],
        }
        comparison["greedy_exact_agrees"] = abs(row["greedy_exact_count"] / 30 - logged[prefix + "greedy_acc"]) < metrics_tolerance
        comparison["either_exact_agrees"] = abs(row["either_exact_count"] / 30 - logged[prefix + "acc"]) < metrics_tolerance
        comparison["sampled_almost_agrees"] = abs(row["sampled_almost_count_zero_equal_allowed"] / 3000 - logged[prefix + "avg_almost_correctness"]) < metrics_tolerance
        comparison["greedy_literal_pair_agrees"] = abs(row["greedy_raw_pair_exact_count"] / 30 - logged[prefix + "greedy_acc"]) < metrics_tolerance
        comparison["either_literal_pair_agrees"] = abs(row["either_raw_pair_exact_count"] / 30 - logged[prefix + "acc"]) < metrics_tolerance
        all_greedy &= comparison["greedy_exact_agrees"]
        all_union &= comparison["either_exact_agrees"]
        all_almost &= comparison["sampled_almost_agrees"]
        if abs(row["sampled_usable_rational_count"] / 30 - logged[prefix + "well_formed"]) >= metrics_tolerance:
            well_formed_vs_usable_mismatches.append(e)
        reconciliation.append(comparison)
    mismatches = [
        {"epoch_index": row["epoch_index"], "completed_epoch": row["completed_epoch"],
         "raw_greedy_exact_count": row["greedy_exact_count"],
         "raw_greedy_literal_pair_count": row["greedy_raw_pair_exact_count"],
         "logged_greedy_count": round(row["logged_greedy_percent"] * 30),
         "raw_either_exact_count": row["either_exact_count"],
         "raw_either_literal_pair_count": row["either_raw_pair_exact_count"],
         "logged_either_count": round(row["logged_either_percent"] * 30),
         "sampled_almost_count": row["sampled_almost_count_zero_equal_allowed"],
         "logged_sampled_almost_equivalent_count": row["logged_sampled_almost_fraction"] * 3000,
         "greedy_usable_rational_count": row["greedy_usable_rational_count"],
         "greedy_nonintegral_rational_count": row["greedy_nonintegral_rational_count"],
         "greedy_exact_agrees": row["greedy_exact_agrees"],
         "either_exact_agrees": row["either_exact_agrees"],
         "sampled_almost_agrees": row["sampled_almost_agrees"]}
        for row in reconciliation
        if not (row["greedy_exact_agrees"] and row["either_exact_agrees"] and row["sampled_almost_agrees"])
    ]
    return {
        "file_sha256": sha256(path),
        "file_label": file_label, "matched_reference_file_label": reference_label,
        "paper_role_from_verified_example_identities": role,
        "examples": len(data), "unique_k": len(set(ks)), "k_range": [min(ks), max(ks)],
        "same_ordered_examples_as_previous_arithmetic_pool": True,
        "input_and_target_sequence_sha256": signature,
        "problem_keys": ["k", "D_Factor"],
        "all_targets_equal_v3_of_stored_r": True,
        "paper_target_is_negative_of_raw_integer_target": True,
        "paper_target_class_counts": dict(sorted(targets.items())),
        "all_discriminant_products_equal_formula_exactly": True,
        "model_discriminant_formula": "k^14*(k^2-16)/2^24",
        "all_factor_bases_prime_by_exact_trial_division": True,
        "all_bases_strictly_increasing_distinct": True,
        "all_exponents_nonzero_integers": True,
        "negative_exponent_pair_counts": {str(pair): c for pair, c in sorted(negatives.items())},
        "nonintegral_model_discriminants": nonintegral_deltas,
        "distinct_prime_bases_in_pool": len(base_counts),
        "largest_prime_factor_base": max(base_counts),
        "all_examples_have_all_500_epoch_keys_for_both_decoders": True,
        "greedy_exact_log_reconciliation_all_500_epochs": all_greedy,
        "either_exact_log_reconciliation_all_500_epochs": all_union,
        "sampled_almost_with_zero_exact_log_reconciliation_all_500_epochs": all_almost,
        "log_scalar_comparisons": 1500,
        "log_scalar_comparisons_agree_count": sum(
            row[key] for row in reconciliation
            for key in ("greedy_exact_agrees", "either_exact_agrees", "sampled_almost_agrees")),
        "log_metric_mismatches": mismatches,
        "greedy_normalized_and_literal_pair_counts_always_equal": all(
            row["greedy_exact_count"] == row["greedy_raw_pair_exact_count"] for row in rows),
        "well_formed_is_not_json_sampled_usable_rational_fraction": bool(well_formed_vs_usable_mismatches),
        "well_formed_vs_json_usable_count_mismatch_epochs": len(well_formed_vs_usable_mismatches),
        "prediction_totals_across_500_epochs": {
            field: sum(row[field] for row in rows) for field in
            ("greedy_usable_rational_count", "sampled_usable_rational_count",
             "greedy_nonintegral_rational_count", "sampled_nonintegral_rational_count",
             "greedy_explicit_null_count", "sampled_explicit_null_count",
             "greedy_missing_key_count", "sampled_missing_key_count")
        },
    }, reconciliation, examples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--arithmetic-index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    arithmetic = load_json(args.arithmetic_index)
    input_sets = {k: {r["k"] for r in v} for k, v in arithmetic["by_file_label"].items()}
    assert len(input_sets["valid"]) == len(input_sets["test1"]) == 3000
    assert not (input_sets["valid"] & input_sets["test1"])
    summary = {
        "runs": {}, "reference_arithmetic_index_sha256": sha256(args.arithmetic_index),
        "all_evaluation_pools_disjoint": True,
        "scope": "Configuration, log completeness, actual evaluation identities/targets/features, and serialized-guess log reconciliation",
    }
    first_features = {}
    exported_examples = {}
    all_rows = []
    for run in RUNS:
        folder = args.inputs / run
        log_summary, metrics = audit_log(folder / "train.log")
        summary["runs"][run] = {"log": log_summary, "evaluation": {}}
        for label in FILE_LABELS:
            path = folder / ("eval." + label + ".epochs_0_499.json.bz2")
            result, rows, examples = audit_evaluation(path, label, arithmetic, first_features, metrics)
            summary["runs"][run]["evaluation"][label] = result
            exported_examples.setdefault(label, examples)
            all_rows.extend({"run_id": run, **row} for row in rows)
            print(run, label, "checked", result["examples"], "examples and 500 epochs", flush=True)
    parameter_keys = set().union(*(set(summary["runs"][run]["log"]["parameters"]) for run in RUNS))
    differences = {
        key: {run: summary["runs"][run]["log"]["parameters"].get(key) for run in RUNS}
        for key in sorted(parameter_keys)
        if len({summary["runs"][run]["log"]["parameters"].get(key) for run in RUNS}) > 1
    }
    assert set(differences) <= {"command", "dump_path", "exp_id"}
    seeds = {run: summary["runs"][run]["log"]["logged_random_initializations"][0]["base_seed"] for run in RUNS}
    assert len(set(seeds.values())) == 3
    summary.update({
        "identical_configuration_except": differences,
        "same_input_feature_and_target_sequences_across_three_runs": True,
        "same_architecture_and_training_protocol_across_three_runs": True,
        "distinct_logged_worker_base_seeds": seeds,
        "file_label_to_paper_role_verified": {
            label: summary["runs"][RUNS[0]]["evaluation"][label]["paper_role_from_verified_example_identities"]
            for label in FILE_LABELS
        },
        "raw_target_semantics": "Legacy r_k field contains v3(r_k); negate targets and integer predictions for v3(n_k). Exact equality is invariant under this sign change.",
        "model_discriminant_is_not_claimed_minimal": True,
        "log_metric_semantics_verified": {
            "greedy_acc": "Intended greedy exact accuracy; compare to 100 times rational equality success / 3000. Any discrepancies are explicitly listed.",
            "acc": "Intended success of either greedy or sampled prediction / 3000, in percent. Any discrepancies are explicitly listed.",
            "avg_almost_correctness": "Compare with sampled same-sign strict factor-two success, with target=prediction=0 also counted, divided by 3000",
            "well_formed": "Not identical to serialized JSON usable-rational fraction; its underlying string-parser definition is not reconstructed here",
        },
        "total_log_scalar_comparisons": len(all_rows) * 3,
        "log_scalar_comparisons_all_agree": all(
            row[k] for row in all_rows
            for k in ("greedy_exact_agrees", "either_exact_agrees", "sampled_almost_agrees")),
        "log_scalar_comparisons_agree_count": sum(
            row[k] for row in all_rows
            for k in ("greedy_exact_agrees", "either_exact_agrees", "sampled_almost_agrees")),
        "limitations": [
            "The available logs establish three different worker base seeds, not a complete global-seeding recipe or paired initializations.",
            "Both evaluation pools were evaluated at every epoch; no claim is made that test predictions were first computed after selection.",
            "Actual training rows were not supplied for an independent training/evaluation membership audit; the logs report 243999 training rows.",
            "These files establish one repeated v3_fD configuration, not superiority over other feature configurations.",
        ],
        "elapsed_seconds": time.monotonic() - started,
        "maximum_resident_memory_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })
    write_json(args.out / "protocol_summary.json", summary)
    with (args.out / "log_metric_reconciliation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(args.out / "evaluation_discriminant_inputs.json", {
        "source_run_id": RUNS[0],
        "same_input_and_target_sequences_verified_for_runs": list(RUNS),
        "file_label_to_paper_role": summary["file_label_to_paper_role_verified"],
        "model_discriminant_formula": "k^14*(k^2-16)/2^24",
        "raw_target_semantics": summary["raw_target_semantics"],
        "by_file_label": exported_examples,
    })
    print(json.dumps({
        "configuration_difference_keys": list(differences),
        "worker_base_seeds": seeds,
        "log_metric_comparisons": summary["total_log_scalar_comparisons"],
        "log_metric_comparisons_agree": summary["log_scalar_comparisons_agree_count"],
        "paper_role_mapping": summary["file_label_to_paper_role_verified"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_MiB": summary["maximum_resident_memory_MiB"],
    }, indent=2))


if __name__ == "__main__":
    main()
