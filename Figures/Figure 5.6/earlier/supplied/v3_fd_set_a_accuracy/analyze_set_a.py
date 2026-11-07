#!/usr/bin/env python3
"""Recompute conditional v3_fD accuracy on the manuscript's set A.

A = {k : 3 does not divide k and k(k^2-16) is squarefree}.
Plotting uses validation only; test scores use existing global selections.
No model training, inference, or new checkpoint selection is performed.
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

SELECTED = {"1000260_9": 219, "1000260_10": 279, "1000260_11": 297}
ROLES = {"test1": "validation", "valid": "test"}
GROUPS = ("0", "1", ">1")


def read_json(path):
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rational(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    n, d = value
    if type(n) is not int or type(d) is not int or d == 0:
        return None
    return Fraction(n, d)


def vp3(n):
    n = abs(n)
    if n == 0:
        raise ValueError("The 3-adic valuation of zero is not finite")
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v


def factor_table(limit):
    spf = list(range(limit + 1))
    for p in range(2, math.isqrt(limit) + 1):
        if spf[p] == p:
            for n in range(p * p, limit + 1, p):
                if spf[n] == n:
                    spf[n] = p
    return spf


def product_factors(k, spf):
    factors = Counter()
    for value in (k - 4, k, k + 4):
        n = abs(value)
        if n == 0:
            return None
        while n > 1:
            p = spf[n]
            factors[p] += 1
            n //= p
    return factors


def aggregate(values):
    return {"per_run_percent": values, "mean_percent": statistics.mean(values),
            "sample_sd_percentage_points": statistics.stdev(values)}


def group_of(target):
    assert target >= 0
    return str(target) if target <= 1 else ">1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arithmetic_path = args.inputs / "evaluation_arithmetic_inputs.json"
    arithmetic = read_json(arithmetic_path)
    assert arithmetic["file_label_to_paper_role"] == ROLES
    baseline = arithmetic["by_file_label"]
    largest = max(abs(r["k"]) + 4 for rows in baseline.values() for r in rows)
    spf = factor_table(largest)
    members, membership_rows, pool_meta = {}, [], {}
    for label, role in ROLES.items():
        records = {}
        for row in baseline[label]:
            k = row["k"]
            factors = product_factors(k, spf)
            in_a = k % 3 != 0 and factors is not None and all(e == 1 for e in factors.values())
            true_n = vp3(row["answer"][1]) - vp3(row["answer"][0])
            if in_a:
                assert true_n >= 0
            record = {"example_id": row["example_id"], "k": k,
                      "in_A": in_a, "true_v3_n": true_n}
            records[row["example_id"]] = record
            membership_rows.append({"file_label": label, "paper_role": role, **record,
                                    "product_factorization": json.dumps(sorted(factors.items())) if factors is not None else "zero"})
        assert len(records) == len(baseline[label]) == 3000
        members[label] = records
        selected = [r for r in records.values() if r["in_A"]]
        groups = Counter(group_of(r["true_v3_n"]) for r in selected)
        classes = Counter(r["true_v3_n"] for r in selected)
        pool_meta[role] = {
            "all_examples": len(records), "A_examples": len(selected),
            "A_group_sizes": {g: groups[g] for g in GROUPS},
            "A_true_valuation_counts": dict(sorted(classes.items())),
            "A_zero_or_one_examples": groups["0"] + groups["1"],
            "minimum_k": min(r["k"] for r in records.values()),
            "A_contains_k_1": any(r["k"] == 1 for r in selected),
        }

    validation_k = {r["k"] for r in members["test1"].values()}
    test_k = {r["k"] for r in members["valid"].values()}
    assert len(validation_k) == len(test_k) == 3000
    assert not validation_k.intersection(test_k)
    rows, selected_rows, selected_pooled, identities = [], [], [], {}
    hashes = {"evaluation_arithmetic_inputs.json": hashlib.sha256(arithmetic_path.read_bytes()).hexdigest()}
    target_checks, membership_checks = 0, 0
    for run, chosen in SELECTED.items():
        for label, role in ROLES.items():
            path = args.inputs / run / f"eval.{label}.epochs_0_499.json.bz2"
            hashes[f"{run}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
            data = read_json(path)
            assert list(data) == [r["example_id"] for r in baseline[label]]
            assert len(data) == 3000
            epochs = list(range(1, 501)) if role == "validation" else [chosen]
            correct = {e: Counter() for e in epochs}
            invalid = {e: Counter() for e in epochs}
            noninteger = {e: Counter() for e in epochs}
            features, selected_global_correct = [], 0
            for example_id, item in data.items():
                old = members[label][example_id]
                k = item["problem"]["k"]
                assert k == old["k"]
                target = old["true_v3_n"]
                answer = rational(item["answer"])
                assert answer is not None and answer == -target
                target_checks += 1
                features.append([example_id, item["problem"], item["answer"]])
                guesses = item["guesses_greedy"]
                prediction_chosen = rational(guesses.get(str(chosen - 1)))
                selected_global_correct += prediction_chosen == -target
                if not old["in_A"]:
                    continue
                membership_checks += 1
                group = group_of(target)
                for epoch in epochs:
                    pred = rational(guesses.get(str(epoch - 1)))
                    correct[epoch][group] += pred == -target
                    invalid[epoch][group] += pred is None
                    noninteger[epoch][group] += pred is not None and pred.denominator != 1
            if role in identities:
                assert features == identities[role]
            else:
                identities[role] = features
            sizes = pool_meta[role]["A_group_sizes"]
            for epoch in epochs:
                for group in GROUPS:
                    n = sizes[group]
                    assert n > 0
                    row = {"run_id": run, "file_label": label, "paper_role": role,
                           "completed_epoch": epoch, "epoch_index": epoch - 1,
                           "true_group": group, "examples": n,
                           "correct": correct[epoch][group],
                           "invalid": invalid[epoch][group], "noninteger": noninteger[epoch][group],
                           "exact_percent": 100 * correct[epoch][group] / n}
                    rows.append(row)
                    if epoch == chosen:
                        selected_rows.append(row)
            c = correct[chosen]
            for pooled_group, parts in (("all_A", GROUPS), ("A_true_0_or_1", ("0", "1"))):
                n = sum(sizes[g] for g in parts)
                count = sum(c[g] for g in parts)
                selected_pooled.append({"run_id": run, "paper_role": role,
                                        "completed_epoch": chosen, "pooled_group": pooled_group,
                                        "examples": n, "correct": count,
                                        "exact_percent": 100 * count / n,
                                        "overall_examples": 3000,
                                        "overall_correct": selected_global_correct,
                                        "overall_exact_percent": 100 * selected_global_correct / 3000})
            print(f"Processed {run}, {role}, A size {sum(sizes.values())}", flush=True)

    curves = []
    by_key = {}
    for row in rows:
        by_key.setdefault((row["paper_role"], row["completed_epoch"], row["true_group"]), []).append(row)
    for (role, epoch, group), rr in sorted(by_key.items()):
        if role != "validation":
            continue
        assert len(rr) == 3 and len({r["examples"] for r in rr}) == 1
        agg = aggregate([r["exact_percent"] for r in rr])
        mean, sd = agg["mean_percent"], agg["sample_sd_percentage_points"]
        curves.append({"paper_role": role, "completed_epoch": epoch, "true_group": group,
                       "examples_per_run": rr[0]["examples"], "mean_percent": mean,
                       "sample_sd_percentage_points": sd,
                       "display_band_low_percent": max(0, mean - sd),
                       "display_band_high_percent": min(100, mean + sd),
                       **{r["run_id"] + "_correct": r["correct"] for r in rr}})
    assert len(curves) == 1500
    aggregates, pooled_aggregates = {}, {}
    for role in ROLES.values():
        aggregates[role], pooled_aggregates[role] = {}, {}
        for group in GROUPS:
            rr = [r for r in selected_rows if r["paper_role"] == role and r["true_group"] == group]
            assert len(rr) == 3
            aggregates[role][group] = {"examples_per_run": rr[0]["examples"],
                                       "correct_counts": [r["correct"] for r in rr],
                                       **aggregate([r["exact_percent"] for r in rr])}
        for group in ("all_A", "A_true_0_or_1"):
            rr = [r for r in selected_pooled if r["paper_role"] == role and r["pooled_group"] == group]
            pooled_aggregates[role][group] = {"examples_per_run": rr[0]["examples"],
                                              "correct_counts": [r["correct"] for r in rr],
                                              **aggregate([r["exact_percent"] for r in rr])}
    summary = {
        "definition": "A={k: 3 does not divide k and k(k^2-16) is squarefree}",
        "membership_method": "Merge exact prime factorizations of abs(k-4), abs(k), abs(k+4); no repeated prime and 3 does not divide k",
        "small_k_policy": "No extra k>=5 restriction; k=1 is included if present and satisfies the definition",
        "target": "v3(n_k)=-v3(r_k); raw target and prediction are negated",
        "groups": list(GROUPS),
        "metric": "Exact greedy valuation prediction conditional on membership of A and the true valuation group",
        "higher_group_requires_exact_value": True,
        "invalid_outputs_remain_failures_in_denominator": True,
        "file_label_to_paper_role": ROLES,
        "runs": list(SELECTED), "selected_completed_epochs": SELECTED,
        "selection": "Same existing global validation maxima within completed epochs1-300, earliest tie; no subgroup selection",
        "plotted_pool": "validation", "plotted_completed_epochs": [1, 500],
        "test_scoring_scope": "Only each run's existing selected checkpoint",
        "pools": pool_meta, "selected_group_accuracies": aggregates,
        "selected_pooled_accuracies": pooled_aggregates,
        "sample_standard_deviation_divisor": 2,
        "display_band_policy": "Mean +/- sample SD, restricted to [0,100]% for display only",
        "smoothing": "none", "raw_targets_checked_against_stored_rational_r_k": target_checks,
        "A_membership_records_seen_across_runs": membership_checks,
        "ordered_ids_parameters_features_targets_match_across_runs": True,
        "validation_test_k_overlap": 0,
        "source_sha256": hashes,
    }
    write_csv(args.out / "set_a_membership.csv", membership_rows)
    write_csv(args.out / "set_a_per_epoch_metrics.csv", rows)
    write_csv(args.out / "set_a_validation_curves.csv", curves)
    write_csv(args.out / "set_a_selected_group_metrics.csv", selected_rows)
    write_csv(args.out / "set_a_selected_pooled_metrics.csv", selected_pooled)
    (args.out / "set_a_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"pools": pool_meta, "selected_group_accuracies": aggregates,
                      "selected_pooled_accuracies": pooled_aggregates}, indent=2))


if __name__ == "__main__":
    main()
