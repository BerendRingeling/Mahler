#!/usr/bin/env python3
"""Audit Figure 5.7: conditional exact accuracy on nine arithmetic sets.

All scores are restricted to examples whose TRUE v3(n_k) is 0 or 1.
Validation is scored throughout training; test only at existing selections.
"""
from __future__ import annotations
import argparse
import bz2
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SETS = ("A", "A'", "A''", "B", "B'", "B''", "C", "C'", "C''")
SELECTED = {"1000260_9": 219, "1000260_10": 279, "1000260_11": 297}
ROLES = {"test1": "validation", "valid": "test"}


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


def decode_pair(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    a, b = value
    if type(a) is not int or type(b) is not int or b == 0:
        return None
    return a, b


def exact(pair, target):
    return pair is not None and pair[0] == target * pair[1]


def vp3(n):
    n = abs(n)
    assert n > 0
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v


def factor_table(limit):
    spf = list(range(limit + 1))
    for p in range(2, math.isqrt(limit) + 1):
        if spf[p] == p:
            for n in range(p*p, limit+1, p):
                if spf[n] == n:
                    spf[n] = p
    return spf


def factors(value, spf):
    n = abs(value)
    if n == 0:
        return None
    out = Counter()
    while n > 1:
        p = spf[n]
        out[p] += 1
        n //= p
    return out


def membership(k, spf):
    fk, fm, fp = factors(k, spf), factors(k-4, spf), factors(k+4, spf)
    if fk is None or fm is None or fp is None:
        raise ValueError("Zero discriminant parameter is outside this evaluation dataset")
    fd, ft = fm + fp, fk + fm + fp
    sf_d = all(e == 1 for e in fd.values())
    sf_t = all(e == 1 for e in ft.values())
    B = all(e == 1 for p, e in fd.items() if p != 2)
    Bp = B and k % 8 != 4
    C = all(e < 2 for p, e in fd.items() if p % 4 == 3)
    Cp = C and k % 8 != 4
    flags = {
        "A": k % 3 != 0 and sf_t,
        "A'": k % 3 != 0 and sf_d,
        "A''": sf_d,
        "B": B,
        "B'": Bp,
        "B''": Bp and all(e < 3 or p % 3 == 2 for p, e in fk.items()),
        "C": C,
        "C'": Cp,
        "C''": Cp and all(e < 3 or p % 3 == 2 for p, e in ft.items()),
    }
    assert not flags["A"] or flags["A'"]
    assert not flags["A'"] or flags["A''"]
    assert not flags["B''"] or flags["B'"]
    assert not flags["B'"] or flags["B"]
    assert not flags["C''"] or flags["C'"]
    assert not flags["C'"] or flags["C"]
    return flags, fk, fd, ft


def aggregate(values):
    return {"per_run_percent": values, "mean_percent": statistics.mean(values),
            "sample_sd_percentage_points": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--prior-a", type=Path, required=True,
                        help="Previously verified Figure5.6 selected pooled CSV")
    parser.add_argument("--overall", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arithmetic_path = args.inputs / "evaluation_arithmetic_inputs.json"
    arithmetic = read_json(arithmetic_path)
    assert arithmetic["file_label_to_paper_role"] == ROLES
    baseline = arithmetic["by_file_label"]
    spf = factor_table(max(abs(r["k"]) + 4 for rr in baseline.values() for r in rr))
    with args.prior_a.open() as f:
        prior_a = {(r["run_id"], r["paper_role"]): r for r in csv.DictReader(f)
                   if r["pooled_group"] == "A_true_0_or_1"}
    with args.overall.open() as f:
        overall = {(r["run_id"], r["paper_role"], int(r["completed_epoch"])): r for r in csv.DictReader(f)}
    metadata, records, membership_rows = {}, {}, []
    for label, role in ROLES.items():
        records[label] = {}
        all_sizes, restricted_sizes = Counter(), Counter()
        restricted_truth = {name: Counter() for name in SETS}
        for old in baseline[label]:
            k = old["k"]
            target = vp3(old["answer"][1]) - vp3(old["answer"][0])
            flags, fk, fd, ft = membership(k, spf)
            restricted = target in (0, 1)
            row = {"file_label": label, "paper_role": role, "example_id": old["example_id"],
                   "k": k, "true_v3_n": target, "true_zero_or_one": restricted,
                   **{name: flags[name] for name in SETS},
                   "factor_k": json.dumps(sorted(fk.items())),
                   "factor_k2_minus_16": json.dumps(sorted(fd.items())),
                   "factor_product": json.dumps(sorted(ft.items()))}
            membership_rows.append(row)
            records[label][old["example_id"]] = row
            for name in SETS:
                if flags[name]:
                    all_sizes[name] += 1
                    if restricted:
                        restricted_sizes[name] += 1
                        restricted_truth[name][target] += 1
        assert len(records[label]) == 3000
        metadata[role] = {"examples": 3000, "minimum_k": min(r["k"] for r in baseline[label]),
                          "sets": {name: {"all_examples": all_sizes[name],
                                          "true_zero_or_one_examples": restricted_sizes[name],
                                          "true_zero": restricted_truth[name][0],
                                          "true_one": restricted_truth[name][1]}
                                   for name in SETS}}
    vk = {r["k"] for r in records["test1"].values()}
    tk = {r["k"] for r in records["valid"].values()}
    assert len(vk) == len(tk) == 3000 and not vk.intersection(tk)
    per_epoch, selected_rows, identities = [], [], {}
    hashes = {"evaluation_arithmetic_inputs.json": hashlib.sha256(arithmetic_path.read_bytes()).hexdigest()}
    target_checks, global_checks, a_checks = 0, 0, 0
    for run, chosen in SELECTED.items():
        for label, role in ROLES.items():
            path = args.inputs / run / f"eval.{label}.epochs_0_499.json.bz2"
            hashes[f"{run}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
            data = read_json(path)
            assert list(data) == [r["example_id"] for r in baseline[label]]
            epochs = list(range(1, 501)) if role == "validation" else [chosen]
            keys = [str(e-1) for e in epochs]
            correct = [[0]*len(epochs) for _ in SETS]
            invalid = [[0]*len(epochs) for _ in SETS]
            noninteger = [[0]*len(epochs) for _ in SETS]
            features, global_correct = [], 0
            for example_id, item in data.items():
                old = records[label][example_id]
                assert item["problem"]["k"] == old["k"]
                target = old["true_v3_n"]
                answer = decode_pair(item["answer"])
                assert exact(answer, -target)
                target_checks += 1
                features.append([example_id, item["problem"], item["answer"]])
                guesses = item["guesses_greedy"]
                global_correct += exact(decode_pair(guesses.get(str(chosen-1))), -target)
                if not old["true_zero_or_one"]:
                    continue
                members = [i for i, name in enumerate(SETS) if old[name]]
                for j, key in enumerate(keys):
                    pred = decode_pair(guesses.get(key))
                    if exact(pred, -target):
                        for i in members:
                            correct[i][j] += 1
                    elif pred is None:
                        for i in members:
                            invalid[i][j] += 1
                    elif pred[0] % pred[1] != 0:
                        for i in members:
                            noninteger[i][j] += 1
            if role in identities:
                assert identities[role] == features
            else:
                identities[role] = features
            assert global_correct == int(overall[(run, role, chosen)]["greedy_exact_count"])
            global_checks += 1
            for i, name in enumerate(SETS):
                n = metadata[role]["sets"][name]["true_zero_or_one_examples"]
                assert n > 0
                for j, epoch in enumerate(epochs):
                    row = {"run_id": run, "file_label": label, "paper_role": role,
                           "completed_epoch": epoch, "epoch_index": epoch-1, "set": name,
                           "examples": n, "correct": correct[i][j],
                           "invalid": invalid[i][j], "noninteger": noninteger[i][j],
                           "exact_percent": 100 * correct[i][j] / n}
                    per_epoch.append(row)
                    if epoch == chosen:
                        selected_rows.append(row)
                        if name == "A":
                            old = prior_a[(run, role)]
                            assert row["examples"] == int(old["examples"])
                            assert row["correct"] == int(old["correct"])
                            a_checks += 2
            print(f"Processed {run}, {role}, {len(epochs)} epochs", flush=True)

    aggregates, aggregate_rows = {}, []
    for role in ROLES.values():
        aggregates[role] = {}
        for name in SETS:
            rr = [r for r in selected_rows if r["paper_role"] == role and r["set"] == name]
            assert len(rr) == 3 and len({r["examples"] for r in rr}) == 1
            agg = aggregate([r["exact_percent"] for r in rr])
            aggregates[role][name] = {"examples_per_run": rr[0]["examples"],
                                      "correct_counts": [r["correct"] for r in rr], **agg}
            aggregate_rows.append({"paper_role": role, "set": name, "examples": rr[0]["examples"],
                                   "mean_percent": agg["mean_percent"],
                                   "sample_sd_percentage_points": agg["sample_sd_percentage_points"],
                                   **{r["run_id"] + "_correct": r["correct"] for r in rr}})
    by_curve = {}
    for row in per_epoch:
        if row["paper_role"] == "validation":
            by_curve.setdefault((row["completed_epoch"], row["set"]), []).append(row)
    curves = []
    for (epoch, name), rr in sorted(by_curve.items(), key=lambda x: (x[0][0], SETS.index(x[0][1]))):
        assert len(rr) == 3
        agg = aggregate([r["exact_percent"] for r in rr])
        curves.append({"completed_epoch": epoch, "paper_role": "validation", "set": name,
                       "examples_per_run": rr[0]["examples"], "mean_percent": agg["mean_percent"],
                       "sample_sd_percentage_points": agg["sample_sd_percentage_points"],
                       **{r["run_id"] + "_correct": r["correct"] for r in rr}})
    comparisons = {}
    for role in ROLES.values():
        ranking = sorted(SETS, key=lambda name: aggregates[role][name]["mean_percent"], reverse=True)
        comparisons[role] = {"descending_order_by_selected_mean": ranking,
                             "B_to_Bprime_gain_percentage_points": aggregates[role]["B'"]["mean_percent"]-aggregates[role]["B"]["mean_percent"],
                             "C_to_Cprime_gain_percentage_points": aggregates[role]["C'"]["mean_percent"]-aggregates[role]["C"]["mean_percent"]}
    summary = {
        "sets_in_table_order": list(SETS),
        "definitions": {
            "A": "3 does not divide k; k(k^2-16) squarefree",
            "A'": "3 does not divide k; k^2-16 squarefree",
            "A''": "k^2-16 squarefree",
            "B": "odd part of k^2-16 squarefree",
            "B'": "B with k not congruent to4 modulo8",
            "B''": "B' and p^3 divides k implies p=2 modulo3",
            "C": "p^2 does not divide k^2-16 for every prime p=3 modulo4",
            "C'": "C with k not congruent to4 modulo8",
            "C''": "C' and p^3 divides k(k^2-16) implies p=2 modulo3",
        },
        "membership_method": "Merge exact prime factorizations of abs(k-4),abs(k),abs(k+4) using a smallest-prime-factor sieve",
        "small_k_policy": "No additional k>=5 restriction; evaluation pools contain no k<5",
        "metric": "Greedy exact valuation accuracy restricted to TRUE v3(n_k) in {0,1} within each set",
        "prediction_policy": "Every wrong, nonintegral or invalid prediction remains a failure; no collapse of outputs into binary labels",
        "raw_target_convention": "Raw v3(r_k) is the negative of paper v3(n_k)",
        "file_label_to_paper_role": ROLES, "runs": list(SELECTED),
        "selected_completed_epochs": SELECTED,
        "selection": "Existing global validation choices in epochs1-300, earliest tie; same checkpoint for every set within a run",
        "table_pool": "test", "curve_pool": "validation", "curve_completed_epochs": [1,500],
        "test_scoring_scope": "Only the existing selected checkpoint per run",
        "pools": metadata, "selected_accuracies": aggregates, "comparisons": comparisons,
        "sd": "Sample SD across three runs (divisor2), not a confidence interval",
        "curve_sd_policy": "Computed and retained in CSV; no filled bands in the nine-line display",
        "smoothing": "none", "raw_target_occurrences_checked_against_r_k": target_checks,
        "global_selected_counts_match_prior_raw_audit": global_checks,
        "set_A_selected_count_and_denominator_checks_against_Figure5_6": a_checks,
        "ordered_example_ids_k_targets_and_features_match_across_runs": True,
        "validation_test_k_overlap": 0,
        "source_sha256": hashes,
        "prior_A_csv_sha256": hashlib.sha256(args.prior_a.read_bytes()).hexdigest(),
        "prior_overall_csv_sha256": hashlib.sha256(args.overall.read_bytes()).hexdigest(),
    }
    write_csv(args.out / "sets_membership.csv", membership_rows)
    write_csv(args.out / "sets_per_epoch_metrics.csv", per_epoch)
    write_csv(args.out / "sets_selected_metrics.csv", selected_rows)
    write_csv(args.out / "sets_selected_aggregates.csv", aggregate_rows)
    write_csv(args.out / "sets_validation_curves.csv", curves)
    (args.out / "sets_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"selected_test": aggregates["test"], "selected_validation": aggregates["validation"],
                      "comparisons": comparisons},indent=2))


if __name__ == "__main__":
    main()
