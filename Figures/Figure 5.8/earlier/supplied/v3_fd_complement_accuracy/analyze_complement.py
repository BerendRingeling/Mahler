#!/usr/bin/env python3
"""Derive complement accuracy by an exact partition of audited counts.

Restrict to true v3(n_k) in {0,1}; partition into C'', the special
congruence classes, and their complement. The independent companion
script directly rechecks membership and predictions in the raw exports.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics

RUNS = ("1000260_9", "1000260_10", "1000260_11")
SELECTED = dict(zip(RUNS, (219, 279, 297)))
ROLES = {"test1": "validation", "valid": "test"}
SPECIAL = {0, 5, 22}
FILES = ("memberships.csv", "global_class_metrics.csv", "sets_per_epoch_metrics.csv",
         "congruence_validation_metrics.csv", "congruence_selected_metrics.csv")


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def boolean(value):
    assert value in ("True", "False")
    return value == "True"


def aggregate(values):
    return {"per_run_percent": values, "mean_percent": statistics.mean(values),
            "sample_sd_percentage_points": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    refs = {name:read_csv(args.references/name) for name in FILES}
    global_classes = {(r["run_id"],r["paper_role"],int(r["completed_epoch"]),int(r["true_v3_n"])):r
                      for r in refs["global_class_metrics.csv"]}
    cc = {(r["run_id"],r["paper_role"],int(r["completed_epoch"])):r
          for r in refs["sets_per_epoch_metrics.csv"] if r["set"] == "C''"}
    special = {}
    for r in refs["congruence_validation_metrics.csv"] + refs["congruence_selected_metrics.csv"]:
        key = (r["run_id"],r["paper_role"],int(r["completed_epoch"]))
        if key in special:
            for field in ("examples","true_zero_count","true_one_count","correct_true_one_count"):
                assert int(special[key][field]) == int(r[field])
        special[key] = r

    membership, pools = [], {}
    for label, role in ROLES.items():
        source = [r for r in refs["memberships.csv"] if r["paper_role"] == role]
        assert len(source) == 3000 and {r["file_label"] for r in source} == {label}
        counts, targets = Counter(), {part:Counter() for part in ("global","Cdoubleprime","special","complement")}
        for r in source:
            k, target = int(r["k"]), int(r["true_v3_n"])
            in_cc, in_special = boolean(r["C''"]), k % 27 in SPECIAL
            assert not (in_cc and in_special)
            in_complement = not in_cc and not in_special
            part = "Cdoubleprime" if in_cc else "special" if in_special else "complement"
            restricted = target in (0,1)
            counts[part+"_all"] += 1
            targets[part][target] += 1
            targets["global"][target] += 1
            counts[part+"_restricted"] += restricted
            membership.append({"file_label":label,"paper_role":role,"example_id":r["example_id"],
                               "k":k,"true_v3_n":target,"in_Cdoubleprime":in_cc,
                               "in_special_congruence":in_special,"in_complement":in_complement,
                               "true_zero_or_one":restricted,"included_in_accuracy":in_complement and restricted})
        assert counts["Cdoubleprime_all"]+counts["special_all"]+counts["complement_all"] == 3000
        assert targets["special"][0] == 0
        n = counts["complement_restricted"]
        pools[role] = {
            "all_examples":3000,
            "global_restricted_examples":targets["global"][0]+targets["global"][1],
            "Cdoubleprime_all_examples":counts["Cdoubleprime_all"],
            "Cdoubleprime_restricted_examples":counts["Cdoubleprime_restricted"],
            "special_all_examples":counts["special_all"],
            "special_restricted_examples":counts["special_restricted"],
            "complement_all_examples":counts["complement_all"],
            "complement_restricted_examples":n,
            "complement_true_zero":targets["complement"][0],
            "complement_true_one":targets["complement"][1],
            "constant_zero_percent":100*targets["complement"][0]/n,
            "constant_one_percent":100*targets["complement"][1]/n,
            "complement_true_valuation_histogram":dict(sorted(targets["complement"].items())),
            "Cdoubleprime_and_special_intersection":0,
        }
    vk = {r["k"] for r in membership if r["paper_role"] == "validation"}
    tk = {r["k"] for r in membership if r["paper_role"] == "test"}
    assert len(vk) == len(tk) == 3000 and not vk.intersection(tk)

    rows, selected_rows = [], []
    for run in RUNS:
        for role in ROLES.values():
            epochs = list(range(1,501)) if role == "validation" else [SELECTED[run]]
            meta = pools[role]
            for epoch in epochs:
                key = (run,role,epoch)
                g0, g1 = global_classes[(*key,0)], global_classes[(*key,1)]
                c, s = cc[key], special[key]
                assert int(s["true_zero_count"]) == 0
                global_n = int(g0["examples"])+int(g1["examples"])
                global_correct = int(g0["greedy_correct"])+int(g1["greedy_correct"])
                c_n, c_correct = int(c["examples"]),int(c["correct"])
                # Only true1 occurs in the special union's restricted population.
                # Do NOT subtract its unrestricted all-valuation exact count.
                s_n, s_correct = int(s["true_one_count"]),int(s["correct_true_one_count"])
                n, correct = global_n-c_n-s_n, global_correct-c_correct-s_correct
                assert global_n == meta["global_restricted_examples"]
                assert c_n == meta["Cdoubleprime_restricted_examples"]
                assert s_n == meta["special_restricted_examples"]
                assert n == meta["complement_restricted_examples"]
                assert 0 <= correct <= n
                row = {"run_id":run,"paper_role":role,"completed_epoch":epoch,"epoch_index":epoch-1,
                       "examples":n,"correct":correct,"exact_percent":100*correct/n,
                       "global_restricted_examples":global_n,"global_restricted_correct":global_correct,
                       "Cdoubleprime_restricted_examples":c_n,"Cdoubleprime_restricted_correct":c_correct,
                       "special_restricted_examples":s_n,"special_restricted_correct":s_correct,
                       "constant_zero_percent":meta["constant_zero_percent"],
                       "constant_one_percent":meta["constant_one_percent"]}
                rows.append(row)
                if epoch == SELECTED[run]:
                    selected_rows.append(row)
    assert len(rows) == 1503 and len(selected_rows) == 6
    curves = []
    for epoch in range(1,501):
        rr = [r for r in rows if r["paper_role"] == "validation" and r["completed_epoch"] == epoch]
        assert len(rr) == 3
        a = aggregate([r["exact_percent"] for r in rr])
        mean, sd = a["mean_percent"],a["sample_sd_percentage_points"]
        curves.append({"completed_epoch":epoch,"paper_role":"validation",
                       "examples_per_run":rr[0]["examples"],"mean_percent":mean,
                       "sample_sd_percentage_points":sd,
                       "band_low_percent":max(0,mean-sd),"band_high_percent":min(100,mean+sd),
                       "constant_one_percent":pools["validation"]["constant_one_percent"],
                       **{r["run_id"]+"_correct":r["correct"] for r in rr}})
    selected = {}
    for role in ROLES.values():
        rr = [r for r in selected_rows if r["paper_role"] == role]
        a = aggregate([r["exact_percent"] for r in rr])
        selected[role] = {"examples_per_run":rr[0]["examples"],
                          "correct_counts":[r["correct"] for r in rr],**a,
                          "constant_one_percent":pools[role]["constant_one_percent"],
                          "constant_zero_percent":pools[role]["constant_zero_percent"],
                          "mean_minus_constant_one_percentage_points":a["mean_percent"]-pools[role]["constant_one_percent"]}
    fixed = {str(e):curves[e-1] for e in (1,50,100,150,200,250,300,350,400,450,500)}
    window_rows = []
    for lo, hi in ((200,300),(400,500)):
        for run in RUNS:
            rr = [r for r in rows if r["paper_role"] == "validation" and r["run_id"] == run and lo<=r["completed_epoch"]<=hi]
            assert len(rr) == hi-lo+1
            window_rows.append({"run_id":run,"first_completed_epoch":lo,"last_completed_epoch":hi,
                                "epochs":len(rr),"mean_exact_percent":statistics.mean(r["exact_percent"] for r in rr)})
    window_aggregates = {f"{lo}-{hi}":aggregate([r["mean_exact_percent"] for r in window_rows if r["first_completed_epoch"]==lo])
                         for lo,hi in ((200,300),(400,500))}
    summary = {
        "population":"k not in C'' AND k mod27 not in {0,5,22}, restricted to true v3(n_k) in {0,1}",
        "Cdoubleprime_definition":"p^2 does not divide k^2-16 for all primes p=3mod4; k not4mod8; p^3 divides k(k^2-16) implies p=2mod3",
        "method":"Exact partition of previously independently audited correct counts; independently verified against raw complement predictions in the companion audit",
        "special_subtraction":"correct_true_one_count, not unrestricted exact_correct; no true0 examples occur in this special union in either pool",
        "metric":"Greedy exact valuation accuracy; incorrect and invalid outputs remain failures in the full conditional denominator",
        "runs":list(RUNS),"file_label_to_paper_role":ROLES,"selected_completed_epochs":SELECTED,
        "selection":"Same existing global validation-selected epochs1-300; no new subset checkpoint selection",
        "test_scoring_scope":"Only the existing selected checkpoint of each run",
        "pools":pools,"selected_accuracies":selected,
        "validation_fixed_epochs":fixed,"validation_window_aggregates":window_aggregates,
        "validation_windows_are_descriptive_not_selection_rules":True,
        "sample_SD_divisor":2,"sample_SD_not_confidence_interval":True,
        "curve_pool":"validation","curve_completed_epochs":[1,500],"smoothing":"none",
        "curve_band":"mean +/- sample SD, bounded by0 and100 for display only",
        "partition_count_identities_checked":1503,
        "Cdoubleprime_and_special_disjoint_on_all6000_examples":True,
        "validation_test_k_overlap":0,
        "source_sha256":{name:hashlib.sha256((args.references/name).read_bytes()).hexdigest() for name in FILES},
    }
    write_csv(args.out/"complement_membership.csv",membership)
    write_csv(args.out/"complement_per_epoch_metrics.csv",rows)
    write_csv(args.out/"complement_selected_metrics.csv",selected_rows)
    write_csv(args.out/"complement_validation_curves.csv",curves)
    write_csv(args.out/"complement_validation_windows.csv",window_rows)
    (args.out/"complement_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"pools":pools,"selected":selected,"fixed_epochs":fixed,"windows":window_aggregates},indent=2))


if __name__ == "__main__":
    main()
