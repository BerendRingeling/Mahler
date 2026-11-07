#!/usr/bin/env python3
"""Independent raw-data check of the three rk_wfD repeats.

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
from functools import lru_cache
import hashlib
import json
import math
import resource
import statistics
from fractions import Fraction
from pathlib import Path

RUNS = ("1000262_18", "1000262_19", "1000262_20")
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


@lru_cache(maxsize=None)
def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for d in range(3, math.isqrt(n) + 1, 2):
        if n % d == 0:
            return False
    return True


def inspect(data, baseline, arithmetic, path):
    assert isinstance(data, dict) and len(data) == 3000
    index, features, root_features, feature_records = [], [], [], []
    root_number_counts = collections.Counter()
    problem_field_sets = collections.Counter()
    coverage = {mode: {"missing_epoch_entries": 0, "extra_epoch_entries": 0,
                      "non_dict_records": 0}
                for mode in ("greedy", "sampled")}
    answer_sign_matches_root = 0
    root_number_invalid, malformed_factorizations, product_mismatches = [], [], []
    nonprime_bases, all_bases = [], set()
    factor_lengths, signed_exponents = collections.Counter(), collections.Counter()
    sorted_unique_bases_count = product_match_count = sign_pairs_count = 0
    discriminant_values, sample_encodings = [], []
    for example_id, record in data.items():
        assert isinstance(record, dict)
        answer = record["answer"]
        assert is_rational(answer)
        problem = record["problem"]
        assert isinstance(problem, dict) and type(problem["k"]) is int
        k = problem["k"]
        index.append({"example_id": example_id, "k": k, "answer": canonical(answer)})
        reference = arithmetic[example_id]
        assert reference["k"] == k and reference["answer"] == canonical(answer)
        expected_N = reference["N"]
        assert type(expected_N) is int and expected_N > 0
        # This is the manuscript's stated Weierstrass-model discriminant,
        # not a minimal-discriminant assumption. Negative prime exponents
        # represent its rational denominator and must be retained.
        expected_D = Fraction(k**14 * (k*k - 16), 2**24)
        assert expected_D != 0
        discriminant_values.append(expected_D)
        problem_field_sets[tuple(sorted(problem))] += 1
        w, factorization = problem.get("w"), problem.get("D_Factor")
        features.append([k, w, factorization])
        root_features.append([k, w])
        root_number_counts[str(w)] += 1
        if type(w) is not int or w not in (-1, 1):
            root_number_invalid.append({"example_id": example_id, "value": w})
        else:
            answer_sign = (answer[0]*answer[1] > 0) - (answer[0]*answer[1] < 0)
            answer_sign_matches_root += answer_sign == w
        assert w == reference["w"]
        valid_schema = isinstance(factorization,list) and all(
            isinstance(pair,list) and len(pair)==2 and type(pair[0]) is int
            and type(pair[1]) is int and
            ((pair[0]>=2 and pair[1]!=0) or (pair[0]==-1 and pair[1]==1))
            for pair in factorization)
        reconstructed = None
        if not valid_schema:
            malformed_factorizations.append({"example_id":example_id,"value":factorization})
        else:
            sign_locations = [i for i,pair in enumerate(factorization) if pair[0]==-1]
            if sign_locations and sign_locations != [0]:
                malformed_factorizations.append({"example_id":example_id,
                                                "value":factorization,
                                                "reason":"The optional (-1,1) sign pair must occur only at the start"})
            sign_pairs_count += bool(sign_locations)
            primes = [p for p,e in factorization if p>=2]
            all_bases.update(primes)
            factor_lengths[len(primes)] += 1
            sorted_unique_bases_count += primes == sorted(set(primes))
            signed_exponents.update(str(e) for p,e in factorization if p>=2)
            reconstructed = Fraction(1)
            for p,e in factorization:
                reconstructed *= Fraction(p)**e
            product_match_count += reconstructed == expected_D
            if reconstructed != expected_D:
                product_mismatches.append({"example_id":example_id,"k":k,
                    "reconstructed_D":[reconstructed.numerator,reconstructed.denominator],
                    "expected_D":[expected_D.numerator,expected_D.denominator]})
        feature_records.append({"example_id":example_id,"k":k,"w":w,
            "D_Factor":factorization,
            "D_reconstructed":[reconstructed.numerator,reconstructed.denominator] if reconstructed is not None else None,
            "D_formula":[expected_D.numerator,expected_D.denominator],
            "N_reference":expected_N,"answer":canonical(answer),"stored_answer":answer})
        if len(sample_encodings)<3:
            sample_encodings.append(feature_records[-1])
        for mode in coverage:
            guesses=record.get("guesses_"+mode)
            if not isinstance(guesses,dict):
                coverage[mode]["non_dict_records"] += 1
                continue
            keys=set(guesses)
            coverage[mode]["missing_epoch_entries"] += len(KEY_SET-keys)
            coverage[mode]["extra_epoch_entries"] += len(keys-KEY_SET)
    assert index == baseline, "Example identities or reference rational answers differ"
    for base in sorted(all_bases):
        if not is_prime(base):
            nonprime_bases.append(base)
    info={
        "path":str(path),"compressed_sha256":sha256(path),"examples":len(data),
        "ordered_example_ids_k_canonical_answers_match_rk_baseline":True,
        "all_ids_are_0_to_2999":set(data)=={str(i) for i in range(3000)},
        "distinct_k":len({x["k"] for x in index}),"coverage":coverage,
        "problem_field_sets":{",".join(k):v for k,v in problem_field_sets.items()},
        "root_number_value_counts":dict(root_number_counts),
        "invalid_root_number_values":root_number_invalid,
        "answer_sign_matches_root_number_count":answer_sign_matches_root,
        "root_number_feature_sequence_sha256":hashlib.sha256(json.dumps(root_features,separators=(",",":")).encode()).hexdigest(),
        "full_input_feature_sequence_sha256":hashlib.sha256(json.dumps(features,separators=(",",":")).encode()).hexdigest(),
        "factorization_encoding":"D_Factor is a list of [prime, signed nonzero exponent] integer pairs; optional initial [-1,1] sign pair",
        "factorization_reconstruction_rule":"D = product(Fraction(prime) ** exponent), including sign pair",
        "reference_model_discriminant_formula":"Delta_k = k**14 * (k**2 - 16) / 2**24",
        "model_discriminant_source":"Manuscript eq:Weierstrass and discriminant definition, verified by parent/source agent",
        "not_assumed_minimal_discriminant":True,
        "malformed_factorization_values":malformed_factorizations,
        "factor_product_equals_model_discriminant_count":product_match_count,
        "factor_product_mismatches":product_mismatches,
        "factor_base_primality_method":"Exact trial division through integer square root, cached per base",
        "distinct_factor_bases_checked":len(all_bases),
        "minimum_factor_base":min(all_bases) if all_bases else None,
        "maximum_factor_base":max(all_bases) if all_bases else None,
        "nonprime_factor_bases":nonprime_bases,
        "strictly_increasing_unique_prime_bases_count":sorted_unique_bases_count,
        "number_of_distinct_prime_factors_distribution":dict(factor_lengths),
        "prime_exponent_distribution":dict(signed_exponents),
        "records_with_initial_negative_sign_pair":sign_pairs_count,
        "positive_model_discriminant_count":sum(v>0 for v in discriminant_values),
        "nonintegral_model_discriminant_count":sum(v.denominator!=1 for v in discriminant_values),
        "sample_feature_records":sample_encodings,
    }
    return info,feature_records


def selected_prediction_snapshot(data, arithmetic, chosen, file_label):
    records=[]
    key=str(chosen)
    for example_id,rec in data.items():
        pred=rec["guesses_greedy"].get(key,ABSENT)
        outcome=flags(rec["answer"],pred)
        ref=arithmetic[example_id]
        records.append({"example_id":example_id,"k":rec["problem"]["k"],
                        "w":rec["problem"]["w"],"N":ref["N"],
                        "answer":canonical(rec["answer"]),"stored_answer":rec["answer"],
                        "guess_greedy":None if pred is ABSENT else pred,
                        "canonical_guess_greedy":canonical(pred) if is_rational(pred) else None,
                        "missing":bool(outcome[3]),"invalid":bool(outcome[2]),
                        "exact_correct":bool(outcome[0]),"factor_two_correct":bool(outcome[1]),
                        "factor_two_boundary":bool(outcome[4])})
    return {"file_label":file_label,"paper_role":ROLES[file_label],
            "selected_epoch_index":chosen,"selected_completed_epoch":chosen+1,
            "examples":len(data),"records":records}


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


def perfect_accuracy_diagnostics(validation):
    all_perfect = [e for e in EPOCHS if validation[e]["factor_two_count"] == validation[e]["examples"]]
    window_perfect = [e for e in all_perfect if e < 300]
    last_nonperfect_window = max((e for e in range(300)
                                 if validation[e]["factor_two_count"] != validation[e]["examples"]),
                                default=-1)
    final_perfect_start = last_nonperfect_window + 1
    return {
        "first_100_percent_completed_epoch_among_all500": all_perfect[0]+1 if all_perfect else None,
        "first_100_percent_completed_epoch_in_first300": window_perfect[0]+1 if window_perfect else None,
        "number_of_100_percent_epochs_in_first300": len(window_perfect),
        "number_of_100_percent_epochs_in_all500": len(all_perfect),
        "all_100_percent_epoch_indices_in_first300": window_perfect,
        "first_completed_epoch_from_which_all_remaining_through300_are_perfect":
            final_perfect_start+1 if final_perfect_start < 300 else None,
        "minimum_factor_two_count_in_first300": min(validation[e]["factor_two_count"] for e in range(300)),
        "maximum_factor_two_count_in_first300": max(validation[e]["factor_two_count"] for e in range(300)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--baseline-index", type=Path, required=True)
    ap.add_argument("--arithmetic-index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    baseline = json.loads(args.baseline_index.read_text())
    arithmetic_source = json.loads(args.arithmetic_index.read_text())
    arithmetic = {label:{r["example_id"]:r for r in records}
                  for label,records in arithmetic_source["by_file_label"].items()}
    feature_index = {"file_label_to_paper_role":ROLES,"source_run_id":RUNS[0],
                     "reference_conductor_index_sha256":sha256(args.arithmetic_index),
                     "factorization_encoding":"D_Factor = [[prime, signed exponent], ...], optional initial [-1,1]",
                     "reconstruction":"D = product(Fraction(prime) ** exponent)",
                     "reference_model_discriminant_formula":"Delta_k = k**14 * (k**2 - 16) / 2**24",
                     "not_assumed_minimal_discriminant":True,
                     "source_archives":{},"by_file_label":{}}
    baseline_overlap = len({r["k"] for r in baseline["test1"]} &
                           {r["k"] for r in baseline["valid"]})
    assert baseline_overlap == 0
    result = {
        "method": {
            "file_label_to_paper_role": ROLES,
            "role_basis": "Full ordered k/answer/example-ID match to audited rk baseline",
            "baseline_index_sha256": sha256(args.baseline_index),
            "reference_conductor_index_sha256": sha256(args.arithmetic_index),
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
    snapshots = {"policy":{"decoder":"greedy","selection_metric":"same-sign strict factor-two accuracy","selection_group":"validation","selection_window_completed_epochs":[1,300],"tie_rule":"Earliest validation maximum","test_used_for_selection":False},"file_label_to_paper_role":ROLES,"runs":{}}
    for rid in RUNS:
        print("Reading validation", rid, flush=True)
        val_path = args.inputs / rid / "eval.test1.epochs_0_499.json.bz2"
        with bz2.open(val_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        val_integrity,val_features = inspect(data, baseline["test1"], arithmetic["test1"], val_path)
        if rid == RUNS[0]:
            feature_index["by_file_label"]["test1"] = val_features
            feature_index["source_archives"]["test1"] = {"filename":val_path.name,"sha256":val_integrity["compressed_sha256"]}
        val_metrics = count_epochs(data, EPOCHS)
        choices = {m: select(val_metrics, m) for m in ("factor_two", "exact")}
        factor_two_epoch = choices["factor_two"][0]
        snapshots["runs"][rid] = {"test1":selected_prediction_snapshot(data, arithmetic["test1"], factor_two_epoch, "test1")}
        del data
        gc.collect()
        selected_indices = sorted({299, 499} | {v[0] for v in choices.values()})
        print("Reading test", rid, "at indices", selected_indices, flush=True)
        test_path = args.inputs / rid / "eval.valid.epochs_0_499.json.bz2"
        with bz2.open(test_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
        test_integrity,test_features = inspect(data, baseline["valid"], arithmetic["valid"], test_path)
        if rid == RUNS[0]:
            feature_index["by_file_label"]["valid"] = test_features
            feature_index["source_archives"]["valid"] = {"filename":test_path.name,"sha256":test_integrity["compressed_sha256"]}
        test_metrics = count_epochs(data, selected_indices)
        snapshots["runs"][rid]["valid"] = selected_prediction_snapshot(data, arithmetic["valid"], factor_two_epoch, "valid")
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
            "validation_perfect_accuracy_diagnostics": perfect_accuracy_diagnostics(val_metrics),
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
    result["cross_run_full_input_feature_sequences_identical"] = {
        label: len({result["runs"][rid]["integrity"][label]["full_input_feature_sequence_sha256"]
                    for rid in RUNS}) == 1 for label in ROLES}
    result["factorization_products_match_model_discriminant_for_every_record"] = all(
        info["factor_product_equals_model_discriminant_count"] == info["examples"]
        and not info["malformed_factorization_values"] and not info["nonprime_factor_bases"]
        for run in result["runs"].values() for info in run["integrity"].values())
    feature_index["identical_input_and_answer_sequences_verified_for_runs"] = list(RUNS)
    feature_index["all_factorizations_match_model_discriminant"] = result["factorization_products_match_model_discriminant_for_every_record"]
    feature_path = args.out.parent/"evaluation_discriminant_inputs.json"
    feature_path.write_text(json.dumps(feature_index,indent=2)+"\n")
    result["factorization_input_index"] = feature_path.name
    snapshot_path = args.out.parent/"factor_two_selected_predictions.json"
    snapshot_path.write_text(json.dumps(snapshots,indent=2)+"\n")
    result["selected_prediction_snapshots"] = snapshot_path.name
    result["max_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_epoch_aggregates": result["selected_epoch_aggregates"],
                      "fixed_epoch_aggregates": result["fixed_epoch_aggregates"],
                      "cross_run_root_number_feature_sequences_identical": result["cross_run_root_number_feature_sequences_identical"],
                      "max_process_rss_mib": result["max_process_rss_mib"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
