#!/usr/bin/env python3
"""Independent raw-prediction audit for Figure 5.7's nine arithmetic sets.

Membership uses direct square/cube divisibility of the relevant integers,
not merged factorizations. Accuracy is restricted to TRUE v3(n_k) in {0,1}.
The test pool is scored only at the existing globally selected checkpoints.
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


RUNS = ('1000260_9', '1000260_10', '1000260_11')
ROLES = {'test1': 'validation', 'valid': 'test'}
SELECTED = {'1000260_9': 219, '1000260_10': 279, '1000260_11': 297}
SETS = ('A', "A'", "A''", 'B', "B'", "B''", 'C', "C'", "C''")
BAD_STATUSES = ('missing', 'null', 'malformed', 'zero_denominator', 'noninteger')
MISSING = object()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def rational(value):
    if value is MISSING:
        return 'missing', None
    if value is None:
        return 'null', None
    if not isinstance(value, list) or len(value) != 2 or any(type(x) is not int for x in value):
        return 'malformed', None
    if value[1] == 0:
        return 'zero_denominator', None
    result = Fraction(value[0], value[1])
    return ('integer' if result.denominator == 1 else 'noninteger'), result


def valuation3(value):
    value = abs(value)
    assert value > 0
    power = 0
    while value % 3 == 0:
        value //= 3
        power += 1
    return power


def primes_through(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[:2] = b'\x00\x00'
    for prime in range(2, math.isqrt(limit) + 1):
        if sieve[prime]:
            sieve[prime*prime:limit+1:prime] = b'\x00' * ((limit-prime*prime)//prime + 1)
    return [prime for prime in range(2, limit + 1) if sieve[prime]]


def membership(k, primes):
    """Determine all nine flags by direct divisibility.

    Distinct factors k-4,k,k+4 differ by 4 or 8. Odd primes cannot divide
    two factors. Thus an odd p^2 or p^3 dividing a nonzero product must
    occur wholly within one factor, implying p^2 or p^3 <= |k|+4. Prime 2
    is explicitly included in square tests and is allowed by both cube
    conditions (2 == 2 mod 3). Zero cases are handled separately. No
    lower bound on k is added.
    """
    d = k*k - 16
    product = k*d
    bound = abs(k) + 4
    square_in_d = d == 0
    odd_square_in_d = d == 0
    square_in_product = product == 0
    forbidden_c_square = d == 0
    forbidden_cube_k = k == 0
    forbidden_cube_product = product == 0
    for prime in primes:
        square = prime * prime
        if square > bound:
            break
        if d and d % square == 0:
            square_in_d = True
            if prime != 2:
                odd_square_in_d = True
            if prime % 4 == 3:
                forbidden_c_square = True
        if product and product % square == 0:
            square_in_product = True
        if prime % 3 != 2:
            cube = square * prime
            if k and cube <= abs(k) and k % cube == 0:
                forbidden_cube_k = True
            if product and cube <= bound and product % cube == 0:
                forbidden_cube_product = True
    not_three = k % 3 != 0
    not_four_mod_eight = k % 8 != 4
    flags = {
        'A': not_three and not square_in_product,
        "A'": not_three and not square_in_d,
        "A''": not square_in_d,
        'B': not odd_square_in_d,
        "B'": not odd_square_in_d and not_four_mod_eight,
        "B''": not odd_square_in_d and not_four_mod_eight and not forbidden_cube_k,
        'C': not forbidden_c_square,
        "C'": not forbidden_c_square and not_four_mod_eight,
        "C''": not forbidden_c_square and not_four_mod_eight and not forbidden_cube_product,
    }
    for smaller, larger in (('A', "A'"), ("A'", "A''"), ("A''", 'B'), ('B', 'C'),
                            ("B'", 'B'), ("B''", "B'"), ("C'", 'C'), ("C''", "C'"),
                            ("B'", "C'"), ("B''", "C''")):
        assert not flags[smaller] or flags[larger]
    evidence = {
        'k_squared_minus_16': d, 'k_times_k_squared_minus_16': product,
        'square_divides_d': square_in_d, 'odd_prime_square_divides_d': odd_square_in_d,
        'square_divides_product': square_in_product,
        'prime_3_mod_4_square_divides_d': forbidden_c_square,
        'forbidden_prime_cube_divides_k': forbidden_cube_k,
        'forbidden_prime_cube_divides_product': forbidden_cube_product,
    }
    return flags, evidence


def percent(count, denominator):
    assert denominator > 0
    return float(Fraction(100 * count, denominator))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    assert [row['run_id'] for row in rows] == list(RUNS)
    assert len({row['examples'] for row in rows}) == 1
    result = {'paper_role': rows[0]['paper_role'], 'set_name': rows[0]['set_name'],
              'examples_per_run': rows[0]['examples'], 'true_zero_examples': rows[0]['true_zero_examples'],
              'true_one_examples': rows[0]['true_one_examples'], 'runs': 3}
    for row in rows:
        suffix = row['run_id'].rsplit('_', 1)[1]
        result[f'run_{suffix}_completed_epoch'] = row['completed_epoch']
        result[f'run_{suffix}_exact_correct'] = row['exact_correct']
        result[f'run_{suffix}_accuracy_percent'] = row['exact_accuracy_percent']
    values = [row['exact_accuracy_percent'] for row in rows]
    result['mean_percent'] = statistics.mean(values)
    result['sample_sd_percentage_points'] = statistics.stdev(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.inputs / 'evaluation_arithmetic_inputs.json'
    arithmetic = json.loads(index_path.read_text())
    assert arithmetic['file_label_to_paper_role'] == ROLES
    max_abs_k = max(abs(row['k']) for rows in arithmetic['by_file_label'].values() for row in rows)
    primes = primes_through(math.isqrt(max_abs_k + 4))
    reference, memberships, pool_counts = {}, [], {}
    hashes = {'evaluation_arithmetic_inputs.json': sha256(index_path)}
    for label, role in ROLES.items():
        reference[label] = {}
        counts = {name: Counter() for name in SETS}
        for example in arithmetic['by_file_label'][label]:
            status, stored_r = rational(example['stored_answer'])
            assert status in ('integer', 'noninteger') and stored_r
            target = valuation3(stored_r.denominator) - valuation3(stored_r.numerator)
            k = example['k']
            flags, evidence = membership(k, primes)
            eligible = target in (0, 1)
            record = {'file_label': label, 'paper_role': role, 'example_id': example['example_id'],
                      'k': k, 'true_v3_n': target, 'eligible_true_0_or_1': eligible,
                      **{'in_' + name: flags[name] for name in SETS}, **evidence}
            memberships.append(record)
            reference[label][example['example_id']] = record
            for name in SETS:
                if flags[name]:
                    counts[name]['all_members'] += 1
                    counts[name]['eligible_true_0_or_1'] += eligible
                    counts[name]['true_zero_examples'] += target == 0
                    counts[name]['true_one_examples'] += target == 1
        pool_counts[role] = {name: dict(counts[name]) for name in SETS}
        for values in pool_counts[role].values():
            assert values['eligible_true_0_or_1'] == values['true_zero_examples'] + values['true_one_examples']
    write_csv(args.out / 'nine_set_membership.csv', memberships)
    print(json.dumps({'restricted_sample_sizes': {
        role: {name: values['eligible_true_0_or_1'] for name, values in sets.items()}
        for role, sets in pool_counts.items()}}, separators=(',', ':')), flush=True)

    per_epoch, selected_rows = [], []
    ordered = {}
    invalid_examples = []
    target_checks = 0
    parsed_predictions = 0
    membership_prediction_contributions = 0
    for run in RUNS:
        for label, role in ROLES.items():
            path = args.inputs / run / f'eval.{label}.epochs_0_499.json.bz2'
            hashes[f'{run}/{path.name}'] = sha256(path)
            with bz2.open(path, 'rt', encoding='utf-8') as handle:
                data = json.load(handle, object_pairs_hook=unique_object)
            assert len(data) == 3000 and list(data) == list(reference[label])
            epochs = list(range(1, 501)) if role == 'validation' else [SELECTED[run]]
            correct = {name: [0] * len(epochs) for name in SETS}
            negative = {name: [0] * len(epochs) for name in SETS}
            bad = {name: [Counter() for _ in epochs] for name in SETS}
            sequence = []
            for example_id, example in data.items():
                ref = reference[label][example_id]
                k = example['problem']['k']
                assert type(k) is int and k == ref['k']
                status, raw_target = rational(example['answer'])
                assert status == 'integer' and -raw_target == ref['true_v3_n']
                target = ref['true_v3_n']
                target_checks += 1
                sequence.append((example_id, k, target))
                if not ref['eligible_true_0_or_1']:
                    continue
                active_sets = [name for name in SETS if ref['in_' + name]]
                if not active_sets:
                    continue
                guesses = example['guesses_greedy']
                assert isinstance(guesses, dict)
                for position, epoch in enumerate(epochs):
                    status, prediction = rational(guesses.get(str(epoch - 1), MISSING))
                    parsed_predictions += 1
                    membership_prediction_contributions += len(active_sets)
                    if prediction is not None and -prediction == target:
                        for name in active_sets:
                            correct[name][position] += 1
                    if status != 'integer':
                        for name in active_sets:
                            bad[name][position][status] += 1
                        invalid_examples.append({'run_id': run, 'paper_role': role, 'completed_epoch': epoch,
                                                 'example_id': example_id, 'k': k, 'status': status,
                                                 'sets': active_sets, 'raw_prediction': guesses.get(str(epoch - 1))})
                    elif prediction > 0:
                        for name in active_sets:
                            negative[name][position] += 1
            if label in ordered:
                assert sequence == ordered[label]
            else:
                ordered[label] = sequence
            for position, epoch in enumerate(epochs):
                for name in SETS:
                    counts = pool_counts[role][name]
                    n = counts['eligible_true_0_or_1']
                    row = {'run_id': run, 'file_label': label, 'paper_role': role,
                           'epoch_index': epoch - 1, 'completed_epoch': epoch, 'set_name': name,
                           'examples': n, 'true_zero_examples': counts['true_zero_examples'],
                           'true_one_examples': counts['true_one_examples'], 'exact_correct': correct[name][position],
                           **{'parse_' + status: bad[name][position][status] for status in BAD_STATUSES},
                           'invalid_total': sum(bad[name][position].values()),
                           'negative_integer_predictions': negative[name][position],
                           'exact_accuracy_percent': percent(correct[name][position], n)}
                    assert 0 <= row['exact_correct'] <= n - row['invalid_total']
                    per_epoch.append(row)
                    if epoch == SELECTED[run]:
                        selected_rows.append(row)
            del data

    assert target_checks == 18000
    assert len(per_epoch) == 13527 and len(selected_rows) == 54
    val_ks = {row[1] for row in ordered['test1']}
    test_ks = {row[1] for row in ordered['valid']}
    assert len(val_ks) == len(test_ks) == 3000 and not val_ks.intersection(test_ks)
    selected_index = {(row['run_id'], row['paper_role'], row['set_name']): row for row in selected_rows}
    val_index = {(row['run_id'], row['completed_epoch'], row['set_name']): row
                 for row in per_epoch if row['paper_role'] == 'validation'}
    curves = []
    for epoch in range(1, 501):
        for name in SETS:
            curves.append({'completed_epoch': epoch, 'epoch_index': epoch - 1,
                           **aggregate([val_index[run, epoch, name] for run in RUNS])})
    selected_aggregates = [aggregate([selected_index[run, role, name] for run in RUNS])
                           for role in ROLES.values() for name in SETS]

    rankings, changes = {}, []
    for role in ROLES.values():
        means = {name: sum(Fraction(selected_index[run, role, name]['exact_correct'],
                                    selected_index[run, role, name]['examples']) for run in RUNS) / 3
                 for name in SETS}
        order = sorted(SETS, key=lambda name: (-means[name], SETS.index(name)))
        per_run = {}
        for run in RUNS:
            rates = {name: Fraction(selected_index[run, role, name]['exact_correct'],
                                    selected_index[run, role, name]['examples']) for name in SETS}
            per_run[run] = {'A_rank': 1 + sum(value > rates['A'] for value in rates.values()),
                            'sets_strictly_above_A': [name for name in SETS if rates[name] > rates['A']]}
        rankings[role] = {'ordered_by_selected_mean_accuracy': order,
                          'A_rank_by_mean': 1 + sum(value > means['A'] for value in means.values()),
                          'sets_strictly_above_A_mean': [name for name in SETS if means[name] > means['A']],
                          'sets_tied_with_A_mean': [name for name in SETS if name != 'A' and means[name] == means['A']],
                          'per_run': per_run}
        for base, restricted in (('B', "B'"), ('C', "C'")):
            deltas = []
            result = {'paper_role': role, 'base_set': base, 'restricted_set': restricted,
                      'base_examples': pool_counts[role][base]['eligible_true_0_or_1'],
                      'restricted_examples': pool_counts[role][restricted]['eligible_true_0_or_1'],
                      'base_mean_percent': float(100 * means[base]),
                      'restricted_mean_percent': float(100 * means[restricted]),
                      'mean_change_percentage_points': float(100 * (means[restricted] - means[base]))}
            for run in RUNS:
                a, b = selected_index[run, role, base], selected_index[run, role, restricted]
                delta = 100 * (Fraction(b['exact_correct'], b['examples']) - Fraction(a['exact_correct'], a['examples']))
                deltas.append(float(delta))
                result[f'run_{run.rsplit("_", 1)[1]}_change_percentage_points'] = float(delta)
            result['sample_sd_of_run_changes_percentage_points'] = statistics.stdev(deltas)
            result['all_three_runs_increase'] = all(value > 0 for value in deltas)
            changes.append(result)

    summary = {
        'policy': {'sets_in_order': list(SETS),
                   'definitions': {
                       'A': '3 does not divide k; k(k^2-16) squarefree',
                       "A'": '3 does not divide k; k^2-16 squarefree',
                       "A''": 'k^2-16 squarefree',
                       'B': 'Odd part of k^2-16 squarefree',
                       "B'": 'B, excluding k=4 mod 8',
                       "B''": "B', with every prime p satisfying p^3|k congruent to 2 mod 3",
                       'C': 'No p^2 divides k^2-16 for any prime p=3 mod 4',
                       "C'": 'C, excluding k=4 mod 8',
                       "C''": "C', with every prime p satisfying p^3|k(k^2-16) congruent to 2 mod 3"},
                   'membership_method': 'Direct prime-square/cube divisibility, with bounds justified in membership()',
                   'additional_k_ge_5_restriction': False,
                   'conditioning': 'Membership of the named set AND TRUE v3(n_k) in {0,1}',
                   'prediction_rule': 'Greedy exact rational equality; predictions outside {0,1} are failures',
                   'target_and_prediction': 'Negative of raw decoded v3(r_k)',
                   'invalid_outputs': 'Failures retained in the full conditional denominator',
                   'file_label_to_paper_role': ROLES,
                   'existing_global_selected_completed_epochs': SELECTED,
                   'completed_epoch': 'JSON key + 1',
                   'validation_scoring_completed_epochs': [1, 500],
                   'test_scoring': 'Only the existing selected epoch in each run',
                   'new_checkpoint_selection': False,
                   'sd': 'Sample SD across three run percentages, divisor 2',
                   'interpretation': 'Descriptive comparisons; no significance or network-mechanism inference'},
        'source_sha256': hashes, 'pool_set_counts': pool_counts,
        'small_k_memberships': {str(k): membership(k, primes)[0] for k in range(0, 6)},
        'checks': {'membership_examples': len(memberships), 'membership_flags': len(memberships) * len(SETS),
                   'raw_target_occurrences_checked': target_checks,
                   'ordered_ids_parameters_and_targets_match_across_runs': True,
                   'evaluation_pools_disjoint': True,
                   'scored_epoch_set_rows': len(per_epoch), 'validation_curve_rows': len(curves),
                   'selected_rows': len(selected_rows), 'raw_prediction_records_parsed': parsed_predictions,
                   'prediction_contributions_to_overlapping_sets': membership_prediction_contributions},
        'selected_aggregates': selected_aggregates,
        'selected_accuracy_rankings': rankings,
        'selected_restriction_changes': changes,
        'invalid_prediction_records_in_any_eligible_set': invalid_examples,
    }
    output_rows = {'nine_set_per_epoch_metrics.csv': per_epoch,
                   'nine_set_validation_curves.csv': curves,
                   'nine_set_selected_metrics.csv': selected_rows,
                   'nine_set_selected_aggregates.csv': selected_aggregates,
                   'nine_set_selected_changes.csv': changes}
    for name, rows in output_rows.items():
        write_csv(args.out / name, rows)
    summary['output_csv_sha256'] = {name: sha256(args.out / name)
                                    for name in ('nine_set_membership.csv', *output_rows)}
    (args.out / 'nine_set_audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'checks': summary['checks'],
                      'selected_test': [{k: row[k] for k in ('set_name', 'examples_per_run', 'mean_percent', 'sample_sd_percentage_points')}
                                        for row in selected_aggregates if row['paper_role'] == 'test'],
                      'rankings': rankings, 'changes': changes}, indent=2))


if __name__ == '__main__':
    main()
