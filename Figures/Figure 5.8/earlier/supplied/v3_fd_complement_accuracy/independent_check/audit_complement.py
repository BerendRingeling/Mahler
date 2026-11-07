#!/usr/bin/env python3
"""Direct raw audit of the complement of C'' union k mod 27 in {0,5,22}.

Accuracy is restricted to TRUE v3(n_k) in {0,1}. Validation uses all 500
epochs; test uses only the existing selected checkpoints. No model runs.
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
SPECIAL = (0, 5, 22)
FIXED_EPOCHS = (1, 100, 200, 300, 400, 500)
FIXED_WINDOWS = ((200, 300), (400, 500))
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
    exponent = 0
    while value % 3 == 0:
        value //= 3
        exponent += 1
    return exponent


def primes_through(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[:2] = b'\x00\x00'
    for prime in range(2, math.isqrt(limit) + 1):
        if sieve[prime]:
            sieve[prime*prime:limit+1:prime] = b'\x00' * ((limit-prime*prime)//prime + 1)
    return [prime for prime in range(2, limit + 1) if sieve[prime]]


def in_c_double_prime(k, primes):
    """Direct divisibility check, without merged factorizations.

    An odd prime cannot divide two of k-4,k,k+4, whose differences are
    4 or 8. Hence a relevant square or cube must divide one whole factor
    and is at most |k|+4. Prime 2 is absent from the square condition and
    is allowed by the cube condition. Zero products are handled directly.
    """
    d = k*k - 16
    product = k*d
    if not d or not product or k % 8 == 4:
        return False
    bound = abs(k) + 4
    for prime in primes:
        square = prime * prime
        if square > bound:
            break
        if prime % 4 == 3 and d % square == 0:
            return False
        cube = square * prime
        if prime % 3 != 2 and cube <= bound and product % cube == 0:
            return False
    return True


def percent(count, denominator):
    return float(Fraction(100 * count, denominator))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows):
    assert [row['run_id'] for row in rows] == list(RUNS)
    assert len({row['examples'] for row in rows}) == 1
    result = {'paper_role': rows[0]['paper_role'], 'examples_per_run': rows[0]['examples'],
              'true_zero_examples': rows[0]['true_zero_examples'],
              'true_one_examples': rows[0]['true_one_examples'],
              'constant_zero_accuracy_percent': rows[0]['constant_zero_accuracy_percent'],
              'constant_one_accuracy_percent': rows[0]['constant_one_accuracy_percent'], 'runs': 3}
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
    maximum = max(abs(row['k']) for rows in arithmetic['by_file_label'].values() for row in rows)
    primes = primes_through(math.isqrt(maximum + 4))
    hashes = {'evaluation_arithmetic_inputs.json': sha256(index_path)}
    memberships, references, pools = [], {}, {}
    for label, role in ROLES.items():
        references[label] = {}
        groups = {name: Counter() for name in ('global', 'C_double_prime', 'special', 'complement')}
        overlap = 0
        for example in arithmetic['by_file_label'][label]:
            status, stored_r = rational(example['stored_answer'])
            assert status in ('integer', 'noninteger') and stored_r
            target = valuation3(stored_r.denominator) - valuation3(stored_r.numerator)
            k = example['k']
            c_double = in_c_double_prime(k, primes)
            special = k % 27 in SPECIAL
            complement = not (c_double or special)
            eligible = target in (0, 1)
            overlap += c_double and special
            assert int(c_double) + int(special) + int(complement) == 1
            record = {'file_label': label, 'paper_role': role, 'example_id': example['example_id'],
                      'k': k, 'residue_mod_27': k % 27, 'true_v3_n': target,
                      'true_zero_or_one': eligible, 'in_C_double_prime': c_double,
                      'in_special': special, 'in_union': c_double or special,
                      'in_complement': complement, 'in_scored_complement': complement and eligible}
            memberships.append(record)
            references[label][example['example_id']] = record
            for name, included in (('global', True), ('C_double_prime', c_double),
                                   ('special', special), ('complement', complement)):
                if included:
                    groups[name]['all_examples'] += 1
                    groups[name]['true_zero_examples'] += target == 0
                    groups[name]['true_one_examples'] += target == 1
                    groups[name]['eligible_true_zero_or_one'] += eligible
        assert overlap == 0
        for field in ('all_examples', 'true_zero_examples', 'true_one_examples', 'eligible_true_zero_or_one'):
            assert sum(groups[name][field] for name in ('C_double_prime', 'special', 'complement')) == groups['global'][field]
        n = groups['complement']['eligible_true_zero_or_one']
        pools[role] = {'groups': {name: dict(values) for name, values in groups.items()},
                       'C_double_prime_special_overlap': overlap,
                       'constant_zero_accuracy_percent': percent(groups['complement']['true_zero_examples'], n),
                       'constant_one_accuracy_percent': percent(groups['complement']['true_one_examples'], n)}
    write_csv(args.out / 'complement_membership.csv', memberships)
    print(json.dumps({'pools': pools}, separators=(',', ':')), flush=True)

    rows, selected_rows, invalid_examples = [], [], []
    ordered = {}
    target_checks = 0
    prediction_checks = 0
    for run in RUNS:
        for label, role in ROLES.items():
            path = args.inputs / run / f'eval.{label}.epochs_0_499.json.bz2'
            hashes[f'{run}/{path.name}'] = sha256(path)
            with bz2.open(path, 'rt', encoding='utf-8') as handle:
                data = json.load(handle, object_pairs_hook=unique_object)
            assert len(data) == 3000 and list(data) == list(references[label])
            epochs = list(range(1, 501)) if role == 'validation' else [SELECTED[run]]
            counters = [Counter() for _ in epochs]
            sequence = []
            for example_id, example in data.items():
                ref = references[label][example_id]
                k = example['problem']['k']
                assert type(k) is int and k == ref['k']
                status, raw_target = rational(example['answer'])
                assert status == 'integer' and -raw_target == ref['true_v3_n']
                target = ref['true_v3_n']
                target_checks += 1
                sequence.append((example_id, k, target))
                if not ref['in_scored_complement']:
                    continue
                guesses = example['guesses_greedy']
                assert isinstance(guesses, dict)
                for position, epoch in enumerate(epochs):
                    status, prediction = rational(guesses.get(str(epoch - 1), MISSING))
                    counter = counters[position]
                    counter['examples'] += 1
                    counter['parse_' + status] += 1
                    prediction_checks += 1
                    if prediction is not None:
                        correct = -prediction == target
                        counter['exact_correct'] += correct
                        counter['true_zero_correct'] += correct and target == 0
                        counter['true_one_correct'] += correct and target == 1
                    if status != 'integer':
                        invalid_examples.append({'run_id': run, 'paper_role': role,
                                                 'completed_epoch': epoch, 'example_id': example_id,
                                                 'k': k, 'status': status,
                                                 'raw_prediction': guesses.get(str(epoch - 1))})
            if label in ordered:
                assert sequence == ordered[label]
            else:
                ordered[label] = sequence
            composition = pools[role]['groups']['complement']
            for position, epoch in enumerate(epochs):
                counter = counters[position]
                n = counter['examples']
                assert n == composition['eligible_true_zero_or_one']
                assert counter['true_zero_correct'] + counter['true_one_correct'] == counter['exact_correct']
                row = {'run_id': run, 'file_label': label, 'paper_role': role,
                       'epoch_index': epoch - 1, 'completed_epoch': epoch, 'examples': n,
                       'true_zero_examples': composition['true_zero_examples'],
                       'true_one_examples': composition['true_one_examples'],
                       'exact_correct': counter['exact_correct'],
                       'true_zero_correct': counter['true_zero_correct'],
                       'true_one_correct': counter['true_one_correct'],
                       **{'parse_' + status: counter['parse_' + status] for status in BAD_STATUSES},
                       'invalid_total': sum(counter['parse_' + status] for status in BAD_STATUSES),
                       'exact_accuracy_percent': percent(counter['exact_correct'], n),
                       'constant_zero_accuracy_percent': pools[role]['constant_zero_accuracy_percent'],
                       'constant_one_accuracy_percent': pools[role]['constant_one_accuracy_percent']}
                assert sum(counter['parse_' + status] for status in (*BAD_STATUSES, 'integer')) == n
                rows.append(row)
                if epoch == SELECTED[run]:
                    selected_rows.append(row)
            del data
    assert target_checks == 18000 and len(rows) == 1503 and len(selected_rows) == 6
    val_ks = {row[1] for row in ordered['test1']}
    test_ks = {row[1] for row in ordered['valid']}
    assert len(val_ks) == len(test_ks) == 3000 and not val_ks.intersection(test_ks)
    by_key = {(row['run_id'], row['paper_role'], row['completed_epoch']): row for row in rows}
    curves = [{'completed_epoch': epoch, 'epoch_index': epoch - 1,
               **aggregate([by_key[run, 'validation', epoch] for run in RUNS])}
              for epoch in range(1, 501)]
    selected_aggregates = [aggregate([by_key[run, role, SELECTED[run]] for run in RUNS])
                           for role in ROLES.values()]
    fixed_epochs = [row for row in curves if row['completed_epoch'] in FIXED_EPOCHS]
    window_rows = []
    for first, last in FIXED_WINDOWS:
        for run in RUNS:
            window = [by_key[run, 'validation', epoch] for epoch in range(first, last + 1)]
            mean = 100 * sum(Fraction(row['exact_correct'], row['examples']) for row in window) / len(window)
            window_rows.append({'run_id': run, 'paper_role': 'validation',
                                'first_completed_epoch': first, 'last_completed_epoch': last,
                                'epoch_count': len(window), 'examples_per_epoch': window[0]['examples'],
                                'mean_exact_accuracy_percent': float(mean)})
    window_aggregates = []
    for first, last in FIXED_WINDOWS:
        means = [row['mean_exact_accuracy_percent'] for row in window_rows
                 if row['first_completed_epoch'] == first]
        window_aggregates.append({'paper_role': 'validation', 'first_completed_epoch': first,
                                  'last_completed_epoch': last, 'epoch_count': last - first + 1,
                                  **{f'run_{run.rsplit("_", 1)[1]}_window_mean_percent': value
                                     for run, value in zip(RUNS, means)},
                                  'mean_percent': statistics.mean(means),
                                  'sample_sd_of_run_window_means_percentage_points': statistics.stdev(means)})
    changes = [window_aggregates[1][f'run_{run.rsplit("_", 1)[1]}_window_mean_percent']
               - window_aggregates[0][f'run_{run.rsplit("_", 1)[1]}_window_mean_percent'] for run in RUNS]
    summary = {
        'policy': {'complement': 'NOT (C_double_prime OR k mod 27 in {0,5,22})',
                   'C_double_prime': 'No p^2|k^2-16 for p=3 mod4; k!=4 mod8; every p^3|k(k^2-16) has p=2 mod3',
                   'membership_method': 'Direct prime-square/cube divisibility of the polynomials; no partition-derived prediction counts',
                   'additional_k_ge_5_restriction': False,
                   'conditioning': 'True v3(n_k) in {0,1} within the complement',
                   'decoder': 'greedy', 'metric': 'Exact rational equality, with wrong or invalid outputs retained as failures',
                   'raw_to_paper': 'Negate raw decoded target and prediction',
                   'file_label_to_paper_role': ROLES,
                   'completed_epoch': 'JSON key + 1',
                   'existing_selected_completed_epochs': SELECTED,
                   'validation_scoring_completed_epochs': [1, 500],
                   'test_scoring': 'Only existing selected checkpoints',
                   'new_checkpoint_selection': False,
                   'fixed_validation_epochs': list(FIXED_EPOCHS),
                   'fixed_validation_windows_inclusive': [list(window) for window in FIXED_WINDOWS],
                   'sd': 'Sample SD across three run percentages or three run-window means, divisor 2',
                   'window_interpretation': 'Descriptive fixed windows; epochs are not independent samples'},
        'source_sha256': hashes, 'pools': pools,
        'checks': {'membership_records': len(memberships), 'raw_target_occurrences_checked': target_checks,
                   'C_double_prime_and_special_disjoint_in_both_pools': True,
                   'global_equals_three_way_partition_for_all_and_true_zero_and_true_one': True,
                   'ordered_ids_parameters_and_targets_match_across_runs': True,
                   'evaluation_pools_disjoint': True, 'scored_rows': len(rows),
                   'validation_curve_rows': len(curves), 'selected_rows': len(selected_rows),
                   'raw_prediction_records_scored': prediction_checks},
        'selected_aggregates': selected_aggregates,
        'fixed_validation_epoch_aggregates': fixed_epochs,
        'fixed_validation_window_aggregates': window_aggregates,
        'late_minus_earlier_window_change': {
            'per_run_percentage_points': dict(zip(RUNS, changes)),
            'mean_change_percentage_points': statistics.mean(changes),
            'sample_sd_of_run_changes_percentage_points': statistics.stdev(changes)},
        'invalid_scored_prediction_records': invalid_examples,
    }
    outputs = {'complement_per_epoch_metrics.csv': rows,
               'complement_validation_curves.csv': curves,
               'complement_selected_metrics.csv': selected_rows,
               'complement_selected_aggregates.csv': selected_aggregates,
               'fixed_validation_epoch_aggregates.csv': fixed_epochs,
               'fixed_validation_windows_per_run.csv': window_rows,
               'fixed_validation_window_aggregates.csv': window_aggregates}
    for name, data in outputs.items():
        write_csv(args.out / name, data)
    summary['output_csv_sha256'] = {name: sha256(args.out / name)
                                    for name in ('complement_membership.csv', *outputs)}
    (args.out / 'complement_audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'checks': summary['checks'], 'selected_aggregates': selected_aggregates,
                      'fixed_epoch_results': [{k: row[k] for k in ('completed_epoch', 'mean_percent', 'sample_sd_percentage_points')}
                                              for row in fixed_epochs],
                      'windows': window_aggregates,
                      'late_minus_earlier_change': summary['late_minus_earlier_window_change']}, indent=2))


if __name__ == '__main__':
    main()
