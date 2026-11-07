#!/usr/bin/env python3
"""Independently audit exact valuation accuracy on set A from raw exports.

A = {k: 3 does not divide k and k(k^2-16) is squarefree}.
There is no k>=5 condition. Membership uses direct prime-square division
of the polynomial, not factorization merging. See membership_evidence().
Only the existing selected checkpoints are scored for the test pool.
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
GROUPS = ('0', '1', '>1')
POOLED = ('0_or_1', 'all_A')
STATUSES = ('missing', 'null', 'malformed', 'zero_denominator', 'noninteger', 'integer')
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
    for p in range(2, math.isqrt(limit) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = b'\x00' * ((limit-p*p)//p + 1)
    return [p for p in range(2, limit + 1) if sieve[p]]


def membership_evidence(k, primes):
    """Test p^2 | k(k^2-16) directly, with a sufficient exact prime bound.

    Distinct factors k-4,k,k+4 differ by 4 or 8. An odd prime cannot divide
    two factors, so if its square divides their product it divides one
    individual factor twice; consequently p^2 <= |k|+4. The only possible
    cross-factor repeated prime is 2, and 2^2 <= |k|+4 for every integer k.
    Thus testing the listed squares up to |k|+4 is sufficient. Zero is
    explicitly not squarefree. Negative products use the usual absolute-
    value convention; divisibility itself is unchanged by the sign.
    """
    product = k * (k*k - 16)
    repeated_prime = None
    if product:
        for p in primes:
            if p*p > abs(k) + 4:
                break
            if product % (p*p) == 0:
                repeated_prime = p
                break
    squarefree = product != 0 and repeated_prime is None
    return {'k': k, 'polynomial_k_times_k_squared_minus_16': product,
            'three_does_not_divide_k': k % 3 != 0,
            'polynomial_is_squarefree': squarefree,
            'first_repeated_prime': repeated_prime,
            'in_A': k % 3 != 0 and squarefree}


def percent(count, examples):
    assert examples > 0
    return float(Fraction(100 * count, examples))


def score(counter, status, raw_prediction, target):
    counter['examples'] += 1
    counter['parse_' + status] += 1
    if raw_prediction is not None:
        prediction = -raw_prediction
        counter['exact_correct'] += prediction == target
        counter['negative_integer_prediction_count'] += status == 'integer' and prediction < 0


def metric_row(counter, run, label, epoch, group):
    row = {'run_id': run, 'file_label': label, 'paper_role': ROLES[label],
           'epoch_index': epoch - 1, 'completed_epoch': epoch,
           'true_group': group, 'examples': counter['examples'],
           'exact_correct': counter['exact_correct']}
    for status in STATUSES:
        row['parse_' + status] = counter['parse_' + status]
    row['invalid_total'] = sum(counter['parse_' + status] for status in STATUSES if status != 'integer')
    row['negative_integer_prediction_count'] = counter['negative_integer_prediction_count']
    row['exact_accuracy_percent'] = percent(counter['exact_correct'], counter['examples'])
    assert sum(row['parse_' + status] for status in STATUSES) == row['examples']
    return row


def aggregate(rows):
    assert [row['run_id'] for row in rows] == list(RUNS)
    assert len({row['examples'] for row in rows}) == 1
    result = {'paper_role': rows[0]['paper_role'], 'true_group': rows[0]['true_group'],
              'examples_per_run': rows[0]['examples'], 'runs': 3}
    for row in rows:
        suffix = row['run_id'].rsplit('_', 1)[1]
        result[f'run_{suffix}_completed_epoch'] = row['completed_epoch']
        result[f'run_{suffix}_exact_correct'] = row['exact_correct']
        result[f'run_{suffix}_accuracy_percent'] = row['exact_accuracy_percent']
    values = [row['exact_accuracy_percent'] for row in rows]
    result['mean_percent'] = statistics.mean(values)
    result['sample_sd_percentage_points'] = statistics.stdev(values)
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.inputs / 'evaluation_arithmetic_inputs.json'
    arithmetic = json.loads(index_path.read_text())
    assert arithmetic['file_label_to_paper_role'] == ROLES
    maximum_abs_k = max(abs(row['k']) for rows in arithmetic['by_file_label'].values() for row in rows)
    primes = primes_through(math.isqrt(maximum_abs_k + 4))
    source_hashes = {'evaluation_arithmetic_inputs.json': sha256(index_path)}
    reference = {}
    membership_rows = []
    pool_data = {}
    for label, role in ROLES.items():
        reference[label] = {}
        target_histogram = Counter()
        for row in arithmetic['by_file_label'][label]:
            status, stored_r = rational(row['stored_answer'])
            assert status in ('integer', 'noninteger') and stored_r
            target = valuation3(stored_r.denominator) - valuation3(stored_r.numerator)
            evidence = membership_evidence(row['k'], primes)
            record = {'file_label': label, 'paper_role': role, 'example_id': row['example_id'],
                      **evidence, 'true_v3_n': target}
            membership_rows.append(record)
            reference[label][row['example_id']] = record
            if evidence['in_A']:
                target_histogram[target] += 1
        counts = {'0': target_histogram[0], '1': target_histogram[1],
                  '>1': sum(n for v, n in target_histogram.items() if v > 1)}
        counts['0_or_1'] = counts['0'] + counts['1']
        counts['all_A'] = sum(target_histogram.values())
        pool_data[role] = {'pool_examples': len(reference[label]), 'A_examples': counts['all_A'],
                           'A_target_histogram': dict(sorted(target_histogram.items())),
                           'group_examples': counts,
                           'k_below_5_in_pool': [r['k'] for r in reference[label].values() if r['k'] < 5],
                           'k_below_5_in_A': [r['k'] for r in reference[label].values() if r['in_A'] and r['k'] < 5]}

    validation_rows, selected_rows, weighted_rows = [], [], []
    invalid_examples = []
    raw_targets_checked = 0
    ordered_reference = {}
    scored_prediction_count = 0
    for run in RUNS:
        for label, role in ROLES.items():
            path = args.inputs / run / f'eval.{label}.epochs_0_499.json.bz2'
            source_hashes[f'{run}/{path.name}'] = sha256(path)
            with bz2.open(path, 'rt', encoding='utf-8') as handle:
                data = json.load(handle, object_pairs_hook=unique_object)
            assert len(data) == 3000 and list(data) == list(reference[label])
            epochs = list(range(1, 501)) if role == 'validation' else [SELECTED[run]]
            counters = {epoch: {group: Counter() for group in GROUPS + POOLED} for epoch in epochs}
            sequence = []
            for example_id, example in data.items():
                ref = reference[label][example_id]
                k = example['problem']['k']
                assert type(k) is int and k == ref['k']
                status, raw_target = rational(example['answer'])
                assert status == 'integer' and -raw_target == ref['true_v3_n']
                target = ref['true_v3_n']
                assert target >= 0
                raw_targets_checked += 1
                sequence.append((example_id, k, target))
                if not ref['in_A']:
                    continue
                group = str(target) if target < 2 else '>1'
                guesses = example['guesses_greedy']
                assert isinstance(guesses, dict)
                for epoch in epochs:
                    status, raw_prediction = rational(guesses.get(str(epoch - 1), MISSING))
                    score(counters[epoch][group], status, raw_prediction, target)
                    score(counters[epoch]['all_A'], status, raw_prediction, target)
                    if target in (0, 1):
                        score(counters[epoch]['0_or_1'], status, raw_prediction, target)
                    scored_prediction_count += 1
                    if status != 'integer':
                        invalid_examples.append({'run_id': run, 'paper_role': role,
                                                 'completed_epoch': epoch, 'example_id': example_id,
                                                 'k': k, 'true_v3_n': target, 'status': status,
                                                 'raw_prediction': guesses.get(str(epoch - 1))})
            if label in ordered_reference:
                assert sequence == ordered_reference[label]
            else:
                ordered_reference[label] = sequence
            for epoch in epochs:
                all_metrics = {group: metric_row(counters[epoch][group], run, label, epoch, group)
                               for group in GROUPS + POOLED}
                for group, row in all_metrics.items():
                    assert row['examples'] == pool_data[role]['group_examples'][group]
                all_a, zero_one = all_metrics['all_A'], all_metrics['0_or_1']
                assert sum(all_metrics[g]['examples'] for g in GROUPS) == all_a['examples']
                assert sum(all_metrics[g]['exact_correct'] for g in GROUPS) == all_a['exact_correct']
                assert sum(all_metrics[g]['examples'] for g in ('0', '1')) == zero_one['examples']
                assert sum(all_metrics[g]['exact_correct'] for g in ('0', '1')) == zero_one['exact_correct']
                weighted_all = sum(Fraction(all_metrics[g]['examples'], all_a['examples'])
                                   * Fraction(all_metrics[g]['exact_correct'], all_metrics[g]['examples'])
                                   for g in GROUPS)
                weighted_zero_one = sum(Fraction(all_metrics[g]['examples'], zero_one['examples'])
                                        * Fraction(all_metrics[g]['exact_correct'], all_metrics[g]['examples'])
                                        for g in ('0', '1'))
                assert weighted_all == Fraction(all_a['exact_correct'], all_a['examples'])
                assert weighted_zero_one == Fraction(zero_one['exact_correct'], zero_one['examples'])
                weighted_rows.append({'run_id': run, 'paper_role': role, 'completed_epoch': epoch,
                                      'all_A_examples': all_a['examples'], 'all_A_exact_correct': all_a['exact_correct'],
                                      'true_0_or_1_examples': zero_one['examples'], 'true_0_or_1_exact_correct': zero_one['exact_correct'],
                                      'weighted_all_A_accuracy_percent': float(100 * weighted_all),
                                      'weighted_true_0_or_1_accuracy_percent': float(100 * weighted_zero_one),
                                      'exact_rational_checks_pass': True})
                if role == 'validation':
                    validation_rows.extend(all_metrics[g] for g in GROUPS)
                if epoch == SELECTED[run]:
                    selected_rows.extend(all_metrics[g] for g in GROUPS + POOLED)
            del data

    assert raw_targets_checked == 18000
    assert len(validation_rows) == 4500 and len(selected_rows) == 30
    assert len(weighted_rows) == 1503
    validation_ks = {row[1] for row in ordered_reference['test1']}
    test_ks = {row[1] for row in ordered_reference['valid']}
    assert len(validation_ks) == len(test_ks) == 3000 and not validation_ks.intersection(test_ks)
    curve_index = {(row['run_id'], row['completed_epoch'], row['true_group']): row for row in validation_rows}
    curves = []
    for epoch in range(1, 501):
        for group in GROUPS:
            curves.append({'completed_epoch': epoch, 'epoch_index': epoch - 1,
                           **aggregate([curve_index[run, epoch, group] for run in RUNS])})
    selected_index = {(row['run_id'], row['paper_role'], row['true_group']): row for row in selected_rows}
    selected_aggregates = []
    for role in ROLES.values():
        for group in GROUPS + POOLED:
            selected_aggregates.append(aggregate([selected_index[run, role, group] for run in RUNS]))
    fixed_curves = [row for row in curves if row['completed_epoch'] in (1, 50, 100, 200, 300, 400, 500)]
    summary = {
        'policy': {'set_A_definition': '3 does not divide k and k(k^2-16) is squarefree',
                   'additional_k_ge_5_restriction': False,
                   'membership_method': 'Direct prime-square divisibility of k(k^2-16), with sufficient bound p^2 <= |k|+4',
                   'maximum_prime_tested': primes[-1],
                   'squarefree_zero': False, 'squarefree_negative_integer': 'Squarefree absolute value',
                   'raw_target': 'v3(r_k), verified against the stored rational r_k',
                   'paper_target_and_prediction': 'Negative of decoded raw valuation',
                   'decoder': 'greedy', 'metric': 'Exact rational equality of predicted valuation and true valuation',
                   'higher_group_success': 'Exact valuation required; predicting another value >1 is not a success',
                   'file_label_to_paper_role': ROLES,
                   'completed_epoch': 'JSON key + 1',
                   'existing_global_selected_completed_epochs': SELECTED,
                   'subgroup_checkpoint_selection': False,
                   'validation_scoring_completed_epochs': [1, 500],
                   'test_scoring': 'Only each run\'s existing selected checkpoint',
                   'invalid_outputs': 'Failures; missing, null, malformed, zero-denominator, or noninteger valuation outputs',
                   'sd': 'Sample standard deviation across three run percentages, divisor 2'},
        'source_sha256': source_hashes,
        'pools': pool_data,
        'small_k_membership_checks': {str(k): membership_evidence(k, primes) for k in range(0, 6)},
        'checks': {'membership_records': len(membership_rows), 'raw_target_occurrences_checked': raw_targets_checked,
                   'all_three_runs_share_ordered_ids_parameters_and_targets': True,
                   'evaluation_pools_disjoint': True,
                   'validation_group_epoch_rows': len(validation_rows),
                   'validation_mean_sd_rows': len(curves),
                   'selected_group_and_pooled_rows': len(selected_rows),
                   'weighted_partition_rows': len(weighted_rows),
                   'exact_rational_weighted_checks': 2 * len(weighted_rows),
                   'all_weighted_checks_pass': True,
                   'raw_A_prediction_records_scored': scored_prediction_count},
        'selected_aggregates': selected_aggregates,
        'fixed_validation_curves': fixed_curves,
        'invalid_A_prediction_records': invalid_examples,
    }
    output_rows = {'a_membership.csv': membership_rows,
                   'validation_group_per_epoch.csv': validation_rows,
                   'validation_curve_mean_sd.csv': curves,
                   'selected_group_and_pooled_metrics.csv': selected_rows,
                   'selected_group_and_pooled_aggregates.csv': selected_aggregates,
                   'weighted_partition_checks.csv': weighted_rows}
    for name, rows in output_rows.items():
        write_csv(args.out / name, rows)
    summary['output_csv_sha256'] = {name: sha256(args.out / name) for name in output_rows}
    (args.out / 'set_a_audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'pools': pool_data, 'checks': summary['checks'],
                      'selected_test_results': [row for row in selected_aggregates if row['paper_role'] == 'test']}, indent=2))


if __name__ == '__main__':
    main()
