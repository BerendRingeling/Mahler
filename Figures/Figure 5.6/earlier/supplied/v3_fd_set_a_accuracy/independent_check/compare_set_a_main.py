#!/usr/bin/env python3
"""Compare all requested Figure 5.6 results with the independent raw audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


RUNS = ('1000260_9', '1000260_10', '1000260_11')
GROUPS = ('0', '1', '>1')
POOLED_NAMES = {'all_A': 'all_A', 'A_true_0_or_1': '0_or_1'}
OVERALL_SHA256 = '63795dbdff88aaf5001ce2a2366bffa984e69dd2e3c3ac83e82912013f34967b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-dir', type=Path, required=True)
    parser.add_argument('--independent-dir', type=Path, required=True)
    parser.add_argument('--overall-metrics', type=Path, required=True)
    args = parser.parse_args()
    hashes = {}

    def read(path, as_csv=False):
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        return (list(csv.DictReader(io.StringIO(content.decode(), newline='')))
                if as_csv else json.loads(content))

    main_files = {name: read(args.main_dir / name, True) for name in (
        'set_a_membership.csv', 'set_a_per_epoch_metrics.csv', 'set_a_validation_curves.csv',
        'set_a_selected_group_metrics.csv', 'set_a_selected_pooled_metrics.csv')}
    own_files = {name: read(args.independent_dir / name, True) for name in (
        'a_membership.csv', 'validation_group_per_epoch.csv', 'validation_curve_mean_sd.csv',
        'selected_group_and_pooled_metrics.csv', 'selected_group_and_pooled_aggregates.csv')}
    main_summary = read(args.main_dir / 'set_a_summary.json')
    own_summary = read(args.independent_dir / 'set_a_audit_summary.json')
    overall_rows = read(args.overall_metrics, True)
    counters = {'scalar_field_comparisons': 0, 'exact_comparisons': 0,
                'floating_comparisons': 0, 'structural_checks': 0}
    failures = []

    def structure(actual, expected, label):
        counters['structural_checks'] += 1
        if actual != expected:
            failures.append({'field': label, 'actual': str(actual), 'expected': str(expected)})

    def field(actual, expected, label, floating=False):
        counters['scalar_field_comparisons'] += 1
        if floating:
            counters['floating_comparisons'] += 1
            actual, expected = float(actual), float(expected)
            good = math.isfinite(actual) and math.isfinite(expected)
            good = good and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10)
        else:
            counters['exact_comparisons'] += 1
            if type(expected) is int:
                actual = int(actual)
            good = actual == expected
        if not good:
            failures.append({'field': label, 'actual': actual, 'expected': expected})

    def sequence(actual, expected, label, floating=False):
        structure(len(actual), len(expected), label + '.length')
        for index, (a, e) in enumerate(zip(actual, expected)):
            field(a, e, f'{label}.{index}', floating)

    def indexed(rows, function, label):
        result = {function(row): row for row in rows}
        structure(len(result), len(rows), label + '.unique')
        return result

    def row_key(row):
        return row['run_id'], row['paper_role'], int(row['completed_epoch']), row['true_group']

    def basic_key(row):
        return row['run_id'], row['paper_role'], int(row['completed_epoch'])

    membership_key = lambda row: (row['file_label'], row['example_id'])
    main_membership = indexed(main_files['set_a_membership.csv'], membership_key, 'main_membership')
    own_membership = indexed(own_files['a_membership.csv'], membership_key, 'own_membership')
    structure(set(main_membership), set(own_membership), 'membership_keys')
    structure(len(main_membership), 6000, 'membership_count')
    for key, own in own_membership.items():
        actual = main_membership[key]
        for name in ('file_label', 'paper_role', 'example_id', 'in_A'):
            field(actual[name], own[name], f'membership.{key}.{name}')
        for name in ('k', 'true_v3_n'):
            field(actual[name], int(own[name]), f'membership.{key}.{name}')

    own_selected_rows = own_files['selected_group_and_pooled_metrics.csv']
    own_selected = indexed(own_selected_rows, row_key, 'own_selected')
    own_epoch_rows = own_files['validation_group_per_epoch.csv'] + [
        row for row in own_selected_rows if row['paper_role'] == 'test' and row['true_group'] in GROUPS]
    own_epochs = indexed(own_epoch_rows, row_key, 'own_epochs')
    main_epochs = indexed(main_files['set_a_per_epoch_metrics.csv'], row_key, 'main_epochs')
    main_selected = indexed(main_files['set_a_selected_group_metrics.csv'], row_key, 'main_selected')
    structure(set(main_epochs), set(own_epochs), 'all_epoch_keys')
    structure(len(main_epochs), 4509, 'all_epoch_count')
    expected_selected_keys = {key for key in own_selected if key[-1] in GROUPS}
    structure(set(main_selected), expected_selected_keys, 'selected_group_keys')

    def check_row(actual, own, prefix):
        for name in ('run_id', 'file_label', 'paper_role', 'true_group'):
            field(actual[name], own[name], prefix + '.' + name)
        for name in ('completed_epoch', 'epoch_index', 'examples'):
            field(actual[name], int(own[name]), prefix + '.' + name)
        field(actual['correct'], int(own['exact_correct']), prefix + '.correct')
        invalid = sum(int(own['parse_' + status])
                      for status in ('missing', 'null', 'malformed', 'zero_denominator'))
        field(actual['invalid'], invalid, prefix + '.invalid')
        field(actual['noninteger'], int(own['parse_noninteger']), prefix + '.noninteger')
        field(actual['exact_percent'], own['exact_accuracy_percent'], prefix + '.exact_percent', True)

    for key, own in own_epochs.items():
        check_row(main_epochs[key], own, f'epoch.{key}')
    for key, actual in main_selected.items():
        check_row(actual, own_selected[key], f'selected_group.{key}')

    curve_key = lambda row: (row['paper_role'], int(row['completed_epoch']), row['true_group'])
    main_curves = indexed(main_files['set_a_validation_curves.csv'], curve_key, 'main_curves')
    own_curves = indexed(own_files['validation_curve_mean_sd.csv'], curve_key, 'own_curves')
    structure(set(main_curves), set(own_curves), 'curve_keys')
    structure(len(main_curves), 1500, 'curve_row_count')
    for key, own in own_curves.items():
        actual = main_curves[key]
        for name in ('paper_role', 'true_group'):
            field(actual[name], own[name], f'curve.{key}.{name}')
        for name in ('completed_epoch', 'examples_per_run'):
            field(actual[name], int(own[name]), f'curve.{key}.{name}')
        mean, sd = float(own['mean_percent']), float(own['sample_sd_percentage_points'])
        field(actual['mean_percent'], mean, f'curve.{key}.mean', True)
        field(actual['sample_sd_percentage_points'], sd, f'curve.{key}.sd', True)
        field(actual['display_band_low_percent'], max(0, mean - sd), f'curve.{key}.band_low', True)
        field(actual['display_band_high_percent'], min(100, mean + sd), f'curve.{key}.band_high', True)
        for run in RUNS:
            suffix = run.rsplit('_', 1)[1]
            field(actual[run + '_correct'], int(own[f'run_{suffix}_exact_correct']), f'curve.{key}.{run}.correct')

    field(hashes[str(args.overall_metrics)], OVERALL_SHA256, 'previously_verified_overall_csv_hash')
    overall = indexed(overall_rows, basic_key, 'overall')
    pooled_key = lambda row: (*basic_key(row), POOLED_NAMES[row['pooled_group']])
    main_pooled = indexed(main_files['set_a_selected_pooled_metrics.csv'], pooled_key, 'main_pooled')
    expected_pooled = {key for key in own_selected if key[-1] in POOLED_NAMES.values()}
    structure(set(main_pooled), expected_pooled, 'pooled_selected_keys')
    for key, actual in main_pooled.items():
        own = own_selected[key]
        for name in ('run_id', 'paper_role'):
            field(actual[name], own[name], f'pooled.{key}.{name}')
        field(actual['completed_epoch'], int(own['completed_epoch']), f'pooled.{key}.epoch')
        field(POOLED_NAMES[actual['pooled_group']], own['true_group'], f'pooled.{key}.group')
        field(actual['examples'], int(own['examples']), f'pooled.{key}.examples')
        field(actual['correct'], int(own['exact_correct']), f'pooled.{key}.correct')
        field(actual['exact_percent'], own['exact_accuracy_percent'], f'pooled.{key}.percent', True)
        reference = overall[key[:3]]
        field(actual['overall_examples'], int(reference['examples']), f'pooled.{key}.overall_examples')
        field(actual['overall_correct'], int(reference['greedy_exact_count']), f'pooled.{key}.overall_correct')
        field(actual['overall_exact_percent'], reference['greedy_exact_percent'], f'pooled.{key}.overall_percent', True)

    sequence(main_summary['runs'], list(RUNS), 'summary.runs')
    sequence(main_summary['groups'], list(GROUPS), 'summary.groups')
    field(main_summary['higher_group_requires_exact_value'], True, 'summary.higher_group_exact')
    field(main_summary['invalid_outputs_remain_failures_in_denominator'], True, 'summary.invalid_failures')
    field(main_summary['plotted_pool'], 'validation', 'summary.plotted_pool')
    sequence(main_summary['plotted_completed_epochs'], [1, 500], 'summary.plotted_epochs')
    field(main_summary['sample_standard_deviation_divisor'], 2, 'summary.sd_divisor')
    field(main_summary['smoothing'], 'none', 'summary.smoothing')
    for label, role in own_summary['policy']['file_label_to_paper_role'].items():
        field(main_summary['file_label_to_paper_role'][label], role, 'summary.role.' + label)
    for run, epoch in own_summary['policy']['existing_global_selected_completed_epochs'].items():
        field(main_summary['selected_completed_epochs'][run], epoch, 'summary.selection.' + run)
    for role in ('validation', 'test'):
        actual, own = main_summary['pools'][role], own_summary['pools'][role]
        field(actual['all_examples'], own['pool_examples'], f'summary.pool.{role}.all_examples')
        field(actual['A_examples'], own['A_examples'], f'summary.pool.{role}.A_examples')
        field(actual['A_zero_or_one_examples'], own['group_examples']['0_or_1'], f'summary.pool.{role}.zero_one_examples')
        for group in GROUPS:
            field(actual['A_group_sizes'][group], own['group_examples'][group], f'summary.pool.{role}.{group}.examples')
        structure(actual['A_true_valuation_counts'], own['A_target_histogram'], f'summary.pool.{role}.target_histogram')
        records = [row for row in own_membership.values() if row['paper_role'] == role]
        field(actual['minimum_k'], min(int(row['k']) for row in records), f'summary.pool.{role}.minimum_k')
        has_one = any(int(row['k']) == 1 and row['in_A'] == 'True' for row in records)
        field(actual['A_contains_k_1'], has_one, f'summary.pool.{role}.contains_one')
    for own in own_summary['selected_aggregates']:
        role, group = own['paper_role'], own['true_group']
        if group in GROUPS:
            actual = main_summary['selected_group_accuracies'][role][group]
        else:
            main_name = next(name for name, translated in POOLED_NAMES.items() if translated == group)
            actual = main_summary['selected_pooled_accuracies'][role][main_name]
        prefix = f'summary.selected.{role}.{group}'
        field(actual['examples_per_run'], own['examples_per_run'], prefix + '.examples')
        counts = [own[f'run_{run.rsplit("_", 1)[1]}_exact_correct'] for run in RUNS]
        percentages = [own[f'run_{run.rsplit("_", 1)[1]}_accuracy_percent'] for run in RUNS]
        sequence(actual['correct_counts'], counts, prefix + '.counts')
        sequence(actual['per_run_percent'], percentages, prefix + '.percentages', True)
        field(actual['mean_percent'], own['mean_percent'], prefix + '.mean', True)
        field(actual['sample_sd_percentage_points'], own['sample_sd_percentage_points'], prefix + '.sd', True)
    field(main_summary['raw_targets_checked_against_stored_rational_r_k'],
          own_summary['checks']['raw_target_occurrences_checked'], 'summary.raw_target_checks')
    field(main_summary['A_membership_records_seen_across_runs'],
          3 * sum(pool['A_examples'] for pool in own_summary['pools'].values()), 'summary.A_records_across_runs')
    field(main_summary['validation_test_k_overlap'], 0, 'summary.pool_overlap')
    for name, digest in own_summary['source_sha256'].items():
        field(main_summary['source_sha256'][name], digest, 'summary.source_hash.' + name)

    result = {
        'all_comparisons_pass': not failures, **counters,
        'absolute_float_tolerance': 1e-10,
        'membership_records_compared': 6000,
        'scored_group_rows_compared': 4509,
        'validation_curve_rows_compared': 1500,
        'selected_group_rows_compared': 18,
        'selected_pooled_rows_compared': 12,
        'selected_aggregate_groups_compared': 10,
        'all_membership_flags_verified_using_independent_polynomial_square_division': True,
        'display_band_clipping_checked_separately': True,
        'test_only_at_existing_selected_checkpoints': True,
        'raw_archives_reread_by_comparison': False,
        'unchecked_fields': ['Membership product_factorization representation',
                             'Full feature-dictionary equality metadata',
                             'Plot appearance and wording of prose policy descriptions'],
        'source_sha256': hashes,
        'failure_count': len(failures), 'failures': failures,
    }
    (args.independent_dir / 'set_a_main_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
