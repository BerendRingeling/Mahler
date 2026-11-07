#!/usr/bin/env python3
"""Compare Figure 5.7's full tables and the previously verified A results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


RUNS = ('1000260_9', '1000260_10', '1000260_11')
SETS = ('A', "A'", "A''", 'B', "B'", "B''", 'C', "C'", "C''")
PRIOR_HASHES = {
    'a_membership.csv': '759d786123a6ceaaf1e53393d3c3404ff48212f6bd3626325f9f870faa82bb0f',
    'weighted_partition_checks.csv': 'bcb24511fe00f1dcda60d70523c97228101a1128906637dc64e86c81c6995b95',
}
OVERALL_SHA256 = '63795dbdff88aaf5001ce2a2366bffa984e69dd2e3c3ac83e82912013f34967b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-dir', type=Path, required=True)
    parser.add_argument('--independent-dir', type=Path, required=True)
    parser.add_argument('--previous-a-dir', type=Path, required=True)
    parser.add_argument('--previous-a-selected', type=Path, required=True)
    parser.add_argument('--overall-metrics', type=Path, required=True)
    args = parser.parse_args()
    hashes = {}

    def read(path, as_csv=False):
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        return (list(csv.DictReader(io.StringIO(content.decode(), newline='')))
                if as_csv else json.loads(content))

    main_names = ('sets_membership.csv', 'sets_per_epoch_metrics.csv', 'sets_validation_curves.csv',
                  'sets_selected_metrics.csv', 'sets_selected_aggregates.csv')
    own_names = ('nine_set_membership.csv', 'nine_set_per_epoch_metrics.csv', 'nine_set_validation_curves.csv',
                 'nine_set_selected_metrics.csv', 'nine_set_selected_aggregates.csv')
    main_files = {name: read(args.main_dir / name, True) for name in main_names}
    own_files = {name: read(args.independent_dir / name, True) for name in own_names}
    actual_summary = read(args.main_dir / 'sets_summary.json')
    own_summary = read(args.independent_dir / 'nine_set_audit_summary.json')
    prior_membership = read(args.previous_a_dir / 'a_membership.csv', True)
    prior_weighted = read(args.previous_a_dir / 'weighted_partition_checks.csv', True)
    prior_selected = read(args.previous_a_selected, True)
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

    def basic_key(row):
        return row['run_id'], row['paper_role'], int(row['completed_epoch'])

    def row_key(row, own=False):
        return (*basic_key(row), row['set_name' if own else 'set'])

    member_key = lambda row: (row['file_label'], row['example_id'])
    actual_members = indexed(main_files['sets_membership.csv'], member_key, 'main_membership')
    own_members = indexed(own_files['nine_set_membership.csv'], member_key, 'own_membership')
    structure(set(actual_members), set(own_members), 'membership_keys')
    structure(len(actual_members), 6000, 'membership_examples')
    for key, own in own_members.items():
        actual = actual_members[key]
        for name in ('file_label', 'paper_role', 'example_id'):
            field(actual[name], own[name], f'membership.{key}.{name}')
        for name in ('k', 'true_v3_n'):
            field(actual[name], int(own[name]), f'membership.{key}.{name}')
        field(actual['true_zero_or_one'], own['eligible_true_0_or_1'], f'membership.{key}.eligible')
        for name in SETS:
            field(actual[name], own['in_' + name], f'membership.{key}.{name}')

    own_epochs = indexed(own_files['nine_set_per_epoch_metrics.csv'], lambda r: row_key(r, True), 'own_epochs')
    actual_epochs = indexed(main_files['sets_per_epoch_metrics.csv'], row_key, 'main_epochs')
    own_selected = indexed(own_files['nine_set_selected_metrics.csv'], lambda r: row_key(r, True), 'own_selected')
    actual_selected = indexed(main_files['sets_selected_metrics.csv'], row_key, 'main_selected')
    structure(set(actual_epochs), set(own_epochs), 'epoch_keys')
    structure(set(actual_selected), set(own_selected), 'selected_keys')
    structure(len(actual_epochs), 13527, 'epoch_row_count')
    structure(len(actual_selected), 54, 'selected_row_count')

    def check_row(actual, own, prefix):
        for name in ('run_id', 'file_label', 'paper_role'):
            field(actual[name], own[name], prefix + '.' + name)
        for name in ('completed_epoch', 'epoch_index', 'examples'):
            field(actual[name], int(own[name]), prefix + '.' + name)
        field(actual['set'], own['set_name'], prefix + '.set')
        field(actual['correct'], int(own['exact_correct']), prefix + '.correct')
        invalid = sum(int(own['parse_' + status])
                      for status in ('missing', 'null', 'malformed', 'zero_denominator'))
        field(actual['invalid'], invalid, prefix + '.invalid')
        field(actual['noninteger'], int(own['parse_noninteger']), prefix + '.noninteger')
        field(actual['exact_percent'], own['exact_accuracy_percent'], prefix + '.percent', True)

    for key, own in own_epochs.items():
        check_row(actual_epochs[key], own, f'epoch.{key}')
    for key, own in own_selected.items():
        check_row(actual_selected[key], own, f'selected.{key}')

    actual_curve_key = lambda row: (row['paper_role'], int(row['completed_epoch']), row['set'])
    own_curve_key = lambda row: (row['paper_role'], int(row['completed_epoch']), row['set_name'])
    actual_curves = indexed(main_files['sets_validation_curves.csv'], actual_curve_key, 'main_curves')
    own_curves = indexed(own_files['nine_set_validation_curves.csv'], own_curve_key, 'own_curves')
    structure(set(actual_curves), set(own_curves), 'curve_keys')
    structure(len(actual_curves), 4500, 'curve_row_count')
    for key, own in own_curves.items():
        actual = actual_curves[key]
        field(actual['paper_role'], own['paper_role'], f'curve.{key}.role')
        field(actual['completed_epoch'], int(own['completed_epoch']), f'curve.{key}.epoch')
        field(actual['set'], own['set_name'], f'curve.{key}.set')
        field(actual['examples_per_run'], int(own['examples_per_run']), f'curve.{key}.examples')
        field(actual['mean_percent'], own['mean_percent'], f'curve.{key}.mean', True)
        field(actual['sample_sd_percentage_points'], own['sample_sd_percentage_points'], f'curve.{key}.sd', True)
        for run in RUNS:
            suffix = run.rsplit('_', 1)[1]
            field(actual[run + '_correct'], int(own[f'run_{suffix}_exact_correct']), f'curve.{key}.{run}.correct')

    actual_aggregates = indexed(main_files['sets_selected_aggregates.csv'],
                                 lambda row: (row['paper_role'], row['set']), 'main_aggregates')
    own_aggregates = indexed(own_files['nine_set_selected_aggregates.csv'],
                              lambda row: (row['paper_role'], row['set_name']), 'own_aggregates')
    structure(set(actual_aggregates), set(own_aggregates), 'aggregate_keys')
    for key, own in own_aggregates.items():
        actual = actual_aggregates[key]
        field(actual['paper_role'], own['paper_role'], f'aggregate.{key}.role')
        field(actual['set'], own['set_name'], f'aggregate.{key}.set')
        field(actual['examples'], int(own['examples_per_run']), f'aggregate.{key}.examples')
        field(actual['mean_percent'], own['mean_percent'], f'aggregate.{key}.mean', True)
        field(actual['sample_sd_percentage_points'], own['sample_sd_percentage_points'], f'aggregate.{key}.sd', True)
        for run in RUNS:
            suffix = run.rsplit('_', 1)[1]
            field(actual[run + '_correct'], int(own[f'run_{suffix}_exact_correct']), f'aggregate.{key}.{run}.correct')

    # Compare A against the preserved independent Figure 5.6 data.
    for name, digest in PRIOR_HASHES.items():
        field(hashes[str(args.previous_a_dir / name)], digest, 'prior_A_hash.' + name)
    previous_members = indexed(prior_membership, member_key, 'previous_A_membership')
    structure(set(previous_members), set(own_members), 'previous_A_membership_keys')
    for key, previous in previous_members.items():
        own = own_members[key]
        field(own['in_A'], previous['in_A'], f'previous_A.{key}.membership')
        field(own['k'], int(previous['k']), f'previous_A.{key}.k')
        field(own['true_v3_n'], int(previous['true_v3_n']), f'previous_A.{key}.target')
    previous_weighted = indexed(prior_weighted, basic_key, 'previous_A_weighted')
    current_A = {key[:3]: row for key, row in own_epochs.items() if key[-1] == 'A'}
    structure(set(previous_weighted), set(current_A), 'previous_A_epoch_keys')
    for key, previous in previous_weighted.items():
        own = current_A[key]
        field(own['examples'], int(previous['true_0_or_1_examples']), f'previous_A.{key}.examples')
        field(own['exact_correct'], int(previous['true_0_or_1_exact_correct']), f'previous_A.{key}.correct')
        field(own['exact_accuracy_percent'], previous['weighted_true_0_or_1_accuracy_percent'],
              f'previous_A.{key}.percent', True)
    previous_selected_A = [row for row in prior_selected if row['pooled_group'] == 'A_true_0_or_1']
    structure(len(previous_selected_A), 6, 'previous_A_selected_row_count')
    for previous in previous_selected_A:
        key = basic_key(previous)
        own = current_A[key]
        field(own['examples'], int(previous['examples']), f'previous_A_selected.{key}.examples')
        field(own['exact_correct'], int(previous['correct']), f'previous_A_selected.{key}.correct')
        field(own['exact_accuracy_percent'], previous['exact_percent'], f'previous_A_selected.{key}.percent', True)
    field(hashes[str(args.overall_metrics)], OVERALL_SHA256, 'overall_reference_hash')
    overall = indexed(overall_rows, basic_key, 'overall_reference')
    for previous in previous_selected_A:
        key = basic_key(previous)
        reference = overall[key]
        field(previous['overall_examples'], int(reference['examples']), f'previous_A_selected.{key}.overall_examples')
        field(previous['overall_correct'], int(reference['greedy_exact_count']), f'previous_A_selected.{key}.overall_correct')
        field(previous['overall_exact_percent'], reference['greedy_exact_percent'], f'previous_A_selected.{key}.overall_percent', True)

    sequence(actual_summary['sets_in_table_order'], list(SETS), 'summary.set_order')
    sequence(actual_summary['runs'], list(RUNS), 'summary.runs')
    field(actual_summary['table_pool'], 'test', 'summary.table_pool')
    field(actual_summary['curve_pool'], 'validation', 'summary.curve_pool')
    sequence(actual_summary['curve_completed_epochs'], [1, 500], 'summary.curve_epochs')
    field(actual_summary['smoothing'], 'none', 'summary.smoothing')
    for label, role in own_summary['policy']['file_label_to_paper_role'].items():
        field(actual_summary['file_label_to_paper_role'][label], role, 'summary.role.' + label)
    for run, epoch in own_summary['policy']['existing_global_selected_completed_epochs'].items():
        field(actual_summary['selected_completed_epochs'][run], epoch, 'summary.selection.' + run)
    for role, sets in own_summary['pool_set_counts'].items():
        actual_pool = actual_summary['pools'][role]
        records = [row for row in own_members.values() if row['paper_role'] == role]
        field(actual_pool['examples'], len(records), f'summary.pool.{role}.examples')
        field(actual_pool['minimum_k'], min(int(row['k']) for row in records), f'summary.pool.{role}.minimum_k')
        for name, counts in sets.items():
            actual = actual_pool['sets'][name]
            for main_name, own_name in (('all_examples', 'all_members'),
                                        ('true_zero_or_one_examples', 'eligible_true_0_or_1'),
                                        ('true_zero', 'true_zero_examples'), ('true_one', 'true_one_examples')):
                field(actual[main_name], counts[own_name], f'summary.pool.{role}.{name}.{main_name}')
    for own in own_summary['selected_aggregates']:
        role, name = own['paper_role'], own['set_name']
        actual = actual_summary['selected_accuracies'][role][name]
        prefix = f'summary.selected.{role}.{name}'
        field(actual['examples_per_run'], own['examples_per_run'], prefix + '.examples')
        counts = [own[f'run_{run.rsplit("_", 1)[1]}_exact_correct'] for run in RUNS]
        percentages = [own[f'run_{run.rsplit("_", 1)[1]}_accuracy_percent'] for run in RUNS]
        sequence(actual['correct_counts'], counts, prefix + '.counts')
        sequence(actual['per_run_percent'], percentages, prefix + '.percentages', True)
        field(actual['mean_percent'], own['mean_percent'], prefix + '.mean', True)
        field(actual['sample_sd_percentage_points'], own['sample_sd_percentage_points'], prefix + '.sd', True)
    for role, ranks in own_summary['selected_accuracy_rankings'].items():
        sequence(actual_summary['comparisons'][role]['descending_order_by_selected_mean'],
                 ranks['ordered_by_selected_mean_accuracy'], f'summary.ranking.{role}')
    for change in own_summary['selected_restriction_changes']:
        role, base = change['paper_role'], change['base_set']
        field(actual_summary['comparisons'][role][base + '_to_' + base + 'prime_gain_percentage_points'],
              change['mean_change_percentage_points'], f'summary.change.{role}.{base}', True)
    field(actual_summary['raw_target_occurrences_checked_against_r_k'],
          own_summary['checks']['raw_target_occurrences_checked'], 'summary.target_checks')
    field(actual_summary['validation_test_k_overlap'], 0, 'summary.pool_overlap')
    for name, digest in own_summary['source_sha256'].items():
        field(actual_summary['source_sha256'][name], digest, 'summary.source_hash.' + name)
    field(actual_summary['prior_A_csv_sha256'], hashes[str(args.previous_a_selected)], 'summary.prior_A_csv_hash')
    field(actual_summary['prior_overall_csv_sha256'], OVERALL_SHA256, 'summary.prior_overall_csv_hash')

    result = {
        'all_comparisons_pass': not failures, **counters,
        'absolute_float_tolerance': 1e-10,
        'membership_examples_compared': 6000, 'membership_flags_compared': 54000,
        'per_run_epoch_set_rows_compared': 13527, 'validation_curve_rows_compared': 4500,
        'selected_rows_compared': 54, 'selected_aggregate_rows_compared': 18,
        'previous_A_membership_records_compared': 6000,
        'previous_A_epoch_results_compared': 1503,
        'previous_A_selected_pooled_results_compared': 6,
        'previous_A_source_hashes_preserved': True,
        'raw_archives_reread_by_comparison': False,
        'unchecked_fields': ['Prime-factorization representations',
                             'Full feature-dictionary equality metadata',
                             'Plot appearance and prose policy wording'],
        'source_sha256': hashes, 'failure_count': len(failures), 'failures': failures,
    }
    (args.independent_dir / 'nine_sets_main_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
