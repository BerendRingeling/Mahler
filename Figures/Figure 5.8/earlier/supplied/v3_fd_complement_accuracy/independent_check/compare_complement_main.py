#!/usr/bin/env python3
"""Compare the partition-based Figure 5.8 analysis with direct raw scoring.

The main package's references are checked against preserved hashes from
their earlier independent audits. No raw archive is reread in this step.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path


RUNS = ('1000260_9', '1000260_10', '1000260_11')
REFERENCE_HASHES = {
    'memberships.csv': '19801e239ee76cb9c9eb0782de27fcb01bdebdddc8dc06f72e9f79aa6328a745',
    'global_class_metrics.csv': 'e48427d44a288b0f67b70c9c08b88490a8f170a4963f45ec83698ff8cc3da3fc',
    'sets_per_epoch_metrics.csv': '5e23ccf53c3c1391a2fd419a5aaee718667173837ae91b07ac156a3b639d26fb',
    'congruence_validation_metrics.csv': '8925106c35349ea277e2fc11b95cce605bb0ac753be2cdf6ad168104445a2d70',
    'congruence_selected_metrics.csv': 'f22ee2052c68dd873111e98a52d6868f6632600a25ed2d127715773bca6f02c6',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-dir', type=Path, required=True)
    parser.add_argument('--independent-dir', type=Path, required=True)
    args = parser.parse_args()
    hashes = {}

    def read(path, as_csv=False):
        content = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(content).hexdigest()
        return (list(csv.DictReader(io.StringIO(content.decode(), newline='')))
                if as_csv else json.loads(content))

    names = ('complement_membership.csv', 'complement_per_epoch_metrics.csv',
             'complement_selected_metrics.csv', 'complement_validation_curves.csv')
    actual_files = {name: read(args.main_dir / name, True) for name in names}
    own_files = {name: read(args.independent_dir / name, True) for name in names}
    actual_windows = read(args.main_dir / 'complement_validation_windows.csv', True)
    own_windows = read(args.independent_dir / 'fixed_validation_windows_per_run.csv', True)
    actual_summary = read(args.main_dir / 'complement_summary.json')
    own_summary = read(args.independent_dir / 'complement_audit_summary.json')
    references = {name: read(args.main_dir / 'references' / name, True) for name in REFERENCE_HASHES}
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

    def key(row):
        return row['run_id'], row['paper_role'], int(row['completed_epoch'])

    member_key = lambda row: (row['file_label'], row['example_id'])
    actual_members = indexed(actual_files['complement_membership.csv'], member_key, 'main_membership')
    own_members = indexed(own_files['complement_membership.csv'], member_key, 'own_membership')
    structure(set(actual_members), set(own_members), 'membership_keys')
    structure(len(actual_members), 6000, 'membership_row_count')
    membership_mapping = {
        'file_label': 'file_label', 'paper_role': 'paper_role', 'example_id': 'example_id',
        'in_Cdoubleprime': 'in_C_double_prime', 'in_special_congruence': 'in_special',
        'in_complement': 'in_complement', 'true_zero_or_one': 'true_zero_or_one',
        'included_in_accuracy': 'in_scored_complement',
    }
    for k, own in own_members.items():
        actual = actual_members[k]
        for actual_name, own_name in membership_mapping.items():
            field(actual[actual_name], own[own_name], f'membership.{k}.{actual_name}')
        for name in ('k', 'true_v3_n'):
            field(actual[name], int(own[name]), f'membership.{k}.{name}')

    actual_epochs = indexed(actual_files['complement_per_epoch_metrics.csv'], key, 'main_epochs')
    own_epochs = indexed(own_files['complement_per_epoch_metrics.csv'], key, 'own_epochs')
    actual_selected = indexed(actual_files['complement_selected_metrics.csv'], key, 'main_selected')
    own_selected = indexed(own_files['complement_selected_metrics.csv'], key, 'own_selected')
    structure(set(actual_epochs), set(own_epochs), 'epoch_keys')
    structure(set(actual_selected), set(own_selected), 'selected_keys')
    structure(len(actual_epochs), 1503, 'epoch_row_count')
    structure(len(actual_selected), 6, 'selected_row_count')
    for name, expected in REFERENCE_HASHES.items():
        field(hashes[str(args.main_dir / 'references' / name)], expected, 'preserved_reference_hash.' + name)
        field(actual_summary['source_sha256'][name], expected, 'summary.reference_hash.' + name)

    global_restricted = defaultdict(lambda: {'examples': 0, 'correct': 0, 'classes': set()})
    for row in references['global_class_metrics.csv']:
        k = key(row)
        target = int(row['true_v3_n'])
        if k not in own_epochs or target not in (0, 1):
            continue
        current = global_restricted[k]
        structure(target not in current['classes'], True, f'reference.global.{k}.{target}.unique')
        current['classes'].add(target)
        current['examples'] += int(row['examples'])
        current['correct'] += int(row['greedy_correct'])
    c_rows = [row for row in references['sets_per_epoch_metrics.csv'] if row['set'] == "C''"]
    c_double = indexed(c_rows, key, 'reference_C_double_prime')
    special_rows = references['congruence_validation_metrics.csv'] + [
        row for row in references['congruence_selected_metrics.csv'] if row['paper_role'] == 'test']
    special = indexed(special_rows, key, 'reference_special')
    structure(set(global_restricted), set(own_epochs), 'reference.global.keys')
    structure(set(c_double), set(own_epochs), 'reference.C_double_prime.keys')
    structure(set(special), set(own_epochs), 'reference.special.keys')
    for k, values in global_restricted.items():
        structure(values['classes'], {0, 1}, f'reference.global.{k}.classes')

    def check_row(actual, own, prefix):
        for name in ('run_id', 'paper_role'):
            field(actual[name], own[name], prefix + '.' + name)
        for name in ('completed_epoch', 'epoch_index', 'examples'):
            field(actual[name], int(own[name]), prefix + '.' + name)
        field(actual['correct'], int(own['exact_correct']), prefix + '.correct')
        field(actual['exact_percent'], own['exact_accuracy_percent'], prefix + '.exact_percent', True)
        field(actual['constant_zero_percent'], own['constant_zero_accuracy_percent'], prefix + '.constant_zero', True)
        field(actual['constant_one_percent'], own['constant_one_accuracy_percent'], prefix + '.constant_one', True)
        k = key(own)
        g, c, s = global_restricted[k], c_double[k], special[k]
        field(actual['global_restricted_examples'], g['examples'], prefix + '.global_examples')
        field(actual['global_restricted_correct'], g['correct'], prefix + '.global_correct')
        field(actual['Cdoubleprime_restricted_examples'], int(c['examples']), prefix + '.C_double_examples')
        field(actual['Cdoubleprime_restricted_correct'], int(c['correct']), prefix + '.C_double_correct')
        field(actual['special_restricted_examples'], int(s['true_one_count']), prefix + '.special_examples')
        field(actual['special_restricted_correct'], int(s['correct_true_one_count']), prefix + '.special_correct')
        field(int(s['true_zero_count']), 0, prefix + '.special_true_zero_count')
        field(g['examples'] - int(c['examples']) - int(s['true_one_count']), int(own['examples']), prefix + '.partition_denominator')
        field(g['correct'] - int(c['correct']) - int(s['correct_true_one_count']), int(own['exact_correct']), prefix + '.partition_correct_count')

    for k, own in own_epochs.items():
        check_row(actual_epochs[k], own, f'epoch.{k}')
    for k, own in own_selected.items():
        check_row(actual_selected[k], own, f'selected.{k}')

    curve_key = lambda row: int(row['completed_epoch'])
    actual_curves = indexed(actual_files['complement_validation_curves.csv'], curve_key, 'main_curves')
    own_curves = indexed(own_files['complement_validation_curves.csv'], curve_key, 'own_curves')
    structure(set(actual_curves), set(own_curves), 'curve_keys')
    structure(len(actual_curves), 500, 'curve_row_count')

    def check_curve(actual, own, prefix):
        field(actual['completed_epoch'], int(own['completed_epoch']), prefix + '.epoch')
        field(actual['paper_role'], own['paper_role'], prefix + '.role')
        field(actual['examples_per_run'], int(own['examples_per_run']), prefix + '.examples')
        mean, sd = float(own['mean_percent']), float(own['sample_sd_percentage_points'])
        field(actual['mean_percent'], mean, prefix + '.mean', True)
        field(actual['sample_sd_percentage_points'], sd, prefix + '.sd', True)
        field(actual['band_low_percent'], max(0, mean - sd), prefix + '.band_low', True)
        field(actual['band_high_percent'], min(100, mean + sd), prefix + '.band_high', True)
        field(actual['constant_one_percent'], own['constant_one_accuracy_percent'], prefix + '.constant_one', True)
        for run in RUNS:
            field(actual[run + '_correct'], int(own[f'run_{run.rsplit("_", 1)[1]}_exact_correct']), prefix + '.' + run + '.correct')

    for epoch, own in own_curves.items():
        check_curve(actual_curves[epoch], own, f'curve.{epoch}')
    window_key = lambda row: (row['run_id'], int(row['first_completed_epoch']), int(row['last_completed_epoch']))
    actual_window_index = indexed(actual_windows, window_key, 'main_windows')
    own_window_index = indexed(own_windows, window_key, 'own_windows')
    structure(set(actual_window_index), set(own_window_index), 'window_keys')
    for k, own in own_window_index.items():
        actual = actual_window_index[k]
        field(actual['run_id'], own['run_id'], f'window.{k}.run_id')
        for name in ('first_completed_epoch', 'last_completed_epoch'):
            field(actual[name], int(own[name]), f'window.{k}.{name}')
        field(actual['epochs'], int(own['epoch_count']), f'window.{k}.epochs')
        field(actual['mean_exact_percent'], own['mean_exact_accuracy_percent'], f'window.{k}.mean', True)

    sequence(actual_summary['runs'], list(RUNS), 'summary.runs')
    for label, role in own_summary['policy']['file_label_to_paper_role'].items():
        field(actual_summary['file_label_to_paper_role'][label], role, 'summary.role.' + label)
    for run, epoch in own_summary['policy']['existing_selected_completed_epochs'].items():
        field(actual_summary['selected_completed_epochs'][run], epoch, 'summary.selection.' + run)
    pool_mapping = {
        'all_examples': ('global', 'all_examples'),
        'global_restricted_examples': ('global', 'eligible_true_zero_or_one'),
        'Cdoubleprime_all_examples': ('C_double_prime', 'all_examples'),
        'Cdoubleprime_restricted_examples': ('C_double_prime', 'eligible_true_zero_or_one'),
        'special_all_examples': ('special', 'all_examples'),
        'special_restricted_examples': ('special', 'eligible_true_zero_or_one'),
        'complement_all_examples': ('complement', 'all_examples'),
        'complement_restricted_examples': ('complement', 'eligible_true_zero_or_one'),
        'complement_true_zero': ('complement', 'true_zero_examples'),
        'complement_true_one': ('complement', 'true_one_examples'),
    }
    for role, own in own_summary['pools'].items():
        actual = actual_summary['pools'][role]
        for name, (group, own_name) in pool_mapping.items():
            field(actual[name], own['groups'][group][own_name], f'summary.pool.{role}.{name}')
        field(actual['constant_zero_percent'], own['constant_zero_accuracy_percent'], f'summary.pool.{role}.constant_zero', True)
        field(actual['constant_one_percent'], own['constant_one_accuracy_percent'], f'summary.pool.{role}.constant_one', True)
        field(actual['Cdoubleprime_and_special_intersection'], own['C_double_prime_special_overlap'], f'summary.pool.{role}.overlap')
        histogram = Counter(str(row['true_v3_n']) for row in own_members.values()
                            if row['paper_role'] == role and row['in_complement'] == 'True')
        structure(actual['complement_true_valuation_histogram'], dict(histogram), f'summary.pool.{role}.target_histogram')
    for own in own_summary['selected_aggregates']:
        role = own['paper_role']
        actual = actual_summary['selected_accuracies'][role]
        prefix = 'summary.selected.' + role
        field(actual['examples_per_run'], own['examples_per_run'], prefix + '.examples')
        counts = [own[f'run_{run.rsplit("_", 1)[1]}_exact_correct'] for run in RUNS]
        values = [own[f'run_{run.rsplit("_", 1)[1]}_accuracy_percent'] for run in RUNS]
        sequence(actual['correct_counts'], counts, prefix + '.counts')
        sequence(actual['per_run_percent'], values, prefix + '.percentages', True)
        field(actual['mean_percent'], own['mean_percent'], prefix + '.mean', True)
        field(actual['sample_sd_percentage_points'], own['sample_sd_percentage_points'], prefix + '.sd', True)
        field(actual['constant_zero_percent'], own['constant_zero_accuracy_percent'], prefix + '.constant_zero', True)
        field(actual['constant_one_percent'], own['constant_one_accuracy_percent'], prefix + '.constant_one', True)
        field(actual['mean_minus_constant_one_percentage_points'], own['mean_percent'] - own['constant_one_accuracy_percent'], prefix + '.gain_over_one', True)
    for epoch, actual in actual_summary['validation_fixed_epochs'].items():
        check_curve(actual, own_curves[int(epoch)], 'summary.fixed_epoch.' + epoch)
    for own in own_summary['fixed_validation_window_aggregates']:
        name = f"{own['first_completed_epoch']}-{own['last_completed_epoch']}"
        actual = actual_summary['validation_window_aggregates'][name]
        values = [own[f'run_{run.rsplit("_", 1)[1]}_window_mean_percent'] for run in RUNS]
        sequence(actual['per_run_percent'], values, f'summary.window.{name}.per_run', True)
        field(actual['mean_percent'], own['mean_percent'], f'summary.window.{name}.mean', True)
        field(actual['sample_sd_percentage_points'], own['sample_sd_of_run_window_means_percentage_points'], f'summary.window.{name}.sd', True)
    for name in ('validation_windows_are_descriptive_not_selection_rules',
                 'sample_SD_not_confidence_interval', 'Cdoubleprime_and_special_disjoint_on_all6000_examples'):
        field(actual_summary[name], True, 'summary.' + name)
    field(actual_summary['sample_SD_divisor'], 2, 'summary.SD_divisor')
    field(actual_summary['curve_pool'], 'validation', 'summary.curve_pool')
    sequence(actual_summary['curve_completed_epochs'], [1, 500], 'summary.curve_epochs')
    field(actual_summary['smoothing'], 'none', 'summary.smoothing')
    field(actual_summary['partition_count_identities_checked'], len(own_epochs), 'summary.partition_count')
    field(actual_summary['validation_test_k_overlap'], 0, 'summary.pool_overlap')

    result = {
        'all_comparisons_pass': not failures, **counters,
        'absolute_float_tolerance': 1e-10,
        'membership_rows_compared': 6000,
        'per_run_epoch_rows_compared': 1503,
        'selected_rows_compared': 6,
        'validation_curve_rows_compared': 500,
        'validation_window_rows_compared': 6,
        'fixed_epoch_summary_rows_compared': len(actual_summary['validation_fixed_epochs']),
        'selected_aggregate_pools_compared': 2,
        'window_aggregate_summaries_compared': 2,
        'partition_component_counts_checked_against_previously_audited_sources': True,
        'partition_results_checked_against_independent_direct_raw_scoring': True,
        'reference_hashes_preserved': True,
        'test_scope_only_existing_selected_checkpoints': True,
        'raw_archives_reread_by_comparison': False,
        'unchecked_fields': ['Plot appearance and wording of prose policy descriptions'],
        'source_sha256': hashes, 'failure_count': len(failures), 'failures': failures,
    }
    (args.independent_dir / 'complement_main_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
