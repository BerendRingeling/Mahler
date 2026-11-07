#!/usr/bin/env python3
"""Compare the main Figure 5.5 tables with the independent derivation.

All saved grouped rows, means, sample SDs, display bounds, selected results,
and fixed-epoch summaries are checked. No raw prediction archive is read.
"""
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
ROLES = ('validation', 'test')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-dir', type=Path, required=True)
    parser.add_argument('--independent-dir', type=Path, required=True)
    parser.add_argument('--audited-dir', type=Path, required=True)
    args = parser.parse_args()
    hashes = {}

    def read(path, csv_file=False):
        data = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        if csv_file:
            return list(csv.DictReader(io.StringIO(data.decode('utf-8'), newline='')))
        return json.loads(data)

    main_rows = read(args.main_dir / 'grouped_per_epoch_metrics.csv', True)
    main_curves = read(args.main_dir / 'conditional_curve_metrics.csv', True)
    main_selected = read(args.main_dir / 'selected_group_metrics.csv', True)
    main_summary = read(args.main_dir / 'conditional_summary.json')
    own_rows = read(args.independent_dir / 'conditional_per_run_per_epoch.csv', True)
    own_curves = read(args.independent_dir / 'conditional_mean_sample_sd_per_epoch.csv', True)
    own_selected = read(args.independent_dir / 'conditional_selected_epoch_per_run.csv', True)
    own_selected_aggregates = read(args.independent_dir / 'conditional_selected_epoch_aggregates.csv', True)
    own_summary = read(args.independent_dir / 'conditional_audit_summary.json')
    audit = read(args.audited_dir / 'independent_v3_audit.json')
    prior = read(args.audited_dir / 'main_independent_comparison.json')

    counters = {'scalar_field_comparisons': 0, 'floating_comparisons': 0,
                'exact_comparisons': 0, 'structural_checks': 0}
    failures = []

    def serializable(value):
        return sorted(value) if isinstance(value, set) else value

    def structure(actual, expected, label):
        counters['structural_checks'] += 1
        if actual != expected:
            failures.append({'field': label, 'actual': serializable(actual),
                             'expected': serializable(expected)})

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

    def row_key(row, own=False):
        group = row['group'] if own else row['true_group']
        return row['run_id'], row['paper_role'], int(row['completed_epoch']), group

    def curve_key(row, own=False):
        group = row['group'] if own else row['true_group']
        return row['paper_role'], int(row['completed_epoch']), group

    def index_rows(rows, key_function, name):
        result = {key_function(row): row for row in rows}
        structure(len(result), len(rows), name + '.unique_keys')
        return result

    actual_by_key = index_rows(main_rows, row_key, 'main_rows')
    own_by_key = index_rows(own_rows, lambda r: row_key(r, True), 'independent_rows')
    actual_curves = index_rows(main_curves, curve_key, 'main_curves')
    expected_curves = index_rows(own_curves, lambda r: curve_key(r, True), 'independent_curves')
    actual_selected = index_rows(main_selected, row_key, 'main_selected')
    expected_selected = index_rows(own_selected, lambda r: row_key(r, True), 'independent_selected')
    structure(set(actual_by_key), set(own_by_key), 'all_epoch_row_keys')
    structure(set(actual_curves), set(expected_curves), 'all_curve_row_keys')
    structure(set(actual_selected), set(expected_selected), 'selected_row_keys')
    structure(len(actual_by_key), 9000, 'epoch_row_count')
    structure(len(actual_curves), 3000, 'curve_row_count')
    structure(len(actual_selected), 18, 'selected_row_count')

    def check_row(actual, own, label):
        field(actual['run_id'], own['run_id'], label + '.run_id')
        field(actual['paper_role'], own['paper_role'], label + '.paper_role')
        field(actual['completed_epoch'], int(own['completed_epoch']), label + '.completed_epoch')
        field(actual['true_group'], own['group'], label + '.true_group')
        field(actual['examples'], int(own['examples']), label + '.examples')
        field(actual['correct'], int(own['greedy_exact_count']), label + '.correct')
        field(actual['accuracy_percent'], own['conditional_exact_percent'], label + '.accuracy_percent', True)

    def check_curve(actual, own, label):
        field(actual['paper_role'], own['paper_role'], label + '.paper_role')
        field(actual['completed_epoch'], int(own['completed_epoch']), label + '.completed_epoch')
        field(actual['true_group'], own['group'], label + '.true_group')
        field(actual['examples_per_run'], int(own['examples_per_run']), label + '.examples_per_run')
        for run in RUNS:
            suffix = run.rsplit('_', 1)[1]
            field(actual[run + '_correct'], int(own[f'run_{suffix}_correct']), label + f'.{run}.correct')
        mean, sd = float(own['mean_percent']), float(own['sample_sd_percentage_points'])
        field(actual['mean_percent'], mean, label + '.mean_percent', True)
        field(actual['sample_sd_percentage_points'], sd, label + '.sample_sd_percentage_points', True)
        field(actual['display_band_lower_percent'], max(0, mean - sd), label + '.display_band_lower_percent', True)
        field(actual['display_band_upper_percent'], min(100, mean + sd), label + '.display_band_upper_percent', True)

    for key, own in own_by_key.items():
        check_row(actual_by_key[key], own, f'epoch.{key}')
    for key, own in expected_curves.items():
        check_curve(actual_curves[key], own, f'curve.{key}')
    for key, own in expected_selected.items():
        check_row(actual_selected[key], own, f'selected.{key}')

    sequence(main_summary['runs'], list(RUNS), 'summary.runs')
    field(main_summary['plotted_pool'], 'validation', 'summary.plotted_pool')
    sequence(main_summary['plotted_completed_epochs'], [1, 500], 'summary.plot_epoch_window')
    field(main_summary['class_rows_input'], 27000, 'summary.class_rows_input')
    field(main_summary['grouped_epoch_rows'], 9000, 'summary.grouped_epoch_rows')
    field(main_summary['weighted_group_totals_match_all_overall_counts'], True,
          'summary.weighted_group_totals_match_all_overall_counts')
    field(main_summary['smoothing'], None, 'summary.smoothing')
    for file_label, role in own_summary['policy']['paper_role_mapping'].items():
        field(main_summary['file_label_to_paper_role'][file_label], role, f'summary.pool.{file_label}')
    for run in RUNS:
        epoch = own_summary['existing_global_selections'][run]['completed_epoch']
        field(main_summary['selected_epochs'][run], epoch, f'summary.selection.{run}')
        source_file = audit['runs'][run]['files']['test1']
        expected_epochs = [i + 1 for i, hist in enumerate(source_file['paper_integer_prediction_histograms'])
                           if hist.get('1', 0) == source_file['examples']]
        sequence(main_summary['validation_epochs_predicting_one_for_all_examples'][run],
                 expected_epochs, f'summary.all_predictions_one.{run}')
    for role in ROLES:
        for group in GROUPS:
            field(main_summary['group_sizes'][role][group],
                  own_summary['group_denominators'][role][group], f'summary.group_sizes.{role}.{group}')
    for own in own_selected_aggregates:
        role, group = own['paper_role'], own['group']
        actual = main_summary['selected_group_accuracies'][role][group]
        prefix = f'summary.selected.{role}.{group}'
        field(actual['examples_per_run'], int(own['examples_per_run']), prefix + '.examples_per_run')
        counts = [int(own[f'run_{run.rsplit("_", 1)[1]}_correct']) for run in RUNS]
        percentages = [float(own[f'run_{run.rsplit("_", 1)[1]}_percent']) for run in RUNS]
        sequence(actual['correct_counts'], counts, prefix + '.correct_counts')
        sequence(actual['percentages'], percentages, prefix + '.percentages', True)
        field(actual['mean_percent'], own['mean_percent'], prefix + '.mean_percent', True)
        field(actual['sample_sd_percentage_points'], own['sample_sd_percentage_points'],
              prefix + '.sample_sd_percentage_points', True)
    for epoch, groups in main_summary['fixed_validation_epochs'].items():
        for group, actual in groups.items():
            own = expected_curves['validation', int(epoch), group]
            check_curve(actual, own, f'summary.fixed.{epoch}.{group}')

    source_mapping = {
        'all_per_epoch_class_metrics.csv': 'main_class_metrics',
        'all_per_epoch_metrics.csv': 'main_per_epoch_metrics',
        'v3_audit_summary.json': 'main_summary',
    }
    for name, source_key in source_mapping.items():
        field(main_summary['source_sha256'][name], prior['source_hashes'][source_key],
              'source_hash.' + name)
    for path, digest in hashes.items():
        filename = Path(path).name
        if filename in own_summary['output_csv_sha256']:
            field(digest, own_summary['output_csv_sha256'][filename],
                  'independent_output_hash.' + filename)
    field(hashes[str(args.audited_dir / 'independent_v3_audit.json')],
          prior['source_hashes']['independent_audit'], 'preserved_independent_audit_hash')

    result = {
        'all_comparisons_pass': not failures,
        **counters,
        'absolute_float_tolerance': 1e-10,
        'run_pool_epoch_group_rows_compared': len(main_rows),
        'mean_sd_curve_rows_compared': len(main_curves),
        'selected_run_group_rows_compared': len(main_selected),
        'selected_aggregate_groups_compared': len(own_selected_aggregates),
        'all_500_epochs_both_pools_all_three_runs': True,
        'all_mean_and_sample_sd_fields_checked': True,
        'display_band_clipping_checked_separately': True,
        'constant_one_epoch_lists_checked_from_previously_audited_histograms': True,
        'raw_prediction_archives_read': False,
        'unchecked_scope': ['plot appearance and plotting-environment metadata',
                            'wording of prose policy descriptions'],
        'source_sha256': hashes,
        'failure_count': len(failures),
        'failures': failures,
    }
    destination = args.independent_dir / 'conditional_main_comparison.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'source_sha256'}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
