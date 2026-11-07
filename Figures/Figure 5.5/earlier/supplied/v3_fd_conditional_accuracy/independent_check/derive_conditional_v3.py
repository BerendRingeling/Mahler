#!/usr/bin/env python3
"""Independently group already audited v3(n_k) exact-accuracy counts.

This script uses the preserved independent Figure 5.4 audit. It does not
read raw prediction archives, run a model, or choose separate epochs for
the conditional groups. All calculations use Python's standard library.

Example:
  python3 derive_conditional_v3.py --audited-dir PATH_TO_INDEPENDENT_CHECK \
    --out OUTPUT_DIRECTORY
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics


RUNS = ('1000260_9', '1000260_10', '1000260_11')
ROLES = {'test1': 'validation', 'valid': 'test'}
GROUPS = ('0', '1', '>1')
EXISTING_SELECTIONS = {'1000260_9': 219, '1000260_10': 279, '1000260_11': 297}
FIXED_EPOCHS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 278, 300, 400, 500)
WINDOWS = ((1, 50), (51, 100), (101, 200), (201, 300),
           (301, 400), (401, 500), (1, 300), (1, 500))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_of(value):
    assert value >= 0, 'Unexpected negative true paper valuation'
    return str(value) if value < 2 else '>1'


def rate(correct, examples):
    return float(Fraction(100 * correct, examples))


def aggregate(rows):
    assert len(rows) == 3
    assert tuple(row['run_id'] for row in rows) == RUNS
    assert len({row['examples'] for row in rows}) == 1
    percentages = [row['conditional_exact_percent'] for row in rows]
    result = {
        'file_label': rows[0]['file_label'],
        'paper_role': rows[0]['paper_role'],
        'group': rows[0]['group'],
        'examples_per_run': rows[0]['examples'],
        'run_count': 3,
    }
    for run, row in zip(RUNS, rows):
        suffix = run.rsplit('_', 1)[1]
        result[f'run_{suffix}_completed_epoch'] = row['completed_epoch']
        result[f'run_{suffix}_correct'] = row['greedy_exact_count']
        result[f'run_{suffix}_percent'] = row['conditional_exact_percent']
    result['mean_percent'] = statistics.mean(percentages)
    result['sample_sd_percentage_points'] = statistics.stdev(percentages)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audited-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = {
        'class_metrics': 'independent_per_epoch_class_metrics.csv',
        'overall_metrics': 'independent_per_epoch_metrics.csv',
        'independent_audit': 'independent_v3_audit.json',
        'prior_comparison': 'main_independent_comparison.json',
    }
    paths = {key: args.audited_dir / name for key, name in names.items()}
    hashes = {key: sha256(path) for key, path in paths.items()}
    audit = json.loads(paths['independent_audit'].read_text())
    prior = json.loads(paths['prior_comparison'].read_text())
    assert prior['all_comparisons_pass'] and prior['failure_count'] == 0
    assert hashes['class_metrics'] == prior['source_hashes']['independent_class_metrics']
    assert hashes['independent_audit'] == prior['source_hashes']['independent_audit']
    assert prior['per_class_metric_rows_compared'] == 27000
    assert prior['per_epoch_metric_rows_compared'] == 3000

    class_rows = read_csv(paths['class_metrics'])
    overall_rows = read_csv(paths['overall_metrics'])
    assert len(class_rows) == 27000 and len(overall_rows) == 3000
    bins = defaultdict(lambda: [0, 0])
    seen_class_keys = set()
    class_counts = defaultdict(dict)
    for row in class_rows:
        run, label = row['run_id'], row['file_label']
        epoch, number = int(row['epoch_index']), int(row['completed_epoch'])
        value = int(row['true_v3_n'])
        examples, correct = int(row['examples']), int(row['greedy_correct'])
        assert run in RUNS and label in ROLES and row['paper_role'] == ROLES[label]
        assert 0 <= epoch < 500 and number == epoch + 1
        assert 0 < examples and 0 <= correct <= examples
        assert math.isclose(float(row['greedy_exact_percent']), rate(correct, examples),
                            rel_tol=0, abs_tol=1e-10)
        full_key = (run, label, epoch, value)
        assert full_key not in seen_class_keys
        seen_class_keys.add(full_key)
        class_counts[run, label, epoch][value] = examples
        group = group_of(value)
        counts = bins[run, label, epoch, group]
        counts[0] += examples
        counts[1] += correct

    overall_by_key = {}
    for row in overall_rows:
        key = (row['run_id'], row['file_label'], int(row['epoch_index']))
        assert key not in overall_by_key
        overall_by_key[key] = row
    expected_keys = {(run, label, epoch) for run in RUNS for label in ROLES
                     for epoch in range(500)}
    assert set(overall_by_key) == expected_keys
    assert set(class_counts) == expected_keys
    assert len(bins) == 9000

    grouped_rows = []
    grouped_by_key = {}
    weighted_checks = []
    denominators = {role: {} for role in ROLES.values()}
    for run in RUNS:
        for label, role in ROLES.items():
            expected_class_counts = audit['runs'][run]['files'][label]['paper_target_class_counts']
            expected_class_counts = {int(k): v for k, v in expected_class_counts.items()}
            for epoch in range(500):
                assert class_counts[run, label, epoch] == expected_class_counts
                original = overall_by_key[run, label, epoch]
                audited = audit['runs'][run]['files'][label]['per_epoch_metrics'][epoch]
                original_n = int(original['examples'])
                original_correct = int(original['greedy_exact_count'])
                assert original_n == audited['examples'] == 3000
                assert original_correct == audited['greedy_exact_count']
                rows_here = []
                for group in GROUPS:
                    examples, correct = bins[run, label, epoch, group]
                    row = {
                        'run_id': run, 'file_label': label, 'paper_role': role,
                        'epoch_index': epoch, 'completed_epoch': epoch + 1,
                        'group': group, 'examples': examples,
                        'greedy_exact_count': correct,
                        'conditional_exact_percent': rate(correct, examples),
                    }
                    grouped_rows.append(row)
                    grouped_by_key[run, label, epoch, group] = row
                    rows_here.append(row)
                    previous_n = denominators[role].setdefault(group, examples)
                    assert previous_n == examples
                sum_examples = sum(row['examples'] for row in rows_here)
                sum_correct = sum(row['greedy_exact_count'] for row in rows_here)
                weighted = sum(Fraction(row['examples'], sum_examples)
                               * Fraction(row['greedy_exact_count'], row['examples'])
                               for row in rows_here)
                overall_fraction = Fraction(original_correct, original_n)
                assert sum_examples == original_n
                assert sum_correct == original_correct
                assert weighted == overall_fraction
                assert math.isclose(float(original['greedy_exact_percent']),
                                    float(100 * overall_fraction),
                                    rel_tol=0, abs_tol=1e-10)
                weighted_checks.append({
                    'run_id': run, 'file_label': label, 'paper_role': role,
                    'epoch_index': epoch, 'completed_epoch': epoch + 1,
                    'sum_group_examples': sum_examples,
                    'overall_examples': original_n,
                    'sum_group_exact_correct': sum_correct,
                    'overall_exact_correct': original_correct,
                    'weighted_conditional_percent': float(100 * weighted),
                    'overall_exact_percent': float(100 * overall_fraction),
                    'exact_rational_check_pass': True,
                })

    # Verify, but do not replace, the pre-existing GLOBAL validation choices.
    selections = {}
    for run in RUNS:
        existing = audit['runs'][run]['selection']
        chosen = existing['selected_completed_epoch']
        assert chosen == EXISTING_SELECTIONS[run]
        val_counts = [sum(bins[run, 'test1', epoch, group][1] for group in GROUPS)
                      for epoch in range(300)]
        peak = max(val_counts)
        ties = [epoch + 1 for epoch, count in enumerate(val_counts) if count == peak]
        assert chosen == ties[0]
        assert ties == existing['tied_completed_epochs']
        selections[run] = {
            'completed_epoch': chosen, 'epoch_index': chosen - 1,
            'validation_exact_count': peak, 'tied_completed_epochs': ties,
            'separately_selected_for_groups': False,
        }

    grouped_aggregates = []
    aggregate_by_key = {}
    weighted_mean_checks = 0
    for label, role in ROLES.items():
        for epoch in range(500):
            for group in GROUPS:
                rows = [grouped_by_key[run, label, epoch, group] for run in RUNS]
                result = {'epoch_index': epoch, 'completed_epoch': epoch + 1, **aggregate(rows)}
                grouped_aggregates.append(result)
                aggregate_by_key[label, epoch, group] = result
            weighted_mean = sum(
                Fraction(denominators[role][group], 3000)
                * sum(Fraction(bins[run, label, epoch, group][1],
                               bins[run, label, epoch, group][0]) for run in RUNS) / 3
                for group in GROUPS
            )
            overall_mean = sum(Fraction(int(overall_by_key[run, label, epoch]['greedy_exact_count']),
                                        3000) for run in RUNS) / 3
            assert weighted_mean == overall_mean
            weighted_mean_checks += 1

    selected_rows = []
    selected_aggregates = []
    for label in ROLES:
        for group in GROUPS:
            rows = [grouped_by_key[run, label, EXISTING_SELECTIONS[run] - 1, group]
                    for run in RUNS]
            selected_rows.extend(rows)
            selected_aggregates.append(aggregate(rows))
    selected_weighted_checks = 0
    for label, role in ROLES.items():
        weighted_mean = sum(
            Fraction(denominators[role][group], 3000)
            * sum(Fraction(bins[run, label, EXISTING_SELECTIONS[run] - 1, group][1],
                           bins[run, label, EXISTING_SELECTIONS[run] - 1, group][0])
                  for run in RUNS) / 3
            for group in GROUPS
        )
        expected = sum(Fraction(int(overall_by_key[run, label, EXISTING_SELECTIONS[run] - 1]
                                   ['greedy_exact_count']), 3000) for run in RUNS) / 3
        assert weighted_mean == expected
        selected_weighted_checks += 1

    windows = []
    for label, role in ROLES.items():
        for group in GROUPS:
            for first, last in WINDOWS:
                per_run_values = {
                    run: [grouped_by_key[run, label, epoch - 1, group]
                          ['conditional_exact_percent'] for epoch in range(first, last + 1)]
                    for run in RUNS
                }
                values = [value for run in RUNS for value in per_run_values[run]]
                windows.append({
                    'paper_role': role, 'group': group,
                    'first_completed_epoch': first, 'last_completed_epoch': last,
                    'run_epoch_count': len(values),
                    'mean_percent_across_run_epoch_values': statistics.mean(values),
                    'minimum_percent_across_run_epoch_values': min(values),
                    'maximum_percent_across_run_epoch_values': max(values),
                    'below_90_percent_count': sum(value < 90 for value in values),
                    'below_95_percent_count': sum(value < 95 for value in values),
                    'exactly_100_percent_count': sum(value == 100 for value in values),
                    **{f'run_{run.rsplit("_", 1)[1]}_window_mean_percent':
                       statistics.mean(per_run_values[run]) for run in RUNS},
                })

    fixed_rows = [row for row in grouped_aggregates if row['completed_epoch'] in FIXED_EPOCHS]
    extrema = []
    for label, role in ROLES.items():
        for group in GROUPS:
            values = [aggregate_by_key[label, epoch, group]['mean_percent'] for epoch in range(500)]
            low, high = min(values), max(values)
            extrema.append({
                'paper_role': role, 'group': group,
                'mean_curve_minimum_percent': low,
                'mean_curve_minimum_completed_epochs': [i + 1 for i, value in enumerate(values) if value == low],
                'mean_curve_maximum_percent': high,
                'mean_curve_maximum_completed_epochs': [i + 1 for i, value in enumerate(values) if value == high],
            })

    summary = {
        'policy': {
            'source': 'Previously independently audited per-true-valuation correct counts',
            'new_raw_prediction_reads': False, 'new_model_evaluation': False,
            'decoder': 'greedy', 'groups': list(GROUPS),
            'conditioning_variable': 'True v3(n_k), already sign-converted in the source table',
            'group_gt1_correctness': 'Exact predicted valuation equals the true valuation; not merely prediction > 1',
            'group_accuracy': 'Sum of exact correct counts divided by sum of examples in the group',
            'class_percentages_are_not_unweighted_averaged': True,
            'paper_role_mapping': ROLES, 'completed_epoch_window': [1, 500],
            'selection': 'Existing global validation greedy exact maximum in epochs 1--300, earliest tie',
            'group_specific_epoch_selection': False, 'test_used_for_selection': False,
            'sd': 'Sample standard deviation across the three run percentages, divisor 2',
            'same_examples_recur_across_runs': True,
        },
        'source_files': {key: {'filename': names[key], 'sha256': hashes[key]} for key in names},
        'preserved_hashes_match_previous_independent_comparison': True,
        'previous_independent_scalar_comparisons': prior['scalar_comparison_count'],
        'checks': {
            'source_class_rows': len(class_rows), 'source_overall_rows': len(overall_rows),
            'per_run_group_epoch_rows': len(grouped_rows),
            'aggregate_group_epoch_rows': len(grouped_aggregates),
            'all_group_counts_reproduce_overall_counts': True,
            'all_group_denominators_reproduce_overall_denominators': True,
            'exact_rational_weighted_accuracy_checks': len(weighted_checks),
            'exact_rational_weighted_mean_checks': weighted_mean_checks,
            'exact_rational_selected_weighted_mean_checks': selected_weighted_checks,
            'all_three_existing_global_selections_reproduced': True,
            'all_checks_pass': True,
        },
        'group_denominators': denominators,
        'existing_global_selections': selections,
        'selected_epoch_group_aggregates': selected_aggregates,
        'fixed_epoch_group_aggregates': fixed_rows,
        'mean_curve_extrema': extrema,
        'interpretation': [
            'At the globally selected epochs, true-1 accuracy is high, true-0 accuracy is intermediate, and exact accuracy within >1 is very low.',
            'Near-100% true-1 accuracy all the time is not supported: there are early transient drops and a substantial decline late in training.',
            'The model retains high true-1 accuracy while true-0 accuracy improves over the middle of the first 300 epochs.',
            'Accuracy within >1 increases late but remains much lower than the other two groups at the selected epochs and at epoch 500.',
            'Window summaries describe observed trajectories; they are not independent observations or confidence intervals.',
        ],
    }

    outputs = {
        'conditional_per_run_per_epoch.csv': grouped_rows,
        'conditional_mean_sample_sd_per_epoch.csv': grouped_aggregates,
        'conditional_selected_epoch_per_run.csv': selected_rows,
        'conditional_selected_epoch_aggregates.csv': selected_aggregates,
        'conditional_fixed_epoch_aggregates.csv': fixed_rows,
        'conditional_training_window_diagnostics.csv': windows,
        'conditional_weighted_count_checks.csv': weighted_checks,
    }
    for name, rows in outputs.items():
        write_csv(args.out / name, rows)
    summary['output_csv_sha256'] = {name: sha256(args.out / name) for name in outputs}
    (args.out / 'conditional_audit_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({
        'checks': summary['checks'], 'group_denominators': denominators,
        'selected_epoch_group_aggregates': selected_aggregates,
    }, indent=2))


if __name__ == '__main__':
    main()
