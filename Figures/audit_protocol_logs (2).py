#!/usr/bin/env python3
"""Compare all nine conductor/factorization training logs without loading predictions.

Three families are audited: rk_wN (9,10,11), rk_wfN (21,22,23), and
rk_wfD (18,19,20), with experiment prefix 1000262_. Only the Python
standard library is required. Inputs are read without modification;
no model is executed and no checkpoint is selected.
"""
import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

EXPECTED_EPOCHS = list(range(500))
IDENTITY_KEYS = {"command", "dump_path", "exp_id"}
FAMILY_KEYS = {"features", "reload_data", "eval_data"}


def audit(path, family):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lines = text.splitlines()
    config = {}
    for line in lines:
        if "The experiment will be stored" in line:
            break
        chunks = line.split(" - ", 3)
        payload = chunks[3] if line.startswith("INFO - ") and len(chunks) == 4 else line
        match = re.match(r"^\s*([A-Za-z][A-Za-z0-9_]*):\s*(.*)$", payload)
        if match:
            config[match.group(1)] = match.group(2)
    starts = [int(x) for x in re.findall(r"Starting epoch (\d+)", text)]
    ends = [int(x) for x in re.findall(r"End of epoch (\d+)", text)]
    metrics = [json.loads(line.split("__log__:", 1)[1])
               for line in lines if "__log__:" in line]
    indices = [int(item["epoch"]) for item in metrics]
    parameters = [int(x) for x in re.findall(r"Number of parameters:\s*(\d+)", text)]
    seeds = []
    loss_records = []
    for line in lines:
        match = re.search(
            r"Initialized random generator for worker (\d+), with seed (\[[^\]]+\]) \(base seed=(\d+)\)", line)
        if match:
            seeds.append({"worker": int(match.group(1)),
                          "seed_tuple": json.loads(match.group(2)),
                          "base_seed": int(match.group(3)),
                          "source_message": line})
        if "LOSS:" in line:
            payload = line.split(" - ", 3)[3]
            loss_records.append({
                "step": int(re.match(r"\s*(\d+)\s*-", payload).group(1)),
                "loss": float(re.search(r"\bLOSS:\s*(\S+)", payload).group(1)),
                "learning_rate": float(re.search(r"\bLR:\s*(\S+)", payload).group(1))})
    finite_losses = all(math.isfinite(row["loss"]) and math.isfinite(row["learning_rate"])
                        for row in loss_records)
    nonfinite_metrics = [{"epoch": row["epoch"], "field": key, "value": str(value)}
                         for row in metrics for key, value in row.items()
                         if isinstance(value, (int, float)) and not math.isfinite(value)]
    warnings = [line for line in lines
                if re.match(r"^(WARNING|ERROR|CRITICAL)\b", line)
                or re.search(r"\b(Traceback|RuntimeError|out of memory)\b", line)]
    sample_counts = [int(x) for x in re.findall(
        r"almost_correctness: .*?\(over (\d+) samples\)", text)]
    saves = re.findall(r"Saving checkpoint to (.+?) \.\.\.", text)
    by_epoch = {int(row["epoch"]): row for row in metrics}
    fixed = {str(e): {key: value for key, value in by_epoch.get(e, {}).items()
                     if key == "epoch" or key.endswith(("_greedy_acc", "_acc",
                         "_avg_almost_correctness", "_xe_loss", "_well_formed"))}
             for e in (299, 499)}
    all_split_metrics = {}
    for label in ("valid", "test1"):
        prefix = label + "_ELLIPTIC_CURVE_"
        columns = sorted({key for row in metrics for key in row if key.startswith(prefix)})
        all_split_metrics[label] = {
            "columns": columns,
            "present_at_every_epoch": {
                key: all(key in row for row in metrics) for key in columns},
            "greedy_exact_zero_through_index_299": all(
                row.get(prefix + "greedy_acc") == 0 for row in metrics if row["epoch"] < 300)}
    return {
        "family": family, "filename": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw), "line_count": len(lines), "configuration": config,
        "number_of_parameters": parameters, "first_log_line": lines[0], "last_log_line": lines[-1],
        "seed_evidence": seeds,
        "epoch_coverage": {
            "starts_count": len(starts), "ends_count": len(ends), "metric_records_count": len(metrics),
            "starts_equal_0_through_499": starts == EXPECTED_EPOCHS,
            "ends_equal_0_through_499": ends == EXPECTED_EPOCHS,
            "metric_indices_equal_0_through_499": indices == EXPECTED_EPOCHS,
            "missing_metric_epochs": sorted(set(EXPECTED_EPOCHS) - set(indices)),
            "duplicate_metric_epochs": [e for e, n in collections.Counter(indices).items() if n > 1]},
        "loss_reporting": {
            "record_count": len(loss_records),
            "steps_equal_200_through_500000_by_200": [row["step"] for row in loss_records] == list(range(200, 500001, 200)),
            "first_record": loss_records[0] if loss_records else None,
            "last_record": loss_records[-1] if loss_records else None,
            "finite": finite_losses,
            "learning_rates": sorted({row["learning_rate"] for row in loss_records})},
        "evaluation_summary_count": len(sample_counts),
        "evaluation_sample_counts": dict(collections.Counter(sample_counts)),
        "split_metric_coverage": all_split_metrics,
        "checkpoint_saves": {"count": len(saves), "unique_paths": sorted(set(saves))},
        "warning_or_error_messages": warnings,
        "nonfinite_logged_metrics": nonfinite_metrics,
        "fixed_epoch_logged_values": fixed,
    }


def compare(runs):
    keys = sorted(set().union(*(set(record["configuration"]) for record in runs.values())))
    shared, differences = {}, {}
    for key in keys:
        values = {run_id: record["configuration"].get(key) for run_id, record in runs.items()}
        if len(set(values.values())) == 1:
            shared[key] = next(iter(values.values()))
        else:
            differences[key] = values
    return {"shared_configuration": shared, "configuration_differences": differences,
            "non_identity_configuration_differences": {k: v for k, v in differences.items() if k not in IDENTITY_KEYS}}



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conductor-inputs', type=Path, required=True)
    parser.add_argument('--factor-conductor-inputs', type=Path, required=True)
    parser.add_argument('--factor-discriminant-inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    families = {
        'rk_wN': {'ids': ('1000262_9','1000262_10','1000262_11'),
                  'root': args.conductor_inputs, 'features': 'k,w,N'},
        'rk_wfN': {'ids': ('1000262_21','1000262_22','1000262_23'),
                   'root': args.factor_conductor_inputs, 'features': 'k,w,N_Factor'},
        'rk_wfD': {'ids': ('1000262_18','1000262_19','1000262_20'),
                   'root': args.factor_discriminant_inputs, 'features': 'k,w,D_Factor'},
    }
    runs = {}
    for family, spec in families.items():
        for rid in spec['ids']:
            runs[rid] = audit(spec['root']/rid/'train.log', family)
    comparisons = {
        'within_'+family: compare({rid:runs[rid] for rid in spec['ids']})
        for family,spec in families.items()}
    comparisons['all_nine_runs'] = compare(runs)
    unexpected_cross_family = {
        key:values for key,values in comparisons['all_nine_runs']['configuration_differences'].items()
        if key not in IDENTITY_KEYS | FAMILY_KEYS}
    base_seeds = {rid:[seed['base_seed'] for seed in record['seed_evidence']]
                  for rid,record in runs.items()}
    new_ids = families['rk_wfD']['ids']
    issues = []
    for rid,record in runs.items():
        coverage = record['epoch_coverage']
        for key in ('starts_equal_0_through_499','ends_equal_0_through_499',
                    'metric_indices_equal_0_through_499'):
            if not coverage[key]:
                issues.append({'run_id':rid,'issue':key})
        if not record['loss_reporting']['steps_equal_200_through_500000_by_200']:
            issues.append({'run_id':rid,'issue':'Unexpected loss-reporting steps'})
        if not record['loss_reporting']['finite'] or record['nonfinite_logged_metrics']:
            issues.append({'run_id':rid,'issue':'Nonfinite numeric output'})
        if record['warning_or_error_messages']:
            issues.append({'run_id':rid,'issue':'Warning or error messages'})
        expected_features = families[record['family']]['features']
        cfg = record['configuration']
        if cfg.get('features') != expected_features or cfg.get('class_feature') != 'r_k' or cfg.get('ec_output') != 'r_k':
            issues.append({'run_id':rid,'issue':'Unexpected input or target configuration'})
        if cfg.get('base') != '10':
            issues.append({'run_id':rid,'issue':'Unexpected token base'})
        if record['number_of_parameters'] != [7517184]:
            issues.append({'run_id':rid,'issue':'Unexpected parameter count'})
        if record['evaluation_summary_count']!=1000 or record['evaluation_sample_counts']!={3000:1000}:
            issues.append({'run_id':rid,'issue':'Unexpected evaluation sample counts'})
        if record['checkpoint_saves']['count']!=500 or len(record['checkpoint_saves']['unique_paths'])!=1:
            issues.append({'run_id':rid,'issue':'Unexpected rolling checkpoint saves'})
        if cfg.get('epoch_size')!='32000' or cfg.get('batch_size')!='32' or cfg.get('max_epoch')!='500':
            issues.append({'run_id':rid,'issue':'Unexpected epoch or batch configuration'})
        if not all(all(fields.values()) for fields in
                   (v['present_at_every_epoch'] for v in record['split_metric_coverage'].values())):
            issues.append({'run_id':rid,'issue':'Incomplete per-epoch split metric columns'})
    for family in families:
        if comparisons['within_'+family]['non_identity_configuration_differences']:
            issues.append({'family':family,'issue':'Within-family differences beyond experiment identity'})
    if unexpected_cross_family:
        issues.append({'issue':'Unexpected cross-family configuration difference',
                       'keys':sorted(unexpected_cross_family)})
    selected_settings = ('features','class_feature','ec_output','base','max_epoch','epoch_size',
        'batch_size','optimizer','architecture','enc_emb_dim','dec_emb_dim','n_enc_layers',
        'n_dec_layers','n_enc_heads','n_dec_heads','max_len','max_output_len','eval_only',
        'reload_checkpoint','run_id','validation_metrics','stopping_criterion')
    result = {
        'audit_scope':'All nine rk_wN, rk_wfN, rk_wfD logs; no prediction loading, checkpoint selection or model execution.',
        'families':{family:{'run_ids':list(spec['ids']),'features':spec['features']}
                    for family,spec in families.items()},
        'new_run_ids':list(new_ids),
        'runs':runs,'comparisons':comparisons,
        'selected_settings':{rid:{key:record['configuration'].get(key) for key in selected_settings}
                             for rid,record in runs.items()},
        'unexpected_cross_family_configuration_differences':unexpected_cross_family,
        'logged_data_worker_base_seeds':base_seeds,
        'one_logged_data_worker_seed_per_run':all(len(values)==1 for values in base_seeds.values()),
        'new_logged_data_worker_base_seeds_distinct':len({value for rid in new_ids for value in base_seeds[rid]})==3,
        'all_nine_logged_data_worker_base_seeds_distinct':len({value for values in base_seeds.values() for value in values})==9,
        'issues':issues,
        'interpretation':[
            'All families have target r_k and inputs k,w plus N, N_Factor, or D_Factor respectively.',
            'This audit compares recorded training settings; actual evaluation examples and feature values are checked separately from prediction exports.',
            'The logs provide no evidence of an architecture, optimizer, or epoch-budget difference between the three families if no unexpected difference is reported.',
            'Recorded data-worker base seeds do not document every model, global, or library seed. Experiment suffixes do not establish matched-seed pairs.',
            'The configured run_id is 0 for these logs; it is not the suffix of exp_id.',
            'Both evaluation groups are logged at every epoch. This does not establish which results influenced historical decisions.',
            'An epoch comprises 32000 training examples, or 1000 updates at batch size 32; it is not a complete pass through the 243999-example dataset.',
            'Rolling checkpoint saves do not establish retention of every checkpoint. Saved predictions suffice for the requested metric calculations.',
            'Greedy, sampled and either-output accuracy fields have distinct interpretations; the direct-prediction audit determines the metric correspondence.'
        ]
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    brief = {
        'output':str(args.out),'issues':issues,
        'within_family_differing_keys':{family:list(comparisons['within_'+family]['configuration_differences'])
                                       for family in families},
        'all_nine_differing_keys':list(comparisons['all_nine_runs']['configuration_differences']),
        'unexpected_cross_family_differences':unexpected_cross_family,
        'logged_data_worker_base_seeds':base_seeds,
        'all_nine_logged_data_worker_base_seeds_distinct':result['all_nine_logged_data_worker_base_seeds_distinct'],
        'selected_new_settings':{rid:result['selected_settings'][rid] for rid in new_ids},
        'new_run_coverage':{rid:{'epoch_coverage':runs[rid]['epoch_coverage'],
                                'loss_records':runs[rid]['loss_reporting']['record_count'],
                                'evaluation_summaries':runs[rid]['evaluation_summary_count'],
                                'evaluation_sample_counts':runs[rid]['evaluation_sample_counts'],
                                'checkpoint_saves':runs[rid]['checkpoint_saves']['count'],
                                'parameter_count':runs[rid]['number_of_parameters'],
                                'nonfinite_metrics':runs[rid]['nonfinite_logged_metrics'],
                                'warnings':runs[rid]['warning_or_error_messages']}
                            for rid in new_ids}}
    print(json.dumps(brief,indent=2))


if __name__=='__main__':
    main()
