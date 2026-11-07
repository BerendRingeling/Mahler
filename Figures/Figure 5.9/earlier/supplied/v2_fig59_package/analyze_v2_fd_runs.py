#!/usr/bin/env python3
"""Recompute the three-run Figure 5.9 validation curve from raw run ZIPs.

Important naming convention in Alberto's exported files:
  eval.test1.epochs_0_499.json.bz2 = paper VALIDATION set
  eval.valid.epochs_0_499.json.bz2 = paper TEST set

Greedy exact-match accuracy is computed directly from the JSON predictions.
The selected checkpoint is the earliest epoch attaining the largest validation
accuracy among epochs 1,...,300. Test accuracy is evaluated only at that fixed
checkpoint.
"""

from __future__ import annotations
import argparse
import bz2
import csv
import json
import math
import statistics
import zipfile
from pathlib import Path


def load_json_from_zip(zip_path: Path, basename: str):
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if n.endswith('/' + basename) or n == basename]
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one {basename} in {zip_path}, found {matches}")
        return json.loads(bz2.decompress(zf.read(matches[0])))


def greedy_accuracy(data: dict, epoch: int) -> float:
    key = str(epoch)
    correct = 0
    for row in data.values():
        if row.get('guesses_greedy', {}).get(key) == row.get('answer'):
            correct += 1
    return 100.0 * correct / len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zips', nargs='+', type=Path, help='Raw run ZIPs')
    ap.add_argument('--outdir', type=Path, default=Path('.'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    run_curves = []
    selected = []

    for zp in args.zips:
        run_id = zp.stem
        validation = load_json_from_zip(zp, 'eval.test1.epochs_0_499.json.bz2')
        test = load_json_from_zip(zp, 'eval.valid.epochs_0_499.json.bz2')
        if len(validation) != 3000 or len(test) != 3000:
            raise RuntimeError(f"Unexpected evaluation sizes for {run_id}: {len(validation)}, {len(test)}")

        curve = [greedy_accuracy(validation, e) for e in range(500)]
        run_curves.append((run_id, curve))

        candidates = [(curve[e], e) for e in range(1, 301)]
        best_acc = max(a for a, _ in candidates)
        best_epoch = min(e for a, e in candidates if a == best_acc)
        test_acc = greedy_accuracy(test, best_epoch)
        selected.append((run_id, best_epoch, best_acc, test_acc))

    curve_path = args.outdir / 'v2_fd_validation_curve.csv'
    with curve_path.open('w', newline='') as f:
        w = csv.writer(f)
        header = ['epoch'] + [rid for rid, _ in run_curves] + ['mean', 'sample_sd']
        w.writerow(header)
        for e in range(500):
            vals = [curve[e] for _, curve in run_curves]
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            w.writerow([e] + [f'{v:.12f}' for v in vals] + [f'{statistics.mean(vals):.12f}', f'{sd:.12f}'])

    checkpoints_path = args.outdir / 'selected_checkpoints.csv'
    with checkpoints_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run_id', 'selected_epoch', 'validation_accuracy_pct', 'test_accuracy_pct'])
        for row in selected:
            w.writerow([row[0], row[1], f'{row[2]:.12f}', f'{row[3]:.12f}'])
        tests = [r[3] for r in selected]
        w.writerow(['MEAN', '', '', f'{statistics.mean(tests):.12f}'])
        w.writerow(['SAMPLE_SD', '', '', f'{statistics.stdev(tests):.12f}'])

    summary_path = args.outdir / 'summary.txt'
    tests = [r[3] for r in selected]
    with summary_path.open('w') as f:
        f.write('Figure 5.9 three-run audit\n')
        f.write('Validation = eval.test1; test = eval.valid (raw file naming).\n')
        f.write('Selection: earliest maximum validation greedy exact-match among epochs 1-300.\n\n')
        for rid, ep, va, ta in selected:
            f.write(f'{rid}: selected epoch {ep}; validation {va:.2f}%; test {ta:.2f}%\n')
        f.write(f'\nMean test accuracy: {statistics.mean(tests):.2f}%\n')
        f.write(f'Sample SD over three runs: {statistics.stdev(tests):.2f}%\n')

    print(curve_path)
    print(checkpoints_path)
    print(summary_path)
    for rid, ep, va, ta in selected:
        print(f'{rid}: epoch={ep}, validation={va:.6f}%, test={ta:.6f}%')
    print(f'mean test={statistics.mean(tests):.6f}%, sample SD={statistics.stdev(tests):.6f}%')


if __name__ == '__main__':
    main()
