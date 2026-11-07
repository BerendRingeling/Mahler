#!/usr/bin/env python3
"""Compute the three-run validation lower-bound curve from the raw run ZIPs.

Paper naming convention:
  eval.test1.epochs_0_499.json.bz2 = validation set
  eval.valid.epochs_0_499.json.bz2 = untouched test set

The stored model target/prediction is v_2(r_k), so v_2(n_k) = -v_2(r_k).
"""
from pathlib import Path
import bz2
import csv
import json
import statistics
import zipfile
from sympy import factorint

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
RUNS = ["1000261_12", "1000261_13", "1000261_14"]
SELECTED_EPOCH = {"1000261_12": 261, "1000261_13": 221, "1000261_14": 203}


def omega_odd(n: int) -> int:
    n = abs(int(n))
    if n == 0:
        return 0
    return sum(1 for p in factorint(n) if p % 2 == 1)


def vhat(k: int) -> int:
    c = 4 if (k % 2 == 1 or k % 16 == 0) else 2
    return omega_odd(k) + 2 * omega_odd(k * k - 16) - c


def pred_v2_n(g):
    if not isinstance(g, (list, tuple)) or not g:
        return None
    try:
        return -int(g[0])
    except Exception:
        return None


def load_json_from_zip(run: str, member: str):
    zpath = RAW / f"{run}.zip"
    with zipfile.ZipFile(zpath) as zf:
        raw = zf.read(f"{run}/{member}")
    return json.loads(bz2.decompress(raw).decode("utf-8"))


# Validation objects (paper validation set).
val_objs = {
    run: load_json_from_zip(run, "eval.test1.epochs_0_499.json.bz2")
    for run in RUNS
}

ks = [int(val_objs[RUNS[0]][str(i)]["problem"]["k"]) for i in range(3000)]
for run in RUNS[1:]:
    ks_run = [int(val_objs[run][str(i)]["problem"]["k"]) for i in range(3000)]
    assert ks_run == ks, f"Validation k order differs in {run}"

vh = [vhat(k) for k in ks]
curves = {run: [] for run in RUNS}
for run in RUNS:
    obj = val_objs[run]
    for epoch in range(500):
        ok = 0
        for i, h in enumerate(vh):
            g = obj[str(i)].get("guesses_greedy", {}).get(str(epoch))
            p = pred_v2_n(g)
            if p is not None and p >= h:
                ok += 1
        curves[run].append(100.0 * ok / 3000.0)

curve_csv = RESULTS / "lower_bound_curve.csv"
with curve_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", *RUNS, "mean", "sample_sd"])
    for epoch in range(500):
        xs = [curves[r][epoch] for r in RUNS]
        w.writerow([epoch, *xs, statistics.mean(xs), statistics.stdev(xs)])

# Also record the lower-bound proportion on the untouched test set at the
# validation-selected checkpoint used for each run.
rows = []
for run in RUNS:
    obj = load_json_from_zip(run, "eval.valid.epochs_0_499.json.bz2")
    epoch = SELECTED_EPOCH[run]
    ok = 0
    for i in range(3000):
        item = obj[str(i)]
        k = int(item["problem"]["k"])
        h = vhat(k)
        p = pred_v2_n(item.get("guesses_greedy", {}).get(str(epoch)))
        if p is not None and p >= h:
            ok += 1
    rows.append((run, epoch, ok, 100.0 * ok / 3000.0))

checkpoint_csv = RESULTS / "selected_checkpoint_lower_bound.csv"
with checkpoint_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_id", "selected_epoch", "count_satisfying_lower_bound", "test_pct"])
    for row in rows:
        w.writerow(row)
    vals = [row[3] for row in rows]
    w.writerow(["MEAN", "", "", statistics.mean(vals)])
    w.writerow(["SAMPLE_SD", "", "", statistics.stdev(vals)])

print(f"Wrote {curve_csv}")
print(f"Wrote {checkpoint_csv}")
