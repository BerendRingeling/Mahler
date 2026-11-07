#!/usr/bin/env python3
"""Compute pooled three-run confusion matrices for the v2 validation experiment.

Paper naming convention:
  eval.test1.epochs_0_499.json.bz2 = validation set
  eval.valid.epochs_0_499.json.bz2 = untouched test set

The stored target/prediction is v_2(r_k), hence v_2(n_k)=-v_2(r_k).
For each fixed epoch, this script pools the 3,000 validation predictions
from each of the three runs (9,000 observations total) and records:
  (1) model prediction vs deterministic predictor P-hat(k),
  (2) model prediction vs true v_2(n_k).
"""
from pathlib import Path
import bz2, csv, json, math, zipfile
from collections import Counter
from sympy import factorint

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
RUNS = ["1000261_12", "1000261_13", "1000261_14"]
EPOCHS = [0, 8, 40, 100, 210, 300]


def omega_odd(n: int) -> int:
    n = abs(int(n))
    if n == 0:
        return 0
    return sum(1 for p in factorint(n) if p % 2 == 1)


def v2_int(n: int) -> int:
    n = abs(int(n))
    if n == 0:
        return 10**9
    e = 0
    while n % 2 == 0:
        e += 1
        n //= 2
    return e


def vhat(k: int) -> int:
    c = 4 if (k % 2 == 1 or k % 16 == 0) else 2
    return omega_odd(k) + 2 * omega_odd(k * k - 16) - c


def s_of_k(k: int) -> float:
    m = max(v2_int(k - 8), v2_int(k + 8))
    return 1.5 * max(1.0 - 2.0 ** (5 - m), 0.0)


def phat(k: int) -> int:
    x = vhat(k) + s_of_k(k) + 1.0
    return int(math.floor(x + 0.5))


def decode_v2_n(g):
    if not isinstance(g, (list, tuple)) or len(g) < 2:
        return None
    try:
        num = int(g[0])
        den = int(g[1])
    except Exception:
        return None
    if den != 1:
        return None
    return -num


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


def load_json(run: str):
    with zipfile.ZipFile(RAW / f"{run}.zip") as zf:
        raw = zf.read(f"{run}/eval.test1.epochs_0_499.json.bz2")
    return json.loads(bz2.decompress(raw).decode("utf-8"))


objs = {run: load_json(run) for run in RUNS}
ks = [int(objs[RUNS[0]][str(i)]["problem"]["k"]) for i in range(3000)]
for run in RUNS[1:]:
    assert [int(objs[run][str(i)]["problem"]["k"]) for i in range(3000)] == ks

P = [phat(k) for k in ks]
truth = [decode_v2_n(objs[RUNS[0]][str(i)]["answer"]) for i in range(3000)]
assert all(t is not None for t in truth)

summary_rows = []
for epoch in EPOCHS:
    pred_pool, phat_pool, truth_pool = [], [], []
    for run in RUNS:
        obj = objs[run]
        for i in range(3000):
            pred = decode_v2_n(obj[str(i)].get("guesses_greedy", {}).get(str(epoch)))
            if pred is None:
                continue
            pred_pool.append(pred)
            phat_pool.append(P[i])
            truth_pool.append(truth[i])

    rp = pearson(pred_pool, phat_pool)
    rt = pearson(pred_pool, truth_pool)
    summary_rows.append((epoch, len(pred_pool), rp, rt))

    cp = Counter(zip(pred_pool, phat_pool))
    ct = Counter(zip(pred_pool, truth_pool))
    with (RESULTS / f"confusion_Phat_epoch_{epoch}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["predicted_v2_n", "P_hat", "count"])
        for (x, y), c in sorted(cp.items()): w.writerow([x, y, c])
    with (RESULTS / f"confusion_truth_epoch_{epoch}.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["predicted_v2_n", "true_v2_n", "count"])
        for (x, y), c in sorted(ct.items()): w.writerow([x, y, c])

with (RESULTS / "correlations.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "pooled_valid_predictions", "pearson_pred_vs_Phat", "pearson_pred_vs_truth"])
    w.writerows(summary_rows)

(RESULTS / "summary.txt").write_text("\n".join(
    f"epoch {e}: n={n}, pooled r(pred,P_hat)={rp:.12f}, pooled r(pred,truth)={rt:.12f}"
    for e, n, rp, rt in summary_rows
) + "\n")
print(RESULTS / "correlations.csv")
