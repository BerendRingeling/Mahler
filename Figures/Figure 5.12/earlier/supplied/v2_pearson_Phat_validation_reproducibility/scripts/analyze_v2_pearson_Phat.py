#!/usr/bin/env python3
"""Compute the three-run Pearson-correlation curve for Figure v2_pearson_Phat_validation.

Paper naming convention:
  eval.test1.epochs_0_499.json.bz2 = validation set
  eval.valid.epochs_0_499.json.bz2 = untouched test set

The stored model target/prediction is v_2(r_k), so v_2(n_k) = -v_2(r_k).
At each epoch, Pearson r is computed between the model's greedy integer
predictions of v_2(n_k) and the deterministic predictor \hat P(k), on the
3,000 validation examples for which the greedy prediction is an integer.
"""
from pathlib import Path
import bz2, csv, json, math, statistics, zipfile
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
    # The argument of round is never half-integral for the values here.
    x = vhat(k) + s_of_k(k) + 1.0
    return int(math.floor(x + 0.5))


def pred_v2_n(g):
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
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 0 or sy <= 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sx * sy)


def load_json_from_zip(run: str, member: str):
    zpath = RAW / f"{run}.zip"
    with zipfile.ZipFile(zpath) as zf:
        raw = zf.read(f"{run}/{member}")
    return json.loads(bz2.decompress(raw).decode("utf-8"))


val_objs = {
    run: load_json_from_zip(run, "eval.test1.epochs_0_499.json.bz2")
    for run in RUNS
}
ks = [int(val_objs[RUNS[0]][str(i)]["problem"]["k"]) for i in range(3000)]
for run in RUNS[1:]:
    ks_run = [int(val_objs[run][str(i)]["problem"]["k"]) for i in range(3000)]
    assert ks_run == ks, f"Validation k order differs in {run}"

ph = [phat(k) for k in ks]
curves = {run: [] for run in RUNS}
counts = {run: [] for run in RUNS}

for run in RUNS:
    obj = val_objs[run]
    for epoch in range(500):
        model_vals = []
        predictor_vals = []
        for i in range(3000):
            g = obj[str(i)].get("guesses_greedy", {}).get(str(epoch))
            p = pred_v2_n(g)
            if p is not None:
                model_vals.append(p)
                predictor_vals.append(ph[i])
        curves[run].append(pearson(model_vals, predictor_vals))
        counts[run].append(len(model_vals))

curve_csv = RESULTS / "v2_pearson_Phat_validation_curve.csv"
with curve_csv.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", *RUNS, "mean", "sample_sd", *[f"{r}_nvalid" for r in RUNS]])
    for epoch in range(500):
        xs = [curves[r][epoch] for r in RUNS]
        good = [x for x in xs if not math.isnan(x)]
        mean = statistics.mean(good)
        sd = statistics.stdev(good) if len(good) > 1 else 0.0
        w.writerow([epoch, *xs, mean, sd, *[counts[r][epoch] for r in RUNS]])

# Summary values used in the manuscript discussion.
mean_first_301 = [statistics.mean([curves[r][e] for r in RUNS]) for e in range(301)]
peak_value = max(mean_first_301)
peak_epoch = mean_first_301.index(peak_value)
selected_vals = [curves[r][SELECTED_EPOCH[r]] for r in RUNS]

summary = RESULTS / "summary.txt"
summary.write_text(
    f"Mean Pearson-r peak on epochs 0-300: epoch {peak_epoch}, r={peak_value:.12f}\n"
    f"Mean Pearson r at epoch 300: {mean_first_301[300]:.12f}\n"
    f"Selected-epoch correlations: {selected_vals}\n"
    f"Selected-epoch mean: {statistics.mean(selected_vals):.12f}\n"
    f"Selected-epoch sample SD: {statistics.stdev(selected_vals):.12f}\n"
)

print(f"Wrote {curve_csv}")
print(f"Wrote {summary}")
