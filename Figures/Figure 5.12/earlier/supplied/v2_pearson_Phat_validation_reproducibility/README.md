# `v2_pearson_Phat_validation` reproducibility bundle

This bundle reproduces the revised Pearson-correlation figure for the 2-adic experiment from the three raw runs:

- `1000261_12`
- `1000261_13`
- `1000261_14`

Paper naming convention used by the scripts:

- `eval.test1.epochs_0_499.json.bz2` = validation set
- `eval.valid.epochs_0_499.json.bz2` = untouched test set

The stored target/prediction is `v_2(r_k)`, so the scripts convert it to `v_2(n_k)=-v_2(r_k)`.

The deterministic predictor is

`P_hat(k) = round(v_hat(k) + s(k) + 1)`

with the definitions used in the manuscript. Pearson correlations are computed at each epoch between the greedy integer model predictions and `P_hat(k)` on the 3,000 validation examples.

## Requirements

Python 3 with:

- `sympy`
- `matplotlib`

## Reproduce

From the root of this bundle:

```bash
python scripts/analyze_v2_pearson_Phat.py
python scripts/make_v2_pearson_Phat_validation.py
```

This writes:

- `results/v2_pearson_Phat_validation_curve.csv`
- `results/summary.txt`
- `results/v2_pearson_Phat_validation.pdf`
- `results/v2_pearson_Phat_validation.png`

The plot uses epochs 0--300 to match the original figure window. The line is the mean over the three runs; the light band is one sample standard deviation on either side.
