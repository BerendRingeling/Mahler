# v2_lower_bound_validation reproducibility bundle

This bundle reproduces the manuscript figure file `v2_lower_bound_validation.pdf` from the three raw runs.

## Contents

### Raw runs
- `raw/1000261_12.zip`
- `raw/1000261_13.zip`
- `raw/1000261_14.zip`

These are the three base-2 runs with inputs `k` and `factor(Delta_k)` and target `v2(n_k)`.

### Scripts
- `scripts/analyze_v2_lower_bound.py` reads the raw ZIPs directly and computes the validation lower-bound curve and the untouched-test lower-bound proportions at the validation-selected checkpoints.
- `scripts/make_v2_lower_bound_validation.py` produces the final PDF/PNG figure from the curve CSV.

### Derived results
- `results/lower_bound_curve.csv`
- `results/selected_checkpoint_lower_bound.csv`
- `results/v2_lower_bound_validation.pdf`
- `results/v2_lower_bound_validation.png`

## Reproduce everything

From the bundle root:

```bash
python scripts/analyze_v2_lower_bound.py
python scripts/make_v2_lower_bound_validation.py
```

Dependencies: Python 3, `sympy`, `numpy`, and `matplotlib`.

## Dataset naming

In these run archives, `eval.test1...` is the paper's validation set and `eval.valid...` is the untouched test set.

## Quantity plotted

For each validation example and epoch, the stored greedy prediction is a prediction of `v_2(r_k)`. We convert it to `v_2(n_k) = -v_2(r_k)` and check whether it satisfies the conjectural lower bound

`hat v(k) <= v_2(n_k)`, with

`hat v(k) = omega_odd(k) + 2 omega_odd(k^2 - 16) - c(k)`,

where `c(k)=4` when `k` is odd or `16 | k`, and `c(k)=2` otherwise.

The curve is the mean percentage over the three runs; the shaded band is one sample standard deviation.
