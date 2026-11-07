# Figure 5.9 reproducibility bundle

This bundle reproduces the revised three-run Figure 5.9 for the p=2 experiment with inputs k and factor(Delta_k).

## Raw runs

- `raw/1000261_12.zip`
- `raw/1000261_13.zip`
- `raw/1000261_14.zip`

The exported filenames use Alberto's historical convention:

- `eval.test1.epochs_0_499.json.bz2` = the paper's VALIDATION set
- `eval.valid.epochs_0_499.json.bz2` = the paper's TEST set

## Reproduce the numerical curve

From this directory, run:

```bash
python analyze_v2_fd_runs.py raw/1000261_12.zip raw/1000261_13.zip raw/1000261_14.zip --outdir results
```

This writes:

- `results/v2_fd_validation_curve.csv`
- `results/selected_checkpoints.csv`
- `results/summary.txt`

The selected checkpoint for each run is the earliest epoch attaining the maximum validation greedy exact-match accuracy among epochs 1 through 300. The test set is used only after that epoch has been fixed.

## Reproduce Figure 5.9

```bash
python make_v2_fd_validation_accuracy.py --csv results/v2_fd_validation_curve.csv --outdir results
```

This writes:

- `results/v2_fd_validation_accuracy.pdf`
- `results/v2_fd_validation_accuracy.png`

## Expected selected-checkpoint test results

- 1000261_12: epoch 261; test 37.03%
- 1000261_13: epoch 221; test 31.37%
- 1000261_14: epoch 203; test 36.00%
- mean test accuracy: 34.80%
- sample standard deviation: 3.02 percentage points
