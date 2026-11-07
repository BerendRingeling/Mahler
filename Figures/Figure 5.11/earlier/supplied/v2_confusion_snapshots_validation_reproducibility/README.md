# v2_confusion_snapshots_validation

Reproducibility package for `v2_confusion_snapshots_validation.pdf`.

## Raw runs
- `1000261_12.zip`
- `1000261_13.zip`
- `1000261_14.zip`

The paper naming convention is:
- `eval.test1.epochs_0_499.json.bz2` = validation set
- `eval.valid.epochs_0_499.json.bz2` = untouched test set

## What is plotted
At epochs 0, 8, 40, 100, 210, and 300, the validation predictions from the three runs are pooled (9,000 predictions when all are valid).

Top panels: model prediction of `v2(n_k)` versus the deterministic predictor `P_hat(k)`.

Bottom panels: model prediction versus the true `v2(n_k)`.

The stored model output is `v2(r_k)`, so the scripts convert using `v2(n_k) = -v2(r_k)`.
Panel titles report the Pearson correlation computed on the same pooled observations shown in the heat map.

## Reproduce
From the package directory:

```bash
python scripts/analyze_v2_confusion_snapshots.py
python scripts/make_v2_confusion_snapshots_validation.py
```

Outputs are written to `results/`.
