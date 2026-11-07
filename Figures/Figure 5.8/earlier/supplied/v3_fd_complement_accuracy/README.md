# Figure 5.8: accuracy outside C'' and the mod-27 classes

This package replaces the paragraph beginning `Finally, we consider the
complement` and the following figure with label `fig:outside` in the current
manuscript. Stop at that figure's `\end{figure}`. The following paragraph
beginning `After the experiment giving` is outside this edit.

## Files to use in the paper

- `v3_fd_complement_accuracy.pdf`: vector figure for LaTeX.
- `v3_fd_complement_accuracy.png`: preview of the same figure.
- `figure_5_8_replacement.tex`: the complete replacement paragraph and figure.

Place the PDF alongside the paper's existing figure PDFs and use the supplied
replacement. It retains `\label{fig:outside}` and uses
`width=0.65\linewidth`. The manuscript itself has not been modified.

## What is being measured

The population consists of parameters satisfying BOTH:

1. k is outside C''.
2. k modulo 27 is not 0, 5 or 22.

Accuracy is then restricted to examples whose TRUE valuation v3(n_k) is 0
or 1. Predictions outside these two values are not excluded from the
denominator or converted to a binary label: they count as incorrect.
Incorrect, missing or invalid predictions count as failures.

C'' has the following definition from the manuscript:

- p^2 does not divide k^2-16 for every prime p congruent to 3 modulo 4;
- k is not congruent to 4 modulo 8;
- if p^3 divides k(k^2-16), then p is congruent to 2 modulo 3.

No additional k >= 5 restriction is introduced. The complement of C''
need not lie outside every other family considered in the paper.

The plotted curve is GREEDY EXACT validation accuracy, averaged across the
three runs at each completed epoch from 1 to 500. The band is one sample
standard deviation across those three percentages, with divisor 3-1. It is
not a confidence interval. No smoothing is used.

## Evaluation roles and fixed checkpoint choices

The export labels do not themselves specify the roles used in the paper.
The verified mapping is:

| Export filename label | Role in the manuscript |
| --- | --- |
| test1 | validation |
| valid | test |

Each pool contains 3,000 examples, identical across runs. The validation and
test pools have disjoint k values. Raw model targets are v3(r_k); negating
the raw target and decoded prediction gives the paper's v3(n_k).

The selected checkpoints are the same global validation-selected
checkpoints already used in the preceding figures. They were selected
within completed epochs 1 to 300. There is no new selection for this subset.

| Run | Selected completed epoch | JSON epoch key |
| --- | ---: | ---: |
| 1000260_9 | 219 | 218 |
| 1000260_10 | 279 | 278 |
| 1000260_11 | 297 | 296 |

This analysis scores validation predictions at all 500 epochs and test
predictions only at those existing selected checkpoints. The original
compressed exports contain predictions for both pools at every epoch;
the analysis does not claim that these test predictions were first
generated after checkpoint selection. Training examples and checkpoints
are not required to reproduce these figure metrics from the saved outputs.

## Checked results

| Population | Examples | True 0 | True 1 | Constant-1 accuracy |
| --- | ---: | ---: | ---: | ---: |
| Restricted validation complement | 551 | 276 | 275 | 49.9092558984% |
| Restricted test complement | 540 | 266 | 274 | 50.7407407407% |

At the selected checkpoints:

| Run | Validation correct / 551 | Test correct / 540 | Test accuracy |
| --- | ---: | ---: | ---: |
| 1000260_9 | 372 | 367 | 67.9629629630% |
| 1000260_10 | 369 | 369 | 68.3333333333% |
| 1000260_11 | 375 | 363 | 67.2222222222% |

The selected test result is 67.8395061728% with sample SD
0.5657500858 percentage points: (67.84 +/- 0.57)% in the paper.
The mean is 17.0987654321 percentage points above the matched constant-1
test baseline. The selected validation mean is 67.5136116152%, with
sample SD 0.5444646098 percentage points.

The curve rises in earlier training and declines later. Mean validation
accuracy is 62.7949183303% at epoch 300, 56.5033272837% at epoch 400,
and 46.9449485783% at epoch 500. Descriptive, fixed windows of epochs
200-300 and 400-500 have means 63.4328224111% and 50.6022653082%,
respectively. Both windows include 101 epochs; these windows were not
used to select any checkpoint, and epochs are not treated as independent
replicates. The old assertion that accuracy is still increasing at the end
has therefore been removed.

The result supports accuracy above the stated baseline on these examples
within the computed range. It does not identify a new arithmetic rule or
establish extrapolation to larger k.

## Appearance

The figure preserves the original 9.2 by 5.2 inch canvas, DejaVu Sans
fonts, brown (#8c564b) 2-point line without markers, grey dashed baseline,
light grey grid, and 40-100% vertical range. The legend has been moved
from the lower right to the empty upper right because the new declining
curve reaches the original legend area. The new mean/SD display matches
the three-run reporting used in the preceding revised figures.

The caption explicitly states that the vertical axis starts at 40%.
Five early mean values fall below the displayed range. A few band values
also fall below it, including late lower bounds. Every value is retained
in the CSV files; no data are deleted or altered to match the old shape.
`figure_style_and_data_audit.json` records the exact affected epochs and
plotting parameters. The band is bounded to 0-100% before axis clipping.

## Reproduce the figure

From the extracted package directory, with Python 3, NumPy and Matplotlib:

```bash
python3 build_complement_figure.py
```

This redraws the PDF and PNG from `complement_validation_curves.csv`.
The installed package versions used for the supplied plot are recorded in
`figure_style_and_data_audit.json`. No trained model or GPU is needed.

## Reproduce the statistical calculations

The main calculation uses an exact partition of previously audited
counts. The reference CSVs are included under `references/`:

```bash
python3 analyze_complement.py --references references --out recomputed
```

For each pool and checkpoint, true-0/1 examples split into C'', the special
congruence classes, and the complement. C'' and the special classes are
disjoint on all 6,000 evaluation examples. The complement correct count is
the global true-0/1 correct count minus the C'' correct count minus the
special-classes TRUE-1 correct count. There are no true-0 examples in the
special classes in these two pools. Subtracting the special classes'
unrestricted correct count would be wrong because some of their correctly
predicted higher-valued examples fall outside the conditional population.
The main script checks 1,503 count identities.

An independent calculation checks membership directly using prime-square
and prime-cube divisibility, checks targets against the stored rational
answers, and scores the original saved greedy predictions directly:

```bash
python3 independent_check/audit_complement.py --inputs raw_inputs --out raw_recomputed
```

The six original compressed prediction exports and the arithmetic input
index are under `raw_inputs/`. The index supplies k and the stored rational
answer for each example ID. It is an input index, not a new model result.
The independent outputs, comparison script, comparison report and exact
reproduction instructions are under `independent_check/`.

The direct calculation and the partition calculation agree. Full
membership lists, per-run counts, all 500 validation means and sample
standard deviations, selected test counts, fixed descriptive windows, and
source hashes are supplied. The LaTeX replacement was compiled in an
isolated amsart layout with the manuscript's text width; there were no
unresolved references or overfull boxes in the final proof.

## File guide

- `complement_membership.csv`: all 6,000 evaluation examples and subset flags.
- `complement_per_epoch_metrics.csv`: 1,500 validation rows and 3 selected test rows.
- `complement_selected_metrics.csv`: selected validation and test counts.
- `complement_validation_curves.csv`: 500 means, sample SDs and per-run counts.
- `complement_validation_windows.csv`: fixed descriptive window means by run.
- `complement_summary.json`: definitions, exact aggregates, checks and input hashes.
- `source_excerpts/`: original Figure 5.8, original LaTeX block, edit boundaries
  and style notes, retained for comparison only.
- `package_manifest.json`: file sizes and SHA-256 hashes for this package.
