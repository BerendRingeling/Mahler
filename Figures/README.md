# Figures

The PDF files directly in each numbered folder match the figure files referenced in the revised manuscript. Figure 3.2 is drawn in TikZ in the manuscript. Its folder includes the TikZ source and a high-resolution PDF extract of the diagram.

Earlier PDFs and notebooks are in the relevant `earlier/` folders. The supplied historical scripts are kept for reference and may require the inputs from their original audit packages.

| Figure | PDF files |
| --- | --- |
| 3.1 | [L_Ek_2_full.pdf](Figure%203.1/L_Ek_2_full.pdf) |
| 3.2 | [parity_set_inclusions.pdf](Figure%203.2/parity_set_inclusions.pdf) |
| 3.3 | [correlation_v2.pdf](Figure%203.3/correlation_v2.pdf) |
| 3.4 | [v2_data_vs_model.pdf](Figure%203.4/v2_data_vs_model.pdf) |
| 5.1 | [rk_root_number_validation_corrected.pdf](Figure%205.1/rk_root_number_validation_corrected.pdf) |
| 5.2 | [rk_conductor_factorization_validation_corrected.pdf](Figure%205.2/rk_conductor_factorization_validation_corrected.pdf) |
| 5.3 | [rk_l2_distribution_test_corrected.pdf](Figure%205.3/rk_l2_distribution_test_corrected.pdf) |
| 5.4 | [v3_fd_validation_accuracy_corrected.pdf](Figure%205.4/v3_fd_validation_accuracy_corrected.pdf) |
| 5.5 | [v3_fd_conditional_accuracy_corrected.pdf](Figure%205.5/v3_fd_conditional_accuracy_corrected.pdf) |
| 5.6 | [v3_fd_set_a_accuracy_corrected.pdf](Figure%205.6/v3_fd_set_a_accuracy_corrected.pdf) |
| 5.7 | [v3_fd_sets_comparison_corrected.pdf](Figure%205.7/v3_fd_sets_comparison_corrected.pdf) |
| 5.8 | [v3_fd_complement_accuracy_corrected.pdf](Figure%205.8/v3_fd_complement_accuracy_corrected.pdf) |
| 5.9 | [v2_fd_validation_accuracy_corrected.pdf](Figure%205.9/v2_fd_validation_accuracy_corrected.pdf) |
| 5.10 | [v2_lower_bound_validation_corrected.pdf](Figure%205.10/v2_lower_bound_validation_corrected.pdf) |
| 5.11 | [v2_november_confusion_Phat.pdf](Figure%205.11/v2_november_confusion_Phat.pdf), [v2_november_confusion_truth.pdf](Figure%205.11/v2_november_confusion_truth.pdf) |
| 5.12 | [v2_pearson_Phat_validation_corrected.pdf](Figure%205.12/v2_pearson_Phat_validation_corrected.pdf) |
| 5.13 | [v2_november_correct_given_G0.pdf](Figure%205.13/v2_november_correct_given_G0.pdf) |
| 7.1 | [histogram_all_v3_0_250K.pdf](Figure%207.1/histogram_all_v3_0_250K.pdf) |
| 7.2 | [histogram_all_v2_0_250K.pdf](Figure%207.2/histogram_all_v2_0_250K.pdf) |

## Regenerating the revised Section 5 figures

Download and unpack the public experiment archive so that the run families are in `Figures/MahlerExperiments/`. From `Figures/`, use Python with NumPy, pypdf and ReportLab installed.

```sh
python reproduction/regenerate.py MahlerExperiments --output-dir regenerated-figures
```

Use `sage -python` in place of `python` to run with SageMath's Python environment. The command reads the archived predictions and uses the included vector plot templates and valuation baselines. It writes the 14 revised Section 5 PDFs to the specified output directory.

The SageMath scripts for Figures 3.1, 3.3, 3.4, 7.1 and 7.2 use `MahlerExperiments/data/output_full.txt`. Run each script from its own numbered folder. The original notebooks are alongside the scripts.

The experiment archive is distributed separately.
