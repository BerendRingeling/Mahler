#!/usr/bin/env python3
"""Rebuild the test-set L(E_k,2) distribution figure from saved predictions.

Example:
  python3 build_l2_distribution.py --inputs inputs --out rebuilt

No model inference or L-function evaluation is performed. Reference L-values
are reconstructed from the stored rational candidates and the paper's formulas.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

from paper_plot_style import (
    BLUE, ORANGE, FIGURE_SIZE, apply_paper_style, paper_style_metadata,
)

RUNS = ("1000262_18", "1000262_19", "1000262_20")
ROLES = {"test1": "validation", "valid": "test"}
BIN_WIDTH = 0.02
FULL_MAX = 9.0
MAIN_MAX = 2.0  # Fixed diagnostic threshold, retained from the numerical audit.
DISPLAY_RANGE = (0.4, 1.8)
PLOT_BASENAME = "rk_l2_distribution_test"


def load_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mahler_measure(k):
    """Evaluate the paper's HG series, for positive integer k > 4.

    The 4F3 series starts at 1 and successive terms have ratio
    z*(j+3/2)^2*(j+1)/(j+2)^3, where z=16/k^2<1. Each ratio is
    less than z, so the omitted positive series tail is at most
    last_term*z/(1-z). This bounds series truncation, not rounding.
    All supplied test k are at least 81. The independent checker
    uses 80-digit Decimal arithmetic and a separately written series.
    """
    if not isinstance(k, int) or k <= 4:
        raise ValueError("This implementation requires integer k > 4")
    z = 16.0 / (k * k)
    term, terms = 1.0, [1.0]
    for j in range(10000):
        term *= z * (j + 1.5) ** 2 * (j + 1) / (j + 2) ** 3
        terms.append(term)
        tail = term * z / (1.0 - z)
        if tail < 1e-17:
            break
    else:
        raise ArithmeticError("Hypergeometric series did not converge")
    value = math.log(k) - 2.0 / (k * k) * math.fsum(terms)
    return value, len(terms), tail


def distribution_summary(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    return {
        "examples": len(values),
        "minimum": min(values), "maximum": max(values),
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "median": statistics.median(values),
        "quantiles_numpy_linear": {
            str(p): float(np.quantile(a, p)) for p in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)
        },
        "below_0_5": sum(v < 0.5 for v in values),
        "above_2": sum(v > 2.0 for v in values),
        "below_0": sum(v < 0 for v in values),
        "zero": sum(v == 0 for v in values),
        "nonfinite": sum(not math.isfinite(v) for v in values),
    }


def check_selections(inputs, snapshots):
    result = {}
    for run in RUNS:
        with (inputs / "per_run" / run / "per_epoch_metrics.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        validation = [r for r in rows if r["split"] == "test1"
                      and 1 <= int(r["epoch_number"]) <= 300]
        assert len(validation) == 300
        assert {int(r["epoch_number"]) for r in validation} == set(range(1, 301))
        maximum = max(int(r["greedy_factor_two_count"]) for r in validation)
        ties = sorted(int(r["epoch_number"]) for r in validation
                      if int(r["greedy_factor_two_count"]) == maximum)
        chosen = ties[0]
        test_rows = [r for r in rows if r["split"] == "valid"
                     and int(r["epoch_number"]) == chosen]
        assert len(test_rows) == 1
        test_row = test_rows[0]
        for label in ROLES:
            record = snapshots["runs"][run][label]
            assert record["selected_completed_epoch"] == chosen
            assert record["selected_epoch_index"] == chosen - 1
            assert record["paper_role"] == ROLES[label]
            assert len(record["records"]) == record["examples"] == 3000
        result[run] = {
            "completed_epoch": chosen, "epoch_index": chosen - 1,
            "validation_maximum_count": maximum, "tied_completed_epochs": ties,
            "test_factor_two_count_from_original_audit": int(test_row["greedy_factor_two_count"]),
            "denominator": 3000,
        }
    return result


def make_plot(histograms, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    apply_paper_style()
    edges = np.array([r["bin_left"] for r in histograms] + [histograms[-1]["bin_right"]])
    widths = np.diff(edges)
    reference = np.array([r["reference_count"] for r in histograms])
    mean = np.array([r["model_mean_count"] for r in histograms])
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, constrained_layout=True)
    # Retain the approved gray-reference/blue-model overlay. Solid fills
    # avoid the unintended brown produced by transparent blue and orange.
    # The shared Section 3 typography and axes remain in place.
    ax.bar(edges[:-1], reference, width=widths, align="edge",
           color="#c3c7cb", edgecolor="#9fa5aa", linewidth=0.35, alpha=1.0,
           label="Reference", zorder=2)
    ax.bar(edges[:-1] + 0.025 * widths, mean, width=0.95 * widths,
           align="edge", color=BLUE, edgecolor="white",
           linewidth=0.45, alpha=1.0, label="Model (mean of 3 runs)", zorder=3)
    ymax = math.ceil(max(float(reference.max()), float(mean.max())) * 1.06 / 50) * 50
    ax.set(xlabel=r"$L(E_k,2)$", ylabel="Count per run",
           xlim=DISPLAY_RANGE, ylim=(0, ymax))
    ax.set_xticks(np.arange(DISPLAY_RANGE[0], DISPLAY_RANGE[1] + 0.01, 0.2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.legend(loc="upper right")
    fig.savefig(output / (PLOT_BASENAME + ".pdf"),
                metadata={"Creator": "build_l2_distribution.py", "CreationDate": None})
    fig.savefig(output / (PLOT_BASENAME + ".png"), dpi=200, facecolor="white")
    plt.close(fig)
    return {"matplotlib": matplotlib.__version__, "numpy": np.__version__,
            **paper_style_metadata(), "png_dpi": 200,
            "design": "Section 3 typography and axes; solid gray reference bars behind blue mean-model bars; count axis; no grid",
            "reference_color": "#c3c7cb", "model_color": BLUE,
            "reference_alpha": 1.0, "model_alpha": 1.0,
            "histogram_geometry": "Overlaid; full-width reference bars and centered 0.95-bin-width model bars",
            "inset": False, "sd_band": False}


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).resolve().parent / "inputs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    arithmetic = load_json(args.inputs / "evaluation_arithmetic_inputs.json")
    snapshots = load_json(args.inputs / "factor_two_selected_predictions.json")
    assert arithmetic["file_label_to_paper_role"] == ROLES
    assert snapshots["file_label_to_paper_role"] == ROLES
    assert snapshots["policy"]["test_used_for_selection"] is False
    selected = check_selections(args.inputs, snapshots)
    pool = arithmetic["by_file_label"]["valid"]
    validation_pool = arithmetic["by_file_label"]["test1"]
    assert len(pool) == len(validation_pool) == 3000
    assert len({x["k"] for x in pool}) == len(pool)
    assert len({x["example_id"] for x in pool}) == len(pool)
    assert not ({x["k"] for x in pool} & {x["k"] for x in validation_pool})
    reference_rows, reference_values = [], []
    for item in pool:
        k, w, N = item["k"], item["w"], item["N"]
        assert w in (-1, 1) and N > 0
        target = Fraction(*item["answer"])
        assert target == Fraction(*item["stored_answer"]) and target != 0
        mu, terms, tail = mahler_measure(k)
        L = 4 * math.pi ** 2 * mu * float(Fraction(w, N) / target)
        assert math.isfinite(L) and L > 0
        reference_values.append(L)
        reference_rows.append({
            "example_id": item["example_id"], "k": k, "w": w, "N": N,
            "target_numerator": target.numerator, "target_denominator": target.denominator,
            "mu": mu, "reference_L2": L,
            "hypergeometric_terms": terms, "hypergeometric_truncation_tail_bound": tail,
        })
    pred_rows, predictions, run_summaries = [], {}, {}
    ref_min, ref_max = min(reference_values), max(reference_values)
    for run in RUNS:
        records = snapshots["runs"][run]["valid"]["records"]
        values, relative_errors, absolute_errors, ratio_logs = [], [], [], []
        correct = 0
        for item, reference, snapshot in zip(pool, reference_rows, records):
            for key in ("example_id", "k", "w", "N", "answer", "stored_answer"):
                assert item[key] == snapshot[key], (run, key, item["example_id"])
            assert not snapshot["missing"] and not snapshot["invalid"]
            guess = Fraction(*snapshot["canonical_guess_greedy"])
            assert guess == Fraction(*snapshot["guess_greedy"]) and guess != 0
            target = Fraction(*item["answer"])
            assert guess * target > 0
            ratio = target / guess
            factor_two = Fraction(1, 2) < guess / target < 2
            assert factor_two == snapshot["factor_two_correct"]
            correct += factor_two
            L = 4 * math.pi ** 2 * reference["mu"] * float(Fraction(item["w"], item["N"]) / guess)
            assert math.isfinite(L) and L > 0
            assert math.isclose(L / reference["reference_L2"], float(ratio), rel_tol=1e-14)
            values.append(L)
            relative_errors.append(float(abs(ratio - 1)))
            absolute_errors.append(abs(L - reference["reference_L2"]))
            ratio_logs.append(math.log2(float(ratio)))
            pred_rows.append({
                "run_id": run, "selected_completed_epoch": selected[run]["completed_epoch"],
                "example_id": item["example_id"], "k": item["k"],
                "guess_numerator": guess.numerator, "guess_denominator": guess.denominator,
                "reference_L2": reference["reference_L2"], "induced_L2": L,
                "induced_over_reference": float(ratio),
                "absolute_relative_error": float(abs(ratio - 1)),
                "factor_two_correct": int(factor_two),
            })
        assert len(values) == 3000
        assert correct == selected[run]["test_factor_two_count_from_original_audit"]
        predictions[run] = values
        run_summaries[run] = {
            **distribution_summary(values), "selection": selected[run],
            "test_factor_two_count_recomputed": correct,
            "test_factor_two_accuracy_percent": 100 * correct / 3000,
            "below_reference_minimum": sum(v < ref_min for v in values),
            "above_reference_maximum": sum(v > ref_max for v in values),
            "median_absolute_relative_error": statistics.median(relative_errors),
            "p95_absolute_relative_error": float(np.quantile(relative_errors, 0.95)),
            "mean_absolute_error": statistics.mean(absolute_errors),
            "mean_log2_induced_over_reference": statistics.mean(ratio_logs),
        }
    edges = np.linspace(0, FULL_MAX, round(FULL_MAX / BIN_WIDTH) + 1)
    widths = np.diff(edges)
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    count_matrix = np.array([np.histogram(predictions[run], bins=edges)[0] for run in RUNS])
    assert int(reference_counts.sum()) == 3000
    assert all(int(x) == 3000 for x in count_matrix.sum(axis=1))
    densities = count_matrix / (3000 * widths)
    ref_density = reference_counts / (3000 * widths)
    mean = densities.mean(axis=0)
    sd = densities.std(axis=0, ddof=1)
    histograms = []
    for i in range(len(widths)):
        histograms.append({
            "bin_left": float(edges[i]), "bin_right": float(edges[i + 1]),
            "reference_count": int(reference_counts[i]), "reference_density": float(ref_density[i]),
            **{run + "_count": int(count_matrix[j, i]) for j, run in enumerate(RUNS)},
            "model_mean_count": float(count_matrix[:, i].mean()),
            "model_sample_sd_count": float(count_matrix[:, i].std(ddof=1)),
            "model_mean_density": float(mean[i]), "model_sample_sd_density": float(sd[i]),
        })
    for run in RUNS:
        values = predictions[run]
        run_summaries[run]["histogram_mass_at_or_below_2"] = sum(v <= MAIN_MAX for v in values) / 3000
    display_counts = {
        "reference": {
            "displayed": sum(DISPLAY_RANGE[0] <= v <= DISPLAY_RANGE[1] for v in reference_values),
            "below": sum(v < DISPLAY_RANGE[0] for v in reference_values),
            "above": sum(v > DISPLAY_RANGE[1] for v in reference_values),
        }
    }
    for run in RUNS:
        values = predictions[run]
        display_counts[run] = {
            "displayed": sum(DISPLAY_RANGE[0] <= v <= DISPLAY_RANGE[1] for v in values),
            "below": sum(v < DISPLAY_RANGE[0] for v in values),
            "above": sum(v > DISPLAY_RANGE[1] for v in values),
            "outside": sum(v < DISPLAY_RANGE[0] or v > DISPLAY_RANGE[1] for v in values),
            "displayed_fraction": sum(DISPLAY_RANGE[0] <= v <= DISPLAY_RANGE[1] for v in values) / 3000,
        }
    input_files = [args.inputs / "evaluation_arithmetic_inputs.json",
                   args.inputs / "factor_two_selected_predictions.json"]
    input_files += [args.inputs / "per_run" / run / "per_epoch_metrics.csv" for run in RUNS]
    summary = {
        "policy": {
            "paper_role": "test", "file_label": "valid", "examples": 3000,
            "model_inputs": ["k", "w", "D_Factor"], "decoder": "greedy",
            "selection": "Earliest maximum validation factor-two count among completed epochs 1--300, separately per run",
            "test_used_for_selection": False,
            "reference_formula": "omega*4*pi^2*mu_k/(N_k*r_k)",
            "induced_formula": "omega*4*pi^2*mu_k/(N_k*s_k)",
            "exact_ratio_identity": "induced/reference = r_k/s_k",
            "reference_origin": "Stored rational candidates, not a new independent L-function evaluation",
            "N_role": "Reference conductor used in postprocessing; the model input is its stated discriminant factorization",
            "no_prediction_omissions": True,
            "reference_histogram_counted_once": True,
            "model_summary": "Plotted mean bin counts across three runs; sample SD (ddof=1) retained in data tables",
            "sd_is_not_confidence_interval": True,
        },
        "test_k_range": [min(x["k"] for x in pool), max(x["k"] for x in pool)],
        "reference": distribution_summary(reference_values),
        "runs": run_summaries,
        "histogram": {
            "bin_width": BIN_WIDTH, "number_of_bins": len(widths), "full_range": [0, FULL_MAX],
            "main_display_range": list(DISPLAY_RANGE), "inset": None, "plotted_quantity": "Bin count, model averaged across three runs",
            "normalization": "Each run and reference divided by 3000 and bin width over the full range",
            "no_renormalization_of_main_display": True,
            "reference_density_integral": float(np.sum(ref_density * widths)),
            "model_mean_density_integral": float(np.sum(mean * widths)),
            "display_window_counts": display_counts,
            "reference_total_count": int(reference_counts.sum()),
            "per_run_total_counts": {run: int(count_matrix[j].sum()) for j, run in enumerate(RUNS)},
            "sd_displayed": False,
            "full_range_counts_preserved_in_table": True,
        },
        "input_sha256": {str(f.relative_to(args.inputs)): hashlib.sha256(f.read_bytes()).hexdigest() for f in input_files},
        "python_version": sys.version,
    }
    write_csv(args.out / "l2_reference_values.csv", reference_rows)
    write_csv(args.out / "l2_predictions.csv", pred_rows)
    write_csv(args.out / "l2_histogram_counts.csv", histograms)
    if not args.no_plot:
        summary["plot_environment"] = make_plot(histograms, args.out)
    write_json(args.out / "distribution_summary.json", summary)
    print(json.dumps({"reference_examples": len(reference_rows), "prediction_rows": len(pred_rows),
                      "selected_epochs": {r: selected[r]["completed_epoch"] for r in RUNS},
                      "figure": None if args.no_plot else PLOT_BASENAME + ".pdf"}, indent=2))


if __name__ == "__main__":
    main()
