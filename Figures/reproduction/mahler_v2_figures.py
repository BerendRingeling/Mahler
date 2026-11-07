"""Regenerate the two three-run p=2 plots using the vector styles in Mahler (20).

Usage (NumPy and pypdf required):
    sage -python mahler_v2_figures.py mahler_uploads.zip "Mahler (20).pdf" --output-dir .
"""
import argparse
import bz2
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from pypdf import PdfReader

from mahler_v3_figures import fs, update_plot


def read_validation(archive, run):
    directory = f"mahler_uploads/v2/v2_fD/1000261_{run}/"
    log = archive.read(directory + "train.log").decode()
    assert "tokenized_validate_base_2.data" in log and "Creating valid iterator" in log
    with archive.open(directory + "eval.valid.epochs_0_499.json.bz2") as member:
        with bz2.open(member, "rt", encoding="utf-8") as stream:
            rows = json.load(stream).values()

    exact, satisfies = [], []
    for row in rows:
        k = row["problem"]["k"]
        odd_count = lambda n: sum(p != 2 for p in fs(abs(n)))
        bound = odd_count(k) + 2 * (odd_count(k - 4) + odd_count(k + 4))
        bound -= 4 if k % 2 or k % 16 == 0 else 2
        numerator, denominator = row["answer"]
        correct = np.zeros(301, dtype=bool)
        respects_bound = np.zeros(22, dtype=bool)
        for epoch in range(301):
            prediction = row["guesses_greedy"].get(str(epoch))
            if not (
                isinstance(prediction, list) and len(prediction) == 2
                and all(isinstance(value, int) for value in prediction)
                and prediction[1]
            ):
                continue
            a, b = prediction
            correct[epoch] = a * denominator == b * numerator
            if epoch < 22:
                respects_bound[epoch] = -a / b >= bound
        exact.append(correct)
        satisfies.append(respects_bound)
    return 100 * np.mean(exact, axis=0), 100 * np.mean(satisfies, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive:
        accuracy, bound = map(np.array, zip(*(read_validation(archive, r) for r in (12, 13, 14))))
    assert np.array_equal(accuracy[:, :300].argmax(axis=1), [261, 272, 211])
    assert np.isclose(bound[:, 7].mean(), 99.7888888889)
    manuscript = PdfReader(args.manuscript)
    update_plot(manuscript, args.output_dir, "v2_fd_validation_accuracy", 32, 21, 10, 5,
                [accuracy], [636], [1434], tick_indices=(648, 682, 894, 928),
                min_path_points=250)
    update_plot(manuscript, args.output_dir, "v2_lower_bound_validation", 34, 23, 64, 4,
                [bound], [78], [800], tick_indices=(90, 124, 370, 404),
                x_tick_step=3, min_path_points=15, markers=(806, 850))


if __name__ == "__main__":
    main()
