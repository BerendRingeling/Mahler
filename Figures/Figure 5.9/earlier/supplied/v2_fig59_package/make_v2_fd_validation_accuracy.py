#!/usr/bin/env python3
"""Plot revised Figure 5.9 from v2_fd_validation_curve.csv."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, default=Path('v2_fd_validation_curve.csv'))
    ap.add_argument('--outdir', type=Path, default=Path('.'))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    epoch, mean, sd = [], [], []
    with args.csv.open(newline='') as f:
        for row in csv.DictReader(f):
            e = int(row['epoch'])
            if e > 300:
                continue
            epoch.append(e)
            mean.append(float(row['mean']))
            sd.append(float(row['sample_sd']))

    lo = [m-s for m,s in zip(mean,sd)]
    hi = [m+s for m,s in zip(mean,sd)]

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.fill_between(epoch, lo, hi, color='#1f77b4', alpha=0.09, linewidth=0)
    ax.plot(epoch, mean, color='#1f77b4', linewidth=1.35,
            marker='o', markersize=1.2, markeredgewidth=0,
            label='greedy exact-match')

    ax.set_title('Validation greedy accuracy vs. epoch', fontsize=13)
    ax.set_xlabel('epoch', fontsize=11)
    ax.set_ylabel('validation greedy accuracy (%)', fontsize=11)
    ax.set_xlim(0, 300)
    ax.set_ylim(10, 41)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    ax.set_yticks([10, 15, 20, 25, 30, 35, 40])
    ax.grid(True, color='0.84', linewidth=0.8)
    ax.legend(loc='lower right', frameon=True, framealpha=1.0, fontsize=10)
    fig.tight_layout()

    pdf = args.outdir / 'v2_fd_validation_accuracy.pdf'
    png = args.outdir / 'v2_fd_validation_accuracy.png'
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == '__main__':
    main()
