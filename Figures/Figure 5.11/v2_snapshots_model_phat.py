#!/usr/bin/env python
# coding: utf-8

# In[1]:


# confusion_matrices_Phat.sage
#
# Display two grids of confusion-matrix snapshots over the 3000 validation k,
# with the Pearson correlation coefficient r shown in every panel.
#
# In both grids, the model prediction is on the x-axis:
#
#   (A) x = predicted v_2(n_k)  vs  y = true v_2(n_k)
#   (B) x = predicted v_2(n_k)  vs  y = P_hat(k)
#
# Nothing is saved to disk.
#
# Run:
#     sage confusion_matrices_Phat.sage

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


DATA_DIR = (
    os.path.dirname(os.path.abspath(__file__))
    if '__file__' in dir()
    else '.'
)

SNAP_EPOCHS = [0, 8, 40, 100, 210, 306]

GRID_WSPACE = float(0.06)
GRID_HSPACE = float(0.08)


def epoch_of(path):
    return int(path.rsplit('.', 1)[1])


files = sorted(
    glob.glob(
        os.path.join(
            DATA_DIR,
            '../MahlerExperiments/v2/v2_fD/v2_fD run 3 (best run)/eval.valid.elliptic_curve.*'
        )
    ),
    key=epoch_of
)

if not files:
    files = sorted(
        glob.glob(
            os.path.join(DATA_DIR, 'eval.valid.elliptic_curve.*')
        ),
        key=epoch_of
    )

if not files:
    raise RuntimeError(
        'No eval.valid.elliptic_curve.* files found in:\n%s'
        % DATA_DIR
    )


# ----------------------------------------------------------------------
# Deterministic predictor:
#
#     P_hat(k) = round(v_hat(k) + s(k) + 1)
# ----------------------------------------------------------------------

def omega_odd(m):
    m = ZZ(m).abs()

    if m == 0:
        return 0

    return sum(
        1 for p in m.prime_divisors()
        if p != 2
    )


def vhat(k):
    k = ZZ(k)

    c = 4 if (k % 2 == 1 or k % 16 == 0) else 2

    return (
        omega_odd(k)
        + 2 * (
            omega_odd(k - 4)
            + omega_odd(k + 4)
        )
        - c
    )


def s_func(k):
    k = ZZ(k)

    M = max(
        (k - 8).valuation(2),
        (k + 8).valuation(2)
    )

    return QQ(3) / 2 * max(
        1 - 2 ** (5 - M),
        0
    )


_pcache = {}


def P_hat(k):
    if k not in _pcache:
        value = vhat(k) + s_func(k) + 1
        _pcache[k] = float(round(float(value)))

    return _pcache[k]


# ----------------------------------------------------------------------
# Parse true value, model prediction and P_hat(k)
# ----------------------------------------------------------------------

BLOCK = re.compile(
    r"'k':\s*(?P<k>\d+)"
    r".*?answer=\[(?P<t>-?\d+)"
    r".*?greedy\s*\n\s*decoded=\[(?P<p>-?\d+)",
    re.S
)


def load(path):
    true_values = []
    predicted_values = []
    phat_values = []

    with open(path, 'r') as input_file:
        text = input_file.read()

    for match in BLOCK.finditer(text):
        k = int(match.group('k'))

        true_values.append(
            abs(int(match.group('t')))
        )

        predicted_values.append(
            abs(int(match.group('p')))
        )

        phat_values.append(
            P_hat(k)
        )

    if not true_values:
        print('Warning: no examples found in %s' % path)

    return (
        np.array(true_values, dtype=float),
        np.array(predicted_values, dtype=float),
        np.array(phat_values, dtype=float)
    )


data = {
    epoch_of(path): load(path)
    for path in files
}

# Remove files for which no examples were parsed.
data = {
    epoch: values
    for epoch, values in data.items()
    if len(values[0]) > 0
}

if not data:
    raise RuntimeError(
        'Files were found, but no examples could be parsed.'
    )


# ----------------------------------------------------------------------
# Common integer grid
# ----------------------------------------------------------------------

V = int(
    max(
        max(
            true_values.max(),
            predicted_values.max(),
            np.round(phat_values).max()
        )
        for true_values, predicted_values, phat_values
        in data.values()
    )
)

edges = np.arange(-0.5, V + 1.5, 1.0)

extent = [
    float(-0.5),
    float(V + 0.5),
    float(-0.5),
    float(V + 0.5)
]

cmap = plt.get_cmap('viridis').copy()

# Empty cells use the darkest value of the colormap.
cmap.set_bad(cmap(0.0))


def pearson(x, y):
    # Pearson correlation is undefined if one array is constant.
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')

    return float(np.corrcoef(x, y)[0, 1])


def hist2d(xvals, yvals):
    x_rounded = np.clip(
        np.round(xvals),
        0,
        V
    )

    y_rounded = np.clip(
        np.round(yvals),
        0,
        V
    )

    # Rows correspond to y and columns correspond to x.
    H, _, _ = np.histogram2d(
        y_rounded,
        x_rounded,
        bins=[edges, edges]
    )

    H[H == 0] = np.nan

    return H


# ----------------------------------------------------------------------
# Construct one snapshot grid
# ----------------------------------------------------------------------

def make_snapshot_grid(
    tag,
    xkey,
    ykey,
    xlabel,
    ylabel,
    title
):
    def xy(epoch):
        true_values, predicted_values, phat_values = data[epoch]

        choices = {
            'true': true_values,
            'pred': predicted_values,
            'phat': phat_values
        }

        return choices[xkey], choices[ykey]

    snapshots = [
        epoch for epoch in SNAP_EPOCHS
        if epoch in data
    ]

    if not snapshots:
        raise RuntimeError(
            'None of the requested snapshot epochs are available.\n'
            'Requested: %s\n'
            'Available: %s'
            % (SNAP_EPOCHS, sorted(data))
        )

    max_count = float(
        max(
            np.nanmax(hist2d(*xy(epoch)))
            for epoch in snapshots
        )
    )

    def draw(axis, epoch):
        x_values, y_values = xy(epoch)

        image = axis.imshow(
            hist2d(x_values, y_values),
            origin='lower',
            extent=extent,
            cmap=cmap,
            norm=LogNorm(
                vmin=float(1),
                vmax=max_count
            ),
            aspect='equal'
        )

        axis.plot(
            [float(-0.5), float(V + 0.5)],
            [float(-0.5), float(V + 0.5)],
            color='red',
            linewidth=float(1)
        )

        axis.set_xlim(
            float(-0.5),
            float(V + 0.5)
        )

        axis.set_ylim(
            float(-0.5),
            float(V + 0.5)
        )

        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)

        r = pearson(x_values, y_values)

        axis.set_title(
            r'epoch %d   ($r = %.2f$)'
            % (epoch, r)
        )

        return image

    number_of_columns = min(3, len(snapshots))

    number_of_rows = int(
        np.ceil(
            len(snapshots)
            / float(number_of_columns)
        )
    )

    figure, axes = plt.subplots(
        number_of_rows,
        number_of_columns,
        figsize=(
            float(4.6 * number_of_columns),
            float(4.3 * number_of_rows)
        ),
        layout='constrained',
        num=title
    )

    axes_flat = np.atleast_1d(axes).ravel()

    image = None

    for axis, epoch in zip(axes_flat, snapshots):
        image = draw(axis, epoch)

    for axis in axes_flat[len(snapshots):]:
        axis.axis('off')

    try:
        figure.get_layout_engine().set(
            wspace=float(GRID_WSPACE),
            hspace=float(GRID_HSPACE)
        )
    except Exception:
        pass

    figure.colorbar(
        image,
        ax=axes_flat.tolist(),
        shrink=float(0.7),
        label='count (log scale)'
    )

    figure.suptitle(title)

    return figure


# ----------------------------------------------------------------------
# Create both figures
# ----------------------------------------------------------------------

make_snapshot_grid(
    'true_vs_pred',
    xkey='pred',
    ykey='true',
    xlabel=r'predicted $v_2(n_k)$',
    ylabel=r'true $v_2(n_k)$',
    title=r'True vs. predicted $v_2(n_k)$'
)

make_snapshot_grid(
    'pred_vs_Phat',
    xkey='pred',
    ykey='phat',
    xlabel=r'predicted $v_2(n_k)$',
    ylabel=r'$\hat{P}(k)$',
    title=r'Model prediction vs. predictor $\hat{P}(k)$'
)


# Display both snapshot grids and save nothing.
plt.show()


# In[ ]:




