"""Plot styling matched to the paper's Section 3, Figures 3.1 and 3.4.

Those figures use DejaVu Sans, source font size 10 pt, source spine width
0.8 pt, a white background, no grid, and the Matplotlib default palette.
At width=0.65\\linewidth, a 9-inch source figure gives approximately the
same printed label size as the Section 3 figures included at scale 0.5.
This module changes presentation only; it contains no data or metrics.
"""

FIGURE_SIZE = (9.0, 5.0)
BLUE = "#1f77b4"
ORANGE = "#ff7f0e"
GREEN = "#2ca02c"
GRAY = "#666666"
LINE_WIDTH = 1.5
BAND_ALPHA = 0.15


def apply_paper_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def paper_style_metadata():
    return {
        "reference": "Manuscript Section 3, Figures 3.1 and 3.4",
        "font": "DejaVu Sans",
        "mathtext_fontset": "dejavusans",
        "source_font_size_pt": 10,
        "source_axes_label_size_pt": 10,
        "source_tick_label_size_pt": 10,
        "source_legend_size_pt": 10,
        "source_spine_width_pt": 0.8,
        "figure_size_inches": list(FIGURE_SIZE),
        "grid": False,
        "palette": [BLUE, ORANGE, GREEN],
        "latex_width": "0.65\\linewidth",
        "scientific_data_and_metric_policy_changed": False,
    }
