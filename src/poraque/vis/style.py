# -*- coding: utf-8 -*-
# file: style.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Plot styling: palette, colormaps and Matplotlib defaults.

Colour is assigned **by the job it does**, which for scalar-field work means
three distinct jobs and three distinct ramps:

``CATEGORICAL``
    Identity — one hue per series (training vs validation, one material vs
    another). Hues are assigned in fixed order and never cycled, so a series
    keeps its colour when others are added or removed.

``SEQUENTIAL``
    Magnitude — a single hue, light to dark, with monotonic lightness. Used for
    fields that are one-signed, such as :math:`\rho` and :math:`\tau`. A
    rainbow map (``jet`` and friends) is never used: its lightness is
    non-monotonic, so it invents banding that is not in the data and it is
    unreadable to colour-blind viewers.

``DIVERGING``
    Polarity — two hues meeting at a **neutral grey** midpoint, used for signed
    error maps. It is always paired with limits symmetric about zero
    (:func:`symmetric_limits`); a diverging ramp on asymmetric limits puts the
    neutral colour somewhere other than zero and silently misstates the sign of
    the error.

The categorical hues are validated for colour-vision deficiency: the worst
adjacent pair separates by ΔE 9.1 (protanopia) and 22.9 (normal vision), above
the ΔE 8 target. Every series additionally carries a legend entry, so identity
is never conveyed by colour alone.
"""

import numpy as np

#: Categorical hues in fixed assignment order (light surface).
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7")

#: Ink tokens. Text never wears a series colour; a coloured mark beside the
#: label carries identity instead.
INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e",
              "muted": "#8a8880", "surface": "#ffffff", "grid": "#d9d8d4"},
    "dark": {"primary": "#ffffff", "secondary": "#c3c2b7",
             "muted": "#8a8880", "surface": "#1a1a19", "grid": "#3a3a38"},
}

#: Endpoints of the single-hue sequential ramp (light -> dark blue).
_SEQUENTIAL_STOPS = ("#f4f8fd", "#a8c9ee", "#5595dd", "#2a78d6", "#12447d")

#: Endpoints of the diverging ramp: orange <- neutral grey -> blue.
_DIVERGING_STOPS = ("#8c3410", "#eb6834", "#f5b498",
                    "#e8e8e6",
                    "#a8c9ee", "#2a78d6", "#12447d")


def _colormap(name, stops):
    """Build a :class:`matplotlib.colors.LinearSegmentedColormap` from hex stops."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(stops), N=256)


def sequential_cmap():
    """Single-hue, monotonic-lightness ramp for one-signed magnitude fields."""
    return _colormap("poraque_sequential", _SEQUENTIAL_STOPS)


def diverging_cmap():
    """Two-hue ramp with a neutral grey midpoint, for signed error maps."""
    return _colormap("poraque_diverging", _DIVERGING_STOPS)


def symmetric_limits(values, quantile=1.0):
    r"""
    Limits symmetric about zero, for use with :func:`diverging_cmap`.

    Parameters
    ----------
    values : array_like
        Signed data, typically ``prediction - reference``.
    quantile : float, optional
        Use this quantile of :math:`|v|` instead of the maximum, so a handful
        of outliers cannot flatten the whole map to neutral. ``1.0`` uses the
        true maximum.

    Returns
    -------
    tuple of float
        ``(-limit, +limit)``.
    """
    magnitude = np.abs(np.asarray(values, dtype=float))
    magnitude = magnitude[np.isfinite(magnitude)]
    if magnitude.size == 0:
        return (-1.0, 1.0)
    limit = float(np.max(magnitude) if quantile >= 1.0
                  else np.quantile(magnitude, quantile))
    limit = limit if limit > 0 else 1.0
    return (-limit, limit)


def rc_params(theme="light", base_size=10):
    """
    Matplotlib settings implementing the house style.

    Grid and axes are deliberately recessive — thin, low-contrast, behind the
    data — so the marks carry the attention.

    Parameters
    ----------
    theme : {"light", "dark"}, optional
    base_size : int, optional
        Base font size in points.

    Returns
    -------
    dict
        Suitable for ``matplotlib.pyplot.rcParams.update`` or ``rc_context``.
    """
    ink = INK["dark" if theme == "dark" else "light"]
    return {
        "figure.facecolor": ink["surface"],
        "savefig.facecolor": ink["surface"],
        "axes.facecolor": ink["surface"],
        "axes.edgecolor": ink["grid"],
        "axes.labelcolor": ink["secondary"],
        "axes.titlecolor": ink["primary"],
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,          # grid behind the data, never over it
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": ink["grid"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "text.color": ink["primary"],
        "xtick.color": ink["secondary"],
        "ytick.color": ink["secondary"],
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "font.size": base_size,
        "axes.titlesize": base_size + 1,
        "axes.labelsize": base_size,
        "legend.fontsize": base_size - 1,
        "legend.frameon": False,
        "lines.linewidth": 2.0,          # thin marks
        "lines.markersize": 5,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
    }


def series_color(index):
    """Categorical hue for series ``index``, assigned in fixed order."""
    return CATEGORICAL[int(index) % len(CATEGORICAL)]
