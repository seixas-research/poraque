# -*- coding: utf-8 -*-
# file: parity.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Parity statistics accumulated over a whole dataset, in bounded memory.

A parity plot of one structure says how the operator does on that structure.
The one a report needs is over the whole split, and a split is too large to
hold: 97 platinum cells at 48³ are 11 million voxels, a Materials Project set
at the same resolution is billions, and Matplotlib given either as points does
not finish.

:class:`ParityAccumulator` keeps none of the voxels. For each structure it draws
a fixed-size random sample, and folds it into three things whose size does not
grow with the dataset:

* a **sparse 2D histogram** on a fixed fine grid --- a dict from bin to count ---
  in log space for positive values and linear space for all of them, so neither
  scale has to be decided before the last structure has been seen;
* **running sums** (:math:`n`, :math:`\sum x`, :math:`\sum y`, :math:`\sum x^2`,
  :math:`\sum y^2`, :math:`\sum(x-y)^2`, :math:`\sum|x-y|`) from which relative
  :math:`L^2`, :math:`R^2`, MAE and RMSE of the sample are exact;
* counts of structures, sampled voxels, and voxels left off the log histogram.

No range has to be known in advance either: a bin is an integer index at a fixed
width, so a value outside everything seen before simply occupies a new key.
:meth:`ParityAccumulator.histogram` rebins onto display edges shared by every
split, when the figure is drawn.

The sample is uniform **per structure**, not per voxel, so a 32-atom bulk cell and
a 55-atom nanoparticle weigh the same whatever their grids --- the question the
plot answers is how the operator does across structures.
"""

import numpy as np

#: Voxels drawn per structure.
SAMPLES_PER_STRUCTURE = 10_000

#: Fine bin width in log10 units: 1/200 of a decade.
LOG_BIN_WIDTH = 0.005

#: Fine linear bins span this many widths of the first structure's reference
#: range; later structures extend it with new keys.
LINEAR_BINS = 2000


class ParityAccumulator:
    """
    Reference against prediction, over many structures, without keeping either.

    Parameters
    ----------
    samples : int, optional
        Voxels drawn per structure, without replacement; every voxel of a
        smaller one.
    seed : int, optional
        The draw for the k-th structure added is seeded by ``(seed, k)``, so the
        same data in the same order gives the same plot.

    Examples
    --------
    >>> accumulator = ParityAccumulator(samples=1000)
    >>> rng = np.random.default_rng(0)
    >>> for _ in range(3):
    ...     x = rng.random((10, 10, 10)) + 0.1
    ...     accumulator.add(x, x * 1.01)
    >>> accumulator.structures, accumulator.count
    (3, 3000)
    >>> round(accumulator.metrics()["relative_l2"], 6)
    0.01
    """

    def __init__(self, samples=SAMPLES_PER_STRUCTURE, seed=0):
        self.samples = int(samples)
        self.seed = int(seed)
        self.structures = 0
        self.count = 0
        self.nonpositive = 0
        self._sums = np.zeros(7)        # x, y, x^2, y^2, (x-y)^2, |x-y|, xy
        self._log = {}
        self._linear = {}
        self._linear_width = None

    # ------------------------------------------------------------------ #
    def add(self, reference, prediction):
        """
        Fold one structure in.

        Parameters
        ----------
        reference, prediction : ScalarField or array_like
            Same shape. A field contributes its ``data``; a two-channel density
            should be reduced to the channel being plotted first.
        """
        x = np.asarray(getattr(reference, "data", reference),
                       dtype=float).ravel()
        y = np.asarray(getattr(prediction, "data", prediction),
                       dtype=float).ravel()
        if x.shape != y.shape:
            raise ValueError(f"reference has {x.size} values and prediction "
                             f"{y.size}; a parity pair is voxel for voxel.")
        rng = np.random.default_rng((self.seed, self.structures))
        self.structures += 1
        if x.size > self.samples:
            chosen = rng.choice(x.size, self.samples, replace=False)
            x, y = x[chosen], y[chosen]
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if not x.size:
            return

        difference = x - y
        self._sums += (x.sum(), y.sum(), (x * x).sum(), (y * y).sum(),
                       (difference ** 2).sum(), np.abs(difference).sum(),
                       (x * y).sum())
        self.count += int(x.size)

        if self._linear_width is None:
            span = float(x.max() - x.min())
            self._linear_width = (span if span > 0 else
                                  max(abs(float(x.max())), 1.0)) / LINEAR_BINS
        _fold(self._linear, np.floor(x / self._linear_width),
              np.floor(y / self._linear_width))

        positive = (x > 0) & (y > 0)
        self.nonpositive += int((~positive).sum())
        if positive.any():
            _fold(self._log,
                  np.floor(np.log10(x[positive]) / LOG_BIN_WIDTH),
                  np.floor(np.log10(y[positive]) / LOG_BIN_WIDTH))

    # ------------------------------------------------------------------ #
    def metrics(self):
        """Relative L2, R², MAE and RMSE over every sampled voxel."""
        n = self.count
        if not n:
            return {"relative_l2": float("nan"), "r2": float("nan"),
                    "mae": float("nan"), "rmse": float("nan"), "voxels": 0,
                    "structures": self.structures}
        sx, sy, sxx, syy, sdd, sad, _ = self._sums
        variance = sxx - sx * sx / n
        return {
            "relative_l2": float(np.sqrt(sdd / sxx)) if sxx > 0 else float("nan"),
            "r2": float(1.0 - sdd / variance) if variance > 0 else float("nan"),
            "mae": float(sad / n),
            "rmse": float(np.sqrt(sdd / n)),
            "voxels": int(n),
            "structures": int(self.structures),
        }

    def positive_share(self):
        """Share of sampled voxels the log histogram holds."""
        return 1.0 - self.nonpositive / self.count if self.count else 0.0

    def extent(self, log):
        """``(lower, upper)`` over both axes of the chosen histogram."""
        table = self._log if log else self._linear
        if not table:
            return None
        keys = np.array(list(table.keys()), dtype=np.int64)
        i, j = keys >> 32, (keys & 0xFFFFFFFF) - (1 << 31)
        low, high = int(min(i.min(), j.min())), int(max(i.max(), j.max())) + 1
        if log:
            return 10.0 ** (low * LOG_BIN_WIDTH), 10.0 ** (high * LOG_BIN_WIDTH)
        return low * self._linear_width, high * self._linear_width

    def histogram(self, edges, log):
        """
        Counts on display ``edges`` shared by every split: ``(bins, bins)``,
        reference along the first axis.
        """
        table = self._log if log else self._linear
        counts = np.zeros((len(edges) - 1, len(edges) - 1))
        if not table:
            return counts
        keys = np.array(list(table.keys()), dtype=np.int64)
        values = np.array(list(table.values()), dtype=float)
        i = (keys >> 32).astype(float) + 0.5
        j = ((keys & 0xFFFFFFFF) - (1 << 31)).astype(float) + 0.5
        if log:
            x, y = 10.0 ** (i * LOG_BIN_WIDTH), 10.0 ** (j * LOG_BIN_WIDTH)
        else:
            x, y = i * self._linear_width, j * self._linear_width
        a = np.clip(np.searchsorted(edges, x, side="right") - 1, 0,
                    len(edges) - 2)
        b = np.clip(np.searchsorted(edges, y, side="right") - 1, 0,
                    len(edges) - 2)
        np.add.at(counts, (a, b), values)
        return counts

    @property
    def occupied_bins(self):
        """Keys held, which is what memory grows with --- not voxels."""
        return len(self._log) + len(self._linear)


def _fold(table, i, j):
    """Add a sample's (i, j) fine-bin indices to a sparse count table."""
    keys = (i.astype(np.int64) << 32) + (j.astype(np.int64) + (1 << 31))
    unique, counts = np.unique(keys, return_counts=True)
    for key, count in zip(unique.tolist(), counts.tolist()):
        table[key] = table.get(key, 0) + count
