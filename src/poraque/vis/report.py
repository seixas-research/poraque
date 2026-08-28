# -*- coding: utf-8 -*-
# file: report.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Visual evaluation of a trained neural operator against reference DFT data.

:class:`TrainingReport` produces the three figures that answer three different
questions, which is why they are three figures and not one:

**Did it train?**  :meth:`loss_curves`
    Objective against epoch. Training loss and validation error are drawn in
    *separate stacked panels* rather than on twin y-axes: they are different
    quantities (a normalised objective and a physical relative :math:`L^2`), and
    a second y-scale invites the reader to compare two curves whose vertical
    positions have no shared meaning.

**Is it right in space?**  :meth:`field_comparison`
    A cross-section of the reference field, the prediction, and their
    difference. Reference and prediction share one colour scale — plotting them
    on independent scales is the classic way to make a bad prediction look
    good. The difference panel uses a diverging ramp on limits symmetric about
    zero, so neutral always means zero error.

**Is it right in distribution?**  :meth:`parity`
    Predicted against true voxel values. With :math:`10^6` points a scatter is a
    solid blob, so the default is a 2D density histogram; the identity line and
    the fitted metrics sit on top.

All figures are written to a results directory and the paths returned, so a
training script can archive them without knowing anything about Matplotlib.
"""

import os

import numpy as np

from .style import (
    INK,
    diverging_cmap,
    rc_params,
    sequential_cmap,
    series_color,
    symmetric_limits,
)


def _as_array(field):
    """
    Accept a :class:`~poraque.fields.ScalarField` or a raw array.

    A multi-channel field is refused rather than reduced. A
    :class:`~poraque.fields.SpinDensity` stacks two channels into ``data``, and
    every figure here draws *one* quantity: silently taking the first channel
    would produce a plausible picture of the wrong thing, and passing the stack
    through produced a Matplotlib error deep inside a run that had already
    finished training. The caller chooses the channel, so it can label it.
    """
    values = np.asarray(getattr(field, "data", field), dtype=float)
    if values.ndim != 3:
        raise ValueError(
            f"a field figure draws one 3D quantity, got shape {values.shape}. "
            f"For a spin-polarised field pass the channel to draw -- `.total` "
            f"or `.magnetization` -- rather than the field itself.")
    return values


def _thin_log_minor_labels(panel, subs=(2.0, 5.0)):
    """
    Label only a couple of the decade subdivisions on a log x-axis.

    Over a range narrower than about two decades Matplotlib labels the *minor*
    log ticks as well as the decades, so an axis spanning 1.2 to 61 gets
    ``2x10^0, 3x10^0, 4x10^0, ... 6x10^1``. Each is wide in mathtext, and
    horizontally they overprint into unreadable overlap -- the y-axis survives
    the same labels only because vertical tick spacing dwarfs the text height.

    Labelling the 2 and 5 subdivisions keeps the decade structure legible at a
    fraction of the width. The ticks themselves are untouched; only which of
    them carry text changes.
    """
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator

    if panel.get_xscale() != "log":
        return
    panel.xaxis.set_minor_locator(LogLocator(base=10.0, subs=subs, numticks=12))
    panel.xaxis.set_minor_formatter(LogFormatterSciNotation(base=10.0,
                                                            labelOnlyBase=False))


def _rotate_if_crowded(figure, panel, rotation=30, pad=2.0):
    """
    Rotate the x tick labels, but only when they actually collide.

    Measuring beats rotating unconditionally: a linear parity axis labelled
    ``1 2 3 4 5`` is perfectly readable upright, and tilting it would be a
    cosmetic regression. So the figure is laid out once, the drawn label boxes
    are compared, and the rotation is applied only if two of them overlap.

    Returns
    -------
    bool
        Whether the labels were rotated.
    """
    figure.draw_without_rendering()
    labels = [label for label in panel.get_xticklabels(which="both")
              if label.get_text() and label.get_visible()]
    boxes = sorted((label.get_window_extent() for label in labels),
                   key=lambda box: box.x0)
    crowded = any(later.x0 - earlier.x1 < pad
                  for earlier, later in zip(boxes, boxes[1:]))
    if crowded:
        for label in labels:
            label.set_rotation(rotation)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
    return crowded


def _metrics(prediction, reference):
    """Relative L2, R^2, MAE and RMSE between two fields."""
    prediction = prediction.ravel()
    reference = reference.ravel()
    difference = prediction - reference
    total = np.sum((reference - reference.mean()) ** 2)
    return {
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(reference)),
        "r2": float(1.0 - np.sum(difference ** 2) / total) if total > 0 else float("nan"),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
    }


class TrainingReport:
    """
    Generate and save the evaluation figures for one model.

    Parameters
    ----------
    output_dir : str, optional
        Directory for the figures; created on demand.
    theme : {"light", "dark"}, optional
        Dark mode is a *selected* set of tokens, not an inverted light mode.
    dpi : int, optional
        Raster resolution.
    fmt : str, optional
        Image format (``png``, ``pdf``, ``svg``).
    prefix : str, optional
        Prepended to every filename, e.g. the task or fold name.

    Examples
    --------
    >>> report = TrainingReport("results/plots", prefix="chg2tau")   # doctest: +SKIP
    >>> report.loss_curves(history)                                  # doctest: +SKIP
    >>> report.field_comparison(reference, prediction,
    ...                         label=r"$\\tau$", unit="eV/Ang^3")   # doctest: +SKIP
    >>> report.parity(reference, prediction)                         # doctest: +SKIP
    """

    def __init__(self, output_dir="results/plots", theme="light", dpi=160,
                 fmt="png", prefix=""):
        self.output_dir = str(output_dir)
        self.theme = theme
        self.dpi = int(dpi)
        self.fmt = str(fmt).lstrip(".")
        self.prefix = str(prefix)
        self.ink = INK["dark" if theme == "dark" else "light"]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _path(self, name):
        os.makedirs(self.output_dir, exist_ok=True)
        stem = f"{self.prefix}_{name}" if self.prefix else name
        return os.path.join(self.output_dir, f"{stem}.{self.fmt}")

    def _save(self, figure, name):
        import matplotlib.pyplot as plt

        path = self._path(name)
        figure.savefig(path, dpi=self.dpi, facecolor=self.ink["surface"])
        plt.close(figure)
        return path

    def _context(self, base_size=10):
        import matplotlib.pyplot as plt

        return plt.rc_context(rc_params(self.theme, base_size))

    # ------------------------------------------------------------------ #
    # 1. Loss curves
    # ------------------------------------------------------------------ #
    def loss_curves(self, history, name="loss_curves", title=None, log_scale=True):
        """
        Plot the training objective and validation error against epoch.

        Parameters
        ----------
        history : dict
            ``{"train_loss": [...], "val_error": [...]}`` as returned by
            :func:`~poraque.ml.training.train`. ``val_error`` may be empty.

            ``train_loss`` is the **total** objective — data fidelity plus
            every weighted physics term — because that is the quantity the
            optimiser stepped on and the only one comparable between runs.
        name : str, optional
            Filename stem.
        title : str, optional
        log_scale : bool, optional
            Log y-axis. Losses fall over orders of magnitude, and on a linear
            axis everything after the first few epochs is a flat line against
            the floor.

        Returns
        -------
        str
            Path written.
        """
        import matplotlib.pyplot as plt

        train = np.asarray(history.get("train_loss", []), dtype=float)
        validation = np.asarray(history.get("val_error", []), dtype=float)
        if train.size == 0:
            raise ValueError("history contains no 'train_loss' entries.")
        has_validation = validation.size > 0

        # With eval_every > 1 the validation series is shorter than the
        # training one and its points sit at specific epochs. Plotting it
        # against 1..N would silently compress the curve onto the wrong x-axis.
        val_epochs = np.asarray(history.get("val_epoch", []), dtype=float)
        if val_epochs.size != validation.size:
            val_epochs = np.arange(1, validation.size + 1, dtype=float)

        with self._context():
            rows = 2 if has_validation else 1
            figure, axes = plt.subplots(
                rows, 1, figsize=(7.0, 3.2 * rows), sharex=True,
                constrained_layout=True,
            )
            axes = np.atleast_1d(axes)

            panels = [(axes[0], train,
                       np.arange(1, train.size + 1, dtype=float),
                       "Training objective", 0)]
            if has_validation:
                panels.append((axes[1], validation, val_epochs,
                               "Validation relative $L^2$ (physical units)", 1))

            for axis, values, epochs, label, slot in panels:
                colour = series_color(slot)
                # Mark the individual points on a sparse series, so a
                # ten-point validation curve does not read as a smooth one.
                marker = "o" if values.size <= 40 else None
                axis.plot(epochs, values, color=colour, label=label,
                          marker=marker, markersize=3)
                if log_scale and np.all(values > 0):
                    axis.set_yscale("log")

                # Direct label at the final point, rather than a number on
                # every point: the endpoint is the value the reader wants.
                axis.scatter([epochs[-1]], [values[-1]], s=28, color=colour,
                             zorder=3, edgecolor=self.ink["surface"], linewidth=2)
                axis.annotate(f"{values[-1]:.4g}",
                              xy=(epochs[-1], values[-1]),
                              xytext=(-6, 10), textcoords="offset points",
                              ha="right", fontsize=9, color=self.ink["primary"])
                if values.size and np.isfinite(values).any():
                    best = int(np.nanargmin(values))
                    if best != values.size - 1:
                        axis.scatter([epochs[best]], [values[best]], s=28,
                                     facecolor=self.ink["surface"],
                                     edgecolor=colour, linewidth=1.8, zorder=3)
                        axis.annotate(f"best {values[best]:.4g}",
                                      xy=(epochs[best], values[best]),
                                      xytext=(6, -12), textcoords="offset points",
                                      fontsize=8, color=self.ink["secondary"])
                # One series per panel: the axis label names it, so a legend
                # box would only repeat the same words.
                axis.set_ylabel(label, fontsize=9)

            axes[-1].set_xlabel("Epoch")
            figure.suptitle(title or "Training history", x=0.01, ha="left",
                            fontsize=12, color=self.ink["primary"])
            return self._save(figure, name)

    # ------------------------------------------------------------------ #
    # 2. Field cross-sections
    # ------------------------------------------------------------------ #
    def field_comparison(self, reference, prediction, name="field_slice",
                         axis=2, index=None, label="field", unit="",
                         title=None, log=False, difference_quantile=0.999):
        """
        Compare a 2D cross-section of the reference and predicted 3D fields.

        Parameters
        ----------
        reference, prediction : ScalarField or numpy.ndarray
            Fields of identical shape.
        name : str, optional
            Filename stem.
        axis : {0, 1, 2}, optional
            Axis to slice along.
        index : int, optional
            Slice index; defaults to the middle of ``axis``.
        label : str, optional
            Quantity name used in panel titles.
        unit : str, optional
            Physical unit, shown on the colour bars.
        title : str, optional
        log : bool, optional
            Logarithmic colour scale, for strictly positive fields spanning
            several decades (a valence density does).
        difference_quantile : float, optional
            Quantile of ``|error|`` setting the diverging limits, so a few
            outlying voxels cannot wash out the whole error map.

        Returns
        -------
        str
            Path written.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize

        reference_data = _as_array(reference)
        prediction_data = _as_array(prediction)
        if reference_data.shape != prediction_data.shape:
            raise ValueError(
                f"Shape mismatch: reference {reference_data.shape} vs "
                f"prediction {prediction_data.shape}."
            )
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {axis!r}.")

        index = reference_data.shape[axis] // 2 if index is None else int(index)
        reference_slice = np.take(reference_data, index, axis=axis)
        prediction_slice = np.take(prediction_data, index, axis=axis)
        difference = prediction_slice - reference_slice

        # ONE shared scale for reference and prediction. Independent scales
        # would hide exactly the errors this figure exists to reveal.
        finite = np.concatenate([reference_slice.ravel(), prediction_slice.ravel()])
        finite = finite[np.isfinite(finite)]
        if log and finite.min() <= 0:
            log = False           # LogNorm cannot show non-positive values
        if log:
            norm = LogNorm(vmin=max(finite.min(), 1e-8), vmax=finite.max())
        else:
            norm = Normalize(vmin=finite.min(), vmax=finite.max())

        low, high = symmetric_limits(difference, difference_quantile)
        metrics = _metrics(prediction_data, reference_data)
        axis_names = ("a", "b", "c")
        remaining = [axis_names[i] for i in range(3) if i != axis]

        with self._context():
            figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.1),
                                        constrained_layout=True)
            common = dict(origin="lower", aspect="equal", interpolation="nearest")

            for panel, values, panel_title in (
                (axes[0], reference_slice, f"DFT reference {label}"),
                (axes[1], prediction_slice, f"FNO prediction {label}"),
            ):
                image = panel.imshow(values.T, cmap=sequential_cmap(), norm=norm,
                                     **common)
                panel.set_title(panel_title)
                panel.set_xlabel(f"grid index along {remaining[0]}")
                panel.grid(False)
            axes[0].set_ylabel(f"grid index along {remaining[1]}")

            bar = figure.colorbar(image, ax=axes[:2], fraction=0.046, pad=0.02)
            bar.set_label(f"{label} [{unit}]" if unit else label)
            bar.outline.set_visible(False)

            error_image = axes[2].imshow(difference.T, cmap=diverging_cmap(),
                                         vmin=low, vmax=high, **common)
            axes[2].set_title("Prediction $-$ reference")
            axes[2].set_xlabel(f"grid index along {remaining[0]}")
            axes[2].grid(False)
            error_bar = figure.colorbar(error_image, ax=axes[2], fraction=0.046,
                                        pad=0.02)
            error_bar.set_label(f"error [{unit}]" if unit else "error")
            error_bar.outline.set_visible(False)

            headline = (title or f"{label}: slice {index} along axis "
                                 f"{axis_names[axis]}")
            figure.suptitle(
                f"{headline}          "
                f"relative $L^2$ = {metrics['relative_l2']:.4f}   "
                f"$R^2$ = {metrics['r2']:.4f}   "
                f"MAE = {metrics['mae']:.4g} {unit}".rstrip(),
                x=0.01, ha="left", fontsize=11, color=self.ink["primary"],
            )
            return self._save(figure, name)

    # ------------------------------------------------------------------ #
    # 3. Parity
    # ------------------------------------------------------------------ #
    def pareto(self, front, knee=None, name="pareto", title=None,
               annotate=6):
        r"""
        The accuracy/complexity front of a symbolic search, with its knee.

        A search does not return *an* expression, it returns a trade: every
        candidate that was the best of its length. This is that trade drawn,
        which is the plot a reader needs to decide what to quote.

        The loss axis is logarithmic. A front spans orders of magnitude, and on
        a linear axis every candidate but the most accurate collapses onto the
        floor -- the knee would then always be the shortest expression,
        whatever it cost.

        Two points are marked: the **lowest loss** (the leftmost value on the
        y axis) and the **knee**, the candidate nearest the ideal corner once
        both axes are rescaled to :math:`[0, 1]`. Where they coincide, nothing
        was traded away and the plot says so.

        Parameters
        ----------
        front : sequence of dict
            Entries with ``complexity`` and ``loss``; ``expression`` is used
            for the annotations when present.
        knee : dict, optional
            The chosen knee, from
            :func:`~poraque.ml.symbolic.pareto_knee`. Computed here when
            omitted, so the plot is usable on a bare front.
        name : str, optional
        title : str, optional
        annotate : int, optional
            How many expressions to label. Beyond a handful the labels overlap
            and the curve is what carries the message.

        Returns
        -------
        str or None
            Path written, or ``None`` for a front with nothing plottable.
        """
        usable = [e for e in front
                  if e.get("complexity") is not None
                  and e.get("loss") is not None
                  and np.isfinite(e["loss"]) and e["loss"] > 0]
        if not usable:
            return None
        usable = sorted(usable, key=lambda e: e["complexity"])

        if knee is None:
            from ..ml.symbolic import pareto_knee

            knee = pareto_knee(usable)

        complexity = np.array([e["complexity"] for e in usable], dtype=float)
        loss = np.array([e["loss"] for e in usable], dtype=float)
        best = usable[int(np.argmin(loss))]

        with self._context():
            import matplotlib.pyplot as plt

            figure, axes = plt.subplots(figsize=(7.0, 4.4))
            axes.step(complexity, loss, where="post", color=self.ink["muted"],
                      linewidth=1.2, zorder=1)
            axes.scatter(complexity, loss, s=34, color=self.ink["secondary"],
                         zorder=2, label="front")

            def mark(entry, colour, marker, text):
                if not entry:
                    return
                axes.scatter([entry["complexity"]], [entry["loss"]], s=170,
                             marker=marker, facecolor=colour, zorder=4,
                             edgecolor=self.ink["surface"], linewidth=1.8,
                             label=text)

            same = (knee and knee.get("complexity") == best["complexity"]
                    and knee.get("loss") == best["loss"])
            mark(best, series_color(0), "o", "lowest loss")
            if not same:
                mark(knee, series_color(1), "D", "Pareto knee")

            # Labelled sparsely and only on the interesting end: a formula
            # printed beside every point is unreadable at any figure size.
            marked = {id(best), id(knee)}
            step = max(1, len(usable) // max(1, annotate))
            for index, entry in enumerate(usable):
                if index % step and id(entry) not in marked:
                    continue
                text = str(entry.get("expression", ""))
                if len(text) > 26:
                    text = text[:25] + "\u2026"
                axes.annotate(text, (entry["complexity"], entry["loss"]),
                              textcoords="offset points", xytext=(6, 6),
                              fontsize=7, color=self.ink["muted"])

            axes.set_yscale("log")
            axes.set_xlabel("complexity (nodes)")
            axes.set_ylabel("loss")
            axes.set_title(title or "Accuracy against complexity")
            axes.grid(True, which="both", alpha=0.3)
            axes.legend(frameon=False, loc="upper right")
            if same:
                axes.text(0.02, 0.04,
                          "the knee is the lowest-loss expression:\n"
                          "nothing was traded away",
                          transform=axes.transAxes, fontsize=8,
                          color=self.ink["muted"])
            figure.tight_layout()
            return self._save(figure, name)

    def parity(self, reference, prediction, name="parity", label="field",
               unit="", title=None, bins=200, max_points=200_000,
               scatter=False, log=False, validation=None,
               split_labels=("training", "validation"),
               prediction_label="FNO prediction"):
        """
        Predicted against true voxel values, with the identity line.

        Passing ``validation`` puts the held-out set in a **second panel**
        beside the training one, which is the plot that answers whether the
        operator generalises: if the held-out density sits wider about the
        identity line, that spread *is* the generalisation gap, read off
        directly rather than inferred from two summary numbers.

        The panels share their axes, their bin edges and one colour scale, so
        the two sides are directly comparable. Each is normalised to the share
        of *its own* voxels, because two structures need not have the same grid
        size and raw counts would then paint the larger one denser for no
        physical reason.

        Parameters
        ----------
        reference, prediction : ScalarField or numpy.ndarray
            The training set, or the only set when ``validation`` is omitted.
        name : str, optional
        label : str, optional
        unit : str, optional
        title : str, optional
        bins : int, optional
            Bins per axis for the density histogram.
        max_points : int, optional
            Subsample size per set when points are drawn individually.
        scatter : bool, optional
            Draw individual points instead of a density map. Only sensible for
            small grids — a million points render as one opaque blob, which
            shows the extent of the data but nothing about where it
            concentrates. Applies per panel, so it composes with ``validation``.
        log : bool, optional
            Log-scale both axes, for strictly positive fields.
        validation : tuple of (ScalarField or ndarray, ScalarField or ndarray), optional
            ``(reference, prediction)`` for the held-out set. The two sets need
            not be the same size — they are usually different structures.
        split_labels : tuple of str, optional
            Legend names for the two sets.
        prediction_label : str, optional
            What produced the y values. The default names the operator; a
            distilled expression is not the operator, and labelling its plot
            as though it were would misattribute the error.

        Returns
        -------
        str
            Path written.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        sets = [(split_labels[0], _as_array(reference).ravel(),
                 _as_array(prediction).ravel())]
        if validation is not None:
            try:
                validation_reference, validation_prediction = validation
            except (TypeError, ValueError):
                raise ValueError(
                    "validation must be a (reference, prediction) pair."
                ) from None
            sets.append((split_labels[1],
                         _as_array(validation_reference).ravel(),
                         _as_array(validation_prediction).ravel()))

        for series_name, series_x, series_y in sets:
            if series_x.shape != series_y.shape:
                raise ValueError(f"{series_name}: reference and prediction "
                                 f"must have the same size.")

        # The log decision is taken across every set at once: axes are shared,
        # so one set falling back to linear would silently rescale the other.
        finite = [np.isfinite(sx) & np.isfinite(sy) for _, sx, sy in sets]
        valid = finite
        if log:
            positive = [f & (sx > 0) & (sy > 0)
                        for f, (_, sx, sy) in zip(finite, sets)]
            # An undertrained model can predict a non-positive field
            # everywhere, which would leave nothing to plot. Fall back to
            # linear axes rather than crashing on an empty reduction: a
            # diagnostic plot is least dispensable exactly when the model is
            # behaving badly.
            if sum(p.sum() for p in positive) < 0.01 * sum(f.sum()
                                                           for f in finite):
                log = False
            else:
                valid = positive

        if not any(mask.any() for mask in valid):
            raise ValueError("No finite points to plot.")

        drawn = [(series_name, sx[mask], sy[mask],
                  _metrics(sy[mask], sx[mask]))
                 for (series_name, sx, sy), mask in zip(sets, valid)
                 if mask.any()]
        suffix = f" [{unit}]" if unit else ""
        comparing = len(drawn) > 1

        # Shared across panels: one range, one set of bin edges. A bin has to
        # mean the same thing on both sides or the comparison is decorative.
        lower = float(min(min(x.min(), y.min()) for _, x, y, _ in drawn))
        upper = float(max(max(x.max(), y.max()) for _, x, y, _ in drawn))
        # Log axes demand log-spaced bins. With linear bins the cells render
        # enormous at the low end and invisible at the high end, which reads as
        # structure in the data that is not there.
        if log:
            edges = np.logspace(np.log10(lower), np.log10(upper), bins + 1)
        else:
            edges = np.linspace(lower, upper, bins + 1)

        histograms = []
        if not scatter:
            for _, x, y, _ in drawn:
                counts, _, _ = np.histogram2d(x, y, bins=[edges, edges])
                # Share of *this* set's voxels, not a raw count: two structures
                # need not have the same number of voxels, and comparing counts
                # would then paint the larger one denser for no physical reason.
                histograms.append(counts / max(counts.sum(), 1))
            positive = [h[h > 0] for h in histograms]
            floor = float(min(p.min() for p in positive if p.size))
            ceiling = float(max(h.max() for h in histograms))
            norm = LogNorm(vmin=floor, vmax=ceiling)

        with self._context():
            if comparing:
                figure, axes = plt.subplots(
                    1, 2, figsize=(9.8, 5.0), sharex=True, sharey=True,
                    constrained_layout=True)
                panels = list(axes)
            else:
                figure, single = plt.subplots(figsize=(5.6, 5.4),
                                              constrained_layout=True)
                panels = [single]

            mesh = None
            for index, (panel, (series_name, x, y, series_metrics)) in enumerate(
                    zip(panels, drawn)):
                if scatter:
                    if x.size > max_points:
                        picked = np.random.default_rng(0).choice(
                            x.size, max_points, replace=False)
                        x, y = x[picked], y[picked]
                    panel.scatter(x, y, s=4, alpha=0.25, linewidths=0,
                                  color=series_color(index), label="voxels")
                else:
                    mesh = panel.pcolormesh(edges, edges, histograms[index].T,
                                            cmap=sequential_cmap(), norm=norm)

                panel.plot([lower, upper], [lower, upper], linestyle="--",
                           linewidth=1.6, color=self.ink["muted"], zorder=4,
                           label="perfect agreement")
                if log:
                    panel.set_xscale("log")
                    panel.set_yscale("log")
                panel.set_xlim(lower, upper)
                panel.set_ylim(lower, upper)
                panel.set_aspect("equal")
                panel.set_xlabel(f"DFT reference {label}{suffix}")
                panel.set_ylabel(f"{prediction_label} {label}{suffix}")
                panel.legend(loc="upper left", fontsize=8 if comparing else None)

                if comparing:
                    panel.set_title(
                        f"{series_name} — rel $L^2$ "
                        f"{series_metrics['relative_l2']:.4f}",
                        fontsize=10, color=self.ink["primary"])
                panel.text(
                    0.98, 0.04,
                    f"relative $L^2$ = {series_metrics['relative_l2']:.4f}\n"
                    f"$R^2$ = {series_metrics['r2']:.4f}\n"
                    f"MAE = {series_metrics['mae']:.4g}{suffix}\n"
                    f"RMSE = {series_metrics['rmse']:.4g}{suffix}",
                    transform=panel.transAxes, ha="right", va="bottom",
                    fontsize=8 if comparing else 9,
                    color=self.ink["secondary"],
                )

            if comparing:
                # Only the leftmost panel keeps its y-axis; the axes are shared,
                # so repeating the labels between the panels is pure clutter.
                for panel in panels:
                    panel.label_outer()

            if mesh is not None:
                # One colourbar for every panel. Separate bars would each
                # autoscale and two different densities could then wear the
                # same colour, which is the specific lie this plot must not tell.
                bar = figure.colorbar(mesh, ax=panels, fraction=0.046, pad=0.02)
                bar.set_label("share of the structure's voxels per bin")
                bar.outline.set_visible(False)

            figure.suptitle(title or f"Parity: {label}", x=0.01, ha="left",
                            fontsize=12, color=self.ink["primary"])

            # Both axes carry the same labels, but only the x-axis runs out of
            # room for them. Thin them first, then rotate only if they still
            # collide. `constrained_layout` re-reserves the margin afterwards,
            # so no tight_layout call is needed -- and adding one would fight
            # the constrained layout this figure was created with.
            for panel in panels:
                _thin_log_minor_labels(panel)
                _rotate_if_crowded(figure, panel)
            return self._save(figure, name)

