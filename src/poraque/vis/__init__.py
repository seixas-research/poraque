# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Visualization of training runs and field predictions.

:class:`~poraque.vis.report.TrainingReport` renders the three figures that
answer three separate questions about a trained operator — did it train
(loss curves), is it right in space (field cross-sections), is it right in
distribution (parity) — and writes them to a results directory.

Colour is assigned by the job it does: a single-hue sequential ramp for
one-signed magnitudes, a two-hue diverging ramp with a neutral midpoint and
zero-symmetric limits for signed errors, and a CVD-validated categorical set
for series identity. See :mod:`poraque.vis.style`.

Matplotlib is imported lazily, so ``import poraque.vis`` is cheap and the rest
of the package never depends on it::

    from poraque.vis import TrainingReport

    report = TrainingReport("results/plots", prefix="chg2tau")
    report.full_report(history=history, reference=tau_dft, prediction=tau_fno,
                       label=r"$\tau$", unit="eV/Ang^3")
"""

_LAZY = {
    "TrainingReport": "poraque.vis.report",
    "ModelReport": "poraque.vis.pdf_report",
    "CATEGORICAL": "poraque.vis.style",
    "INK": "poraque.vis.style",
    "diverging_cmap": "poraque.vis.style",
    "rc_params": "poraque.vis.style",
    "sequential_cmap": "poraque.vis.style",
    "series_color": "poraque.vis.style",
    "symmetric_limits": "poraque.vis.style",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    """Resolve public names on first use, so Matplotlib loads only if needed."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    try:
        module = import_module(module_name)
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            f"poraque.vis.{name} requires Matplotlib: pip install matplotlib"
        ) from error

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
