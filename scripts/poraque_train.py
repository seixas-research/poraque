#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_train.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Train and evaluate the Fourier Neural Operators on a DFT dataset.

The run is defined by a **YAML configuration file**; command-line flags
override individual entries, so one committed config can be swept from the
shell without being edited or copied. The resolved configuration is written
next to the results, recording exactly what ran.

Tasks
-----
``ext2chg``
    ``EXTCAR`` -> ``CHGCAR``, the Hohenberg-Kohn map.
``chg2tau``
    ``CHGCAR`` -> ``TAUCAR``, the kinetic energy density functional.

The two are **independent**. ``task: ext2chg`` builds one network, trains one
objective and writes one model; nothing about :math:`\tau` is instantiated,
loaded or differentiated. That is what makes the vast public archives of
charge densities usable: they publish no kinetic energy density, and needing
one would rule them out entirely. A task the data cannot supply is reported and
skipped rather than failing the run, so ``task: all`` on a density-only archive
trains what it can.

Data sources
------------
``data.data_paths`` is a **list of directories**, and every entry has the same
shape: subdirectories, one per material, each holding that material's
volumetric files::

    data:
      data_paths:
        - data/vasp/structures  # local DFT runs
        - data/MP               # a poraque-mp download
        - data/cache/res32      # a cache from an earlier run

so local runs and a Materials Project download train as one dataset with
nothing said about where either came from. What a material's directory *holds*
decides how it is read — inputs beside the density mean the external potential
is computed from them, a density alone means it is computed from the density's
own header — and ``TAUCAR`` is optional per material throughout. See
:mod:`poraque.data.sources`, and note the caveat there about mixing two
definitions of :math:`V_{\rm ext}` — the run warns when it happens.

Method notes
------------
**Downsampling is spectral.** The native VASP grids are reduced by Fourier
truncation (:mod:`poraque.fields.resample`), the exact band-limited projection
for a plane-wave field: periodicity and the electron count survive to machine
precision. Interpolation would alias, break periodicity at the cell boundary
and shift the integral.

**One protocol, one variation.** A run trains a single model per task on a
train/validation split sized by ``training.valid_fraction`` (a fifth by default),
and reports metrics in **physical units** on the held-out structures. Setting
``enable_kfold`` swaps that for K-fold cross-validation, which is the only
other protocol; nothing else changes how training is organised.

Usage
-----
Installed (``pip install -e .``), this is the ``poraque-train`` console command
and runs from any directory::

    poraque-train --config configs/train.yaml
    poraque-train --config configs/train.yaml --epochs 500
    poraque-train --config configs/train.yaml --device mps
    poraque-train --config configs/train.yaml --kfold

Running this file directly — ``python scripts/poraque_train.py`` — is
equivalent, and needs nothing installed.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Run straight from a checkout, without installing, by preferring the in-tree
# package. Installed as the ``poraque-train`` console script this module sits in
# site-packages, that directory does not exist, and the installed package wins.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)


# There used to be a `_preimport_symbolic_engine(sys.argv[1:])` here, which
# read the config before `import torch` so that PySR's Julia runtime could be
# loaded first: importing `juliacall` after `torch` can segfault the process,
# which juliacall warns about itself citing pytorch#78829, and the crash
# arrived *after* training had finished and took the run with it. The engine is
# `poraque.ml.gp` now -- NumPy and SciPy -- so there is no second runtime, no
# load order to respect, and nothing for this to do.
import torch  # noqa: E402

from poraque import banner  # noqa: E402
from poraque.fields import (  # noqa: E402
    FIELD_DTYPES,
    charge_channel,
    field_integral,
    set_default_dtype,
)
from poraque.ml import (  # noqa: E402
    resolve_bundle_path,
    FieldOperator,
    FieldPairDataset,
    save_bundle,
    train,
)
from poraque.ml.config import BUNDLE_SUFFIX, TrainingConfig  # noqa: E402
from poraque.ml.data import CACHE_MEMORY_BUDGET  # noqa: E402
from poraque.ml.device import (  # noqa: E402
    describe_device,
    device_report,
    enable_tf32,
    resolve_device,
)
from poraque.ml.distributed import (  # noqa: E402
    barrier,
    describe as describe_distributed,
    discover as discover_distributed,
    initialize as initialize_distributed,
    shutdown as shutdown_distributed,
)
from poraque.ml.fno import PRECISIONS  # noqa: E402
from poraque.ml.training import OPTIMIZERS  # noqa: E402
from poraque.ml.losses import PhysicsInformedLoss  # noqa: E402
from poraque.ml.symbolic import (  # noqa: E402
    DATA_LOSSES,
    FEATURE_SCHEMES,
    TEMPLATES,
    result_to_dict,
    symbolic_physics,
)
from poraque.ml.tasks import TASKS, resolve_task  # noqa: E402

#: Display label and unit per field, for figures.
FIELD_LABELS = {
    "EXTCAR": (r"$V_{\mathrm{ext}}$", r"eV"),
    "CHGCAR": (r"$\rho$", r"e/$\AA^3$"),
    "TAUCAR": (r"$\tau$", r"eV/$\AA^3$"),
}


# ===================================================================== #
# Output naming
#
# Every artefact of a run is built from one string, `task.name`. Two runs that
# differ in anything worth keeping -- a chemical space, a resolution, a set of
# physics weights -- differ in that name, and so cannot overwrite each other.
# Everything lands under one directory per run:
#
#     models/<name>/<name>.poraque    log/    plots/    report/
#
# The helpers below are the only places those paths are formed. They read the
# config rather than mutating it, so re-running with the config a run archived
# beside its results reproduces the same paths instead of nesting a second copy
# of the name inside the first.
# ===================================================================== #
def model_name(config):
    """The run's name, falling back to the historical default."""
    return config.run_name()


def bundle_path(config):
    """
    Where the trained weights go: ``<output.root>/<name>/<name>.poraque``.

    Returns
    -------
    str or None
        ``None`` when checkpointing is switched off.
    """
    return config.checkpoint_path()


def plot_directory(config):
    """
    Figures go in the run's own ``plots/`` directory.

    Returns
    -------
    str or None
    """
    return config.plot_dir()


def report_filename(config, task_name, n_tasks=1, kind="report"):
    """
    ``<name>_report.pdf``, or ``<name>_<task>_report.pdf`` when both train.

    The report is per task, so a ``task: all`` run produces two of them and one
    name cannot serve both. Qualifying only in that case keeps the common
    single-task run -- every ``ext2chg`` run on a density archive -- at the
    plain name.

    Parameters
    ----------
    kind : str, optional
        Trailing component, so cross-validation writes a
        ``<name>_kfold_report.pdf`` that cannot be mistaken for the single-fit
        report beside it.
    """
    stem = model_name(config)
    if n_tasks >= 2:
        stem += f"_{task_name}"
    return f"{stem}_{kind}.pdf"


class Tee:
    """
    Write to the terminal and, when there is one, a log file.

    ``path=None`` means terminal only, which is what ``output.write_log:
    false`` asks for. Without that case the toggle crashed the run before it
    started, on ``os.path.dirname(None)``.

    ``silent=True`` swallows everything, and is what every rank but the first
    gets under DDP. Four ranks opening one path with ``"w"`` truncate each
    other's output and interleave the survivors, so a four-GPU run's log ends
    up less readable than a one-GPU run's and its progress table unparseable.
    The silencing is here rather than at each of the several hundred call
    sites, which is the only version of it that can be kept correct.
    """

    def __init__(self, path, silent=False):
        self.path = None if silent else path
        self.silent = bool(silent)
        self.handle = None
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self.handle = open(self.path, "w")

    def __call__(self, message=""):
        if self.silent:
            return
        print(message)
        if self.handle is not None:
            self.handle.write(str(message) + "\n")
            self.handle.flush()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


# ===================================================================== #
# Cache construction
# ===================================================================== #
def resolved_spin(data):
    """
    Whether this dataset's cache carries a magnetisation channel.

    ``data.spin`` may be ``"auto"``, which is a question rather than an answer,
    and the question can only be settled by reading the sources. Answered here
    so that :func:`cache_tag` and :func:`build_cache` cannot disagree — a tag
    that said one thing while the build did another would silently reuse a
    one-channel cache for a two-channel run.

    Errors while probing are swallowed and reported as "no spin": this runs
    only to name a directory, and the real read a moment later raises a far
    better message than a half-built path can.

    Parameters
    ----------
    data : DataConfig

    Returns
    -------
    bool
    """
    if data.spin is True:
        return True
    if data.spin is False:
        return False

    from poraque.data import discover_records, resolve_source

    options = {"potcar_dir": data.potcar_dir, "sigma": data.sigma,
               "gaussian_blur": data.gaussian_blur,
               "blur_method": data.blur_method, "pattern": data.pattern,
               "format": data.format}
    try:
        sources = [resolve_source(path, **options) for path in data.paths()]
        records = discover_records(sources, required=("CHGCAR",))
        return any(record.source.is_spin_polarized(record)
                   for record in records)
    except Exception:                                   # noqa: BLE001
        return False


def cache_tag(data):
    """
    Directory name encoding everything that changes the stored fields.

    Without this, switching ``--gaussian-blur`` would silently reuse the
    previous cache and the "comparison" would compare a model against itself.
    The paths are folded in too, because a cache built from one archive and a
    cache built from a mixture are different datasets under the same
    resolution.
    """
    tag = f"res{data.resolution}"
    if data.gaussian_blur:
        tag += f"_blur{data.gaussian_blur:g}{data.blur_method[:4]}"
    if data.sigma:
        tag += f"_sig{data.sigma:g}"
    if data.potcar_dir:
        # A tabulated potential and a Gaussian one are different fields, not
        # different roundings of one, so they must never share a cache.
        tag += "_potcar"
    if resolved_spin(data):
        # A cache carrying the magnetisation channel is a different dataset
        # from one that does not, so it gets its own directory. Resolved from
        # the data, not from the literal setting: `spin: auto` on ISPIN = 2
        # sources caches two channels and must not share a directory with a
        # `spin: false` build of the same sources.
        tag += "_spin"

    paths = data.paths()
    if len(paths) > 1:
        # Short and stable: the basenames of the archives, in the order given.
        names = "-".join(os.path.basename(os.path.normpath(p)) for p in paths)
        tag += f"_{names}"
    return tag


def build_cache(config, log):
    """
    Downsample every material under ``data.data_paths`` into one dataset.

    Each path is detected independently — a directory of DFT runs, an archive
    of standalone ``CHGCAR`` files, or a prepared cache — and all of them land
    in the same per-material layout, so nothing downstream knows or cares which
    a given material came from.

    Every field a source can supply is written, not only the ones the current
    task needs: a cache built for ``ext2chg`` from an archive that also holds
    :math:`\\tau` serves ``chg2tau`` afterwards with no rebuild.

    Returns
    -------
    str
        Cache directory, laid out so
        :class:`~poraque.ml.data.FieldPairDataset` reads it unchanged.
    """
    from poraque.data import build_field_cache, discover_records, resolve_source
    from poraque.data.cache import build_paw_reference

    data = config.data
    paths = data.paths()
    target = os.path.join(data.cache, cache_tag(data))

    log(f"Cache: {target}")
    log(f"  sources: {len(paths)} path(s)")

    build_field_cache(
        paths, target, resolution=data.resolution,
        potcar_dir=data.potcar_dir, sigma=data.sigma,
        gaussian_blur=data.gaussian_blur, blur_method=data.blur_method,
        pattern=data.pattern, format=data.format, log=log,
        # Passed through as given, INCLUDING "auto", which build_field_cache
        # resolves against the sources. It used to be flattened here with
        # `data.spin is True`, which made "auto" -- the default -- mean *no
        # spin*, silently discarding every ISPIN = 2 magnetisation block on
        # the way into the cache.
        spin=data.spin,
        # One chunked fields.h5 per material instead of three text files, when
        # asked for. Both are in the cache fingerprint, so changing either
        # rebuilds rather than mixing layouts in one directory.
        storage=data.storage, compression=data.compression,
        compression_level=data.compression_level,
        # Whether a configured POTCAR library that cannot serve an element is
        # an error or a documented degradation. Either way, what it *did* serve
        # goes into the cache fingerprint.
        strict_potcar=data.strict_potcar,
    )

    # The PAW augmentation records travel with the weights, so a prediction can
    # be written as an ICHARG=1 restart without a reference calculation beside
    # it. They come from the *native-resolution* sources, not the cache: the
    # one-centre terms are on-site quantities and do not live on the FFT grid
    # at all, so downsampling neither changes nor carries them.
    sources = [resolve_source(path, pattern=data.pattern, format=data.format,
                              potcar_dir=data.potcar_dir)
               for path in paths]
    # The augmentation records now come from the ISOLATED ATOMS by default
    # (data.paw_source), not from averaging the training set. They are a
    # per-element, transferable, provenance-carrying quantity, which is what
    # they have to be once slabs and clusters join the set: a training-set
    # average is a property of whatever happened to be in it.
    library = None
    if data.paw_source == "atomic" and data.atomic_reference:
        from poraque.fields.atomic import resolve_library

        try:
            library = resolve_library(data.atomic_reference, cache=target,
                                      log=log)
        except (FileNotFoundError, ValueError) as error:
            log(f"  PAW reference: {error}")
    build_paw_reference(discover_records(sources, required=("CHGCAR",)),
                        target, log, library=library, source=data.paw_source)
    return target


def load_paw_reference(cache):
    """The cached per-element PAW table, or an empty dict."""
    from poraque.data.cache import load_paw_reference as _load

    return _load(cache)


def trainable_tasks(names, cache, log):
    """
    Drop the requested tasks the cached data cannot supply both fields for.

    A dataset does not always carry every field. A Materials Project download
    is the clear case — it publishes the charge density and nothing else, so
    ``chg2tau`` has no :math:`\\tau` to regress onto and no amount of pipeline
    work invents one — but a partial local dataset behaves the same way.

    Failing here, before a model is built, is the whole point: the alternative
    is a ``ValueError`` from the dataset constructor that names a missing file
    rather than the task that needed it, and (with ``task: all``) throws away
    the task that *would* have trained.

    Parameters
    ----------
    names : list of str
        Requested task names.
    cache : str
        Cache directory to inspect.
    log : callable

    Returns
    -------
    list of str
        The subset that can actually be trained, in the requested order.

    Raises
    ------
    SystemExit
        If none of them can, naming what was found instead.
    """
    from poraque.ml.data import discover_materials

    keep, dropped = [], []
    for name in names:
        task = resolve_task(name)
        if discover_materials(cache, task.required_files):
            keep.append(name)
        else:
            dropped.append(task)

    for task in dropped:
        log(f"\n  SKIPPING {task.name}: no material under {cache} has both "
            f"{task.input_field} and {task.target_field}.")
        log(f"      {task.description.rstrip('.')} cannot be learned from data "
            f"that carries no {task.target_field}, and nothing reconstructs "
            f"one. The remaining task(s) train normally.")

    if not keep:
        available = sorted({name for entry in os.listdir(cache)
                            if os.path.isdir(os.path.join(cache, entry))
                            for name in os.listdir(os.path.join(cache, entry))})
        raise SystemExit(
            f"None of the requested tasks {names} can be trained on {cache}: "
            f"the cached materials hold {available or 'no fields at all'}."
        )
    return keep


def dataset_elements(cache):
    """
    Every chemical element the cached materials contain.

    Read from the cached ``CHGCAR`` headers, which are a few hundred bytes
    each, so this costs nothing next to the fields themselves.
    """
    from poraque.fields.vasp.volumetric import read_structure_header

    elements = set()
    for entry in sorted(os.listdir(cache)):
        path = os.path.join(cache, entry, "CHGCAR")
        if os.path.exists(path):
            elements.update(read_structure_header(path).elements)
    return sorted(elements)


def chemistry_caveat(n_structures, elements):
    """
    One sentence bounding what a dataset of this breadth can support.

    Computed rather than asserted: a hard-coded "all of one element" was true
    of the original single-element dataset and is a false claim about a
    Materials Project chemical space, which is exactly the kind of caveat a
    reader trusts without checking.
    """
    if len(elements) <= 1:
        return (f"{n_structures} structure(s), all of "
                f"{elements[0] if elements else 'one element'}: nothing here "
                f"speaks to transfer across chemistry.")
    return (f"{n_structures} structure(s) spanning {', '.join(elements)}: this "
            f"measures transfer within that chemical space, not beyond it.")


# ===================================================================== #
# Metrics
# ===================================================================== #
def metrics(prediction, target, grid=None):
    r"""
    Error metrics in the physical units of the fields.

    Parameters
    ----------
    prediction, target : array_like
        The predicted and reference fields.
    grid : FieldGrid, optional
        Enables the three metrics that are integrals over the cell rather than
        sums over voxels, and cannot be taken without one:

        ``relative_h1``
            The relative :math:`H^1` error, values and gradients together.
        ``integral_error``
            :math:`|\int\hat f - \int f|`, which for ``ext2chg`` is the
            electron-count error in electrons.
        ``jsd``
            The Jensen-Shannon divergence, when both fields are non-negative
            densities.

    Returns
    -------
    dict

    Notes
    -----
    **Nothing here is told what the run was optimising**, and that is what
    keeps two runs with different objectives comparable. ``relative_h1`` is
    therefore *not* the :math:`H^1` objective
    :class:`~poraque.ml.losses.SobolevLoss` minimises: that one is two
    separately-normalised terms combined with ``training.sobolev_weight``, and
    its value depends on a run setting. This is the textbook relative Sobolev
    norm,

    .. math::

        \frac{\lVert \hat f - f \rVert^2_{L^2}
               + \lVert \nabla\hat f - \nabla f \rVert^2_{L^2}}
              {\lVert f \rVert^2_{L^2} + \lVert \nabla f \rVert^2_{L^2}}
        \quad\text{(square-rooted)},

    which has no free parameter and means the same thing in every report.
    """
    predicted_field, reference_field = prediction, target
    prediction = np.asarray(prediction, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    difference = prediction - target
    total = np.sum((target - target.mean()) ** 2)
    spread = np.ptp(target) or 1.0
    values = {
        "mse": float(np.mean(difference ** 2)),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(target)),
        "nrmse_range": float(np.sqrt(np.mean(difference ** 2)) / spread),
        "r2": float(1.0 - np.sum(difference ** 2) / total) if total > 0
        else float("nan"),
    }
    if grid is not None:
        values["relative_h1"] = sobolev_error(
            predicted_field, reference_field, grid)
        values["integral_error"] = integral_error(
            predicted_field, reference_field, grid)
        values["jsd"] = shape_divergence(predicted_field, reference_field, grid)
    return values


def sobolev_error(prediction, target, grid):
    r"""
    Relative :math:`H^1` error: values and gradients in one number.

    .. math::

        \sqrt{\frac{\lVert \Delta f \rVert^2
                     + \lVert \nabla \Delta f \rVert^2}
                    {\lVert f \rVert^2 + \lVert \nabla f \rVert^2}}

    The gradients are spectral, which is exact for the band-limited periodic
    fields a plane-wave grid carries --- a finite-difference stencil would put
    its own truncation error into a number meant to describe the model's.

    Why it earns a column beside the relative :math:`L^2`: a prediction can
    match a density pointwise and still be *rough*, and the von Weizsacker
    kinetic energy --- the dominant term wherever the density is inhomogeneous
    --- depends on :math:`\nabla\rho` rather than on :math:`\rho`. A model
    with a small :math:`L^2` and a large :math:`H^1` is one whose downstream
    energetics will be worse than its headline number suggests.

    Returns
    -------
    float or None
        ``None`` if the target has no gradient to speak of (a constant field),
        where the ratio is 0/0.
    """
    from poraque.fields.density import spectral_gradient

    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.shape != target.shape or prediction.shape != tuple(grid.shape):
        # A spin-polarised (2, ...) stack, or a shape the grid does not
        # describe: the FFT would be taken over the wrong axes and return a
        # plausible number. Better to report nothing.
        return None

    difference = prediction - target
    gradient_difference = spectral_gradient(difference, grid)
    gradient_target = spectral_gradient(target, grid)

    numerator = (np.sum(difference ** 2)
                 + sum(np.sum(component ** 2)
                       for component in gradient_difference))
    denominator = (np.sum(target ** 2)
                   + sum(np.sum(component ** 2)
                         for component in gradient_target))
    if not denominator > 0:
        return None
    return float(np.sqrt(numerator / denominator))


def integral_error(prediction, target, grid):
    r"""
    :math:`|\int \hat f\,d^3r - \int f\,d^3r|` over the cell.

    For ``ext2chg`` this is the **electron-count error in electrons**, and it is
    the one metric here that a pointwise loss controls worst: a per-voxel MSE is
    nearly indifferent to a uniform 2 % error in :math:`\rho`, while the
    electrostatic terms built from that density are of order :math:`10^4` eV,
    so the same 2 % moves a total energy by tens of eV --- by a different amount
    for every structure, so it does not cancel in a difference.

    Signed inside the absolute value, not summed as absolute voxel errors: the
    quantity being asked about is whether the *totals* agree. A prediction that
    is 1 % high in one half of the cell and 1 % low in the other conserves
    charge exactly, and should read zero here and badly under ``mae``.

    Returns
    -------
    float or None
        ``None`` when the field's shape is not the grid's.
    """
    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    if prediction.shape != target.shape or prediction.shape != tuple(grid.shape):
        return None
    return float(abs(np.sum(prediction - target)) * grid.volume_element)


def shape_divergence(prediction, target, grid):
    r"""
    Jensen-Shannon divergence between the predicted and reference densities.

    Both fields are first turned into **probability densities** — clamped at
    zero and divided by their own spatial integral, so each integrates to
    exactly one — and the divergence is taken between those. That is what
    makes the number a divergence rather than an arbitrary functional: without
    the normalisation it would mostly report the difference in electron count,
    which :math:`\int\rho` already reports directly and far more legibly.

    The consequence is that ``jsd`` measures **shape** alone. A prediction
    carrying 5 % too much charge, distributed identically, scores zero here and
    badly on the integral; a prediction with the right total in the wrong place
    does the reverse. Neither number implies the other, which is why both are
    reported.

    Returns
    -------
    float or None
        The divergence in nats, or ``None`` for a signed field — a potential is
        not a density and has no distribution to compare.
    """
    from poraque.ml.committee import jensen_shannon_divergence

    reference = np.asarray(target, dtype=float)
    # A field that is negative over a substantial fraction of the cell is not a
    # density that rang from band-limiting; it is a different kind of object,
    # and flooring it would produce a number that looks meaningful.
    if np.count_nonzero(reference < 0) > 0.01 * reference.size:
        return None
    try:
        return float(jensen_shannon_divergence(prediction, reference, grid)["jsd"])
    except ValueError:
        return None


#: Fallback width of the label column, used when the caller does not size it.
MIN_LABEL_WIDTH = 22

def metrics_label_width(names):
    """
    Width of the label column, sized so every row of the table lines up.

    Computed rather than fixed because a structure named more descriptively
    than ``struct_000`` would otherwise overflow the hard-coded width and
    quietly shove the numeric columns to the right.

    Parameters
    ----------
    names : iterable of str
        Every structure identifier the section will print.
    """
    labels = [f"{name} ({tag})" for name in names
              for tag in ("train", "VALIDATION")]
    return max([MIN_LABEL_WIDTH] + [len(label) for label in labels])


#: Columns of the per-structure table, as ``(heading, key, format)``.
#:
#: One definition for the heading, the rule and the rows, so the three cannot
#: drift apart -- the failure a hand-counted rule produces is a dashed line one
#: character short of the words above it.
METRIC_COLUMNS = (
    ("split", "split", "<10s"),
    ("MSE", "mse", "11.5g"),
    ("MAE", "mae", "11.5g"),
    ("RMSE", "rmse", "11.5g"),
    ("rel L2", "relative_l2", "10.4f"),
    ("R2", "r2", "9.4f"),
    ("JSD", "jsd", "10.3e"),
)


def metric_columns(rows):
    """
    Columns to print, dropping any the data does not carry.

    ``jsd`` is undefined for a signed field, so a run whose target is a
    potential would otherwise print a column of blanks.
    """
    present = {key for row in rows for key, value in row.items()
               if value is not None}
    return [column for column in METRIC_COLUMNS
            if column[1] in present or column[1] == "split"]


def format_metrics_header(width, columns=METRIC_COLUMNS):
    """
    Heading and rule for the per-structure table.

    Each title takes its own column's alignment, so a left-aligned value does
    not sit under a right-aligned word; and the rule is measured from the
    heading rather than counted, so it cannot end up short of it.

    The unit belongs on the section title, not here: hung off the end of the
    heading it fell outside the rule and read as a stray column.

    Returns
    -------
    list of str
        The heading and its rule.
    """
    heading = f"    {'structure':<{width}s}" + "".join(
        f" {title:{'<' if spec.startswith('<') else '>'}{_column_width(spec)}s}"
        for title, _, spec in columns)
    return [heading, "    " + "-" * (len(heading) - 4)]


def _column_width(spec):
    """Field width declared by a format spec such as ``11.5g``."""
    digits = spec.lstrip("<>^")
    return int(digits.split(".")[0].rstrip("sgfe") or 0)


def format_metrics_row(name, split, values, width=MIN_LABEL_WIDTH,
                       columns=METRIC_COLUMNS):
    """One aligned row of the per-structure table."""
    cells = []
    for _, key, spec in columns:
        value = split if key == "split" else values.get(key)
        if value is None:
            cells.append(" " * _column_width(spec))
        else:
            cells.append(f"{value:{spec}}")
    return f"    {name:<{width}s}" + "".join(f" {cell}" for cell in cells)


def format_aggregate(label, rows, log):
    """
    Mean / min / max of every metric across a set of structures.

    Printed for the training set and, when one exists, for the validation set
    -- the two answer different questions, and quoting only the first invites
    a training fit to be read as a generalisation estimate.
    """
    if not rows:
        return
    log(f"\n  --- {label} ({len(rows)} structures) ---")
    heading = f"      {'metric':<12s} {'mean':>12s}   {'min':>11s}   {'max':>11s}"
    log(heading)
    log("      " + "-" * (len(heading) - 6))
    for key in ("mse", "mae", "rmse", "relative_l2", "r2", "jsd"):
        values = [m[key] for m in rows if m.get(key) is not None]
        if not values:
            continue
        log(f"      {key:<12s} {np.mean(values):12.5g}   "
            f"{np.min(values):11.5g}   {np.max(values):11.5g}")


def build_loss(config, task_name):
    """
    Assemble the objective from the ``training`` section of the config.

    ``physics_informed`` is handed to the loss rather than resolved here: the
    answer under ``"auto"`` is a statement about the weights, and the weights
    are what the loss is built from. Resolving it in two places is how the
    objective and the training loop's decision about whether to decode a
    prediction come to disagree.
    """
    physics = config.training.physics_weights()
    enable = config.training.physics_informed_enabled
    if enable is False:
        # Reported, because a block of non-zero weights sitting under a switch
        # that turns them off is exactly the configuration someone will later
        # read as evidence that they applied.
        stated = [key for key, value in physics.items() if value]
        if stated:
            import warnings

            warnings.warn(
                f"training.physics_informed.enable is false, so "
                f"{', '.join(sorted(stated))} in the same block are inert. "
                f"Remove the switch to honour them.",
                RuntimeWarning, stacklevel=2,
            )
    if physics["euler_lagrange_weight"] and enable is not False:
        import warnings

        warnings.warn(
            "The Euler-Lagrange residual is evaluated WITHOUT v_xc, which is "
            "of the same order as the kinetic potential it is weighed "
            "against; data.xc is not consulted during training. Treat the "
            "residual as a consistency measure, not a correctness one.",
            RuntimeWarning, stacklevel=2,
        )
    return PhysicsInformedLoss(
        task=task_name,
        loss=config.training.loss,
        sobolev_weight=config.training.sobolev_weight,
        **physics,
        physics_informed=enable,
    )


# ===================================================================== #
# Leave-one-out driver
# ===================================================================== #
def resolve_baseline(task, config, cache, log):
    """
    The isolated-atom baseline for delta-density mode, or ``None``.

    Returns ``None`` for ``chg2tau`` whatever the config says: there is no
    atomic superposition of a kinetic energy density, and silently ignoring the
    flag there is better than refusing a config that sets it once for a run
    training both tasks.
    """
    if not config.data.delta_density:
        return None
    if task.target_field != "CHGCAR":
        log(f"      NOTE: data.delta_density is ignored for {task.name} -- "
            f"its target is {task.target_field}, which has no atomic "
            f"superposition.")
        return None
    reference = config.data.atomic_reference
    from poraque.fields.atomic import resolve_library

    # Unset and set-but-absent are the same failure to a run: neither can name
    # the atoms the target is defined against. They are raised as one message
    # for that reason -- `atomic_reference` has a default, and a default that
    # resolves on the machine the data lives on resolves nowhere else, so the
    # path being wrong is the likelier of the two cases rather than the rarer.
    try:
        library = (resolve_library(reference, cache=cache, log=log)
                   if reference else None)
    except FileNotFoundError as exc:
        library, why = None, str(exc)
    else:
        why = None

    if library is None:
        where = (f"data.atomic_reference is {reference!r}, and {why}"
                 if why else
                 "data.atomic_reference is unset, so nothing says where the "
                 "isolated atoms are")
        raise ValueError(
            f"data.delta_density is on (the default since 2026-08-26) and the "
            f"target is rho - rho_sup, but {where}.\n\n"
            "Point it at a directory holding one subdirectory per element, "
            "each a single-atom calculation, or at a database built from "
            "them:\n\n"
            "    data:\n"
            "      atomic_reference: data/vasp/isolated_atoms\n\n"
            "A directory is ingested on the spot and memoised into the cache. "
            "To build the database ahead of time instead:\n\n"
            "    poraque-atoms <atom dirs> --output atomic_reference.json\n\n"
            "Or set `data.delta_density: false` to train on the absolute "
            "density, which is what runs before this date did.")

    missing = _uncovered_elements(cache, library)
    if missing:
        raise ValueError(
            f"The isolated-atom database covers {library.elements()} but the "
            f"dataset also contains {missing}. A superposition missing whole "
            f"atoms is wrong in a way that looks entirely plausible, so it is "
            f"refused rather than built partially. Add an isolated-atom "
            f"calculation for {missing}, or set data.delta_density: false.")

    log(f"      delta-density mode: target is rho - rho_sup, baseline from "
        f"{len(library)} atom(s) {library.elements()}")
    log(f"      library fingerprint {library.fingerprint[:16]} "
        f"(recorded in the checkpoint)")
    return library


def _uncovered_elements(cache, library):
    """
    Elements in the cached dataset that the atomic database does not cover.

    Checked here, once, rather than at the first batch: the failure is a
    property of the pair (dataset, library) and is knowable before a single
    epoch runs. Discovering it two hours in, from a KeyError inside a
    DataLoader worker, is the same information delivered uselessly.

    A cache that cannot be read yields no complaint -- the dataset layer raises
    its own, better-worded error a moment later.
    """
    from poraque.fields.atomic import base_element
    from poraque.fields.vasp.volumetric import read_structure_header

    if not os.path.isdir(cache):
        return []

    covered, seen = set(library.elements()), set()
    for entry in sorted(os.listdir(cache)):
        density = os.path.join(cache, entry, "CHGCAR")
        if not os.path.exists(density):
            continue
        try:
            structure = read_structure_header(density)
        except (OSError, ValueError):
            continue
        seen.update(base_element(symbol) for symbol in structure.symbols)
    return sorted(seen - covered)


def build_operator(task, train_set, config, log):
    """Construct the operator, attaching the Pauli head when requested."""
    source_transform = train_set.input_transform
    target_transform = train_set.target_transform

    head = {}
    if config.model.pauli_residual and task.name == "chg2tau":
        from poraque.ml import fit_pauli_scale, pauli_bound_violation

        scale = (config.model.pauli_scale if config.model.pauli_scale
                 else fit_pauli_scale(train_set))
        head = {"pauli_residual": True, "pauli_scale": scale,
                "learn_pauli_scale": config.model.learn_pauli_scale}
        log(f"      head: tau = tau_vW[rho] + s*softplus(f)   s = {scale:.4f} eV/Ang^3")
        for entry in pauli_bound_violation(train_set):
            if entry["violations"]:
                log(f"      note: {entry['material']} violates tau >= tau_vW at "
                    f"{entry['violations']}/{entry['points']} points "
                    f"({100 * entry['fraction']:.4f} %)")

    # The dataset knows its channel counts (two for a spin-polarised density);
    # the operator must be built to match or a 2-channel target would train
    # against a broadcast 1-channel prediction with no error anywhere.
    in_channels, out_channels = train_set.channels
    if (in_channels, out_channels) != (1, 1):
        log(f"      channels: {in_channels} in, {out_channels} out "
            f"(spin-polarised density)")

    # init_seed reaches FieldOperator, which isolates the draw from the global
    # stream; the manual_seed here keeps the ambient behaviour when it is unset.
    torch.manual_seed(config.training.seed)
    operator = FieldOperator(
        task, input_transform=source_transform, target_transform=target_transform,
        device=config.training.device,
        strict_device=config.training.strict_device,
        training_resolution=config.data.resolution,
        init_seed=config.training.init_seed,
        in_channels=in_channels, out_channels=out_channels,
        # From the *dataset*, not re-resolved from the config: the operator's
        # baseline has to be bit-for-bit the one the targets were built with,
        # and reading the file twice is one more chance for them to differ.
        baseline=getattr(train_set, "baseline", None),
        **config.model_kwargs(), **head,
    )
    if config.model.precision != "float32":
        operator.set_precision(config.model.precision)
    log(f"      model: {type(operator.model).__name__} width={config.model.width} "
        f"modes={config.model.modes} layers={config.model.n_layers}  "
        f"({operator.model.n_parameters():,} parameters)")
    if config.model.precision != "float32":
        log(f"      precision: {config.model.precision} — roughly twice the "
            f"time and memory of the float32 default")
    report_mode_selection(train_set, config, log)
    return operator


def resolve_strict_device(config, log):
    """
    Resolve ``training.device``, reporting rather than raising a traceback.

    ``strict_device`` exists to stop a run early, and a stack trace is the wrong
    way to say so: the reader of a job's error file needs the diagnosis and the
    device report, not the four frames it took to get there. Both go into the
    log, which is where a batch job's output is actually read.

    Returns
    -------
    torch.device

    Raises
    ------
    SystemExit
        When ``strict_device`` is set and the request cannot be honoured.
    """
    try:
        return resolve_device(config.training.device,
                              strict=config.training.strict_device)
    except RuntimeError as error:
        log("")
        log(f"  {error}")
        log("")
        for line in device_report(config.training.device):
            log(f"    {line}")
        raise SystemExit(
            f"training.device: {config.training.device!r} could not be "
            f"honoured and training.strict_device is set, so this run stops "
            f"here rather than spending a GPU allocation on the CPU. "
            f"Set strict_device: false to allow the fallback.") from None


def loader_settings(config, distributed=None):
    """
    The ``train()`` keywords that govern *how* data reaches the device.

    Collected in one place because both training paths — the split fit and each
    fold of the cross-validation — must pass all of them, and the k-fold path
    has already dropped a keyword the split path had (``dtype``, once) and
    quietly changed what it was measuring.

    Parameters
    ----------
    config : TrainingConfig
    distributed : DistributedContext, optional
        Passed straight through. A fold that forgot it would train on the
        *whole* dataset on every rank and average four identical gradients,
        which is not an error anywhere and is four times the work for the same
        model — hence its being in this dict rather than at the call sites.

    Returns
    -------
    dict
    """
    return {
        "num_workers": config.training.num_workers,
        "pin_memory": config.training.pin_memory,
        "distributed": distributed,
    }


def format_bytes(count):
    """``12.3 MiB`` for a byte count, in binary units."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024.0
    return f"{value:.1f} GiB"  # pragma: no cover - unreachable


def report_field_cache(train_set, validation, config, log):
    """
    Say whether the decoded fields are held in RAM, and what that costs.

    Worth a line of its own because it is the single largest performance
    setting in the file and its effect is invisible from the results: a run
    that re-parses every field every epoch produces exactly the same numbers as
    one that does not, ten times slower. On the data this was measured on the
    loader was 59 % of the training loop and the GPU sat at 2-4 %.

    The size is reported whichever way ``auto`` went, since "caching was
    declined" is only actionable beside the number that declined it.
    """
    cost = train_set.cache_bytes + (validation.cache_bytes if validation else 0)
    requested = config.data.cache_in_memory
    if train_set.cache:
        log(f"  field cache         : in RAM, ~{format_bytes(cost)} "
            f"(data.cache_in_memory: {requested})")
        return
    log("  field cache         : OFF -- every epoch re-reads and re-parses "
        "every field")
    log(f"                        ~{format_bytes(cost)} decoded "
        f"(data.cache_in_memory: {requested}"
        + (f", budget {format_bytes(CACHE_MEMORY_BUDGET)}"
           if str(requested).lower() == "auto" else "") + ")")


def report_mode_selection(train_set, config, log):
    r"""
    Say how many modes the run will actually use, per structure.

    ``mode_selection: physical`` truncates at a constant wavevector rather than
    a constant index, keeping :math:`\lfloor G_{\max} L_i / 2\pi \rfloor` modes
    along an axis of length :math:`L_i`. That count is then **capped** by
    ``modes`` and by the grid, so ``g_max`` can only ever take modes away.

    Which makes a badly chosen ``g_max`` invisible: it costs capacity silently,
    with no error and nothing in the log. A ``g_max`` of 6 on a 4 Å cell keeps
    3 of 8 modes — about 5 % of the spectral parameters the model allocated —
    and the run would otherwise look identical to one that used all of them.
    """
    if config.model.mode_selection != "physical" or config.model.g_max is None:
        return

    ceiling = config.model.modes
    counts = []
    # The cells come from the dataset's own samples rather than from a second
    # read of the files: they are already the geometry `physical_modes` will be
    # handed at forward time.
    for index in range(len(train_set)):
        cell = np.asarray(train_set[index]["cell"], dtype=float)
        lengths = np.linalg.norm(cell, axis=-1)
        counts.append([min(int(ceiling),
                           max(1, int(np.floor(config.model.g_max * length
                                               / (2.0 * np.pi)))))
                       for length in lengths])
    if not counts:
        return

    counts = np.asarray(counts)
    lowest, highest = int(counts.min()), int(counts.max())
    log(f"      mode selection: physical, g_max = {config.model.g_max:g} "
        f"1/Ang -> {lowest}-{highest} of {ceiling} modes retained")
    if highest < ceiling:
        starved = int((counts < ceiling).all(axis=1).sum())
        log(f"      NOTE: g_max truncates every structure below the {ceiling} "
            f"modes the model allocates.")
        log(f"      {starved}/{len(counts)} structures use fewer on every "
            f"axis; the unused spectral weights are dead parameters.")
        log(f"      A cell needs L >= {ceiling * 2 * np.pi / config.model.g_max:.1f} "
            f"Ang to supply {ceiling} modes at this g_max. Raise g_max, or "
            f"lower modes.")


def load_pretrained_operator(task, train_set, validation, config, log):
    r"""
    Start from a trained checkpoint instead of a fresh initialisation.

    Two things are taken from the checkpoint rather than the config, and both
    matter:

    **The architecture**, inferred from the stored tensors. A width or mode
    count remembered in the config that disagreed with the weights could only
    load mismatched tensors, so the tensors are the authority.

    **The normalizations.** The datasets are re-pointed at the checkpoint's
    transforms, discarding the ones just fitted to this data. Refitting would
    rescale the network's inputs out from under weights trained against the old
    scale, which throws away most of what pre-training bought — the model would
    spend the fine-tune relearning the scale rather than the chemistry.

    Returns
    -------
    tuple of (FieldOperator, dict)
        The operator, and a record of what was loaded for the report.
    """
    from poraque.ml import bundle_tasks, freeze_lifting_layers, load_bundle

    settings = config.fine_tuning
    path = settings.pretrained_checkpoint
    if not os.path.exists(path):
        raise SystemExit(
            f"fine_tuning.pretrained_checkpoint={path!r} does not exist. "
            f"Train a base model first, or point at one.")

    available = bundle_tasks(path)
    if task.name not in available:
        raise SystemExit(
            f"{path} holds no {task.name!r} model; it contains {available}. "
            f"Fine-tuning needs a base model for the task being trained.")

    operator = load_bundle(path, task.name, device=config.training.device)
    log(f"      fine-tuning from : {path}")
    log(f"      architecture     : inferred from the checkpoint "
        f"({operator.model.n_parameters():,} parameters)")

    # The checkpoint's normalizations replace the ones fitted above.
    for dataset in (train_set, validation):
        if dataset is not None:
            dataset.input_transform = operator.input_transform
            dataset.target_transform = operator.target_transform
    log(f"      transforms       : taken from the checkpoint "
        f"(in {operator.input_transform}  out {operator.target_transform})")

    counts = {"frozen": 0, "trainable": operator.model.n_parameters()}
    adaptation = "full fine-tune"
    if settings.use_lora:
        from poraque.ml.lora import apply_lora

        # Everything is frozen and the adapters added; `freeze_lifting_layers`
        # would be a no-op on top of that and is skipped rather than applied
        # to nothing, so the report cannot claim two mechanisms are running.
        counts = apply_lora(operator.model, rank=settings.lora_rank,
                            alpha=settings.lora_alpha,
                            dropout=settings.lora_dropout)
        # Recorded on the operator so `state()` can write a checkpoint that
        # holds the adapter and names the base, instead of the whole model.
        operator.lora = {
            "rank": int(settings.lora_rank),
            "alpha": float(settings.lora_alpha),
            "dropout": float(settings.lora_dropout),
            "base_checkpoint": os.path.abspath(path),
        }
        adaptation = (f"LoRA r={settings.lora_rank} "
                      f"alpha={settings.lora_alpha:g}")
        share = 100.0 * counts["trainable"] / max(
            counts["trainable"] + counts["frozen"], 1)
        log(f"      LoRA             : {counts['adapters']} adapted layer(s), "
            f"rank {settings.lora_rank}, alpha {settings.lora_alpha:g}")
        log(f"      trainable        : {counts['trainable']:,} of "
            f"{counts['trainable'] + counts['frozen']:,} parameters "
            f"({share:.3f}%); the rest is frozen")
        log("      NOTE: the checkpoint will hold the ADAPTER only and name "
            "this base;")
        log(f"            it cannot be loaded without {os.path.basename(path)}.")
        if settings.freeze_lifting_layers:
            log("      NOTE: fine_tuning.freeze_lifting_layers is ignored "
                "under LoRA, which")
            log("            freezes every base weight already.")
    elif settings.freeze_lifting_layers:
        counts = freeze_lifting_layers(operator.model)
        adaptation = "full fine-tune, lifting frozen"
        log(f"      frozen           : lifting path, "
            f"{counts['frozen']:,} parameters; {counts['trainable']:,} "
            f"remain trainable")

    return operator, {
        "pretrained checkpoint": path,
        "fine-tuned": "yes",
        "adaptation": adaptation,
        "fine-tuning learning rate": f"{settings.learning_rate:g}",
        "lifting layers": ("frozen" if (settings.freeze_lifting_layers
                                        or settings.use_lora)
                           else "trainable"),
        "trainable parameters": f"{counts['trainable']:,}",
        "frozen parameters": f"{counts['frozen']:,}",
    }


def loss_summary(history):
    """
    Final-epoch objective, for the report table.

    One number, whatever the objective was made of. A physics-informed run's
    ``train_loss`` already *is* the total the optimiser stepped on -- data
    fidelity plus every weighted constraint -- and that total is what a reader
    compares against another run.

    Returns
    -------
    dict
        Rows to merge into the report summary.
    """
    if not history.get("train_loss"):
        return {}
    return {"final train loss": f"{history['train_loss'][-1]:.5f}"}


def extract_resource_usage(history):
    """
    Remove the cost measurements from ``history`` and return them.

    :func:`poraque.ml.train` records what the run *cost* — wall time per epoch,
    and on CUDA the peak allocated and reserved device memory — beside what it
    *achieved*. They are separated here because :func:`split_history` sorts by
    type rather than by meaning: a list becomes a plotted curve and a scalar
    becomes an early-stopping summary, so left in place a byte count would be
    filed under early stopping and the timings would be serialised a second
    time as a loss curve.

    Parameters
    ----------
    history : dict
        Mutated: the three keys are popped.

    Returns
    -------
    dict
        Always all three keys, ``None`` where the backend does not report one,
        so a reader can tell "not measured" from "absent from this version".
    """
    return {
        "seconds_per_epoch": history.pop("seconds_per_epoch", None),
        "peak_vram_bytes": history.pop("peak_vram_bytes", None),
        "peak_vram_reserved_bytes": history.pop("peak_vram_reserved_bytes", None),
    }


def split_history(history):
    """
    Separate the per-epoch curves in ``history`` from the scalar summaries.

    :func:`poraque.ml.train` returns lists keyed by ``train_loss``,
    ``val_error`` and ``val_epoch``, and -- whenever a validation split exists
    -- four scalars beside them: ``best_epoch``, ``best_error``,
    ``stopped_early`` and ``val_metric`` (the norm the first two are measured
    in). Serialising the two together is what makes this worth a
    function: ``list(map(float, v))`` over an ``int`` raises ``TypeError``, and
    a validation split is the default, so the failure is on the common path.

    Returns
    -------
    tuple of (dict, dict or None)
        The curves as lists of floats, and the scalars, or ``None`` when
        training ran without validation and produced none.
    """
    curves = {key: list(map(float, value)) for key, value in history.items()
              if isinstance(value, list)}
    scalars = {key: value for key, value in history.items()
               if not isinstance(value, list)}
    return curves, (scalars or None)


def resolve_validation_split(dataset, config):
    """
    Choose the structures held out for validation.

    The split is defined by one number, ``training.valid_fraction``: that share
    of the structures is drawn at random and kept back. ``0`` trains on every
    structure, at the cost of turning the reported metrics into a training fit.

    The draw is at the **structure level**: whole materials move together.
    Splitting voxels instead would put the same crystal on both sides, and
    because neighbouring voxels are strongly correlated the validation score
    would look excellent while saying nothing about a new material.

    Parameters
    ----------
    dataset : FieldPairDataset
        The full dataset, for its material list.
    config : TrainingConfig

    Returns
    -------
    tuple of (set of str, str)
        Held-out identifiers, and a human-readable description of the split —
        logged so the record shows the protocol, not only the result.
    """
    names = [m.identifier for m in dataset.materials]
    fraction = float(config.training.valid_fraction or 0.0)

    # Range-check before the zero case: a negative value is a typo, and
    # treating it as "no split" would silently change the protocol rather than
    # reporting the mistake.
    if not 0.0 <= fraction < 1.0:
        raise SystemExit(
            f"training.valid_fraction must lie in [0, 1), got {fraction}."
        )

    if fraction == 0.0:
        return set(), "none - trained on every structure, metrics are TRAINING FIT"
    if len(names) < 2:
        raise SystemExit(
            f"valid_fraction needs at least two structures to split; the "
            f"dataset has {len(names)}."
        )

    # Round to nearest, then clamp so neither side is empty: a fraction that
    # rounds to zero would silently train on everything while the config
    # claims a validation split.
    count = int(round(fraction * len(names)))
    count = max(1, min(count, len(names) - 1))

    seed = config.training.seed
    order = np.random.default_rng(seed).permutation(len(names))
    selected = {names[i] for i in order[:count]}
    return selected, (f"valid_fraction={fraction:g} -> {count}/{len(names)} "
                      f"structures, seed={seed}")


def format_names(names, indent="      ", per_line=None, width=78):
    """
    Structure identifiers as indented, wrapped lines.

    A raw ``['struct_000', 'struct_001', ...]`` of seventeen entries is one
    unreadable line that wraps wherever the terminal happens to end. Laid out
    in columns it can be scanned, and a missing structure is visible.

    Parameters
    ----------
    names : sequence of str
    indent : str, optional
    per_line : int, optional
        Columns; derived from the longest name and ``width`` when omitted.
    width : int, optional
        Target line width, matching the section rules elsewhere.

    Returns
    -------
    list of str
    """
    names = [str(name) for name in names]
    if not names:
        return []
    column = max(len(name) for name in names) + 2
    per_line = per_line or max(1, (width - len(indent)) // column)
    return [indent + "".join(f"{name:<{column}s}" for name in chunk).rstrip()
            for chunk in (names[i:i + per_line]
                          for i in range(0, len(names), per_line))]


def format_shapes(buckets, indent="                        "):
    """
    Grid shapes and how many structures carry each, one per line.

    Replaces a raw list of every structure's shape *and* a separate bucket
    line: the two said the same thing, and neither was readable past a handful
    of structures.

    Parameters
    ----------
    buckets : dict
        ``{(nx, ny, nz): count}``.

    Returns
    -------
    list of str
        The first line carries no indent, so the caller can put it after a
        label; the rest are aligned under it.
    """
    lines = []
    for shape, count in sorted(buckets.items()):
        text = "x".join(str(n) for n in shape)
        lines.append(f"{text:<14s} {count:>3d} "
                     f"structure{'s' if count != 1 else ''}")
    return [lines[0]] + [indent + line for line in lines[1:]] if lines else []


def plot_channel(field):
    """
    The single channel a cross-section or parity figure should draw.

    A :class:`~poraque.fields.SpinDensity` stacks its two channels into
    ``data``, and a figure handed that stack silently becomes nonsense — or,
    as it happened, hands Matplotlib a three-axis "image" and dies after the
    training has already run. The density channel is the one these figures are
    about; the magnetisation has its own column in the metrics table.

    Parameters
    ----------
    field : ScalarField or SpinDensity

    Returns
    -------
    ScalarField or numpy.ndarray
        Unchanged for a single-channel field, so the figure keeps the grid and
        structure it would otherwise have.
    """
    total = getattr(field, "total", None)
    return field if total is None else total


def compare_fields(prediction, target):
    r"""
    Metrics for one predicted field against its reference, channel-correctly.

    **The reduction to the density channel is the point of this function.**
    ``operator.predict`` returns a :class:`~poraque.fields.SpinDensity`
    whenever ``data.spin`` resolved on --- which is every model trained on the
    platinum data --- and ``SpinDensity.data`` is a ``(2, Nx, Ny, Nz)`` stack.
    Handed straight to :func:`metrics`, every number in the report is then
    taken over :math:`\rho` *and* :math:`m` together, in a table whose unit
    column says :math:`e/\mathrm{\AA}^3`.

    That is not a small distortion, and it is not conservative. A prediction
    exact in :math:`\rho` and wrong only in the magnetisation was measured
    reporting ``relative_l2`` 1.3e-3, ``mae`` 7.9e-4 and ``max_abs`` 6.1e-3 ---
    a density error that does not exist. And ``relative_h1`` and
    ``integral_error`` came back ``None``, because a ``(2, ...)`` stack is not
    the grid's shape, so the two metrics added for the split table would have
    been silently absent from every spin-polarised run's report.

    This is the same defect the six inference sites carried until 2026-09-02,
    in a seventh place: :func:`~poraque.fields.base.charge_channel` exists
    precisely so a call site that does not know which kind of field it was
    handed can be written once.

    The magnetisation is **reported rather than folded in**, under
    ``magnetisation_relative_l2``. Dropping it would leave half of a
    two-channel model's output unmeasured, which is the mirror of the mistake
    being fixed.

    Returns
    -------
    dict
        As :func:`metrics`, plus ``magnetisation_relative_l2`` for a
        two-channel field.
    """
    density, reference = charge_channel(prediction), charge_channel(target)
    values = metrics(density.data, reference.data, grid=reference.grid)

    moment = getattr(target, "magnetization", None)
    if moment is not None and getattr(prediction, "magnetization", None) is not None:
        difference = np.asarray(prediction.magnetization) - np.asarray(moment)
        scale = np.linalg.norm(np.asarray(moment))
        values["magnetisation_relative_l2"] = (
            float(np.linalg.norm(difference) / scale) if scale > 0 else None)
    return values


def dataset_metric_probe(operator, dataset):
    """
    Metrics for one material, to decide which columns the table needs.

    Costs one extra prediction. The alternative is buffering every row until
    the last is known, which would hold the whole section back from a terminal
    that is otherwise reporting progress as it goes.
    """
    source, target = dataset.load_fields(0)
    return compare_fields(operator.predict(source), target)


def evaluate_material(operator, dataset, index, log, label,
                      width=MIN_LABEL_WIDTH, split="", columns=None):
    """
    Predict one material and report metrics against its reference field.

    Parameters
    ----------
    width : int, optional
        Label-column width, from :func:`metrics_label_width`. Pass the width
        computed for the whole section, not per row, or the rows will not line
        up with one another.
    split : str, optional
        ``"train"`` or ``"validation"``, printed as its own column rather than
        glued to the name -- so the structure column holds structures.
    columns : sequence, optional
        Subset of :data:`METRIC_COLUMNS`; defaults to all of them.
    """
    source, target = dataset.load_fields(index)
    prediction = operator.predict(source)
    # Through `compare_fields`, not `metrics` directly: a spin-polarised
    # prediction is a (2, ...) stack, and every number in this table is a
    # statement about the density alone.
    values = compare_fields(prediction, target)
    log(format_metrics_row(label, split, values, width,
                           columns or METRIC_COLUMNS))
    return prediction, target, values


def validate_fine_tuning_settings(config):
    """
    Check the fine-tuning settings before anything is trained.

    Every failure here is knowable from the config alone, and every one of them
    would otherwise surface *after* the fit — a missing checkpoint at load time,
    a name collision only when the result is written. Neither is worth an hour
    of GPU to discover.
    """
    settings = config.fine_tuning
    if not settings.enable:
        return

    path = settings.pretrained_checkpoint
    if not path:
        raise SystemExit("fine_tuning.enable is set but "
                         "fine_tuning.pretrained_checkpoint is empty.")
    if not os.path.exists(path):
        legacy = resolve_bundle_path(path)
        if legacy == path:
            raise SystemExit(
                f"fine_tuning.pretrained_checkpoint={path!r} does not exist. "
                f"Train a base model first, or point at one.")
        settings.pretrained_checkpoint = legacy

    if settings.learning_rate <= 0:
        raise SystemExit(
            f"fine_tuning.learning_rate={settings.learning_rate!r} must be "
            f"positive.")

    if settings.use_lora:
        if settings.lora_rank <= 0:
            raise SystemExit(
                f"fine_tuning.lora_rank={settings.lora_rank!r} must be "
                f"positive: a rank-0 correction is identically zero, so the "
                f"run would train nothing and report a flat loss curve.")
        if settings.lora_alpha <= 0:
            raise SystemExit(
                f"fine_tuning.lora_alpha={settings.lora_alpha!r} must be "
                f"positive; it scales the correction by alpha / rank.")
        if not 0.0 <= settings.lora_dropout < 1.0:
            raise SystemExit(
                f"fine_tuning.lora_dropout={settings.lora_dropout!r} must be "
                f"in [0, 1).")

    destination = bundle_path(config)
    if destination:
        if os.path.abspath(destination) == os.path.abspath(
                settings.pretrained_checkpoint):
            raise SystemExit(
                f"the fine-tuned model would be written over its own base "
                f"checkpoint at {destination}. Point output.root "
                f"somewhere else.")


def compute_dtype(config):
    """
    The torch dtype the batches must be produced in.

    Set by ``model.precision``, not by ``data.precision``: the first governs
    what the operator computes in, and a dataset yielding float32 into a
    float64 model fails with ``Input type (float) and bias type (double)
    should be the same``. The two settings are separate on purpose -- fields
    may be held in float64 and fed to a float32 operator -- but the tensor
    handed to the model is the model's business.

    Returns
    -------
    torch.dtype
    """
    return PRECISIONS[config.model.precision][0]


def validate_precision_settings(config):
    """
    Check the two precision settings before anything is read or built.

    Both are names rather than dtypes, so a typo is caught here — at the cost
    of a millisecond, on the command line — instead of surfacing as a numpy or
    torch error somewhere inside the cache build.
    """
    if config.data.precision not in FIELD_DTYPES:
        raise SystemExit(
            f"data.precision={config.data.precision!r} is not known; expected "
            f"one of {sorted(FIELD_DTYPES)}. This is how the volumetric fields "
            f"are stored in memory.")
    if config.model.precision not in PRECISIONS:
        raise SystemExit(
            f"model.precision={config.model.precision!r} is not known; "
            f"expected one of {sorted(PRECISIONS)}. This is what the operator "
            f"computes in.")
    if (config.data.precision == "float16"
            and config.model.precision == "float64"):
        raise SystemExit(
            "data.precision='float16' with model.precision='float64' asks for "
            "double-precision arithmetic on data that carries about three "
            "decimal digits. Raise data.precision, or lower model.precision.")

    # Apple's Metal backend has no float64 at all -- not slow, absent. Caught
    # here rather than left to surface as a TypeError from inside the first
    # conversion, and refused rather than quietly rerouted to the CPU: that
    # substitution changes the run's speed by an order of magnitude and is the
    # user's call to make.
    if config.model.precision == "float64":
        device = resolve_device(config.training.device,
                                strict=config.training.strict_device)
        if device.type == "mps":
            raise SystemExit(
                f"model.precision='float64' cannot run on {describe_device(device)}: "
                f"the Metal backend does not implement double precision.\n"
                f"  Set training.device: cpu to run this in float64, or "
                f"model.precision: float32 to keep the accelerator.\n"
                f"  (data.precision is unaffected — fields may still be held "
                f"in float64 while the operator computes in float32.)")


def validate_loss_settings(config):
    """
    Check ``training.loss`` before the cache is built, not an hour in.

    A typo in an objective is a run that trains on the wrong thing or, with the
    old ``"sobolev"`` spelling, on the right thing under a name that no longer
    says which norm. Either way the place to find out is in the first second.
    """
    from poraque.ml.losses import resolve_data_loss

    try:
        resolve_data_loss(config.training.loss)
    except ValueError as error:
        raise SystemExit(f"training.{error}") from None


def validate_activation_settings(config):
    """
    Resolve ``model.activation`` and ``model.kan_setup`` before anything runs.

    The block is only expanded when the model is built, which is after the
    cache. A typo in ``kan_setup.variant`` would otherwise surface once every
    field has been downsampled, having changed nothing about that work.
    """
    try:
        config.model.activation_kwargs()
    except ValueError as error:
        raise SystemExit(str(error))


def validate_equivariance_settings(config):
    """
    Resolve ``model.equivariant`` and its block before anything runs.

    Same timing argument as :func:`validate_activation_settings`, and one more
    besides: ``equivariant: true`` beside the default ``use_coordinates: true``
    is a contradiction the constructor refuses, and refusing it here means the
    run says so on the command line rather than after the cache is built.
    """
    try:
        config.model.equivariant_kwargs()
    except ValueError as error:
        raise SystemExit(str(error))


def validate_physics_settings(config):
    """
    Resolve ``training.physics_informed`` before anything runs.

    ``true`` with every weight at zero is refused, and refusing it here rather
    than in :func:`build_loss` is the difference between a message on the
    command line and one an hour later, after the cache: the objective is
    built per task, and the first task is built after the fields are prepared.
    """
    from poraque.ml.losses import PhysicsInformedLoss

    try:
        PhysicsInformedLoss(
            task="ext2chg",
            **config.training.physics_weights(),
            physics_informed=config.training.physics_informed_enabled,
        )
    except (ValueError, TypeError) as error:
        raise SystemExit(str(error))


def validate_symbolic_settings(settings):
    """
    Check the symbolic settings before anything is trained.

    Distillation runs *after* the fit, so a typo in ``features`` would
    otherwise surface an hour in, with the search — not the training — as the
    only casualty but the feedback uselessly late. Checked here it costs a
    millisecond and fails on the command line.
    """
    if not settings.enable:
        return
    if settings.template not in TEMPLATES:
        raise SystemExit(
            f"symbolic.template={settings.template!r} is not known; expected "
            f"one of {list(TEMPLATES)}.")
    if settings.features not in FEATURE_SCHEMES:
        raise SystemExit(
            f"symbolic.features={settings.features!r} is not a known feature "
            f"scheme; expected one of {list(FEATURE_SCHEMES)}.")
    if settings.target not in ("model", "reference"):
        raise SystemExit(
            f"symbolic.target={settings.target!r} is not known; expected "
            f"'model' (distil the trained operator) or 'reference' (fit the "
            f"DFT data).")
    if settings.epsilon <= 0:
        raise SystemExit(
            f"symbolic.epsilon={settings.epsilon!r} must be positive: it is a "
            f"density floor and clamps every denominator.")
    if settings.data_loss not in DATA_LOSSES:
        raise SystemExit(
            f"symbolic.data_loss={settings.data_loss!r} is not known; expected "
            f"one of {sorted(DATA_LOSSES)}.")
    physics = symbolic_physics(settings)
    unknown = set(physics) - {"enable", "positivity_weight",
                              "thomas_fermi_weight", "von_weizsacker_weight",
                              "p_infinity"}
    if unknown:
        raise SystemExit(
            f"Unknown key(s) in symbolic.physics: {sorted(unknown)}.\n"
            f"  Note that this block constrains the symbolic *search*. The "
            f"terms that constrain the neural operator -- electron count, "
            f"Euler-Lagrange -- belong in training.physics_informed.")
    if physics["enable"] and float(physics["p_infinity"]) <= 0:
        raise SystemExit(
            f"symbolic.physics.p_infinity={physics['p_infinity']!r} must be "
            f"positive: it is the reduced gradient standing in for the von "
            f"Weizsacker limit, and p is a magnitude.")
    for key in ("positivity_weight", "thomas_fermi_weight",
                "von_weizsacker_weight"):
        if float(physics[key]) < 0:
            raise SystemExit(
                f"symbolic.physics.{key}={physics[key]!r} must not be "
                f"negative: a negative penalty rewards the violation it is "
                f"meant to forbid.")
    # The floor was 10 while PySR was the engine, because its tournament
    # selection draws 10 individuals by default and refuses to run when the
    # population cannot supply them. `poraque.ml.gp` draws three, so the real
    # requirement is now that a tournament can choose between anything at all.
    # Relaxing it rather than leaving it is the point: a limit kept after the
    # dependency that justified it has gone is a limit nobody can explain.
    if settings.population_size < 4:
        raise SystemExit(
            f"symbolic.population_size={settings.population_size} is too "
            f"small: the search selects by tournament and cannot choose "
            f"between fewer than four individuals.")


def save_task_checkpoint(task, operator, config, log):
    """
    Persist one task's weights immediately, before any optional analysis.

    Returns
    -------
    str or None
        Path written, or ``None`` when checkpointing is switched off.
    """
    run = config.run_dir()
    if run is None or not config.output.checkpoint:
        return None

    path = os.path.join(run,
                        f"{model_name(config)}_{task.name}_trained{BUNDLE_SUFFIX}")
    save_bundle(path, {task.name: operator},
                metadata={"note": "single-task safety copy, written before "
                                  "the optional post-training analyses; "
                                  "superseded by the unified bundle"})
    log(f"  weights secured -> {path}")
    return path


def _figure_sink(config):
    """
    Somewhere to write the symbolic parity plot when no plot directory is set.

    Returns ``None`` when no PDF report is being built either — with neither
    output configured there is nothing the figure could appear in, and writing
    it would only litter.
    """
    if not config.report_dir():
        return None

    from poraque.vis import TrainingReport

    return TrainingReport(os.path.join(config.report_dir(),
                                       f"{model_name(config)}_figures"),
                          dpi=config.output.dpi, fmt=config.output.plot_format,
                          save_data=config.output.save_raw_plot_data)


def run_symbolic_distillation(task, dataset, operator, config, log,
                              validation=None, report=None):
    r"""
    Search for a closed-form expression reproducing the trained operator.

    Only ``chg2tau`` is attempted. Distillation looks for a *functional of the
    density*, and ``ext2chg`` maps a potential to a density — the Hohenberg-Kohn
    map, whose whole content is non-local, so a semi-local expression for it
    would be meaningless rather than merely inaccurate.

    Returns
    -------
    SymbolicResult or None
        ``None`` when the feature is off, the task is not ``chg2tau``, or the
        search failed. A failure is reported and swallowed: it arrives after
        the model is already trained, and losing a finished fit to an optional
        analysis would be the worse outcome.
    """
    settings = config.symbolic
    if not settings.enable:
        return None
    if task.name != "chg2tau":
        log(f"\n  symbolic distillation: skipped for {task.name} "
            f"(a density functional is only meaningful for chg2tau).")
        return None

    from poraque.ml.symbolic import distill_dataset

    log(f"\n{'=' * 78}")
    log(f"SYMBOLIC DISTILLATION - {task.name}")
    log("=" * 78)
    start = time.time()
    try:
        result = distill_dataset(dataset, settings, operator=operator, log=log,
                                 validation=validation)
    except ImportError as error:
        log(f"  unavailable: {error}")
        return None
    except (ValueError, RuntimeError) as error:
        log(f"  failed: {error}")
        return None

    # The formula gets the same parity plot the network does, against the same
    # DFT reference, so the two are read on one scale rather than through two
    # differently-defined summary numbers.
    #
    # It is drawn whenever distillation produced something to draw, not only
    # when figures were asked for: the PDF report is meant to carry it, and
    # `plot_figures` and `write_pdf_report` are independent settings, so a run with a
    # report but no figure directory would otherwise silently lose the one plot
    # that shows whether the formula is any good. Held-out data is used when
    # there is any, and the fitted voxels otherwise, with the caption saying
    # which.
    scored, provenance = result.validation, "held-out data"
    if scored.get("reference") is None:
        scored, provenance = result.fitted, "the fitted voxels (training fit)"

    destination = report if report is not None else _figure_sink(config)
    if destination is not None:
        label_text, unit = FIELD_LABELS[task.target_field]
        previous, destination.prefix = (destination.prefix,
                                        f"{task.name}_symbolic")
        try:
            if scored.get("reference") is not None:
                result.parity_plot = destination.parity(
                    scored["reference"], scored["predicted"],
                    name="parity", label=label_text, unit=unit, log=True,
                    prediction_label="symbolic formula",
                    title=f"{task.name} · lowest loss "
                          f"({result.complexity} nodes) on {provenance}")
                log(f"  parity plot  : {result.parity_plot}  [{provenance}]")

            # The knee gets its own parity plot, on the same voxels. Two
            # panels of one number each say less than two plots that can be
            # laid side by side: the question is whether the shorter formula
            # gives anything away, and that is visible rather than summarised.
            knee_scored = (result.knee_validation
                           if result.knee_validation.get("reference") is not None
                           else result.knee_fitted)
            if knee_scored.get("reference") is not None:
                result.knee_parity_plot = destination.parity(
                    knee_scored["reference"], knee_scored["predicted"],
                    name="parity_knee", label=label_text, unit=unit, log=True,
                    prediction_label="symbolic formula",
                    title=f"{task.name} · Pareto knee "
                          f"({result.knee.get('complexity')} nodes) on "
                          f"{provenance}")
                log(f"  knee parity  : {result.knee_parity_plot}")

            if result.pareto:
                result.pareto_plot = destination.pareto(
                    result.pareto, knee=result.knee, name="pareto",
                    title=f"{task.name} · accuracy against complexity")
                log(f"  pareto plot  : {result.pareto_plot}")
        finally:
            destination.prefix = previous

    log(f"\n{result.summary()}")
    log(f"  search time  : {time.time() - start:.1f} s")
    log("")
    log("  NOTE: the features are semi-local, so this is the best semi-local")
    log("  functional matching the operator -- not a reconstruction of it. The")
    log("  residual measures how much of the learned map is non-local.")
    return result


def run_task(task_name, cache, config, log, n_tasks=1, distributed=None):
    r"""
    Train one model for ``task_name`` on a train/validation split.

    This is the ordinary path; K-fold cross-validation is the only variation on
    it. **One** model is fitted across all the training structures, never one
    per material: batches are drawn across structures — the sampler groups by
    grid shape and shuffles both within and across those groups — so a gradient
    step generally mixes several.

    ``training.valid_fraction`` holds back a fifth of the structures by
    default, so the reported score is genuinely held out and ``early_stopping``
    has something to watch. Set it to ``0`` when the goal is the final artefact
    trained on *all* the data — in which case the metrics become a **training
    fit** and carry no generalisation claim.

    Parameters
    ----------
    n_tasks : int, optional
        How many tasks this run trains in total. It only names the report:
        with one task the PDF is ``<name>_report.pdf``, with two it has to
        carry the task as well or the second would overwrite the first.
    distributed : DistributedContext, optional
        The process group, forwarded to :func:`~poraque.ml.training.train`. The
        *data* work above is done identically on every rank — the split, the
        transforms, the field cache — and only the batches are partitioned;
        fitting the transforms per rank on a partition would give the four
        replicas four different normalisations.
    """
    task = resolve_task(task_name)
    log(f"\n{'=' * 78}")
    log(f"TASK  {task.name}:  {task.input_field} -> {task.target_field}")
    log(f"      {task.description}")
    log("=" * 78)

    baseline = resolve_baseline(task, config, cache, log)
    dataset = FieldPairDataset(cache, task=task, spin=config.data.spin,
                               dtype=compute_dtype(config), baseline=baseline,
                               cache=config.data.cache_in_memory)
    validation_names, split_origin = resolve_validation_split(dataset, config)

    train_records = [m for m in dataset.materials
                     if m.identifier not in validation_names]
    test_records = [m for m in dataset.materials
                    if m.identifier in validation_names]

    train_set = FieldPairDataset(cache, task=task, materials=train_records,
                                 spin=config.data.spin,
                                 dtype=compute_dtype(config), baseline=baseline,
                                 cache=config.data.cache_in_memory)
    source_transform, target_transform = train_set.fit_transforms()
    validation = (FieldPairDataset(cache, task=task, materials=test_records,
                                   input_transform=source_transform,
                                   target_transform=target_transform,
                                   spin=config.data.spin,
                                   dtype=compute_dtype(config),
                                   baseline=baseline,
                                   cache=config.data.cache_in_memory)
                  if test_records else None)
    report_field_cache(train_set, validation, config, log)

    shapes = train_set.shapes()
    buckets = {}
    for shape in shapes:
        buckets[tuple(shape)] = buckets.get(tuple(shape), 0) + 1

    log(f"  training structures : {len(train_set)}")
    for line in format_names([m.identifier for m in train_records]):
        log(line)
    if validation_names:
        log(f"  validation          : {len(validation_names)}  ({split_origin})")
        for line in format_names(sorted(validation_names)):
            log(line)
    else:
        log(f"  validation          : none  ({split_origin})")
    shape_lines = format_shapes(buckets)
    log(f"  grid shapes         : {shape_lines[0]}")
    for line in shape_lines[1:]:
        log(line)
    log(f"  batch size          : {config.training.batch_size} "
        f"(capped per bucket; batches mix structures of equal shape)")
    log(f"  transforms          : in {source_transform}  out {target_transform}")

    fine_tuning = None
    if config.fine_tuning.enable:
        operator, fine_tuning = load_pretrained_operator(
            task, train_set, validation, config, log)
    else:
        operator = build_operator(task, train_set, config, log)

    # A fine-tune replaces the base learning rate: continuing at it would walk
    # the weights away from the solution being adapted before a small dataset
    # could constrain them.
    learning_rate = (config.fine_tuning.learning_rate if fine_tuning
                     else config.training.learning_rate)
    if fine_tuning:
        log(f"      learning rate    : {learning_rate:g} "
            f"(fine-tuning; base training uses "
            f"{config.training.learning_rate:g})")

    # Early stopping needs something to watch. With no validation split the
    # library would warn -- rightly, for a caller who asked for both -- but here
    # the two are merely the shipped defaults, and a default configuration must
    # not warn. Say it plainly in the log instead.
    patience = config.training.early_stopping
    if patience and validation is None:
        log(f"  early stopping      : inactive ({patience} epochs requested, "
            f"but nothing is held out to measure improvement against)")
        patience = 0

    log(f"\n  progress (every {config.training.eval_epoch} epochs):")
    start = time.time()
    history = train(
        operator, train_set, validation=validation,
        epochs=config.training.epochs, batch_size=config.training.batch_size,
        learning_rate=learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer=config.training.optimizer,
        scheduler=config.training.scheduler, grad_clip=config.training.grad_clip,
        loss=build_loss(config, task.name), seed=config.training.seed,
        eval_every=config.training.eval_epoch, early_stopping=patience,
        log=log, verbose=True,
        **loader_settings(config, distributed),
    )
    elapsed = time.time() - start
    # Lifted out before anything else reads `history`: they are run *cost*
    # rather than run *quality*, and leaving them in would put a byte count
    # among the early-stopping scalars and a second copy of the per-epoch
    # timings among the loss curves that `split_history` serialises.
    resources = extract_resource_usage(history)
    log(f"\n  trained {len(history['train_loss'])}/{config.training.epochs} epochs in {elapsed:.1f} s   "
        f"loss {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")

    # ---------------- per-material evaluation ---------------- #
    label_text, unit = FIELD_LABELS[task.target_field]
    per_material, figures = {}, []
    report = None
    showcase = None
    figure_dir = plot_directory(config)
    if figure_dir:
        from poraque.vis import TrainingReport

        report = TrainingReport(figure_dir, dpi=config.output.dpi,
                                fmt=config.output.plot_format,
                                prefix=f"{task.name}",
                                save_data=config.output.save_raw_plot_data)
        figures.append(report.loss_curves(
            history, title=f"{task.name} ({len(train_set)} training structures)"))

    unit_note = f"  [{task.target_unit}]" if task.target_unit else ""
    log(f"\n  per-structure results "
        f"({'TRAINING FIT' if not validation_names else 'train / validation'})"
        f"{unit_note}:")
    label_width = metrics_label_width(
        [record.identifier for record in train_records + test_records])

    # The heading is printed once, and the columns are chosen from what the
    # task actually produces -- a signed target has no JSD, and a column of
    # blanks is worse than no column.
    probe = dataset_metric_probe(operator, train_set)
    columns = metric_columns([probe])
    for line in format_metrics_header(label_width, columns):
        log(line)

    for index in range(len(train_set)):
        name = train_records[index].identifier
        prediction, target, values = evaluate_material(
            operator, train_set, index, log, name,
            width=label_width, split="train", columns=columns)
        per_material[name] = {"split": "train", "metrics": values,
                              "predicted_integral": field_integral(prediction),
                              "reference_integral": field_integral(target)}
        if report is not None and index == 0:
            report.prefix = f"{task.name}_{name}"
            showcase = (plot_channel(target), plot_channel(prediction))
            figures.append(report.field_comparison(
                *showcase, label=label_text, unit=unit,
                log=(task.target_field in ("CHGCAR", "TAUCAR")),
                title=f"{task.name} · {name}"))

    held_out = None
    if validation is not None:
        for index in range(len(validation)):
            name = test_records[index].identifier
            prediction, target, values = evaluate_material(
                operator, validation, index, log, name,
                width=label_width, split="validation", columns=columns)
            per_material[name] = {"split": "validation", "metrics": values,
                                  "predicted_integral": field_integral(prediction),
                                  "reference_integral": field_integral(target)}
            if index == 0:
                held_out = (plot_channel(target), plot_channel(prediction))

    # The parity plot is drawn after both loops so it can carry the held-out
    # structure beside the training one. Two clouds on shared axes show the
    # generalisation gap directly -- a validation cloud visibly wider about the
    # identity line is the same story the aggregate numbers tell, but visible
    # rather than inferred.
    if report is not None and showcase is not None:
        figures.append(report.parity(
            *showcase, validation=held_out, label=label_text, unit=unit,
            log=(task.target_field in ("CHGCAR", "TAUCAR"))))

    # ---------------- aggregate ---------------- #
    train_metrics = [v["metrics"] for v in per_material.values()
                     if v["split"] == "train"]
    validation_metrics = [v["metrics"] for v in per_material.values()
                          if v["split"] == "validation"]
    format_aggregate(f"{task.name}: aggregate over training", train_metrics, log)
    # The held-out set gets the same treatment. Quoting only the training
    # aggregate is how a training fit gets read as a generalisation estimate,
    # and the two numbers are usually not close.
    format_aggregate(f"{task.name}: aggregate over VALIDATION",
                     validation_metrics, log)

    if not validation_names:
        log("\n      NOTE: valid_fraction is 0, so nothing was held out and these")
        log("      are TRAINING-FIT numbers. They show the model can represent the")
        log("      data; they are not a generalisation estimate. Raise")
        log("      valid_fraction, or run --kfold, for that.")

    # ---------------- persist ---------------- #
    # The operator is handed back to main(), which writes every task into one
    # unified bundle after the last has trained. Writing per task would either
    # leave the file half-updated on a crash or retain a stale sibling from an
    # earlier run.
    checkpoint = None
    if figures:
        log(f"  figures         -> {figure_dir} ({len(figures)})")
    if report is not None and report.data_files:
        log(f"  raw plot data   -> {figure_dir} "
            f"({len(report.data_files)} files)")

    # ---------------- the weights, before anything optional ---------------- #
    # The unified bundle is written once every task has trained, which is
    # correct but leaves the trained weights unpersisted while the optional
    # analyses below run. Symbolic distillation is a long stochastic search
    # over expressions -- minutes to hours, on data the fit has already
    # produced -- and losing a finished fit to a post-hoc analysis is not a
    # trade worth making, so a single-task copy goes to disk first and the
    # unified bundle supersedes it. (The sharper version of this argument was
    # PySR's Julia runtime, which could take the whole process down after
    # training had finished. That engine is gone; the ordering is still right.)
    save_task_checkpoint(task, operator, config, log)

    # ---------------- symbolic distillation ---------------- #
    symbolic = run_symbolic_distillation(task, train_set, operator, config, log,
                                         validation=validation, report=report)

    # ---------------- PDF report ---------------- #
    pdf = None
    if config.report_dir():
        from poraque.vis import ModelReport

        caveats = [
            chemistry_caveat(len(train_set), dataset_elements(cache)),
        ]
        if not validation_names:
            caveats.append(
                "No structure was held out, so every number in the table is "
                "training fit, not a generalisation estimate."
            )
        if any(row["metrics"].get("magnetisation_relative_l2") is not None
               for row in per_material.values()):
            # The table's units say e/Ang^3 and its numbers are the density's.
            # A two-channel model also predicts a magnetisation, and a reader
            # who assumes the table covers everything the model outputs is
            # reading half of it.
            caveats.append(
                "This is a two-channel model. Every number in the table is "
                "the DENSITY channel; the magnetisation is measured "
                "separately, as magnetisation_relative_l2 in the metrics JSON."
            )
        reporter = ModelReport(config.report_dir())
        summary = {
            "name": model_name(config),
            "weights": bundle_path(config) or "not saved",
            "model": type(operator.model).__name__,
            "parameters": f"{operator.model.n_parameters():,}",
            "training structures": str(len(train_set)),
            # A count, not the shapes themselves. The shapes are one line per
            # distinct grid and the distinct grids are a property of the
            # dataset, so an MP set put dozens of tuples into a summary table
            # meant to be read at a glance. `format_shapes` still writes every
            # one of them to the run log, where length costs nothing.
            "grid resolutions": str(len(buckets)),
            "epochs": str(config.training.epochs),
            "batch size": str(config.training.batch_size),
            "device": describe_device(operator.device),
            "training time": f"{elapsed:.1f} s",
        }
        summary.update(loss_summary(history))
        if fine_tuning:
            # Ahead of the rest: a reader who misses this reads every number
            # below as belonging to a model trained from scratch.
            summary = {**fine_tuning, **summary}
            caveats.insert(0, (
                "This is a FINE-TUNED model, adapted from "
                f"{fine_tuning['pretrained checkpoint']}. Its scores describe "
                "the material family it was specialised on, and say nothing "
                "about the broader set the base model was trained across."))
        pdf = reporter.build(
            task=task.name, per_material=per_material, figures=figures,
            unit=task.target_unit, caveats=caveats,
            filename=report_filename(config, task.name, n_tasks),
            summary=summary,
            configuration={f"{section}.{key}": value
                           for section, values in config.to_dict().items()
                           if isinstance(values, dict)
                           for key, value in values.items()},
            symbolic=symbolic,
        )
        log(f"  PDF report      -> {pdf}")

    curves, stopping = split_history(history)
    return {
        "task": task.name,
        "mode": "split",
        "report": pdf,
        "n_train": len(train_set),
        "train_structures": [m.identifier for m in train_records],
        "validation": sorted(validation_names),
        "grid_shapes": [list(s) for s in shapes],
        "per_material": per_material,
        "checkpoint": checkpoint,
        "figures": figures,
        "seconds": elapsed,
        # What a batch-size study needs, recorded by the run that produced it.
        # Before this the only route to either number was sampling
        # `nvidia-smi` from outside the process, which cannot separate one task
        # of a two-task run from the other and is gone by the time anyone reads
        # the results. `seconds_per_epoch` is measured after a device
        # synchronisation, so it is compute rather than submission.
        **resources,
        "final_train_loss": history["train_loss"][-1],
        "history": curves,
        "early_stopping": stopping,
        "fine_tuning": fine_tuning,
        "symbolic": (result_to_dict(symbolic) if symbolic is not None
                     else None),
        # Not serialised into the JSON summary; main() pops it to build the
        # unified bundle once every task has trained.
        "operator": operator,
    }


def structure_level_folds(names, k, seed=0):
    r"""
    Partition **whole structures** into ``k`` validation groups.

    The split is at the structure level by construction: a fold is a set of
    material names, and every voxel of a material goes to the same side. A
    voxel-level split would place the same material in both training and
    validation, and the resulting score would measure interpolation *within* a
    material rather than transfer *to a new* one — the number would look
    excellent and mean nothing.

    Parameters
    ----------
    names : sequence of str
        Structure identifiers.
    k : int
        Requested number of folds; capped at ``len(names)``.
    seed : int, optional
        Shuffling seed, so the partition is reproducible.

    Returns
    -------
    list of list of str
        ``k`` disjoint groups whose union is ``names``.
    """
    names = list(names)
    k = max(2, min(int(k), len(names)))
    order = np.random.default_rng(seed).permutation(len(names))
    return [[names[i] for i in group]
            for group in np.array_split(order, k) if len(group)]


def run_task_kfold(task_name, cache, config, log, n_tasks=1, distributed=None):
    r"""
    K-fold cross-validation over structures.

    Each fold trains a fresh model on :math:`K-1` groups of structures and
    scores it on the held-out group, in physical units. The result is a
    *generalisation estimate* with a spread, not a deployable model: it fits
    *K* of them. Leave ``enable_kfold`` off for the artefact to ship.

    ``valid_fraction`` is ignored here — the folds define the splits.

    Parameters
    ----------
    distributed : DistributedContext, optional
        Forwarded to each fold. Every rank runs every fold, over its share of
        that fold's batches; the folds are *not* distributed across ranks. That
        would be the better parallelisation — the folds are independent, and
        splitting them needs no communication at all — but it changes what the
        run produces per rank, and this is the version that can be compared
        against a single-device k-fold line for line.
    """
    task = resolve_task(task_name)
    baseline = resolve_baseline(task, config, cache, log)
    dataset = FieldPairDataset(cache, task=task, spin=config.data.spin,
                               dtype=compute_dtype(config), baseline=baseline,
                               cache=config.data.cache_in_memory)
    names = [m.identifier for m in dataset.materials]
    folds = structure_level_folds(names, config.training.k_folds,
                                  config.training.seed)
    by_name = {m.identifier: m for m in dataset.materials}

    log(f"\n{'=' * 78}")
    log(f"TASK  {task.name}:  {task.input_field} -> {task.target_field}   "
        f"[{len(folds)}-FOLD CROSS-VALIDATION]")
    log(f"      {task.description}")
    log("=" * 78)
    log(f"  structures : {len(names)}")
    log(f"  folds      : {len(folds)} (requested {config.training.k_folds})")
    if len(folds) == len(names):
        log("               = leave-one-out, since k equals the structure count")
    for index, group in enumerate(folds, 1):
        log(f"     fold {index}: validate on {group}")
    log("  split is at the STRUCTURE level: no material appears on both sides")

    label_text, unit = FIELD_LABELS[task.target_field]
    records, figures, fold_resources = [], [], []
    # Sized once over every structure, so the column does not shift from fold
    # to fold as different names land in the validation group.
    label_width = metrics_label_width(names)

    for index, group in enumerate(folds, 1):
        log(f"\n  --- fold {index}/{len(folds)}: validate on {group} ---")
        train_records = [by_name[n] for n in names if n not in group]
        val_records = [by_name[n] for n in group]

        train_set = FieldPairDataset(cache, task=task, materials=train_records,
                                     spin=config.data.spin,
                                     dtype=compute_dtype(config),
                                     baseline=baseline,
                                     cache=config.data.cache_in_memory)
        source_transform, target_transform = train_set.fit_transforms()
        # dtype was dropped here once, so a float64 run crashed at the first
        # validation pass with a float/double mismatch. `cache` was dropped
        # here too, in the change that added it -- with the same shape of
        # consequence, a fold that silently re-parsed every file every epoch.
        val_set = FieldPairDataset(cache, task=task, materials=val_records,
                                   input_transform=source_transform,
                                   target_transform=target_transform,
                                   spin=config.data.spin,
                                   dtype=compute_dtype(config),
                                   baseline=baseline,
                                   cache=config.data.cache_in_memory)
        if index == 1:
            report_field_cache(train_set, val_set, config, log)
        log(f"      train on {len(train_set)}: {[m.identifier for m in train_records]}")

        operator = build_operator(task, train_set, config, log)
        start = time.time()
        history = train(
            operator, train_set, validation=val_set,
            epochs=config.training.epochs, batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            optimizer=config.training.optimizer,
            scheduler=config.training.scheduler,
            grad_clip=config.training.grad_clip,
            loss=build_loss(config, task.name), seed=config.training.seed,
            eval_every=config.training.eval_epoch,
            early_stopping=config.training.early_stopping,
            log=log, verbose=True,
            **loader_settings(config, distributed),
        )
        elapsed = time.time() - start
        # Per fold, and kept per fold: K models are fitted here, so a single
        # peak would be whichever fold happened to be largest with nothing
        # saying which.
        fold_resources.append({"fold": index, "seconds": elapsed,
                               **extract_resource_usage(history)})
        log(f"      trained {len(history['train_loss'])}/{config.training.epochs} epochs in {elapsed:.1f} s   "
            f"loss {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")

        # Same table as the single-fit path, headed once per fold: without a
        # heading these rows are seven unlabelled numbers.
        fold_columns = metric_columns(
            [dataset_metric_probe(operator, val_set)])
        for line in format_metrics_header(label_width, fold_columns):
            log(line)
        for position in range(len(val_set)):
            name = val_records[position].identifier
            prediction, target, values = evaluate_material(
                operator, val_set, position, log, name,
                width=label_width, split=f"fold {index}",
                columns=fold_columns)
            records.append({"fold": index, "material": name,
                            "split": f"fold {index}", "metrics": values,
                            "predicted_integral": field_integral(prediction),
                            "reference_integral": field_integral(target)})
            if plot_directory(config) and position == 0:
                from poraque.vis import TrainingReport

                report = TrainingReport(
                    plot_directory(config), dpi=config.output.dpi,
                    fmt=config.output.plot_format,
                    prefix=f"{task.name}_fold{index}_{name}",
                    save_data=config.output.save_raw_plot_data)
                figures.append(report.loss_curves(
                    history, title=f"{task.name} · fold {index}"))
                figures.append(report.field_comparison(
                    target, prediction, label=label_text, unit=unit,
                    log=(task.target_field in ("CHGCAR", "TAUCAR")),
                    title=f"{task.name} · fold {index} · {name}"))

    # ---------------- aggregate across folds ---------------- #
    log(f"\n  --- {task.name}: {len(folds)}-fold summary "
        f"({len(records)} validation structures) ---")
    aggregate = {}
    heading = (f"      {'metric':<12s} {'mean':>12s} {'+/- std':>13s} "
               f"{'min':>11s}   {'max':>11s}")
    log(heading)
    log("      " + "-" * (len(heading) - 6))
    for key in ("mse", "mae", "rmse", "relative_l2", "r2", "jsd"):
        scored = [r["metrics"][key] for r in records
                  if r["metrics"].get(key) is not None]
        if not scored:
            continue
        values = np.array(scored, dtype=float)
        aggregate[key] = {"mean": float(values.mean()), "std": float(values.std()),
                          "min": float(values.min()), "max": float(values.max())}
        log(f"      {key:<12s} {values.mean():12.5g} {values.std():>13.4g} "
            f"{values.min():11.5g}   {values.max():11.5g}")

    log("\n      These ARE generalisation numbers: every score above comes from")
    log("      a model that never saw that structure. The spread across folds")
    log("      matters as much as the mean with a dataset this small.")

    # ---------------- consolidated report ---------------- #
    pdf = None
    if config.report_dir():
        from poraque.vis import ModelReport

        per_material = {r["material"]: {"split": r["split"],
                                        "metrics": r["metrics"]}
                        for r in records}
        reporter = ModelReport(config.report_dir())
        pdf = reporter.build(
            task=task.name, per_material=per_material, figures=figures,
            unit=task.target_unit,
            filename=report_filename(config, task.name, n_tasks,
                                     kind="kfold_report"),
            summary={
                "name": model_name(config),
                "protocol": f"{len(folds)}-fold cross-validation",
                "split level": "structure (whole materials held out)",
                "structures": str(len(names)),
                "validation scores": str(len(records)),
                "epochs per fold": str(config.training.epochs),
                "device": describe_device(
                    resolve_device(config.training.device,
                                   strict=config.training.strict_device)),
                "mean relative L2": f"{aggregate['relative_l2']['mean']:.5g}"
                                    f" +/- {aggregate['relative_l2']['std']:.3g}",
                "mean MAE": f"{aggregate['mae']['mean']:.5g}"
                            f" +/- {aggregate['mae']['std']:.3g}",
                "mean MSE": f"{aggregate['mse']['mean']:.5g}"
                            f" +/- {aggregate['mse']['std']:.3g}",
            },
            caveats=[
                "Every score is from a model that never saw that structure, so "
                "these are generalisation estimates rather than training fit.",
                chemistry_caveat(len(names), dataset_elements(cache)),
                "With few folds the spread is wide; quote the standard "
                "deviation alongside the mean.",
            ],
            configuration={f"{section}.{key}": value
                           for section, values in config.to_dict().items()
                           if isinstance(values, dict)
                           for key, value in values.items()},
        )
        log(f"\n  consolidated report -> {pdf}")

    return {"task": task.name, "mode": "kfold", "n_folds": len(folds),
            "folds": [{"fold": i, "validate_on": g}
                      for i, g in enumerate(folds, 1)],
            "records": records, "aggregate": aggregate,
            "resources": fold_resources,
            "report": pdf, "figures": figures}


# ===================================================================== #
# Entry point
# ===================================================================== #
def build_parser():
    """Command-line interface. Every override defaults to ``None``.

    A ``None`` default is what lets the resolver distinguish "flag absent" from
    "flag set to a falsy value", so ``--early-stopping 0`` disables early
    stopping instead of being mistaken for an unset flag.
    """
    parser = argparse.ArgumentParser(
        description="Train Poraque's Fourier Neural Operators from a YAML config.",
    )
    parser.add_argument("--config", default=None,
                        help="YAML configuration file (defaults are used if omitted)")

    parser.add_argument("--task", dest="task.type", default=None,
                        choices=["all", "ext2chg", "chg2tau"])
    parser.add_argument("--name", dest="task.name", default=None,
                        metavar="NAME",
                        help="name this run's outputs: models/NAME.poraque, "
                             "reports/NAME_report.pdf and a NAME/ subdirectory "
                             "of the plot directory (default: poraque_models)")
    group = parser.add_argument_group("data overrides")
    group.add_argument("--data-paths", dest="data.data_paths", nargs="+",
                       default=None, metavar="DIR",
                       help="one or more dataset directories. Each holds one "
                            "subdirectory per material, whatever produced it "
                            "-- a VASP run tree, a poraque-mp download, a "
                            "cache from an earlier run -- and all of them are "
                            "pooled. What a material's directory holds is "
                            "read, not declared")
    group.add_argument("--cache", dest="data.cache", default=None)
    group.add_argument("--pattern", dest="data.pattern", default=None)
    group.add_argument("--format", dest="data.format", default=None,
                       choices=["auto", "vasp"],
                       help="the DFT code that wrote the files, or 'auto' to "
                            "detect it per directory (default)")
    group.add_argument("--resolution", dest="data.resolution", type=int, default=None)
    group.add_argument("--potcar-dir", dest="data.potcar_dir", default=None,
                       metavar="DIR",
                       help="POTCAR library (<dir>/Ag/POTCAR, ...). Used where "
                            "the data ships no pseudopotentials -- an MP "
                            "download, or a run whose POTCAR was stripped -- "
                            "to build the exact tabulated external potential "
                            "instead of the Gaussian model")
    group.add_argument("--sigma", dest="data.sigma", type=float, default=None,
                       metavar="A",
                       help="Gaussian pseudo-ion width in Angstrom for the "
                            "computed external potential; defaults to the "
                            "POTCAR core radius where one is available")
    group.add_argument("--gaussian-blur", dest="data.gaussian_blur", type=float,
                       default=None,
                       help="Gaussian blur width in Angstrom for the computed "
                            "external potential")
    group.add_argument("--blur-method", dest="data.blur_method", default=None,
                       choices=["spectral", "ndimage"])

    group = parser.add_argument_group("model overrides")
    group.add_argument("--width", dest="model.width", type=int, default=None)
    group.add_argument("--modes", dest="model.modes", type=int, default=None)
    group.add_argument("--layers", dest="model.n_layers", type=int, default=None)
    group.add_argument("--projection", dest="model.projection_channels",
                       type=int, default=None)
    group.add_argument("--pauli-head", dest="model.pauli_residual",
                       action="store_const", const=True, default=None,
                       help="tau = tau_vW[rho] + s*softplus(f) for chg2tau")
    group.add_argument("--no-pauli-head", dest="model.pauli_residual",
                       action="store_const", const=False, default=None)

    group = parser.add_argument_group("training overrides")
    group.add_argument("--epochs", dest="training.epochs", type=int, default=None)
    group.add_argument("--batch-size", dest="training.batch_size", type=int,
                       default=None)
    group.add_argument("--optimizer", dest="training.optimizer", default=None,
                       choices=list(OPTIMIZERS),
                       help="adamw (default), adam or sgd")
    group.add_argument("--learning-rate", dest="training.learning_rate",
                       type=float, default=None)
    group.add_argument("--valid-fraction", dest="training.valid_fraction",
                       type=float, default=None, metavar="F",
                       help="fraction of structures held out for validation "
                            "(default 0.2); 0 trains on every structure and "
                            "reports a training fit")
    group.add_argument("--kfold", dest="training.enable_kfold",
                       action="store_const", const=True, default=None,
                       help="run K-fold cross-validation instead, ignoring "
                            "--valid-fraction")
    group.add_argument("--k-folds", dest="training.k_folds", type=int,
                       default=None, help="number of folds (capped at the "
                                          "number of structures; equal to it "
                                          "is leave-one-out)")
    group.add_argument("--eval-epoch", dest="training.eval_epoch", type=int,
                       default=None, metavar="N",
                       help="evaluate and log every N epochs (default 10)")
    group.add_argument("--early-stopping", dest="training.early_stopping",
                       type=int, default=None, metavar="N",
                       help="stop after N epochs without validation "
                            "improvement and restore the best weights "
                            "(default 300); 0 disables. Needs a validation "
                            "split")
    group.add_argument("--seed", dest="training.seed", type=int, default=None,
                       help="seeds the split, the folds and the batch order")
    group.add_argument("--init-seed", dest="training.init_seed", type=int,
                       default=None, metavar="N",
                       help="seed the weight initialisation only, leaving the "
                            "data pipeline on --seed. Vary it across runs to "
                            "build a query-by-committee ensemble")
    group.add_argument("--device", dest="training.device", default=None,
                       help="auto | cuda | mps | cpu")
    group.add_argument("--strict-device", dest="training.strict_device",
                       action="store_const", const=True, default=None,
                       help="abort instead of falling back to the CPU when "
                            "--device cannot be honoured; put this in every "
                            "batch job, where a silent fallback spends the GPU "
                            "allocation not using a GPU")
    group.add_argument("--distributed", dest="training.distributed",
                       default=None, choices=("auto", "off"),
                       help="auto | off -- form a DistributedDataParallel "
                            "group over NCCL when the launcher describes one "
                            "(a Slurm step with several tasks, or torchrun). "
                            "'off' runs single-GPU inside a multi-task "
                            "allocation, which is how a scaling run is "
                            "bisected. It cannot create ranks: the launcher "
                            "decides the topology")
    group.add_argument("--cache-in-memory", dest="data.cache_in_memory",
                       default=None,
                       help="auto | true | false -- keep decoded fields in RAM "
                            "between epochs (default auto). Off, every epoch "
                            "re-parses every file; measured at 10x on real "
                            "data")
    group.add_argument("--num-workers", dest="training.num_workers", type=int,
                       default=None,
                       help="DataLoader worker processes (default 0). Prefer "
                            "the in-memory cache: added to it, workers make "
                            "training slower, not faster")
    group = parser.add_argument_group("fine-tuning")
    group.add_argument("--fine-tune", dest="fine_tuning.enable",
                       action="store_const", const=True, default=None,
                       help="start from a trained checkpoint instead of a "
                            "fresh initialisation, to specialise it on this "
                            "dataset")
    group.add_argument("--pretrained", dest="fine_tuning.pretrained_checkpoint",
                       default=None, metavar="PATH",
                       help="base bundle to adapt (default: "
                            "models/poraque_models.poraque)")
    group.add_argument("--fine-tune-lr", dest="fine_tuning.learning_rate",
                       type=float, default=None, metavar="LR",
                       help="learning rate for the fine-tune, replacing "
                            "--learning-rate")
    group.add_argument("--freeze-lifting",
                       dest="fine_tuning.freeze_lifting_layers",
                       action="store_const", const=True, default=None,
                       help="hold the input lifting path fixed and train the "
                            "rest")

    group = parser.add_argument_group("symbolic distillation")
    group.add_argument("--symbolic",
                       dest="symbolic.enable",
                       action="store_const", const=True, default=None,
                       help="after training, search for a closed-form "
                            "expression reproducing the chg2tau operator")
    group.add_argument("--symbolic-target", dest="symbolic.target",
                       choices=["model", "reference"], default=None,
                       help="fit the operator's predictions (model) or the "
                            "DFT data (reference)")
    group.add_argument("--symbolic-features", dest="symbolic.features",
                       choices=list(FEATURE_SCHEMES), default=None,
                       help="variables handed to the engine: (rho, p, q), the "
                            "enhancement factor on (p, q), or the dimensional "
                            "(rho, |grad rho|, lap rho)")
    group.add_argument("--symbolic-template", dest="symbolic.template",
                       choices=list(TEMPLATES), default=None,
                       help="factorise the target before the search: "
                            "thomas_fermi fits the enhancement factor F in "
                            "tau = tau_TF * F")
    group.add_argument("--symbolic-epsilon", dest="symbolic.epsilon",
                       type=float, default=None, metavar="RHO",
                       help="vacuum threshold in e/a0^3; denominators are "
                            "clamped at it and voxels below it dropped")
    group.add_argument("--symbolic-iterations", dest="symbolic.iterations",
                       type=int, default=None, metavar="N")

    group = parser.add_argument_group("output overrides")
    group.add_argument("--log", dest="output.log", default=None)
    group.add_argument("--json", dest="output.json", default=None)
    group.add_argument("--output-root", dest="output.root", default=None,
                       metavar="DIR",
                       help="parent of the run folder; every artefact goes in "
                            "<DIR>/<name>/ (default: models)")

    group.add_argument("--no-report", dest="output.write_pdf_report",
                       action="store_false", default=None,
                       help="skip the PDF report")
    group.add_argument("--no-plots", action="store_true",
                       help="skip figure generation")
    return parser


def run(argv=None):
    """Parse ``argv``, train every requested task, and return the results."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Flags that steer this function rather than describing a run; they must
    # not be fed to `apply_overrides`, which would look for a config section
    # named after each of them.
    NOT_SETTINGS = ("config", "no_plots")

    config = (TrainingConfig.from_yaml(args.config) if args.config
              else TrainingConfig())
    overrides = {k: v for k, v in vars(args).items() if k not in NOT_SETTINGS}
    config.apply_overrides(overrides)

    if args.no_plots:
        config.output.plot_figures = False
    validate_fine_tuning_settings(config)
    validate_symbolic_settings(config.symbolic)
    validate_precision_settings(config)
    validate_loss_settings(config)
    validate_activation_settings(config)
    validate_equivariance_settings(config)
    validate_physics_settings(config)

    # Resolved before anything else, because it decides *which* device this
    # process may use and whether it is allowed to write. Never raises: without
    # a Slurm or torchrun launch it returns a disabled context and everything
    # below takes the single-device path it always took.
    context = discover_distributed(config.training.distributed)
    if context:
        # Each rank owns one GPU, and it is `cuda:<local_rank>` rather than
        # whatever `auto` would pick -- which is `cuda:0` for all four,
        # contending for one device and reporting itself as a scaling failure.
        config.training.device = context.device
        initialize_distributed(context,
                               timeout_minutes=config.training.distributed_timeout)
    if not context.is_main:
        # Rank 0 alone writes. Turned off here, on this process's own copy of
        # the config, rather than guarded at each of the dozen places something
        # is written: `bundle_path`, `plot_directory`, `report_dir`,
        # `log_path` and `json_path` all read these flags, so switching them
        # off once switches off every writer including the ones added later.
        # The empty strings are for the two artefacts a config can name
        # explicitly, which would otherwise bypass their own directory flag.
        config.output.checkpoint = False
        config.output.plot_figures = False
        config.output.write_pdf_report = False
        config.output.write_log = False
        config.output.log = ""
        config.output.json = ""
        config.output.save_raw_plot_data = False

    log = Tee(config.log_path(), silent=not context.is_main)
    try:
        # Through the Tee, so the environment that produced a run is recorded
        # in its log rather than only shown once on a terminal that is long
        # gone by the time anyone reads the results.
        banner(log)
        # Process-wide, and set before anything reads a field: this governs how
        # the volumetric data is held in memory, so it has to be in force
        # before the cache is built rather than applied to fields afterwards.
        set_default_dtype(config.data.precision)
        device = resolve_strict_device(config, log)
        log("=" * 78)
        log("Poraque - Fourier Neural Operator training")
        log("=" * 78)
        # The versions, the platform and the interpreter are already on screen:
        # `poraque.banner()` prints them at import. Repeating torch's version
        # here said the same thing twice and pushed what is specific to *this
        # run* -- the device it resolved to, the config it read, where the
        # results will land -- further down the page.
        log(f"  run    : {model_name(config)}")
        log(f"  device : {describe_device(device)}  (requested "
            f"{config.training.device!r})")
        # A device that did not resolve to what was asked for is the failure
        # this run is most likely to be quietly wasting itself on, so when it
        # happens the full report goes into the log -- torch build, CUDA
        # runtime, where torch was imported from, the architectures it carries
        # kernels for -- rather than one line saying it ended up on the CPU.
        requested_kind = str(config.training.device).split(":")[0].lower()
        if requested_kind not in ("auto", "", device.type):
            log(f"  WARNING: {config.training.device!r} was requested and this "
                f"run is on {device.type}. Set training.strict_device: true to "
                f"make that an error instead.")
            for line in device_report(config.training.device):
                log(f"           {line}")
        if enable_tf32(device, config.training.tf32):
            log("  tf32   : enabled for matmul and cudnn (no effect before "
                "Ampere)")
        # Printed whether or not a group formed. The usual failure is a
        # submission script that requests four GPUs and launches one task,
        # which leaves SLURM_NTASKS at 1 and looks from inside the process
        # exactly like the single-GPU run somebody asked for -- so the
        # variables that were actually present go into the log either way.
        for index, line in enumerate(describe_distributed(context)):
            prefix = "  ranks  : " if index == 0 else "           "
            log(f"{prefix}{line}")
        if context:
            log(f"           effective batch = {config.training.batch_size} "
                f"x {context.world_size} ranks = "
                f"{config.training.batch_size * context.world_size}")
        log(f"  config : {args.config or '<built-in defaults>'}")
        # Every artefact now lives under one directory, so the four separate
        # path lines this replaces were four repetitions of the same prefix.
        run_dir = config.run_dir()
        if run_dir:
            log(f"  output : {run_dir}{os.sep}")
            entries = [
                (os.path.basename(bundle_path(config))
                 if bundle_path(config) else None, "weights"),
                ("log/" if config.log_dir() else None, "log, metrics, config"),
                ("plots/" if plot_directory(config) else None, "figures"),
                ("report/" if config.report_dir() else None, "PDF report"),
            ]
            # Sized from the names rather than fixed. The fixed 22 was wide
            # enough for `<name>.pfno` and is not for `<name>.poraque`, which
            # ran the filename straight into its description.
            column = max([len(name) for name, _ in entries if name] + [22]) + 2
            for index, (name, what) in enumerate(entries):
                glyph = "\u2514\u2500\u2500" if index == len(entries) - 1 else "\u251c\u2500\u2500"
                if name:
                    log(f"           {glyph} {name:<{column}s}{what}")
                else:
                    log(f"           {glyph} ({what}: not written)")
        else:
            log("  output : nothing written (output.root is null)")
        log("")
        log("  configuration")
        for line in config.describe().splitlines():
            log(f"    {line}")
        log("")

        # One rank builds the prepared cache; the rest wait and then read it.
        # Four processes spectrally downsampling into one directory write the
        # same files concurrently, and the loser of that race gets a truncated
        # CHGCAR that parses -- the format has no length field -- into a field
        # of the wrong shape. The barrier is the whole of the fix, and it is
        # why `distributed_timeout` defaults to half an hour: this is where the
        # non-writing ranks spend a cold read of the source data.
        if context.is_main:
            cache = build_cache(config, log)
        barrier(context)
        if not context.is_main:
            cache = build_cache(config, log)
        names = trainable_tasks(config.task.names(), cache, log)
        # One protocol, one variation: K-fold cross-validation.
        driver = run_task_kfold if config.training.enable_kfold else run_task
        if config.training.enable_kfold:
            # These two settings do nothing under K-fold, and a run that
            # silently ignores what its config asked for is worse than one
            # that says so.
            if config.fine_tuning.enable:
                log("  NOTE: --kfold ignores fine_tuning.* -- every fold "
                    "trains from scratch.")
            if config.symbolic.enable:
                log("  NOTE: --kfold ignores symbolic distillation -- it "
                    "runs only on a deployable single-split model.")
        results = [result for result in
                   (driver(name, cache, config, log, n_tasks=len(names),
                           distributed=context)
                    for name in names)
                   if result is not None]

        # ---------------- unified checkpoint ---------------- #
        # One file for the whole chain, written after every task has trained,
        # so the two halves cannot drift apart or be mixed across runs.
        operators = {r["task"]: r.pop("operator") for r in results
                     if r.get("operator") is not None}
        bundle = None
        if operators and config.checkpoint_path():
            # <output.root>/<name>/<name>.poraque, with a distinct stem for a
            # fine-tune: it is a specialisation, usually to a narrower set of
            # materials, and writing it over the general model would replace
            # something broad with something narrow, silently and by default.
            bundle = bundle_path(config)

            if (config.fine_tuning.enable
                    and os.path.abspath(bundle) == os.path.abspath(
                        config.fine_tuning.pretrained_checkpoint)):
                raise SystemExit(
                    f"refusing to write the fine-tuned model over its own base "
                    f"checkpoint at {bundle}. Point output.root "
                    f"somewhere else.")

            metadata = {
                "structures": sorted({name for r in results
                                      for name in r.get("train_structures", [])}),
                "resolution": config.data.resolution,
                "epochs": config.training.epochs,
            }
            # Travels with the weights so a prediction can be written as an
            # ICHARG=1 restart without a reference calculation beside it.
            paw = load_paw_reference(cache)
            if paw:
                from poraque.data.cache import _reference_origin

                metadata["paw_reference"] = paw
                metadata["paw_source"] = _reference_origin(paw)
                log(f"  PAW reference   -> stored in the bundle "
                    f"({', '.join(sorted(paw))}) "
                    f"[{metadata['paw_source']}]")
            if config.fine_tuning.enable:
                metadata["fine_tuned_from"] = \
                    config.fine_tuning.pretrained_checkpoint
                metadata["fine_tuning_learning_rate"] = \
                    config.fine_tuning.learning_rate
                metadata["froze_lifting_layers"] = \
                    config.fine_tuning.freeze_lifting_layers
            save_bundle(bundle, operators, metadata=metadata)
            for result in results:
                result["checkpoint"] = bundle
            log(f"\n  models -> {bundle}  ({', '.join(sorted(operators))})")

            # The unified bundle now holds everything the safety copies did.
            # Removed only after it is on disk, so there is no window in which
            # neither exists.
            for task_name in operators:
                stale = os.path.join(
                    config.run_dir(),
                    f"{model_name(config)}_{task_name}_trained{BUNDLE_SUFFIX}")
                if os.path.exists(stale):
                    os.remove(stale)
            if len(operators) < 2:
                # Two different situations, and telling a user to "run with
                # task: all" when they just did -- and the data had no TAUCAR
                # -- sends them round a loop that cannot terminate.
                missing = sorted(set(TASKS) - set(operators))
                log(f"  NOTE: the bundle holds one task, {sorted(operators)}. "
                    f"It predicts that field and")
                log("  nothing further; the ASE calculator needs both halves "
                    "of the chain to")
                log("  reach a total energy.")
                if config.task.type == "all":
                    log(f"  {missing} was skipped because this dataset does "
                        f"not carry its target field.")
                    log("  Point a run at data that does, and save both models "
                        "into one bundle.")
                else:
                    log(f"  Run with task: all to train {missing} as well.")

        log(f"\n{'=' * 78}\nOVERALL\n{'=' * 78}")
        log(f"  {'task':<12s} {'rel L2 (mean)':>14s} {'R2 (mean)':>12s} "
            f"{'MAE (mean)':>14s}   basis")
        for result in results:
            if result.get("mode") == "kfold":
                values = [r["metrics"] for r in result["records"]]
                basis = (f"{result['n_folds']}-fold CV, "
                         f"{len(values)} validation structures")
            else:
                values = [v["metrics"] for v in result["per_material"].values()]
                basis = (f"training fit, {result['n_train']} structures"
                         if not result["validation"] else
                         f"train + {len(result['validation'])} validation")
            log(f"  {result['task']:<12s} "
                f"{np.mean([m['relative_l2'] for m in values]):14.4f} "
                f"{np.mean([m['r2'] for m in values]):12.4f} "
                f"{np.mean([m['mae'] for m in values]):14.5g}   {basis}")

        # The two `cache`es are different things: the positional one is the
        # prepared-cache directory, the keyword is the in-RAM field cache.
        # False rather than the `auto` default, because this instance exists to
        # be counted and thrown away and `auto` would read a header per
        # material to size a cache nothing is ever going to fill.
        n_materials = len(FieldPairDataset(cache, task=names[0], cache=False))
        log("")
        log(f"  NOTE: {chemistry_caveat(n_materials, dataset_elements(cache))}")
        log("  These numbers characterise the pipeline as much as the science.")
        if any(r.get("mode") == "split" and not r.get("validation")
               for r in results):
            log("  For the runs marked 'training fit' above, no structure was held")
            log("  out, so those are not generalisation estimates at all.")

        # Archive the resolved config beside the results, so the run is
        # reproducible even if the source config is later edited. With
        # output.root: null there is nowhere to archive to -- the documented
        # smoke-test setting -- and the run must end cleanly, not crash here
        # after every epoch has already succeeded.
        metrics_path = config.json_path()
        if metrics_path is not None:
            resolved = os.path.splitext(metrics_path)[0] + "_config.yaml"
            os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
            config.to_yaml(resolved)

            with open(metrics_path, "w") as handle:
                json.dump({"config": config.to_dict(), "device": str(device),
                           "results": results}, handle, indent=2, default=float)
            log(f"\n  log             -> {config.log_path()}")
            log(f"  metrics         -> {metrics_path}")
            log(f"  resolved config -> {resolved}")
        else:
            log("\n  output disabled (output.root: null): no log, metrics or "
                "config archived")
        if plot_directory(config):
            log(f"  figures         -> {plot_directory(config)}")
        return results
    finally:
        # In a `finally` because a rank that exits without destroying its group
        # leaves the others inside a collective until the step's wall clock
        # ends -- one process's exception becomes an hour of billed silence.
        shutdown_distributed(context)
        log.close()


def main(argv=None):
    """Console entry point for ``poraque-train``.

    Returns a process exit status, because the ``[project.scripts]`` wrapper
    calls ``sys.exit(main())`` and would treat any other object as an error
    message. :func:`run` returns the result records themselves.
    """
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
