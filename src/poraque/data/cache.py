# -*- coding: utf-8 -*-
# file: cache.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Write any mixture of data layouts out as one prepared dataset.

Training reads a *prepared cache*: one directory per material holding the
fields already on the grid they will be trained on::

    cache/res32/struct_000/{EXTCAR,CHGCAR,TAUCAR}
    cache/res32/mp-124/{EXTCAR,CHGCAR}

:func:`build_field_cache` produces that from anything
:mod:`poraque.data.sources` recognises — DFT calculation directories, bulk
density archives, several of each — so the trainer sees one layout no matter
what it was given.

Why cache at all
----------------
Two reasons, and both are about doing expensive work once.

**Downsampling.** Native grids run from 48³ to 180³ in a public archive, which
is far more resolution than a first training run wants and orders of magnitude
more compute. The reduction is a **Fourier truncation** — the exact
band-limited projection for a plane-wave field — so periodicity and the
electron count survive to machine precision. Interpolation would alias, break
periodicity at the cell boundary and shift the integral.

**The external potential.** It is computed, never read, and computing it means
an FFT per material per epoch otherwise.

Everything a source can supply is written, not only the fields the current task
needs. A cache built for ``ext2chg`` from an archive that also has :math:`\tau`
serves ``chg2tau`` afterwards with no rebuild.
"""

import json
import os
import time

import numpy as np

from ..fields import FieldGrid
from .sources import discover_records, resolve_source

#: Fields written to the cache, in file order.
CACHED_FIELDS = ("EXTCAR", "CHGCAR", "TAUCAR")

#: Per-element PAW augmentation table, written beside the downsampled fields.
PAW_REFERENCE_FILENAME = "paw_reference.json"


def build_field_cache(paths, cache, resolution=32, format="auto", fields=None,
                      charges=None, sigma=None, gaussian_blur=None,
                      blur_method="spectral", pattern=None, code="auto",
                      spin=False, limit=None, log=None):
    r"""
    Downsample every material under ``paths`` into one prepared cache.

    Materials already present are left alone, so an interrupted build resumes
    rather than restarting — which matters when one archive holds a 180³
    density.

    Parameters
    ----------
    paths : str or sequence of str
        Directories to ingest. Each is auto-detected unless ``format`` says
        otherwise, so a mixture of layouts is one call.
    cache : str or pathlib.Path
        Destination.
    resolution : int, optional
        Longest grid axis after spectral downsampling.
    format : str or sequence of str, optional
        ``"auto"``, one name for every path, or one per path.
    fields : sequence of str, optional
        Which fields to write. Defaults to all of :data:`CACHED_FIELDS`; each
        is written for the materials that can supply it and skipped, without
        complaint, for those that cannot.
    charges : dict, optional
        ``{element: Z_val}`` for bulk archives, which carry no ``POTCAR``.
    sigma : float or dict, optional
        Gaussian pseudo-ion width in Å, where a model potential is used.
    gaussian_blur : float, optional
        Blur width in Å for the computed potential.
    blur_method : str, optional
        ``"spectral"`` or ``"ndimage"``.
    pattern : str, optional
        Subdirectory prefix filter for calculation archives.
    code : str, optional
        DFT code name, or ``"auto"``.
    spin : bool, optional
        Cache the magnetisation channel alongside the total density.
    limit : int, optional
        Build at most this many materials, smallest source file first. Useful
        for an end-to-end check against a large archive.
    log : callable, optional
        Receives one line per material.

    Returns
    -------
    str
        The cache directory.
    """
    emit = log or (lambda *_: None)
    fields = tuple(fields or CACHED_FIELDS)
    paths = [str(paths)] if isinstance(paths, (str, os.PathLike)) else \
        [str(path) for path in paths]

    formats = ([format] * len(paths)
               if isinstance(format, (str, type(None))) else list(format))
    if len(formats) != len(paths):
        raise ValueError(
            f"Got {len(formats)} format(s) for {len(paths)} path(s). Pass one "
            f"name for all of them, or one per path.")

    options = {"charges": charges, "sigma": sigma,
               "gaussian_blur": gaussian_blur, "blur_method": blur_method,
               "pattern": pattern, "code": code, "log": emit}
    sources = [resolve_source(path, fmt, **options)
               for path, fmt in zip(paths, formats)]

    records = discover_records(sources, required=("CHGCAR",), log=emit)
    if not records:
        raise ValueError(
            f"No material with a charge density found under {paths}.")
    if limit:
        records = sorted(
            records,
            key=lambda record: os.path.getsize(record.files["CHGCAR"]),
        )[:limit]
        emit(f"  limited to the {len(records)} smallest of the available "
             f"materials")

    cache = str(cache)
    os.makedirs(cache, exist_ok=True)

    for record in records:
        _build_one(record, cache, resolution, fields, spin, emit)

    return cache


def _build_one(record, cache, resolution, fields, spin, emit):
    """Downsample and write one material, unless it is already there."""
    source = record.source
    wanted = [name for name in fields if name in source.provides(record)]
    destination = os.path.join(cache, record.identifier)
    expected = [os.path.join(destination, name) for name in wanted]

    if expected and all(os.path.exists(path) for path in expected):
        emit(f"  {record.identifier}: cached, grid "
             f"{FieldGrid.from_file(expected[0]).shape}")
        return

    started = time.time()
    native = source.grid(record)
    reduced = native
    if resolution:
        from ..fields.resample import downsample_shape

        shape = downsample_shape(native.shape, target_max=resolution)
        reduced = FieldGrid(shape, native.cell, encut=native.encut)

    os.makedirs(destination, exist_ok=True)
    summary, warnings = [], []

    for name in wanted:
        field = source.read(record, name, native, spin=spin)
        if resolution:
            from .dataset import _resample

            reduced_field = _resample(field, reduced.shape, reduced)
        else:
            reduced_field = field
        reduced_field.write(os.path.join(destination, name))

        data = reduced_field.data
        summary.append(f"{name} [{data.min():.3g}, {data.max():.3g}]")

        # rho and tau are non-negative, but band-limiting a field with sharp
        # core peaks rings (Gibbs) and can undershoot slightly. That is an
        # artefact of the truncation, not of the data, and it is why the
        # dataset uses the sign-tolerant `asinh` normalization.
        if name in ("CHGCAR", "TAUCAR") and field.data.min() >= 0:
            negative = int(np.count_nonzero(data < 0))
            if negative:
                warnings.append(
                    f"{name}: {negative} of {data.size} points "
                    f"({100 * negative / data.size:.2f}%) went negative, min "
                    f"{data.min():.3g} (Gibbs ringing from band-limiting)")

    missing = [name for name in fields if name not in wanted]
    note = f"   [no {', '.join(missing)}]" if missing else ""
    emit(f"  {record.identifier}: {tuple(native.shape)} -> "
         f"{tuple(reduced.shape)} in {time.time() - started:.1f} s   "
         + "  ".join(summary) + note)
    for message in warnings:
        emit(f"      note: {message}")


# ---------------------------------------------------------------------- #
# PAW augmentation reference
# ---------------------------------------------------------------------- #
def build_paw_reference(records, cache, log=None):
    r"""
    Average the PAW augmentation records of the training data, per element.

    The one-centre terms are contractions over the converged wavefunctions, so
    no grid-based model predicts them — but ``ICHARG=1`` needs them. Averaging
    the training set's per element gives a transferable table the model bundle
    can carry, so a prediction for a structure with no reference of its own can
    still be written as a restartable ``CHGCAR``.

    Cached beside the fields, because extracting it means reading the tail of
    every native-resolution ``CHGCAR``.

    Parameters
    ----------
    records : sequence of MaterialRecord
        Materials to read, from :func:`~poraque.data.sources.discover_records`.
        The densities themselves are read, not their directories, so a bulk
        archive of ``CHGCAR_<id>.gz`` files contributes exactly as a
        calculation directory does.
    cache : str
        Where to write ``paw_reference.json``.
    log : callable, optional

    Returns
    -------
    dict
        The reference, empty when no source carried any records.
    """
    from ..fields.vasp.augmentation import build_reference

    emit = log or (lambda *_: None)
    path = os.path.join(cache, PAW_REFERENCE_FILENAME)
    if os.path.exists(path):
        with open(path) as handle:
            reference = json.load(handle)
        emit(f"  PAW reference: cached, {sorted(reference)}")
        return reference

    emit("  PAW reference: reading augmentation records from the sources")
    reference = build_reference([record.files["CHGCAR"] for record in records],
                                log=emit)
    if not reference:
        emit("      none found — these calculations carry no PAW records, so "
             "predictions cannot be written as ICHARG=1 restarts")
        return {}

    os.makedirs(cache, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(reference, handle)
    return reference


def load_paw_reference(cache):
    """The cached per-element table, or an empty dict."""
    path = os.path.join(cache, PAW_REFERENCE_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)
