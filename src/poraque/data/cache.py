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

#: Physical unit of each cached field, for the column headers.
FIELD_UNITS = {"EXTCAR": "eV", "CHGCAR": "e/Ang^3", "TAUCAR": "eV/Ang^3"}

#: Per-element PAW augmentation table, written beside the downsampled fields.
PAW_REFERENCE_FILENAME = "paw_reference.json"

#: What was written, per material: the two grid shapes and each field's value
#: range. Read back on a resumed build so the table below can be printed in
#: full without re-reading a single field.
CACHE_SUMMARY_FILENAME = "cache_summary.json"

#: What the cache was built *from* and *with*. "Files exist" is not "cached":
#: a directory built from other sources, another resolution, or another
#: potential construction holds different physics under the same filenames,
#: and reusing it silently would train on the wrong fields.
CACHE_FINGERPRINT_FILENAME = "cache_fingerprint.json"


def build_field_cache(paths, cache, resolution=32, format="auto", fields=None,
                      charges=None, potcar_dir=None, sigma=None,
                      gaussian_blur=None, blur_method="spectral", pattern=None,
                      code="auto", spin=False, limit=None, log=None):
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
    potcar_dir : str, optional
        A ``POTCAR`` library, used wherever the data itself ships none. With it
        the external potential is the exact tabulated one; without it, the
        Gaussian pseudo-ion model.
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

    options = {"charges": charges, "potcar_dir": potcar_dir, "sigma": sigma,
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
    _check_fingerprint(cache, _fingerprint(paths, resolution, formats, fields,
                                           options, spin))

    summary = _load_summary(cache)
    table = _CacheTable([record.identifier for record in records], fields, emit)
    table.header()
    for record in records:
        entry = _build_one(record, cache, resolution, fields, spin, emit,
                           summary.get(record.identifier))
        summary[record.identifier] = entry
        table.row(record.identifier, entry)
    table.footer()
    _write_summary(cache, summary)

    return cache


# ---------------------------------------------------------------------- #
# Progress table
# ---------------------------------------------------------------------- #
class _CacheTable:
    """
    The build log, as one aligned row per material.

    Rows are emitted **as each material finishes** rather than collected and
    printed at the end: a full archive takes minutes to hours, and a table that
    appears only afterwards is not progress. Column widths are therefore fixed
    up front — the identifiers are all known before the first row, and the rest
    are numeric fields of known width.

    Columns are the two grid shapes, which say what the downsampling did, and
    the value range of every cached field, which is the cheapest check that it
    did it correctly: a density that reaches zero, a potential that does not
    straddle zero, or a range differing by orders of magnitude from its
    neighbours are all visible at a glance and all mean something.
    """

    #: "-1.23e+02 .. 4.56e+01" and a little slack.
    RANGE_WIDTH = 22

    def __init__(self, identifiers, fields, emit):
        self.fields = tuple(fields)
        self.emit = emit
        self.label_width = max([len("material")]
                               + [len(name) for name in identifiers])

    def header(self):
        columns = [f"{'material':<{self.label_width}s}",
                   f"{'native grid':>14s}", f"{'cached grid':>13s}"]
        columns += [f"{f'{name} [{FIELD_UNITS[name]}]':<{self.RANGE_WIDTH}s}"
                    for name in self.fields]
        columns.append("time")
        line = "  " + "  ".join(columns)
        self.emit("")
        self.emit(line)
        self.emit("  " + "-" * (len(line) - 2))

    def row(self, identifier, entry):
        columns = [f"{identifier:<{self.label_width}s}",
                   f"{_shape_text(entry.get('native')):>14s}",
                   f"{_shape_text(entry.get('shape')):>13s}"]
        for name in self.fields:
            text = _range_text(entry["ranges"].get(name))
            columns.append(f"{text:<{self.RANGE_WIDTH}s}")
        columns.append("cached" if entry.get("reused")
                       else f"{entry.get('seconds', 0.0):.1f} s")
        self.emit("  " + "  ".join(columns).rstrip())
        for message in entry.get("warnings", ()):
            self.emit(f"      note: {message}")

    def footer(self):
        self.emit("")


def _shape_text(shape):
    """``(40, 40, 40)`` as ``40x40x40``; ``--`` when it is not recorded."""
    if not shape:
        return "--"
    return "x".join(str(int(n)) for n in shape)


def _range_text(bounds):
    """``[min, max]`` as an aligned pair; ``--`` for a field this material lacks."""
    if not bounds:
        return "--"
    low, high = bounds
    return f"{low:.3g} .. {high:.3g}"


def _fingerprint(paths, resolution, formats, fields, options, spin):
    """Everything that decides what a cache directory's files contain."""
    potcar_dir = options.get("potcar_dir")
    return {
        "paths": sorted(os.path.abspath(path) for path in paths),
        "resolution": int(resolution) if resolution else None,
        "formats": [fmt or "auto" for fmt in formats],
        "fields": sorted(fields),
        "spin": bool(spin),
        "charges": options.get("charges"),
        "potcar_dir": os.path.abspath(potcar_dir) if potcar_dir else None,
        "sigma": options.get("sigma"),
        "gaussian_blur": options.get("gaussian_blur"),
        "blur_method": options.get("blur_method"),
        "pattern": options.get("pattern"),
        "code": options.get("code"),
    }


def _check_fingerprint(cache, fingerprint):
    """
    Refuse to extend a cache built with different parameters.

    A matching or absent record proceeds; an absent record beside existing
    material directories is adopted with a warning (caches predate the
    record). A mismatch raises, naming the keys that differ — the caller
    should pick another cache directory or delete this one deliberately.
    """
    # JSON round-trip so tuples/lists and int/float compare canonically.
    fingerprint = json.loads(json.dumps(fingerprint, sort_keys=True))
    path = os.path.join(cache, CACHE_FINGERPRINT_FILENAME)

    recorded = None
    if os.path.exists(path):
        try:
            with open(path) as handle:
                recorded = json.load(handle)
        except (OSError, ValueError):
            recorded = None

    if recorded is not None and recorded != fingerprint:
        differing = sorted(key for key in set(recorded) | set(fingerprint)
                           if recorded.get(key) != fingerprint.get(key))
        raise ValueError(
            f"{cache} was built with different parameters "
            f"({', '.join(differing)} differ; see "
            f"{CACHE_FINGERPRINT_FILENAME}). Reusing it would silently mix "
            f"datasets or potential constructions. Point the build at a "
            f"fresh cache directory, or delete this one to rebuild it."
        )

    if recorded is None:
        has_materials = any(
            os.path.isdir(os.path.join(cache, entry))
            for entry in os.listdir(cache)
        )
        if has_materials:
            import warnings

            warnings.warn(
                f"{cache} predates the cache fingerprint; adopting its "
                f"contents as if built with the current parameters. If this "
                f"cache came from other sources or another potential "
                f"construction, delete it and rebuild.",
                RuntimeWarning, stacklevel=3,
            )

    with open(path, "w") as handle:
        json.dump(fingerprint, handle, indent=1, sort_keys=True)


def _load_summary(cache):
    """The per-material record of an earlier build, or an empty dict."""
    path = os.path.join(cache, CACHE_SUMMARY_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        # A truncated summary is a cosmetic loss -- the ranges get recomputed
        # from the cached fields -- and never a reason to fail a build.
        return {}


def _write_summary(cache, summary):
    with open(os.path.join(cache, CACHE_SUMMARY_FILENAME), "w") as handle:
        json.dump(summary, handle, indent=1, sort_keys=True)


def _build_one(record, cache, resolution, fields, spin, emit, remembered=None):
    """
    Downsample and write one material, unless it is already there.

    Returns
    -------
    dict
        ``native`` and ``shape`` grid shapes, the ``ranges`` of every field
        written, any Gibbs ``warnings``, and either ``seconds`` or
        ``reused``.
    """
    source = record.source
    wanted = [name for name in fields if name in source.provides(record)]
    destination = os.path.join(cache, record.identifier)
    expected = [os.path.join(destination, name) for name in wanted]

    if expected and all(os.path.exists(path) for path in expected):
        return _describe_cached(destination, wanted, record, remembered)

    started = time.time()
    native = source.grid(record)
    reduced = native
    if resolution:
        from ..fields.resample import downsample_shape

        shape = downsample_shape(native.shape, target_max=resolution)
        reduced = FieldGrid(shape, native.cell, encut=native.encut)

    os.makedirs(destination, exist_ok=True)
    ranges, warnings = {}, []

    for name in wanted:
        field = source.read(record, name, native, spin=spin)
        if resolution:
            from .dataset import _resample

            reduced_field = _resample(field, reduced.shape, reduced)
        else:
            reduced_field = field
        reduced_field.write(os.path.join(destination, name))

        data = reduced_field.data
        ranges[name] = [float(data.min()), float(data.max())]

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

    return {"native": list(native.shape), "shape": list(reduced.shape),
            "ranges": ranges, "warnings": warnings,
            "seconds": time.time() - started}


def _describe_cached(destination, wanted, record, remembered):
    """
    Fill a table row for a material that is already cached.

    Prefers the recorded summary, which costs no I/O at all. Falling back to
    reading the cached fields keeps the table complete for a cache built before
    the summary existed — the files are at the training resolution, so it is a
    fraction of the work the build itself avoided, and it happens once because
    the recomputed entry is written back.
    """
    if remembered and set(remembered.get("ranges", {})) >= set(wanted):
        return {**remembered, "warnings": [], "reused": True}

    from .sources import FIELD_CLASSES

    ranges, shape = {}, None
    for name in wanted:
        path = os.path.join(destination, name)
        grid = FieldGrid.from_file(path)
        shape = shape or list(grid.shape)
        data = FIELD_CLASSES[name].read(path, grid=grid).data
        ranges[name] = [float(data.min()), float(data.max())]

    native = (remembered or {}).get("native")
    if native is None:
        try:
            native = list(record.source.grid(record).shape)
        except (OSError, ValueError):
            # The cache outlived its source directory. That is fine -- nothing
            # downstream needs the native shape -- so leave the column blank
            # rather than failing a build that has everything it requires.
            native = None

    return {"native": native, "shape": shape, "ranges": ranges,
            "warnings": [], "reused": True}


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
