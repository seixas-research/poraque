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

Validating :math:`\tau` on the way in
-------------------------------------
:math:`\tau` is the one cached field with no independent check on it. The
external potential is reconstructed from the geometry and can be compared with
a reference; the density integrates to a known electron count, and a wrong one
announces itself immediately. A kinetic energy density does neither, and an
entire dataset of invalid :math:`\tau` once passed through this function
unnoticed (``DELETIONS.md``).

So every :math:`\tau` is now put through :func:`~poraque.data.validation.
validate_tau` **before it is written**, against the density it is paired with
and against the provenance of the run that produced it. A failure aborts the
build naming the material, rather than caching a field that trains a model to
predict nonsense. The gate's verdict for every material is written to
``tau_validation.json`` at the cache root, and each material keeps its own
``tau_provenance.json`` so the record survives a re-cache.
"""

import json
import os
import time

import numpy as np

from ..fields import FieldGrid
from .sources import discover_records, resolve_source
from .validation import (
    MANIFEST_FILENAME,
    TauValidationConfig,
    TauValidationError,
    TauValidationManifest,
    read_tau_provenance,
    validate_tau,
    write_tau_provenance,
)

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

#: Per-material verdict of the kinetic-energy-density gate, at the cache root.
#: Deliberately *not* part of the fingerprint below: the gate does not change
#: what any cached file contains, so tightening a threshold must not invalidate
#: a cache that is otherwise identical. The manifest is the record instead.
TAU_MANIFEST_FILENAME = MANIFEST_FILENAME

#: What the cache was built *from* and *with*. "Files exist" is not "cached":
#: a directory built from other sources, another resolution, or another
#: potential construction holds different physics under the same filenames,
#: and reusing it silently would train on the wrong fields.
CACHE_FINGERPRINT_FILENAME = "cache_fingerprint.json"


#: Filename of a material's HDF5 field store inside a cache directory.
HDF5_FILENAME = "fields.h5"


def cached_paths(destination, fields, storage="files"):
    """
    Where one material's fields live inside its cache directory.

    Two storage layouts, one function, so nothing downstream has to branch on
    the format: ``files`` puts each field in its own ``CHGCAR``-format text
    file, ``hdf5`` puts all of them in one ``fields.h5`` addressed as
    ``fields.h5::CHGCAR``. Both spellings are paths every Poraquê reader
    accepts — see :mod:`poraque.fields.hdf5` for why the dispatch lives that
    deep.

    Parameters
    ----------
    destination : str
        The material's directory inside the cache.
    fields : sequence of str
    storage : str, optional
        ``"files"`` or ``"hdf5"``.

    Returns
    -------
    dict
        ``{field: path}``.
    """
    if storage == "hdf5":
        from ..fields.hdf5 import join_target

        store = os.path.join(destination, HDF5_FILENAME)
        return {name: join_target(store, name) for name in fields}
    if storage != "files":
        raise ValueError(
            f"Unknown storage {storage!r}; expected 'files' or 'hdf5'.")
    return {name: os.path.join(destination, name) for name in fields}


def build_field_cache(paths, cache, resolution=32, format="auto", fields=None,
                      charges=None, potcar_dir=None, sigma=None,
                      gaussian_blur=None, blur_method="spectral", pattern=None,
                      code="auto", spin=False, limit=None, tau_validation=None,
                      storage="files", compression=None, compression_level=4,
                      log=None):
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
    storage : str, optional
        ``"files"`` (a ``CHGCAR``-format text file per field, the default and
        what every existing cache is) or ``"hdf5"`` (one chunked ``fields.h5``
        per material). The stored numbers are identical either way; HDF5 is
        binary, so it is smaller, exact to a float64 ulp rather than to the
        text format's eleven digits, and loads without parsing millions of
        strings.
    compression : str, optional
        ``"gzip"``, ``"lzf"`` or ``None``. HDF5 storage only — a text cache's
        encoding is the format. See
        :func:`poraque.fields.hdf5.compression_options`.
    compression_level : int, optional
        Gzip level 0-9.
    tau_validation : TauValidationConfig or dict, optional
        Settings for the kinetic-energy-density gate; the defaults when
        omitted. Pass ``{"enabled": False}`` to build without it — which is
        recorded in the manifest, so the resulting cache stays distinguishable
        from one that passed.
    log : callable, optional
        Receives one line per material.

    Raises
    ------
    ~poraque.data.validation.TauValidationError
        When a material's :math:`\tau` fails the gate. The build stops there
        rather than continuing past it: a dataset that is half-validated is one
        whose bad half is now harder to find.

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

    if storage not in ("files", "hdf5"):
        raise ValueError(
            f"Unknown data.storage {storage!r}; expected 'files' or 'hdf5'.")
    if compression and storage != "hdf5":
        raise ValueError(
            f"compression={compression!r} was requested with storage="
            f"{storage!r}. Compression is an HDF5 dataset filter; a text cache "
            f"has no equivalent, and honouring the flag silently by ignoring "
            f"it would report a saving that never happened. Set "
            f"data.storage: hdf5, or drop the compression setting.")

    cache = str(cache)
    os.makedirs(cache, exist_ok=True)
    _check_fingerprint(cache, _fingerprint(paths, resolution, formats, fields,
                                           options, spin, storage, compression,
                                           compression_level))

    validation = TauValidationConfig.from_mapping(tau_validation)
    manifest = TauValidationManifest.load(cache)

    summary = _load_summary(cache)
    table = _CacheTable([record.identifier for record in records], fields, emit)
    table.header()
    try:
        for record in records:
            entry = _build_one(record, cache, resolution, fields, spin, emit,
                               summary.get(record.identifier), validation,
                               manifest, storage, compression,
                               compression_level)
            summary[record.identifier] = entry
            table.row(record.identifier, entry)
    finally:
        # Written even when the gate aborts the build. The record of *which*
        # material failed and by how much is the most useful thing to survive a
        # failure, and losing it to the exception would mean re-running the
        # whole ingestion to find out.
        _write_summary(cache, summary)
        if manifest.entries:
            manifest.write(cache)
    table.footer()
    _report_validation(manifest, validation, emit)

    return cache


def _report_validation(manifest, validation, emit):
    """One line on what the gate did, or on the fact that it did nothing."""
    if not validation.enabled:
        emit("  tau validation: DISABLED for this build "
             f"(recorded in {MANIFEST_FILENAME})")
        return
    passed, failed, ungated = manifest.summary()
    if passed or failed or ungated:
        emit(f"  tau validation: {passed} passed, {failed} failed, "
             f"{ungated} ungated  ->  {MANIFEST_FILENAME}")


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


def _fingerprint(paths, resolution, formats, fields, options, spin,
                 storage="files", compression=None, compression_level=4):
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
        # Storage is in the fingerprint because the two layouts are not
        # interchangeable on disk: half a cache as text and half as HDF5 would
        # load, and every reuse check would then be answering about the wrong
        # files. The codec is here for the same reason a `cache_tag` exists --
        # so a cache built at gzip-9 is not silently extended at lzf.
        "storage": storage,
        "compression": (compression or None) and str(compression),
        "compression_level": (int(compression_level)
                              if compression else None),
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


def _build_one(record, cache, resolution, fields, spin, emit, remembered=None,
               validation=None, manifest=None, storage="files",
               compression=None, compression_level=4):
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
    targets = cached_paths(destination, wanted, storage)

    if targets and _all_present(targets, storage):
        return _describe_cached(destination, wanted, record, remembered,
                                storage)

    started = time.time()
    native = source.grid(record)
    reduced = native
    if resolution:
        from ..fields.resample import downsample_shape

        shape = downsample_shape(native.shape, target_max=resolution)
        reduced = FieldGrid(shape, native.cell, encut=native.encut)

    os.makedirs(destination, exist_ok=True)
    ranges, warnings = {}, []
    native_fields = {}

    for name in wanted:
        field = source.read(record, name, native, spin=spin)
        native_fields[name] = field

        # The gate runs on the *native* field, before any downsampling, and
        # before the file is written. Native because that is the data as the
        # DFT code produced it -- band-limiting rings, and a von Weizsaecker
        # bound tested on a truncated field measures the truncation. Before the
        # write because a cache directory holding a rejected tau is exactly the
        # failure mode this whole mechanism exists to prevent.
        if name == "TAUCAR":
            entry = _validate_tau_field(record, field, native_fields, native,
                                        source, spin, validation, emit,
                                        manifest)
            if entry.get("provenance"):
                write_tau_provenance(destination, entry["provenance"])

        if resolution:
            from .dataset import _resample

            reduced_field = _resample(field, reduced.shape, reduced)
        else:
            reduced_field = field

        if storage == "hdf5":
            from ..fields.hdf5 import write_fields

            write_fields(os.path.join(destination, HDF5_FILENAME),
                         {name: reduced_field}, compression=compression,
                         level=compression_level)
        else:
            reduced_field.write(targets[name])

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


def _validate_tau_field(record, tau, native_fields, grid, source, spin,
                        validation, emit, manifest=None):
    r"""
    Put one material's :math:`\tau` through the ingestion gate.

    The density it is checked against is the one already read for this material
    where possible, and read on demand where the caller asked for ``TAUCAR``
    without ``CHGCAR`` -- the pair is the whole point of the check, so there is
    no version of it that runs without a density.

    A **failing** material is recorded in the manifest before the error is
    re-raised. The verdict is the most useful thing to survive an aborted
    build: losing it to the exception would mean re-running the whole ingestion
    to find out which material failed and by how much.

    Returns
    -------
    dict
        The gate record.

    Raises
    ------
    ~poraque.data.validation.TauValidationError
    """
    validation = TauValidationConfig.from_mapping(validation)
    if not validation.enabled:
        emit(f"      {record.identifier}: tau validation disabled")
        entry = {"material": record.identifier, "passed": None,
                 "enabled": False, "settings": validation.as_dict()}
        if manifest is not None:
            manifest.add(record.identifier, entry)
        return entry

    density = native_fields.get("CHGCAR")
    if density is None:
        density = source.read(record, "CHGCAR", grid, spin=spin)

    provenance = read_tau_provenance(record.directory,
                                     tag=validation.required_tag)
    try:
        entry = validate_tau(tau, density, grid, provenance=provenance,
                             config=validation, identifier=record.identifier)
    except TauValidationError as error:
        if manifest is not None and error.record:
            manifest.add(record.identifier, error.record)
        raise

    if manifest is not None:
        manifest.add(record.identifier, entry)
    return entry


def _all_present(targets, storage):
    """
    Whether every field this material needs is already stored.

    For a text cache that is one ``os.path.exists`` per field. For HDF5 the
    fields share a file, so existence of the file says nothing about which
    datasets are in it — an interrupted build leaves a store with a ``CHGCAR``
    and no ``TAUCAR``, and treating that as complete would skip the material
    forever.
    """
    if storage != "hdf5":
        return all(os.path.exists(path) for path in targets.values())

    from ..fields.hdf5 import field_names, split_target

    store = split_target(next(iter(targets.values())))[0]
    if not os.path.exists(store):
        return False
    try:
        present = set(field_names(store))
    except (OSError, ValueError):
        return False
    return all(split_target(path)[1] in present for path in targets.values())


def _describe_cached(destination, wanted, record, remembered,
                     storage="files"):
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

    paths = cached_paths(destination, wanted, storage)
    ranges, shape = {}, None
    for name in wanted:
        path = paths[name]
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
def build_paw_reference(records, cache, log=None, library=None,
                        source="atomic"):
    r"""
    The per-element PAW augmentation table the model bundle carries.

    The one-centre terms are contractions over the converged wavefunctions, so
    no grid-based model predicts them — but ``ICHARG=1`` needs them, and a
    prediction for a structure with no reference calculation of its own has to
    get them from somewhere. Two somewheres, and which one is right depends on
    what the dataset is about to become:

    ``source="atomic"`` (**the default since 2026-08-26**)
        The **isolated atom's own record**, from the database in ``library``.
        A fixed, per-element, transferable quantity with its own provenance —
        which is what makes it the defensible choice once slabs and clusters
        enter the set. A training-set average is a property of whatever
        happened to be in the training set, is undefined for an element that
        was not, and means something different again when half the set is
        surface atoms.

    ``source="material"``
        The previous behaviour: average the training calculations' records per
        element. More accurate *for an element the training set covers* — on
        this project's gold data a free-atom record sat 86.6 % RMS from a bulk
        site against 9.9 % for the average — and that gap is the honest cost of
        the transferability above, not an argument that either is wrong.

    Cached beside the fields either way, because the material route means
    reading the tail of every native-resolution ``CHGCAR``.

    Parameters
    ----------
    records : sequence of MaterialRecord
        Materials to read, for ``source="material"``. Ignored otherwise.
    cache : str
        Where to write ``paw_reference.json``.
    log : callable, optional
    library : AtomicReferenceLibrary, optional
        The isolated atoms, for ``source="atomic"``. Without one that route has
        nothing to read and falls back to the material average, saying so.
    source : {"atomic", "material"}, optional

    Returns
    -------
    dict
        ``{element: {...}}``, empty when nothing carried any records.
    """
    from ..fields.atomic import augmentation_reference
    from ..fields.vasp.augmentation import build_reference

    emit = log or (lambda *_: None)
    path = os.path.join(cache, PAW_REFERENCE_FILENAME)
    if os.path.exists(path):
        with open(path) as handle:
            reference = json.load(handle)
        origin = _reference_origin(reference)
        # A cached table from the *other* source is not this run's table.
        # Reusing it would answer a question nobody asked -- the same failure
        # `cache_tag` exists to prevent one level up -- so the requested source
        # wins and the file is rebuilt.
        wanted = "isolated atoms" if source == "atomic" else "training-set average"
        if origin == wanted or not reference:
            emit(f"  PAW reference: cached, {sorted(reference)} [{origin}]")
            return reference
        emit(f"  PAW reference: cached table is the {origin}, but "
             f"paw_source={source!r} asks for the {wanted} — rebuilding")

    reference = {}
    if source == "atomic":
        if library is not None and len(library):
            reference = augmentation_reference(library)
            if reference:
                emit(f"  PAW reference: from the ISOLATED ATOMS "
                     f"{sorted(reference)} — one record per element, with its "
                     f"own provenance")
            else:
                emit("  PAW reference: the isolated-atom database carries no "
                     "augmentation records (its CHGCARs are not PAW), falling "
                     "back to the training-set average")
        else:
            emit("  PAW reference: no isolated-atom database available, "
                 "falling back to the training-set average")

    if not reference:
        emit("  PAW reference: averaging augmentation records over the "
             "training calculations")
        reference = build_reference(
            [record.files["CHGCAR"] for record in records], log=emit)

    if not reference:
        emit("      none found — these calculations carry no PAW records, so "
             "predictions cannot be written as ICHARG=1 restarts")
        return {}

    os.makedirs(cache, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(reference, handle)
    return reference


def _reference_origin(reference):
    """``isolated atoms`` / ``training-set average`` / ``mixed``, for the log."""
    kinds = {entry.get("source", "material_average")
             for entry in reference.values() if isinstance(entry, dict)}
    if kinds == {"isolated_atom"}:
        return "isolated atoms"
    if "isolated_atom" not in kinds:
        return "training-set average"
    return "mixed"


def load_paw_reference(cache):
    """The cached per-element table, or an empty dict."""
    path = os.path.join(cache, PAW_REFERENCE_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path) as handle:
        return json.load(handle)
