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


def build_field_cache(paths, cache, resolution=32, fields=None,
                      charges=None, potcar_dir=None, sigma=None,
                      gaussian_blur=None, blur_method="spectral", pattern=None,
                      format="auto", spin=False, limit=None,
                      storage="files", compression=None, compression_level=4,
                      strict_potcar=False, log=None):
    r"""
    Downsample every material under ``paths`` into one prepared cache.

    Materials already present are left alone, so an interrupted build resumes
    rather than restarting — which matters when one archive holds a 180³
    density.

    Parameters
    ----------
    paths : str or sequence of str
        Directories to ingest. Each holds one subdirectory per material --
        whatever produced it -- and all of them are pooled.
    cache : str or pathlib.Path
        Destination.
    resolution : int, optional
        Longest grid axis after spectral downsampling.
    fields : sequence of str, optional
        Which fields to write. Defaults to all of :data:`CACHED_FIELDS`; each
        is written for the materials that can supply it and skipped, without
        complaint, for those that cannot.
    charges : dict, optional
        ``{element: Z_val}``, for materials that carry no ``POTCAR``.
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
    format : str, optional
        ``"vasp"``, or ``"auto"`` to detect the code from the files present.
    spin : {"auto", True, False}, optional
        Whether to cache the magnetisation channel alongside the total density.
        ``"auto"`` — the default everywhere above this function — **reads the
        sources** and carries it whenever any of them is ``ISPIN = 2``. ``True``
        demands it and raises if the data has none; ``False`` is a deliberate
        opt-out that discards it.
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
    strict_potcar : bool, optional
        Raise instead of falling back to the Gaussian pseudo-ion model when a
        configured ``potcar_dir`` cannot serve an element. Off by default,
        because the fallback is a legitimate degradation on a workstation; on a
        cluster it is the same failure shape as the silent CPU fallback
        ``training.strict_device`` exists to remove — a job that holds its
        allocation and quietly computes something other than what was
        configured.
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

    options = {"charges": charges, "potcar_dir": potcar_dir, "sigma": sigma,
               "gaussian_blur": gaussian_blur, "blur_method": blur_method,
               "pattern": pattern, "format": format, "log": emit}
    sources = [resolve_source(path, **options) for path in paths]

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

    spin = _resolve_spin(spin, records, emit)
    potcar_source = _resolve_potcar_source(sources, strict_potcar, emit)

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
    _check_fingerprint(cache, _fingerprint(paths, resolution, fields,
                                           options, spin, storage, compression,
                                           compression_level, potcar_source))

    summary = _load_summary(cache)
    table = _CacheTable([record.identifier for record in records], fields, emit)
    table.header()
    try:
        for record in records:
            entry = _build_one(record, cache, resolution, fields, spin, emit,
                               summary.get(record.identifier), storage,
                               compression, compression_level)
            summary[record.identifier] = entry
            table.row(record.identifier, entry)
    finally:
        # Written even when the build aborts partway. The record of what did
        # get cached is the most useful thing to survive a failure, and losing
        # it to the exception would mean re-reading every field to find out.
        _write_summary(cache, summary)
    table.footer()

    return cache


def _resolve_potcar_source(sources, strict, emit):
    r"""
    What the ``POTCAR`` library actually served, per element.

    ``None`` when no library is configured anywhere — a purely Gaussian dataset
    has nothing to have failed, and its fingerprint is then exactly what it was
    before this existed.

    The point is the gap between the two questions. ``potcar_dir`` in the
    fingerprint is the path that was *asked for*; this is what came back.
    Found on Santos Dumont, where a control run landed on a node with ``/prj``
    unmounted, warned six times on stderr, trained against analytic pseudo-ion
    potentials, and wrote a cache whose fingerprint was indistinguishable from
    the real one — including its directory name, ``res32_potcar``. Every later
    run reused it silently, on mounted nodes too, and the warning never came
    back. The run it invalidated had already been quoted as a measurement.

    Parameters
    ----------
    sources : sequence of MaterialSource
    strict : bool
        Raise rather than record a fallback. See ``strict_potcar``.
    emit : callable

    Returns
    -------
    dict or None
        ``{element: "library" | "gaussian"}``.

    Raises
    ------
    RuntimeError
        Under ``strict``, naming the elements the library did not serve.
    """
    resolved = {}
    for source in sources:
        coverage = source.potcar_coverage()
        if coverage is None:
            continue
        for element, served in coverage.items():
            # A library that serves an element for one source and not another
            # is one library and one answer: the pessimistic one, since the
            # dataset then contains at least one analytic potential.
            if not served or element not in resolved:
                resolved[element] = "library" if served else "gaussian"

    if not resolved:
        return None

    fell_back = sorted(element for element, mode in resolved.items()
                       if mode == "gaussian")
    if fell_back and strict:
        raise RuntimeError(
            f"The POTCAR library does not serve {fell_back}, so the external "
            f"potential for every structure containing them would be the "
            f"Gaussian pseudo-ion model -- a different field, not a coarser "
            f"one. strict_potcar is set, so this is an error rather than a "
            f"warning. Unset data.strict_potcar to accept the fallback, or "
            f"check that data.potcar_dir is readable from this node."
        )

    emit("  POTCAR library: " + ", ".join(f"{element}={mode}" for element, mode
                                          in sorted(resolved.items())))
    if fell_back:
        emit(f"      note: {', '.join(fell_back)} fall back to the Gaussian "
             f"pseudo-ion model, and the cache fingerprint records it -- a "
             f"later run with the library available will rebuild rather than "
             f"reuse this")
    return resolved


def _resolve_spin(requested, records, emit):
    r"""
    Turn ``spin="auto"`` into a decision, by looking at the data.

    This is the *first* of the two places that ask whether a dataset is
    spin-polarised, and it is the one that matters: the second
    (:meth:`~poraque.ml.data.FieldPairDataset._resolve_spin`) inspects the
    **cache**, so whatever is dropped here is invisible to it. The two agreeing
    is not evidence of anything if the first one threw the answer away.

    Regression: this function did not exist, and the caller passed
    ``spin=data.spin is True``. With the default ``"auto"`` that is ``False``,
    so every ``ISPIN = 2`` magnetisation block was discarded on the way into
    the cache and the dataset then reported, truthfully, that the cache held
    no spin. Two layers of auto-detection, and the first was an identity test
    against ``True``.

    The decision is taken **once for the whole dataset**, not per material: the
    ML layer sizes one operator for one channel count, so a cache with two
    blocks for some materials and one for others could not be trained from. A
    mixed set therefore resolves to spin, and the unpolarised members get
    :math:`m \equiv 0` — which :class:`~poraque.fields.SpinDensity` documents
    as meaningful rather than wasted, since it makes an ``ISPIN = 2`` operator
    a strict generalisation of an ``ISPIN = 1`` one.

    Parameters
    ----------
    requested : {"auto", True, False}
    records : list of MaterialRecord
    emit : callable

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        When ``spin=True`` is asked of a dataset that carries no magnetisation
        anywhere -- a channel the data has no values for cannot be trained.
    """
    polarized = [record for record in records
                 if record.source.is_spin_polarized(record)]

    if requested is True:
        if not polarized:
            raise ValueError(
                "data.spin: true was requested, but none of the "
                f"{len(records)} material(s) has a magnetisation block. "
                "Training a channel the data has no values for would fit "
                "noise to zero. Use data.spin: auto, or check ISPIN in the "
                "INCARs that produced these densities.")
        resolved = True
    elif requested is False:
        resolved = False
        if polarized:
            emit(f"  spin: DISCARDING the magnetisation block of "
                 f"{len(polarized)} of {len(records)} material(s) "
                 f"(data.spin: false). The cache will hold the total density "
                 f"only.")
    else:
        resolved = bool(polarized)

    if resolved:
        extra = len(records) - len(polarized)
        note = (f"; {extra} unpolarised material(s) get m = 0"
                if extra else "")
        emit(f"  spin: ISPIN = 2 in {len(polarized)} of {len(records)} "
             f"material(s){note}")
    return resolved


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


def _fingerprint(paths, resolution, fields, options, spin,
                 storage="files", compression=None, compression_level=4,
                 potcar_source=None):
    """Everything that decides what a cache directory's files contain."""
    potcar_dir = options.get("potcar_dir")
    return {
        # What the library actually served, per element, against `potcar_dir`
        # just below, which is only what it was asked for. A run on a node
        # where the library was not mounted builds a cache of analytic
        # pseudo-ion potentials -- a different physical quantity, not a worse
        # approximation to the same one -- and without this key that cache is
        # indistinguishable from the real thing and is reused forever.
        # `None` when no library was configured, so a purely Gaussian dataset's
        # fingerprint is unchanged.
        "potcar_source": potcar_source,
        "paths": sorted(os.path.abspath(path) for path in paths),
        "resolution": int(resolution) if resolution else None,
        "fields": sorted(fields),
        "spin": bool(spin),
        "charges": options.get("charges"),
        "potcar_dir": os.path.abspath(potcar_dir) if potcar_dir else None,
        "sigma": options.get("sigma"),
        "gaussian_blur": options.get("gaussian_blur"),
        "blur_method": options.get("blur_method"),
        "pattern": options.get("pattern"),
        "format": options.get("format"),
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

    One narrower case is adopted rather than refused: a recorded fingerprint
    that agrees on every key it *has*, and lacks only keys this version added.
    Those keys record something the older build never asked about, so it cannot
    have disagreed about them — and refusing would invalidate every cache in
    existence on an upgrade, which is a rebuild of hundreds of densities to
    learn nothing. The warning says which keys are unrecorded, because
    "unrecorded" and "recorded as the same" are different states and only one
    of them is evidence.
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
        added = sorted(set(fingerprint) - set(recorded))
        differing = sorted(key for key in set(recorded) | set(fingerprint)
                           if recorded.get(key) != fingerprint.get(key))
        if added and not (set(differing) - set(added)):
            import warnings

            warnings.warn(
                f"{cache} was built before this version recorded "
                f"{', '.join(added)}, so it cannot say what was used for "
                f"{'them' if len(added) > 1 else 'it'}. Adopting the cache and "
                f"recording the current value. Delete it and rebuild if that "
                f"is not what it holds.",
                RuntimeWarning, stacklevel=3,
            )
            # Deliberately *not* `recorded = None`: that would fall into the
            # legacy branch below and warn a second time, with a sentence that
            # is not true of this cache -- it has a fingerprint, and it agrees.
        else:
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
               storage="files", compression=None, compression_level=4):
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

    for name in wanted:
        field = source.read(record, name, native, spin=spin)

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
        this project's platinum data a free-atom record sat 86.6 % RMS from a bulk
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
