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
:mod:`poraque.data.sources` recognizes — DFT run trees, ``poraque-mp``
downloads, several of each — so the trainer sees one layout no matter what it
was given.

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
needs. A cache built for ``ext2pseudo`` from an archive that also has :math:`\tau`
serves ``pseudo2tau`` afterwards with no rebuild.
"""

import contextlib
import json
import multiprocessing
import os
import time

from ..fields import FieldGrid
from ..fields.hdf5 import DEFAULT_FILENAME
from ..fields.resample import downsampled_grid, resample_field
from .sources import discover_records, resolve_source

#: Fields written to the cache, in file order.
CACHED_FIELDS = ("EXTCAR", "CHGCAR", "TAUCAR")

#: Physical unit of each cached field, for the column headers.
FIELD_UNITS = {"EXTCAR": "eV", "CHGCAR": "e/Ang^3", "TAUCAR": "eV/Ang^3"}

#: Per-element PAW augmentation table, written beside the downsampled fields.
PAW_REFERENCE_FILENAME = "paw_reference.json"

#: Per-element radial PAW core densities, read from the POTCARs that built
#: V_ext and written beside the augmentation table.
PAW_PROFILES_FILENAME = "paw_profiles.json"

#: What was written, per material: the two grid shapes and each field's value
#: range. Read back on a resumed build so the table below can be printed in
#: full without re-reading a single field.
CACHE_SUMMARY_FILENAME = "cache_summary.json"

#: What the cache was built *from* and *with*. "Files exist" is not "cached":
#: a directory built from other sources, another resolution, or another
#: potential construction holds different physics under the same filenames,
#: and reusing it silently would train on the wrong fields.
CACHE_FINGERPRINT_FILENAME = "cache_fingerprint.json"


#: Filename of a material's HDF5 field store inside a cache directory — the
#: store module's own default, imported rather than restated.
HDF5_FILENAME = DEFAULT_FILENAME

#: Memory ``workers="auto"`` commits to the pool **when no Slurm allocation
#: says otherwise**, in bytes. Each worker decodes one native-resolution
#: material at a time, and the peak is set by the *largest* grid in the set,
#: not the average: a dataset of 108³ bulk cells and 288³ nanoparticles sizes
#: its workers on the nanoparticles or it swaps on them. 8 GiB, the same order
#: as :data:`~poraque.ml.data.CACHE_MEMORY_BUDGET` and for the same reason — a
#: workstation must not be driven into swap by a default. Inside a Slurm job
#: :func:`_memory_budget` reads the real allocation instead.
WORKER_MEMORY_BUDGET = 8 * 1024 ** 3

#: Total native voxels across the whole set below which ``workers="auto"``
#: stays single-process, because the pool would cost more than it saves.
#:
#: Measured: a ``spawn``ed pool costs 0.03 s bare and **0.49-0.60 s** once each
#: worker has imported the package (roughly constant in the worker count, since
#: they start together). The build itself runs at 1.03 s per million native
#: voxels. With N workers the pool wins when
#: :math:`0.6 + W/N < W`, i.e. :math:`W > 0.6\,N/(N-1)` --- 0.8 s at four
#: workers, which is about 0.8 M voxels. Rounded to 1 M: a handful of 8³ or
#: 12³ synthetic grids (the end-to-end tests) stays serial, and anything that
#: is really a dataset does not. An explicit count is honored regardless ---
#: this bounds ``auto``, not the feature.
WORKER_MINIMUM_VOXELS = 1_000_000

#: Resident memory one worker costs before it touches a grid: the interpreter,
#: NumPy, and the package's own imports. Measured at 197-198 MiB on this
#: machine over two materials three orders of magnitude apart in size; carried
#: at twice that, because the cost of overestimating is one fewer worker and
#: the cost of underestimating is swap.
WORKER_BASE_BYTES = 400 * 1024 ** 2

#: Bytes one worker needs per native voxel of the largest material it may be
#: handed, on top of :data:`WORKER_BASE_BYTES`: the decoded field, the complex
#: copy ``np.fft.fftn`` makes of it, and the text chunk being parsed.
#:
#: Measured by building one material in a process of its own and reading
#: ``ru_maxrss``: a 108³ density (1.26 M voxels) peaked at 498 MiB and a 288³
#: one (23.89 M voxels) at 1558 MiB, so the slope is
#: :math:`(1558-498)/(23.89-1.26) = 47` B per voxel. Carried at 70, which
#: reproduces both measurements from the top side.
WORKER_BYTES_PER_VOXEL = 70

#: The source each worker rebuilds for itself, at module level because that is
#: the only state a ``spawn``ed process keeps between tasks. See
#: :func:`_initialize_worker`.
_WORKER_SOURCES = {}


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
                      strict_potcar=False, log=None, max_spacing=None,
                      workers=0):
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
        Whether to cache the magnetization channel alongside the total density.
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
    workers : int or str, optional
        Processes to build with. ``0`` (the default) or ``1`` is the
        single-process build; ``"auto"`` picks from the cores this process may
        use, a memory budget taken from the largest native grid, and the
        material count. See :func:`resolve_cache_workers` for why materials
        are the axis that scales and threads are not.

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

    # Resolved before the spin probe, because the probe is itself a full pass
    # over every density file and takes the same pool.
    worker_count = resolve_cache_workers(workers, records, emit)
    _report_codec(emit)
    spin = _resolve_spin(spin, records, emit, workers=worker_count,
                         memo_root=os.path.dirname(os.path.abspath(str(cache))))
    potcar_source = _resolve_potcar_source(sources, strict_potcar, emit)

    if storage not in ("files", "hdf5"):
        raise ValueError(
            f"Unknown data.storage {storage!r}; expected 'files' or 'hdf5'.")
    if compression and storage != "hdf5":
        raise ValueError(
            f"compression={compression!r} was requested with storage="
            f"{storage!r}. Compression is an HDF5 dataset filter; a text cache "
            f"has no equivalent, and honoring the flag silently by ignoring "
            f"it would report a saving that never happened. Set "
            f"data.storage: hdf5, or drop the compression setting.")

    cache = str(cache)
    os.makedirs(cache, exist_ok=True)
    _check_fingerprint(cache, _fingerprint(paths, resolution, fields,
                                           options, spin, storage, compression,
                                           compression_level, potcar_source,
                                           max_spacing))

    summary = _load_summary(cache)
    table = _CacheTable([record.identifier for record in records], fields, emit)
    table.header()
    try:
        if worker_count > 1:
            _build_in_parallel(records, cache, resolution, fields, spin,
                               summary, storage, compression,
                               compression_level, max_spacing, worker_count,
                               sources, options, table)
        else:
            for position, record in enumerate(records, start=1):
                entry = _build_one(record, cache, resolution, fields, spin,
                                   summary.get(record.identifier), storage,
                                   compression, compression_level, max_spacing)
                summary[record.identifier] = entry
                table.row(position, record.identifier, entry)
    finally:
        # Written even when the build aborts partway. The record of what did
        # get cached is the most useful thing to survive a failure, and losing
        # it to the exception would mean re-reading every field to find out.
        _write_summary(cache, summary)
    table.footer()

    return cache


#: Environment variables the common BLAS and OpenMP runtimes read their
#: thread count from, at import.
BLAS_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                         "NUMEXPR_NUM_THREADS")


@contextlib.contextmanager
def _worker_threads(workers):
    r"""
    Give each pool worker its share of the cores for BLAS, then restore.

    V_ext's structure factor is a BLAS matrix product
    (:func:`~poraque.fields.external.structure_factor`), and a BLAS library
    starts as many threads as there are cores. Under a pool of ``workers``
    processes that is ``workers x cores`` threads contending for ``cores``.
    The runtimes read their thread count once, when NumPy is imported, and a
    ``spawn`` worker imports it fresh -- so the variables are set in the
    parent's environment while the workers start, and the parent's own BLAS,
    already initialized, is untouched. A variable the user set is respected.
    """
    share = str(max(1, (os.cpu_count() or 1) // max(1, int(workers))))
    previous = {name: os.environ.get(name) for name in BLAS_THREAD_VARIABLES}
    for name, value in previous.items():
        if value is None:
            os.environ[name] = share
    try:
        yield share
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)


def _report_codec(emit):
    r"""
    Load the C text codec here, in the parent, and say which path the build uses.

    Loading first means a missing library is compiled once, before the pool,
    rather than by every spawned worker at once; the workers then find it in
    the cache. Saying so means a build that fell back to Python -- no
    compiler, or ``PORAQUE_C_BACKEND=0`` -- is visible in the log instead of
    merely several times slower.
    """
    from ..fields.vasp import codec

    codec.load(log=emit)
    if codec.available():
        emit(f"  text codec: C ({codec.library_path()})")
    else:
        emit(f"  text codec: Python -- {codec.describe().split(': ', 1)[-1]}; "
             f"the cache builds several times slower. `poraque-compile` says "
             f"why.")


def _build_in_parallel(records, cache, resolution, fields, spin, summary,
                       storage, compression, compression_level, max_spacing,
                       worker_count, sources, options, table):
    r"""
    One process per material, over a pool, into the one cache directory.

    Correct for the same reason two Slurm jobs sharing a cold cache are:
    :func:`_build_one` writes every field to ``.partial-<pid>`` and moves it
    into place with :func:`os.replace`, which is atomic. The comment there
    records the regression that bought the discipline --- two jobs at LNCC
    interleaved inside one ``write`` and left a zero-length density that
    *parsed* --- and the pool inherits it. A cache built by eight workers is
    byte-identical to the serial one.

    What the workers do **not** touch is the bookkeeping. ``cache_summary.json``
    and ``cache_fingerprint.json`` are plain ``open(..., "w")`` writes and stay
    in the parent, which also owns the progress table and the ``finally`` that
    saves the summary when a build aborts. Nothing about the on-disk invariants
    moves into the pool; only the per-material work does.

    Rows are printed by completion rather than by discovery order
    (``imap_unordered``), and ``i/N`` counts materials finished. The table's
    own reason for existing is progress --- an hour-long build whose table
    appears at the end is not progress --- and ordered delivery would hold a
    finished 88 MiB row behind an unfinished 1.6 GiB one.
    """
    _resolve_charges_before_forking(sources, options)
    payloads = _payloads(records, cache, resolution, fields, spin, summary,
                         storage, compression, compression_level, max_spacing)
    roots = tuple(dict.fromkeys(record.source.root for record in records))

    # `spawn` rather than `fork`: NumPy's BLAS starts threads, and forking a
    # threaded process is undefined where those threads hold locks. It also
    # makes the unshippable `log` fail loudly here instead of only on macOS.
    context = multiprocessing.get_context("spawn")
    failures = {}
    with _worker_threads(worker_count), \
            context.Pool(worker_count, initializer=_initialize_worker,
                         initargs=(roots, shippable_options(options))) as pool:
        for position, (identifier, entry) in enumerate(
                pool.imap_unordered(_build_one_shipped, payloads), start=1):
            if "error" in entry:
                failures[identifier] = entry["error"]
                continue
            summary[identifier] = entry
            table.row(position, identifier, entry)

    if failures:
        detail = "; ".join(f"{name}: {why}"
                           for name, why in sorted(failures.items()))
        raise RuntimeError(
            f"{len(failures)} of {len(payloads)} material(s) failed to cache "
            f"({detail}). The rest were written and are recorded in "
            f"{CACHE_SUMMARY_FILENAME}, so a rerun resumes from them.")


def _resolve_charges_before_forking(sources, options):
    r"""
    Settle the inferred valence charges in the parent, once.

    :attr:`CalculationSource.charges` is lazy, and its fallback
    :func:`~poraque.data.sources.infer_valence_charges` **parses every density
    in the set** to solve :math:`\int\rho = \sum_s n_s Z_s`. In the serial
    build that happens at most once. In a pool each worker would do it
    independently, so eight workers would decode the whole archive eight times
    before writing a single field --- 61 GiB per worker on this project's
    platinum set.

    Only datasets that take the Gaussian pseudo-ion route ask the question at
    all: :meth:`CalculationSource._modeled_charges` returns ``None`` where the
    material carries its own ``POTCAR``, and the tabulated route reads ``ZVAL``
    from the ``POTCAR`` instead. So the probe is whether *any* record would
    fall through to the model, and it is an :func:`os.path.exists` per material
    plus the per-element library answer :meth:`potcar_coverage` has already
    memoized.

    Failure is not an error here. ``from_calculation`` raises a better message
    a moment later, naming the elements it could not place, and raising now
    would turn a diagnosable build into a traceback from a worker.
    """
    if options.get("charges"):
        return
    for source in sources:
        if not _needs_modeled_charges(source):
            continue
        try:
            resolved = source.charges
        except (ValueError, OSError):
            return
        if resolved:
            options["charges"] = dict(resolved)
            return


def _needs_modeled_charges(source):
    """Whether any of ``source``'s materials falls back to the model potential."""
    coverage = source.potcar_coverage()
    if coverage is not None and all(coverage.values()):
        return False
    return any(not os.path.exists(os.path.join(record.directory, "POTCAR"))
               for record in source.discover())


# ---------------------------------------------------------------------- #
# Working in parallel over materials
# ---------------------------------------------------------------------- #
def shippable_options(options):
    r"""
    ``options`` with everything a worker process cannot receive removed.

    One key, and it is not optional: ``log``. A
    :class:`~poraque.data.sources.CalculationSource` keeps its options dict, a
    ``MaterialRecord`` keeps its source, and :func:`build_field_cache` puts the
    build's logging callable in those options --- so a record holds a live
    closure over an open log file and ``pickle.dumps(record)`` raises
    ``PicklingError``. Under ``fork`` that never surfaces; ``spawn``, which is
    macOS's default and the only safe start method beside a threaded BLAS, has
    to send it.

    The workers do not log. One writer of a command's output is the rule
    (:mod:`poraque.utils.logging`), and eight processes interleaving rows into
    one terminal would break the aligned table the parent prints anyway.

    Parameters
    ----------
    options : dict

    Returns
    -------
    dict
    """
    return {key: value for key, value in options.items() if key != "log"}


def _initialize_worker(roots, options):
    """
    Rebuild every source inside a worker, once per process.

    The parent sends the roots and the options rather than the sources, because
    a source carries the log (see :func:`shippable_options`) and because
    rebuilding costs one directory probe per *process* where shipping would
    cost a pickle per *material*.
    """
    _WORKER_SOURCES.clear()
    for root in roots:
        _WORKER_SOURCES[root] = resolve_source(root, **options)


def _build_one_shipped(payload):
    r"""
    One material, in a worker process.

    Takes the three strings a record is made of --- identifier, directory and
    the ``{field: path}`` map --- and rebuilds the record against the worker's
    own source. Returns ``(identifier, entry)``, or
    ``(identifier, {"error": ...})`` when the material raises: one unreadable
    density in an archive of thousands should not lose the other thousands,
    and the parent re-raises after the summary is written.

    ``Exception``, not ``BaseException``: a ``KeyboardInterrupt`` must tear the
    pool down rather than be reported as a material that failed to cache, and
    ``_build_one``'s own cleanup already catches ``BaseException`` to remove
    its temporaries before re-raising.
    """
    from ..ml.data import MaterialRecord

    identifier, directory, files, root, settings = payload
    record = MaterialRecord(identifier, directory, files=dict(files),
                            source=_WORKER_SOURCES[root])
    try:
        return identifier, _build_one(
            record, settings["cache"], settings["resolution"],
            settings["fields"], settings["spin"],
            settings["remembered"], settings["storage"],
            settings["compression"], settings["compression_level"],
            settings["max_spacing"])
    except Exception as error:
        return identifier, {"error": f"{type(error).__name__}: {error}"}


def _payloads(records, cache, resolution, fields, spin, summary, storage,
              compression, compression_level, max_spacing):
    """
    One picklable work item per material, **most expensive first**.

    Longest-processing-time-first, because one dataset's materials are not one
    size: across this project's 97 platinum cells the native grids span 1.26 M
    to 23.89 M voxels and the build cost is linear in them
    (:math:`t = 0.47 + 1.03\\,{\\rm s}` per million, :math:`R^2 = 0.996`), a
    factor of 19 between the cheapest and the dearest. Handed out in discovery
    order with eight workers, the eleven 288³ nanoparticles at the end of the
    list become a 25-second tail with seven idle cores behind it. File size is
    a free proxy for voxel count and needs no header read.
    """
    ordered = sorted(records, key=_record_cost, reverse=True)
    return [(record.identifier, record.directory, record.files,
             record.source.root,
             {"cache": cache, "resolution": resolution, "fields": fields,
              "spin": spin, "remembered": summary.get(record.identifier),
              "storage": storage, "compression": compression,
              "compression_level": compression_level,
              "max_spacing": max_spacing})
            for record in ordered]


def _record_cost(record):
    """Bytes of the density file: the proxy the schedule is sorted on."""
    try:
        return os.path.getsize(record.files["CHGCAR"])
    except (KeyError, OSError):
        return 0


def resolve_cache_workers(requested, records, emit=None):
    r"""
    How many processes to build the cache with.

    The build is not the FFT work its Slurm script claims. Measured per
    material on the platinum set at ``resolution: 48``, the spectral
    downsampling is **2.2 %** of the clock and the Fortran ``E17.11`` text
    codec is **86 %** --- one ``np.array(line.split())`` per line read and one
    ``fortran_exponential`` per voxel written, both pure Python. There is no
    BLAS call in the path and every transform goes through ``np.fft``, which is
    single-threaded, so ``OMP_NUM_THREADS`` has nothing to act on. What does
    scale is materials: :func:`_build_one` is independent per material and its
    writes are already multi-process safe.

    Parameters
    ----------
    requested : int or str
        A count, ``0``/``1`` for the single-process build, or ``"auto"``.
    records : sequence of MaterialRecord
        Used by ``"auto"`` for both bounds: there is no point in more workers
        than materials, and the memory estimate comes from the largest file.
    emit : callable, optional

    Returns
    -------
    int
        At least 1.

    Raises
    ------
    ValueError
        On a value that is neither ``"auto"`` nor a non-negative integer ---
        before the cache is built, like every other setting.

    Notes
    -----
    ``"auto"`` takes the smallest of three numbers:

    * the cores **this process may use** --- ``len(os.sched_getaffinity(0))``
      where the platform has it, which is what Slurm's ``--cpus-per-task``
      restricts a task to; ``os.cpu_count()`` elsewhere. The node's core count
      is the wrong number on a shared node;
    * a memory budget --- 80 % of the job's Slurm allocation where there is
      one, else :data:`WORKER_MEMORY_BUDGET` --- divided by
      :data:`WORKER_BASE_BYTES` plus :data:`WORKER_BYTES_PER_VOXEL` times the
      **largest** native grid in the set. See :func:`_memory_budget`;
    * the number of materials.

    The line it logs names which of the three bound, so a job that wanted more
    workers can see whether to ask for cores or for memory.
    """
    if isinstance(requested, str):
        text = requested.strip().lower()
        if text not in ("auto", ""):
            try:
                requested = int(text)
            except ValueError:
                raise ValueError(
                    f"data.cache_workers: {requested!r} is neither 'auto' nor "
                    f"a number of processes. Use 'auto', or 0 for the "
                    f"single-process build."
                ) from None
        else:
            requested = "auto"

    if requested != "auto":
        count = int(requested)
        if count < 0:
            raise ValueError(
                f"data.cache_workers: {count} is negative. Use 0 or 1 for the "
                f"single-process build.")
        return max(1, count)

    cores = _usable_cores()
    shapes = _native_voxels(records)
    total = sum(shapes)
    if total and total < WORKER_MINIMUM_VOXELS:
        if emit:
            emit(f"  workers: 1 (auto: {total / 1e6:.3f} M voxels in total, "
                 f"under the {WORKER_MINIMUM_VOXELS / 1e6:.0f} M at which a "
                 f"pool pays for its own start-up)")
        return 1

    voxels = max(shapes, default=0)
    budget, source = _memory_budget()
    bytes_each = WORKER_BASE_BYTES + voxels * WORKER_BYTES_PER_VOXEL
    by_memory = max(1, int(budget // bytes_each))
    workers = max(1, min(cores, by_memory, len(records) or 1))
    if emit:
        emit(f"  workers: {workers} (auto: {cores} usable core(s); "
             f"{by_memory} within the "
             f"{budget / 1024 ** 3:.1f} GiB {source} at "
             f"~{bytes_each / 1024 ** 2:.0f} MiB each, sized on the largest "
             f"grid at {voxels / 1e6:.2f} M voxels; "
             f"{len(records)} material(s))")
    return workers


def _memory_budget():
    r"""
    Memory ``auto`` may commit to the pool, and where the number came from.

    **The job's own allocation when Slurm states one.** A constant is the wrong
    answer on a cluster: `--cpus-per-task=8 --mem=32G` on Santos Dumont against
    an 8 GiB default picks four workers for a 288³ set and idles half the
    allocation, which is the exact job this feature exists for. Slurm exports
    the allocation, so the budget can be read rather than guessed --
    ``SLURM_MEM_PER_NODE`` (MB for the whole node) or ``SLURM_MEM_PER_CPU``
    times ``SLURM_CPUS_PER_TASK``, whichever is set.

    Carried at 80 % of what Slurm granted: the rest is the parent process, the
    page cache the text reads go through, and the margin between a worker's
    estimated peak and its real one. Slurm's enforcement is a hard cgroup
    limit, so overshooting is an OOM kill of the job, not a slow build.

    :data:`WORKER_MEMORY_BUDGET` is the fallback, for a workstation that has no
    allocation to read.

    Returns
    -------
    budget : int
        Bytes.
    source : str
        ``"Slurm allocation"`` or ``"default"``, for the log line -- "4 within
        the 8 GiB budget" and "4 within the 8 GiB Slurm allocation" are
        different facts and only one of them is worth changing a submission
        script over.
    """
    granted = _int_environment("SLURM_MEM_PER_NODE")
    if not granted:
        per_cpu = _int_environment("SLURM_MEM_PER_CPU")
        cpus = _int_environment("SLURM_CPUS_PER_TASK") or 1
        granted = per_cpu * cpus if per_cpu else 0
    if granted > 0:
        return int(0.8 * granted * 1024 ** 2), "Slurm allocation"
    return WORKER_MEMORY_BUDGET, "default budget"


def _int_environment(name):
    """``int(os.environ[name])``, or 0 for absent, empty or malformed."""
    try:
        return int(str(os.environ.get(name, "")).strip().rstrip("MmGg") or 0)
    except ValueError:
        return 0


def _usable_cores():
    r"""
    Cores this process may use, not the ones the node has.

    ``SLURM_CPUS_PER_TASK`` first, because it is the job's own statement of
    what it asked for and Slurm always exports it. ``sched_getaffinity``
    *usually* agrees --- Slurm narrows the mask through ``task/affinity`` or
    ``task/cgroup`` --- but that is a site configuration, not a guarantee, and
    where the plugin is off the mask is the whole node. Reading the variable
    makes the answer the same on every cluster and visible off one: the
    affinity call cannot be simulated, so a wrong worker count under
    ``--cpus-per-task=4`` would only ever be found on the cluster.

    Then the affinity mask, for a ``taskset``/cgroup-confined process outside
    Slurm, and ``os.cpu_count()`` last, for a workstation.
    """
    allocated = _int_environment("SLURM_CPUS_PER_TASK")
    if allocated > 0:
        return allocated
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                                  # macOS, Windows
        return max(1, os.cpu_count() or 1)


def _native_voxels(records):
    r"""
    Native voxel count per material, for sizing the pool.

    Two numbers come out of this list: the **largest** sets the per-worker
    memory estimate (every worker may be handed it) and the **total** decides
    whether a pool is worth starting at all.

    Read from the headers, not estimated from file sizes: a spin-polarized
    ``CHGCAR`` holds two grid blocks, so bytes-over-eighteen-columns
    overestimates it by a factor of two, and the budget that comes out of it
    would halve the worker count on exactly the datasets that most need them.
    :func:`~poraque.fields.vasp.volumetric.read_grid_header` stops at the
    ``NGXF NGYF NGZF`` line, so this costs a few hundred bytes per material
    however large the grid --- 18 ms for all 97 of the platinum set.

    Falls back to the file-size estimate for anything whose header will not
    parse, because a worker count is not worth failing a build over.

    Returns
    -------
    list of int
    """
    from ..fields.vasp.volumetric import read_grid_header

    counts = []
    for record in records:
        path = record.files.get("CHGCAR")
        if not path:
            continue
        try:
            _, shape = read_grid_header(path)
            counts.append(int(shape[0]) * int(shape[1]) * int(shape[2]))
        except (OSError, ValueError, IndexError, TypeError):
            counts.append(_record_cost(record) // 36)
    return counts


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


def _spin_of(path):
    """
    Whether one density file carries a magnetization block. Pool-safe.

    ``None`` when the file could not be read: :func:`probe_spin` answers
    ``False`` for this run but does not remember it, so a transient failure
    (a file still being written, a flaky filesystem) is probed again next time
    rather than recorded as unpolarized for good.
    """
    from ..fields.spin import is_spin_polarized

    if not path:
        return False
    try:
        return bool(is_spin_polarized(path))
    except (OSError, ValueError):
        return None


#: Per-file answers of the spin probe, beside the caches (``data.cache.dir``).
SPIN_MEMO = "spin_probe.json"


def probe_spin(paths, memo_root=None, workers=1):
    r"""
    Whether each density file carries a magnetization block, remembered.

    The probe (:func:`~poraque.fields.spin.is_spin_polarized`) seeks past the
    grid of an uncompressed ``CHGCAR``, but a compressed one cannot be seeked:
    the 432 gzipped CHON densities (8.9 GB) took 34.5 s to answer, every run,
    and the answer never changes while the file does not. So each answer is
    kept in ``<memo_root>/spin_probe.json``, keyed by the file's absolute path
    and validated by its size and modification time: a changed or replaced
    file is probed again, an unchanged one is not read at all.

    Parameters
    ----------
    paths : list of str or None
        Density files; ``None`` (a material without one) answers ``False``.
    memo_root : str, optional
        Directory for the memo, normally the cache root. ``None`` probes
        without remembering.
    workers : int, optional
        Processes for the compressed files still to be probed.

    Returns
    -------
    list of bool
    """
    from ..fields.io.compressed import is_compressed

    memo_path = os.path.join(memo_root, SPIN_MEMO) if memo_root else None
    memo = {}
    if memo_path and os.path.exists(memo_path):
        try:
            with open(memo_path) as handle:
                memo = json.load(handle)
        except (OSError, ValueError):
            memo = {}

    from ..fields.hdf5 import split_target

    def stamp(path):
        # A store-backed field is `fields.h5::CHGCAR`: the file to stat is the
        # store. A missing file still gets no stamp, and its probe below
        # answers (False) without being remembered.
        try:
            info = os.stat(split_target(path)[0])
        except OSError:
            return None
        return [info.st_size, info.st_mtime_ns]

    keys = [os.path.abspath(path) if path else None for path in paths]
    stamps = [stamp(key) if key else None for key in keys]
    flags = [None] * len(paths)
    todo = []
    for index, (key, current) in enumerate(zip(keys, stamps)):
        if key is None:
            flags[index] = False
        elif current is not None and memo.get(key, {}).get("stamp") == current:
            flags[index] = bool(memo[key]["polarized"])
        else:
            todo.append(index)

    if todo:
        pending = [paths[index] for index in todo]
        # A seekable file answers in milliseconds, less than a spawned worker
        # takes to import the package; only compressed archives need the pool.
        compressed = sum(1 for path in pending if is_compressed(path))
        if workers > 1 and compressed > 1:
            context = multiprocessing.get_context("spawn")
            count = min(workers, len(pending))
            with _worker_threads(count), context.Pool(count) as pool:
                answers = pool.map(_spin_of, pending)
        else:
            answers = [_spin_of(path) for path in pending]
        for index, answer in zip(todo, answers):
            flags[index] = bool(answer)
            if stamps[index] is not None and answer is not None:
                memo[keys[index]] = {"stamp": stamps[index],
                                     "polarized": bool(answer)}
        if memo_path:
            try:
                os.makedirs(memo_root, exist_ok=True)
                partial = f"{memo_path}.partial-{os.getpid()}"
                with open(partial, "w") as handle:
                    json.dump(memo, handle)
                os.replace(partial, memo_path)
            except OSError:
                pass
    return flags


def _polarized_records(records, workers, memo_root=None):
    r"""
    Which records carry a magnetization block (see :func:`probe_spin`).

    Only the path is sent to a worker, so this needs no source and no
    initializer.
    """
    paths = [record.files.get("CHGCAR") for record in records]
    flags = probe_spin(paths, memo_root=memo_root, workers=workers)
    return [record for record, polarized in zip(records, flags) if polarized]


def _resolve_spin(requested, records, emit, workers=1, memo_root=None):
    r"""
    Turn ``spin="auto"`` into a decision, by looking at the data.

    This is the *first* of the two places that ask whether a dataset is
    spin-polarized, and it is the one that matters: the second
    (:meth:`~poraque.ml.data.FieldPairDataset._resolve_spin`) inspects the
    **cache**, so whatever is dropped here is invisible to it. The two agreeing
    is not evidence of anything if the first one threw the answer away.

    Regression: this function did not exist, and the caller passed
    ``spin=data.spin is True``. With the default ``"auto"`` that is ``False``,
    so every ``ISPIN = 2`` magnetization block was discarded on the way into
    the cache and the dataset then reported, truthfully, that the cache held
    no spin. Two layers of auto-detection, and the first was an identity test
    against ``True``.

    The decision is taken **once for the whole dataset**, not per material: the
    ML layer sizes one operator for one channel count, so a cache with two
    blocks for some materials and one for others could not be trained from. A
    mixed set therefore resolves to spin, and the unpolarized members get
    :math:`m \equiv 0` — which :class:`~poraque.fields.SpinDensity` documents
    as meaningful rather than wasted, since it makes an ``ISPIN = 2`` operator
    a strict generalization of an ``ISPIN = 1`` one.

    Parameters
    ----------
    requested : {"auto", True, False}
    records : list of MaterialRecord
    emit : callable
    workers : int, optional
        Processes to run the probe over. It is a full pass over every density
        file -- see :func:`_polarized_records`.

    Returns
    -------
    bool

    Raises
    ------
    ValueError
        When ``spin=True`` is asked of a dataset that carries no magnetization
        anywhere -- a channel the data has no values for cannot be trained.
    """
    if requested is False:
        # The answer is already "one channel": a magnetization block, where
        # there is one, is dropped by the reader. Probing would only count the
        # blocks for a log line, and on a compressed archive -- which cannot be
        # seeked -- that count is a full decompression of every density.
        emit("  spin: off (data.spin: false); any magnetization block is "
             "discarded and the cache holds the total density only.")
        return False

    polarized = _polarized_records(records, workers, memo_root=memo_root)

    if requested is True:
        if not polarized:
            raise ValueError(
                "data.spin: true was requested, but none of the "
                f"{len(records)} material(s) has a magnetization block. "
                "Training a channel the data has no values for would fit "
                "noise to zero. Use data.spin: auto, or check ISPIN in the "
                "INCARs that produced these densities.")
        resolved = True
    else:
        resolved = bool(polarized)

    if resolved:
        extra = len(records) - len(polarized)
        note = (f"; {extra} unpolarized material(s) get m = 0"
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

    The first column counts the materials, ``i/N``, so a long build says how
    far it has got. Then the two grid shapes, which say what the downsampling
    did, and the value range of every cached field, which is the cheapest check that it
    did it correctly: a density that reaches zero, a potential that does not
    straddle zero, or a range differing by orders of magnitude from its
    neighbors are all visible at a glance and all mean something.
    """

    #: "-1.23e+02 .. 4.56e+01" and a little slack.
    RANGE_WIDTH = 22

    def __init__(self, identifiers, fields, emit):
        self.fields = tuple(fields)
        self.emit = emit
        self.total = len(identifiers)
        self.count_width = max(len("file"), 2 * len(str(self.total)) + 1)
        self.label_width = max([len("material")]
                               + [len(name) for name in identifiers])

    def header(self):
        columns = [f"{'file':>{self.count_width}s}",
                   f"{'material':<{self.label_width}s}",
                   f"{'native grid':>14s}", f"{'cached grid':>13s}"]
        columns += [f"{f'{name} [{FIELD_UNITS[name]}]':<{self.RANGE_WIDTH}s}"
                    for name in self.fields]
        columns.append("time")
        line = "  " + "  ".join(columns)
        self.emit("")
        self.emit(line)
        self.emit("  " + "-" * (len(line) - 2))

    def row(self, position, identifier, entry):
        count = f"{position}/{self.total}"
        columns = [f"{count:>{self.count_width}s}",
                   f"{identifier:<{self.label_width}s}",
                   f"{_shape_text(entry.get('native')):>14s}",
                   f"{_shape_text(entry.get('shape')):>13s}"]
        for name in self.fields:
            text = _range_text(entry["ranges"].get(name))
            columns.append(f"{text:<{self.RANGE_WIDTH}s}")
        columns.append("cached" if entry.get("reused")
                       else f"{entry.get('seconds', 0.0):.1f} s")
        self.emit("  " + "  ".join(columns).rstrip())

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
                 potcar_source=None, max_spacing=None):
    """Everything that decides what a cache directory's files contain."""
    potcar_dir = options.get("potcar_dir")
    fingerprint = {
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
    # Only when set, so every cache built before the key existed keeps a
    # fingerprint that still matches its own.
    if max_spacing:
        fingerprint["max_spacing"] = float(max_spacing)
    return fingerprint


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


def _build_one(record, cache, resolution, fields, spin, remembered=None,
               storage="files", compression=None, compression_level=4,
               max_spacing=None):
    """
    Downsample and write one material, unless it is already there.

    Returns
    -------
    dict
        ``native`` and ``shape`` grid shapes, the ``ranges`` of every field
        written, and either ``seconds`` or
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
    reduced = downsampled_grid(native, resolution, max_spacing)

    os.makedirs(destination, exist_ok=True)
    ranges = {}

    # Every field is written under a temporary name in its own directory and
    # moved into place with `os.replace`, which is atomic. Two processes
    # sharing a cold cache -- two jobs in one allocation, or a resubmission
    # overlapping its predecessor -- otherwise interleave inside a single
    # `write` and leave a torn file behind, and the text format has no length
    # field with which to notice: a truncated `CHGCAR` parses, into a field of
    # the wrong shape, and `_all_present` then calls that material done
    # forever. Observed at LNCC as a zero-length density on a cache two jobs
    # built at once. It is the between-jobs form of the race the DDP barrier
    # already handles within one job.
    #
    # What this does *not* do is stop the two builds duplicating the work.
    # Only a lock would, and a lock that behaves across a shared parallel
    # filesystem is a much larger promise than this function should make;
    # duplicated work costs minutes, a torn field costs the dataset.
    marker = f".partial-{os.getpid()}"
    store = os.path.join(destination, HDF5_FILENAME)
    pending = {}

    try:
        for name in wanted:
            field = source.read(record, name, native, spin=spin)
            reduced_field = (resample_field(field, reduced.shape, grid=reduced)
                             if reduced is not native else field)
            records = (_augmentation_for(record, reduced_field)
                       if name == "CHGCAR" else None)

            # Registered *before* the write, not after: a writer that raises
            # partway has already created the file, and a temporary the
            # cleanup below does not know about is litter nothing will ever
            # remove.
            if storage == "hdf5":
                from ..fields.hdf5 import write_fields

                # One store holds every field, so the whole store is built
                # beside its final name and moved once, after the loop.
                temporary = _partial(store, marker)
                pending[temporary] = store
                write_fields(temporary, {name: reduced_field},
                             compression=compression, level=compression_level,
                             augmentation=({name: records} if records
                                           else None))
            else:
                temporary = _partial(targets[name], marker)
                pending[temporary] = targets[name]
                reduced_field.write(temporary, augmentation=records or None)

            # The range column shows a band-limited density dipping below
            # zero (Gibbs ringing around the core peaks), which is an artifact
            # of the truncation and why the dataset uses the sign-tolerant
            # asinh transform; it is not reported separately.
            data = reduced_field.data
            ranges[name] = [float(data.min()), float(data.max())]
    except BaseException:
        # A half-written temporary is litter, not a cache entry, and it names
        # a process that is no longer running.
        for temporary in pending:
            with contextlib.suppress(OSError):
                os.remove(temporary)
        raise

    for temporary, final in pending.items():
        os.replace(temporary, final)

    return {"native": list(native.shape), "shape": list(reduced.shape),
            "ranges": ranges,
            "seconds": time.time() - started}


def _augmentation_for(record, density):
    r"""
    The PAW augmentation records to write beside a cached density.

    Copied verbatim from the native file. The records are **on-site**
    quantities --- occupancies of the projector channels inside each
    augmentation sphere --- so downsampling the grid neither changes them nor
    has anything to say about them, and a cache that dropped them dropped the
    only per-atom target the data carries.

    One set per density channel, because that is how the cached file is laid
    out: a spin-polarized density is written as two grid blocks with a set
    after each, a density read as one channel from a spin-polarized run keeps
    the total's set alone. The first set keeps the ``MAGMOM`` line VASP writes
    after it, which a reader skips by the declared record lengths and a VASP
    restart expects to find.

    Returns
    -------
    list of list of str
        Empty when the calculation wrote none --- a norm-conserving run, or a
        density stored without them.
    """
    from ..fields.spin import SpinDensity
    from ..fields.vasp.volumetric import read_augmentation_blocks

    path = record.files.get("CHGCAR")
    if not path:
        return []
    try:
        _, blocks = read_augmentation_blocks(path)
    except (OSError, ValueError):
        return []
    channels = 2 if isinstance(density, SpinDensity) else 1
    return [list(block) for block in blocks[:channels]]


def _partial(path, marker):
    """
    A sibling temporary that keeps ``path``'s suffix.

    The suffix is not cosmetic: :meth:`ScalarField.write` dispatches text
    against HDF5 on it, and so does :func:`poraque.fields.hdf5.is_hdf5_path`.
    ``fields.h5`` becomes ``fields.partial-1234.h5`` rather than
    ``fields.h5.partial-1234``; a cache field has no extension at all and
    simply gains one.
    """
    root, extension = os.path.splitext(path)
    return f"{root}{marker}{extension}"


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
        return {**remembered, "reused": True}

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
            "reused": True}


# ---------------------------------------------------------------------- #
# PAW augmentation reference
# ---------------------------------------------------------------------- #
def build_paw_reference(records, cache, log=None, library=None,
                        source="atomic"):
    r"""
    The per-element PAW augmentation table the model bundle carries.

    The one-center terms are contractions over the converged wavefunctions, so
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
        The previous behavior: average the training calculations' records per
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
    from ..fields.vasp.augmentation import RECORD_SCHEMA, build_reference

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
        # And a table built before records were read to their declared length
        # is not reused at all: on spin-polarized data its records carry the
        # MAGMOM line, so it is the wrong length whichever source built it.
        current = all(isinstance(entry, dict)
                      and entry.get("schema", 0) >= RECORD_SCHEMA
                      for entry in reference.values())
        if not current:
            emit("  PAW reference: cached table predates reading records to "
                 "their declared length — rebuilding")
        elif origin == wanted or not reference:
            emit(f"  PAW reference: cached, {sorted(reference)} [{origin}]")
            return reference
        else:
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


def build_paw_profiles(records, cache, log=None):
    r"""
    The radial PAW core charge density of every element in the training set.

    Read from **the POTCAR that built** :math:`V_{\rm ext}`, in the precedence
    :meth:`~poraque.data.sources.CalculationSource._external_potential` uses:
    a ``POTCAR`` in the material's own directory first, and the source's
    ``potcar_dir`` library when there is none. A profile from any other file
    would describe a pseudopotential the model was not trained against.

    One POTCAR is read per element, not per material --- the first material
    that contains it decides --- so a set mixing two variants of one element
    (``Fe`` beside ``Fe_pv``) records only the first. The profile's ``titel``
    names which one.

    Cached beside the fields as ``paw_profiles.json`` and reused when present,
    as :func:`build_paw_reference` is. That is not only a saving: on a cluster
    the cache is built in a CPU job and training runs in a GPU job that may
    see a different filesystem, and a GPU node without the POTCAR library
    mounted must not overwrite the profiles the cache job read with nothing.

    Parameters
    ----------
    records : sequence of MaterialRecord
        The training materials; each carries the source it came from.
    cache : str
        Where to write ``paw_profiles.json``.
    log : callable, optional

    Returns
    -------
    dict
        ``{atomic_number: profile}`` as
        :attr:`~poraque.fields.vasp.potcar.PotcarSingle.core_profile` returns
        it. Empty when no material had a POTCAR to read --- a Gaussian-potential
        dataset has no core density to record.
    """
    from ..fields.vasp.potcar import Potcar, read_potcar_text
    from ..fields.vasp.volumetric import read_structure_header

    emit = log or (lambda *_: None)
    path = os.path.join(cache, PAW_PROFILES_FILENAME)
    if os.path.exists(path):
        profiles = _read_profiles(path)
        # A table written before the projector channels were recorded cannot
        # lay out an occupancy record, which ext2chg needs; it is rebuilt.
        if profiles and all(entry.get("projector_channels")
                            for entry in profiles.values()):
            emit(f"  PAW core profiles: cached, "
                 f"{sorted(entry['element'] for entry in profiles.values())}")
            return profiles
        if profiles:
            emit("  PAW core profiles: cached table predates the projector "
                 "channels — rebuilding")

    profiles, unserved = {}, set()
    for record in records:
        density = record.files.get("CHGCAR")
        if not density:
            continue
        try:
            elements = read_structure_header(
                density.split("::", 1)[0]).elements
        except (OSError, ValueError):
            continue
        wanted = [element for element in dict.fromkeys(elements)
                  if element not in {entry["element"]
                                     for entry in profiles.values()}]
        if not wanted:
            continue

        entries = []
        own = os.path.join(record.directory, "POTCAR")
        if os.path.exists(own):
            try:
                entries = list(Potcar.from_string(read_potcar_text(own),
                                                  parse_tables=True))
            except (OSError, ValueError) as error:
                emit(f"  PAW core profiles: {own} unreadable ({error})")
        elif getattr(record, "source", None) is not None \
                and record.source.options.get("potcar_dir"):
            entries = [record.source._library_entry(element)
                       for element in wanted]

        for entry in entries:
            if entry is None or entry.element not in wanted:
                continue
            profile = entry.core_profile
            if profile is None:
                unserved.add(entry.element)
                continue
            profiles[profile["atomic_number"]] = profile
        unserved.update(element for element in wanted
                        if element not in {entry["element"]
                                           for entry in profiles.values()})

    unserved -= {entry["element"] for entry in profiles.values()}
    if not profiles:
        emit("  PAW core profiles: none -- no material had a PAW POTCAR to "
             "read, so the checkpoint carries no core densities")
        return {}

    for number, profile in sorted(profiles.items()):
        error = abs(profile["core_electrons"]
                    - profile["expected_core_electrons"])
        # sqrt(4 pi) times the integral is Z - ZVAL to 1e-4 on every dataset
        # this was measured against, so a gap means a misread table.
        flag = ("" if error < 1e-3 * max(1.0, profile[
            "expected_core_electrons"]) else "   !! does not match Z - ZVAL")
        emit(f"  PAW core profile: {profile['element']} (Z={number}, "
             f"{profile['titel']}) {profile['core_electrons']:.4f} core "
             f"electrons on {len(profile['r'])} radial points{flag}")
    if unserved:
        emit(f"  PAW core profiles: no PAW POTCAR for {sorted(unserved)}")

    os.makedirs(cache, exist_ok=True)
    with open(path, "w") as handle:
        json.dump({str(number): profile
                   for number, profile in profiles.items()}, handle)
    return profiles


def _read_profiles(path):
    """``paw_profiles.json`` with its keys back as atomic numbers."""
    with open(path) as handle:
        return {int(number): profile
                for number, profile in json.load(handle).items()}


def load_paw_profiles(cache):
    r"""
    Everything PAW the checkpoint stores, per element, keyed by atomic number.

    Two tables, merged because they answer one question --- what a VASP
    ``CHGCAR`` of this element needs that no grid model predicts:

    * the **core density** from :func:`build_paw_profiles` (``r``,
      ``core_density``, ``pseudo_core_density``, ...);
    * the **augmentation occupancies** from :func:`build_paw_reference`, under
      ``"augmentation"`` --- the one-center records ``--add-paw`` writes.

    An element may carry either or both: a Gaussian-potential dataset has
    augmentation records from its CHGCARs and no POTCAR to take a core from.

    Returns
    -------
    dict
        ``{atomic_number: {"element": ..., ...}}``, empty when the cache holds
        neither table.
    """
    from ..fields.vasp.poscar import symbol_to_z

    path = os.path.join(cache, PAW_PROFILES_FILENAME)
    profiles = _read_profiles(path) if os.path.exists(path) else {}
    for element, entry in load_paw_reference(cache).items():
        number = int(symbol_to_z(element))
        profiles.setdefault(number, {"element": element,
                                     "atomic_number": number})
        profiles[number]["augmentation"] = entry
    return profiles
