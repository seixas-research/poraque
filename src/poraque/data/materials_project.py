# -*- coding: utf-8 -*-
# file: materials_project.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Fetch training data from the Materials Project.

:class:`MPDataFetcher` turns a *chemical space* — a set of elements, optionally
narrowed by band gap, crystal system or stability — into a local dataset of
charge densities that :mod:`poraque.data.mp_dataset` reads directly:

    >>> from poraque.data import MPDataFetcher
    >>> fetcher = MPDataFetcher(["Pt", "Pd", "Ni"], outdir="data/MP")   # doctest: +SKIP
    >>> print(fetcher.estimate())        # exact size, nothing transferred  # doctest: +SKIP
    >>> fetcher.run(max_size_mb=20)      # summary, structures, CHGCARs     # doctest: +SKIP

A chemical space is *every* material whose composition uses only the given
elements, across all stoichiometries and structures — the Pt-Pd-Ni space
contains the three elementals, the binaries and the ternaries alike.

Authentication
--------------
The API key is read, in order, from the ``api_key`` argument, the ``MP_API_KEY``
environment variable, a ``.env`` file in the working directory, and finally
``~/.env``. Both files are loaded with :mod:`dotenv`, which is why the key never
has to appear in a config file, a notebook or a shell history — the three places
it most easily leaks from. It is held on the instance and passed to
:class:`~mp_api.client.MPRester`; :meth:`close` (and the context-manager form)
drops the session that holds it.

Sizing before downloading
-------------------------
:meth:`estimate` reports the **exact** transfer, not a model of it. Charge
densities live as objects in the public ``materialsproject-parsed`` S3 bucket,
so the fetcher resolves each material's static-calculation task ID and issues an
S3 ``HEAD`` request, reading ``Content-Length`` without transferring a byte of
payload. Storage is reported three ways because they differ a lot:

    download    bytes pulled over the network (the gzipped S3 object)
    gz on disk  bytes kept if CHGCARs stay gzipped   (x0.879 of download)
    unzipped    bytes kept if CHGCARs are expanded   (x2.99 of gz on disk)

Both ratios were measured against the fully downloaded Pt-Pd-Ni-Cu-Ag set
(85 objects: 1044.6 MB downloaded, 917.7 MB gzipped, 2.74 GB unzipped) rather
than guessed.

**There is no reason to unzip.** Poraquê reads gzipped volumetric files in
place (:mod:`poraque.fields.io.compressed`), so :meth:`decompress` exists only
for interoperating with tools that cannot, and costs roughly a threefold
storage multiplier when used.

The whole database
------------------
``--all`` (or an omitted ``elements``) selects **every** material the index says
has a charge density, not a chemical space. That is one query, not :math:`2^n`:
the summary endpoint takes ``has_props=["charge_density"]`` and answers
server-side, so the alternative — pulling every summary document in the database
and discarding the ones without a density — never happens. The search filters
still apply, which is how a subset of the whole database is taken.

Sizing that set exactly would mean one S3 ``HEAD`` per object, tens of thousands
of them. ``--sample N`` measures ``N`` randomly chosen objects instead and
extrapolates from the sample mean, and the report states which method it used —
an estimate whose method is unstated is a number nobody can plan against. The
sample is drawn with a fixed seed, because charge-density sizes are strongly
right-tailed and an estimate that moved every time it was asked would be useless.

Two routes to the same bytes
----------------------------
There is no separate "bulk" mechanism to choose between, and it is worth saying
so plainly because the question comes up every time:

1. **The mp-api client.** ``MPRester.get_charge_density_from_material_id``
   resolves the material's newest ``GGA Static``/``GGA+U Static`` task and calls
   ``get_charge_density_from_task_id``.
2. **AWS Open Data.** That method is itself a thin wrapper over
   ``_query_open_data(bucket="materialsproject-parsed",
   key="chgcars/<AlphaID>.json.gz")`` — the public S3 bucket, unsigned, no key
   required for the payload.

They are the same object. What makes bulk practical is not picking route 2 over
route 1 but doing the *resolution* in batch: the client issues two queries per
material, and :meth:`MPDataFetcher._resolve_keys` issues two for the whole set.
Sizing already goes straight to the bucket with unsigned ``HEAD`` requests
(:meth:`MPDataFetcher._head_sizes`), for which there is no client route at all.
Downloading goes through the client, which is the documented path and the one
that stays correct when MP changes its key scheme.

Command line
------------
::

    poraque-mp --elements Pt Pd Ni --estimate          # dry run, writes nothing
    poraque-mp --elements Pt Pd Ni --outdir data/MP --max-size-mb 20
    poraque-mp --elements Si O --band-gap 0.5 6.0 --crystal-system Cubic

    # every material with a charge density, sized from a sample of 20
    poraque-mp --all --estimate --sample 20

    # the trainable subset of the whole database, as compressed HDF5
    poraque-mp --all --num-sites 1 12 --max-size-mb 20 \
        --outdir data/MP --hdf5 --compression gzip

``--estimate`` is a **pure dry run**: it reports to the console and leaves no
file behind. Everything else writes into ``--outdir`` (``--output`` is accepted
too), which defaults to the current directory.

What a download leaves behind
-----------------------------
**One directory per material, named by its id**, holding that material's
files::

    data/MP/
        summary.csv  summary.json  chgcar_estimate.csv
        manifest.json  manifest.csv
        structures/mp-124.cif  structures/mp-81.cif
        mp-124/CHGCAR.gz
        mp-81/CHGCAR.gz
        mp-126/fields.h5                    # --hdf5

That is the shape of a VASP run, and it is the shape
:class:`~poraque.data.sources.BulkDensitySource` reads — so a download drops
straight into a training config with no rearranging::

    data:
      root: data/MP

It replaces a flat ``chgcar/CHGCAR_mp-124.gz``, where the material id lived in
the filename and every density in the archive shared one directory. The cost of
that arrangement was not tidiness: nothing could be placed *beside* a density
— a structure file, a POTCAR record, a second field — without inventing a
second naming convention for it, and a per-material directory needs none.

**The old layout is still read, and still resumed from.** An archive already on
disk keeps working, `decompress`/`recompress` see both, and an interrupted
download picks up where it stopped rather than re-fetching thousands of objects
to rename them.

Downloading in bulk
-------------------
A run over thousands of objects is interrupted, rate-limited and partially
failed as a matter of course, so :meth:`MPDataFetcher.download` treats all three
as normal:

* **Resume** from ``manifest.json``, rewritten after every single file.
  An entry marked ``downloaded`` or ``cached`` is settled; a ``failed`` one is
  retried, since that is the only way the set ever reaches completeness. A file
  on disk with no manifest entry is honoured too, so the two mechanisms overlap
  rather than depending on each other.
* **Backoff** on rate limits and dropped connections — exponential, with jitter
  so throttled workers do not all return together. A *permanent* failure is
  recorded on the first attempt; retrying a missing object makes the same error
  slower, not rarer.
* **Provenance** per entry: material id, formula, site count, grid dimensions,
  file size, attempts, the bucket, the client versions and the retrieval date.
  An archive assembled over weeks is not one dataset, and without a per-entry
  date there is no way afterwards to tell a stale file from a current one.

``--hdf5`` stores each density as a chunked, compressed Poraquê field store
(:mod:`poraque.fields.hdf5`) instead of a ``CHGCAR``. The values are identical —
pymatgen holds :math:`\rho\Omega` on the grid, which is exactly the convention
the store keeps, so it is a re-encoding and not a conversion. What is lost is
the PAW augmentation block, which no HDF5 layout here carries: a store trains
perfectly well and cannot seed a VASP ``ICHARG=1`` restart.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import os
import random
import shutil
import sys
import time

import numpy as np
from datetime import datetime, timezone

from .. import banner
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

# --- measured constants -----------------------------------------------------
# Calibrated on the Pt-Pd-Ni-Cu-Ag set (85 charge densities); see the module
# docstring. Ratios, not fits: they convert one measured total into another.
DISK_PER_DOWNLOAD = 0.879  # gzipped-on-disk / bytes-downloaded
UNZIP_EXPANSION = 2.99     # unzipped / gzipped-on-disk

CHGCAR_BUCKET = "materialsproject-parsed"
CHGCAR_PREFIX = "chgcars"

#: Ids per id-filtered API call. The server refuses a filter beyond a few
#: thousand — "List of material/molecule IDs provided is too long" — which the
#: whole-database query meets on its first call. 1000 is comfortably inside it.
ID_BATCH = 1000

#: Calculation types whose charge density MP publishes. A relaxation's density
#: is the last ionic step's and is not what the reported energy belongs to.
STATIC_CALC_TYPES = {"GGA Static", "GGA+U Static"}

#: Summary fields an *estimate* needs. Deliberately without ``structure``: it
#: is by far the heaviest field in the document, and sizing a set never looks
#: at a geometry. Over one chemical space the difference is a few seconds; over
#: the whole database — a hundred thousand materials, each with a full
#: :class:`~pymatgen.core.Structure` — it is the difference between a query
#: that answers and one that does not.
ESTIMATE_FIELDS = [
    "material_id", "formula_pretty", "chemsys", "nelements", "nsites",
    "symmetry", "volume", "energy_above_hull", "is_stable", "band_gap",
    "is_metal", "is_magnetic", "theoretical", "deprecated", "has_props",
]

#: Summary fields requested from the API. ``structure`` is the expensive one
#: and is kept because it is what :meth:`MPDataFetcher.write_structures` writes.
SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "chemsys", "nelements", "nsites",
    "symmetry", "volume", "density", "density_atomic", "energy_per_atom",
    "formation_energy_per_atom", "energy_above_hull", "is_stable", "band_gap",
    "is_metal", "is_magnetic", "ordering", "total_magnetization",
    "theoretical", "deprecated", "has_props", "structure",
]

#: Columns of ``summary.csv``, flattened out of the summary documents.
CSV_COLUMNS = [
    "material_id", "formula_pretty", "chemsys", "nelements", "nsites",
    "spacegroup_symbol", "spacegroup_number", "crystal_system", "volume",
    "density", "density_atomic", "energy_per_atom",
    "formation_energy_per_atom", "energy_above_hull", "is_stable", "band_gap",
    "is_metal", "is_magnetic", "ordering", "total_magnetization",
    "theoretical", "has_charge_density",
]

MANIFEST_COLUMNS = ["material_id", "formula_pretty", "nsites", "file",
                    "size_mb", "compressed", "status", "error",
                    "grid", "retrieved", "attempts"]

#: JSON manifest written beside the CSV one. This is the file a resumed run
#: reads back: the CSV is for a human, and parsing money-critical state out of
#: a spreadsheet format is how a resume quietly loses a column.
MANIFEST_JSON = "manifest.json"

#: Transient failures worth retrying, by exception *name* rather than by class.
#: botocore and urllib3 raise from a dozen modules and importing them all here
#: to name the classes would make this module depend on the whole transport
#: stack for the sake of an isinstance check.
TRANSIENT_ERRORS = (
    "ClientError", "ConnectionError", "ConnectionClosedError", "EndpointConnectionError",
    "HTTPError", "IncompleteRead", "ProtocolError", "ReadTimeout", "ReadTimeoutError",
    "RemoteDisconnected", "RequestException", "ResponseStreamingError",
    "SSLError", "Timeout", "TimeoutError", "TooManyRequestsError",
    "ChunkedEncodingError", "MPRestError",
)

#: Written beside the CHGCARs as a human-readable record of what a fetch did.
#: Nothing reads it back; the dataset layer discovers files directly.
MANIFEST_FILENAME = "manifest.csv"

#: Filename a downloaded density is given inside its own material directory.
#: Bare ``CHGCAR``, as VASP writes it: the material is named by the directory,
#: so repeating the id in the filename would say the same thing twice and would
#: make the file unreadable to anything that keys on VASP's own names.
DENSITY_FILENAME = "CHGCAR"

#: Filename of the HDF5 field store, when ``--hdf5`` is used. The same name the
#: prepared cache uses (:mod:`poraque.data.cache`), because it is the same
#: thing: one material, one store, addressed as ``fields.h5::CHGCAR``.
STORE_FILENAME = "fields.h5"

#: Where downloads used to go: one flat directory of ``CHGCAR_<id>.gz``.
#: Kept because archives already on disk are in that layout and must still be
#: read and, more importantly, still *resumed* -- re-fetching tens of thousands
#: of objects because the layout moved is not a trade anyone would accept.
LEGACY_SUBDIRECTORY = "chgcar"


def retrieval_provenance():
    """
    Who fetched this, from where, with what, and when.

    Recorded per entry in the manifest. A charge-density archive assembled over
    weeks is not one dataset — objects are re-parsed and task ids are
    superseded — and without a per-entry date and client version there is no
    way afterwards to tell a stale file from a current one.

    Returns
    -------
    dict
    """
    from importlib.metadata import PackageNotFoundError, version

    def _version(package):
        try:
            return version(package)
        except PackageNotFoundError:                            # pragma: no cover
            return None

    return {
        "retrieved": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mp_api_version": _version("mp-api"),
        "emmet_core_version": _version("emmet-core"),
        "pymatgen_version": _version("pymatgen"),
        "bucket": CHGCAR_BUCKET,
        "prefix": CHGCAR_PREFIX,
    }


def _is_transient(error):
    """Whether ``error`` is worth another attempt."""
    names = {type(error).__name__} | {
        base.__name__ for base in type(error).__mro__}
    return bool(names & set(TRANSIENT_ERRORS))


def with_retries(call, attempts=4, base_delay=2.0, label="", log=print,
                 sleep=time.sleep):
    """
    Run ``call``, retrying transient failures with exponential backoff.

    The Materials Project rate-limits, and a bulk fetch is exactly the workload
    that meets the limit. A 429 or a dropped connection halfway through a
    thousand-object run is not a reason to lose the run, and it is also not a
    reason to hammer: the delay doubles, with jitter so a set of workers that
    were throttled together do not all come back together.

    A *permanent* failure — a missing object, a malformed document — is raised
    on the first attempt. Retrying it would turn a clear error into a slow one.

    Parameters
    ----------
    call : callable
        Taking no arguments.
    attempts : int, optional
        Total tries, not retries; ``1`` disables retrying.
    base_delay : float, optional
        Seconds before the first retry.
    label : str, optional
        Prefixed to the log line.
    log : callable, optional
    sleep : callable, optional
        Injected so tests do not actually wait.

    Returns
    -------
    tuple
        ``(result, attempts_used)``.
    """
    last = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            return call(), attempt
        except Exception as error:                              # noqa: BLE001
            last = error
            if attempt >= attempts or not _is_transient(error):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            delay *= 0.5 + random.random()          # jitter, so retries spread
            log(f"{label}retrying in {delay:.1f}s after "
                f"{type(error).__name__}: {error}")
            sleep(delay)
    raise last                                                  # pragma: no cover


def _format_bytes(count):
    """Human-readable byte count."""
    if count >= 1e9:
        return f"{count / 1e9:.2f} GB"
    if count >= 1e6:
        return f"{count / 1e6:.1f} MB"
    return f"{count / 1e3:.0f} kB"


@dataclass
class Estimate:
    """
    Exact download and storage figures for a chemical space's densities.

    Attributes
    ----------
    elements : list of str
        The space these figures describe.
    n_materials : int
        Materials matching the search.
    n_advertised : int
        Materials whose ``has_props`` claims a charge density.
    n_available : int
        Materials whose S3 object is actually present. The two differ: the
        index can advertise a density the bucket does not hold.
    sizes : dict
        ``{material_id: bytes}``, with ``-1`` for an advertised but absent
        object.
    rows : list of dict
        Per-material records, largest first.
    method : str
        ``"exact"`` when every object was measured with an S3 ``HEAD``, or
        ``"sampled"`` when a random subset was measured and the total
        extrapolated. Reported in :meth:`__str__` because an estimate whose
        method is unstated is a number nobody can act on: the difference here
        is between "1.9 TB" and "1.9 TB, give or take a few hundred GB".
    sampled : int
        How many objects were actually measured. Equals :attr:`n_available` for
        an exact estimate.
    """

    elements: list
    n_materials: int
    n_advertised: int
    n_available: int
    sizes: dict = field(repr=False, default_factory=dict)
    rows: list = field(repr=False, default_factory=list)
    method: str = "exact"
    sampled: int = 0

    @property
    def label(self):
        """How this selection is named: a chemical space, or the database."""
        return "-".join(self.elements) + " space" if self.elements \
            else "all elements with a charge density"

    @property
    def measured_bytes(self):
        """Bytes actually observed, over however many objects were measured."""
        return sum(size for size in self.sizes.values() if size > 0)

    @property
    def mean_bytes(self):
        """Mean size of a measured object, the basis of an extrapolation."""
        good = [size for size in self.sizes.values() if size > 0]
        return (sum(good) / len(good)) if good else 0.0

    @property
    def is_sampled(self):
        """Whether the totals below are extrapolated rather than measured."""
        return self.method == "sampled"

    @property
    def download_bytes(self):
        """
        Bytes that would cross the network.

        Measured exactly when every object was HEADed, and extrapolated from
        the sample's mean otherwise. The extrapolation is a mean and not a
        median on purpose: what is being predicted is a *total*, and a total is
        the mean times the count however skewed the distribution is. The median
        would understate it badly here, because charge-density sizes are heavily
        right-tailed — a handful of large cells carry a large share of the bytes.
        """
        if self.is_sampled:
            return self.mean_bytes * self.n_advertised
        return self.measured_bytes

    @property
    def gz_disk_bytes(self):
        """Storage if the CHGCARs are kept gzipped, as Poraquê reads them."""
        return self.download_bytes * DISK_PER_DOWNLOAD

    @property
    def unzipped_bytes(self):
        """Storage if the CHGCARs are expanded — roughly 3x the gzipped figure."""
        return self.gz_disk_bytes * UNZIP_EXPANSION

    @property
    def n_missing(self):
        """Advertised by the index but not actually fetchable."""
        return self.n_advertised - self.n_available

    def files_under(self, max_size_mb):
        """
        ``(count, bytes)`` of the objects at or below a size cap.

        Scaled up from the sample when the estimate is a sampled one, so the
        row means the same thing in both modes: how many of the *whole* set
        would survive the cap, and what they would weigh.
        """
        kept = [s for s in self.sizes.values() if 0 < s <= max_size_mb * 1e6]
        if not self.is_sampled or not self.sampled:
            return len(kept), sum(kept)
        scale = self.n_advertised / self.sampled
        return int(round(len(kept) * scale)), sum(kept) * scale

    def __str__(self):
        good = sorted(size for size in self.sizes.values() if size > 0)
        if not good:
            return "Estimate: no charge densities available"

        def percentile(fraction):
            return good[min(int(len(good) * fraction), len(good) - 1)]

        rule = "=" * 66
        lines = [
            rule,
            f"  CHGCAR ESTIMATE - {self.label}",
            rule,
            f"  Materials matched            : {self.n_materials}",
            f"  Advertising a charge density : {self.n_advertised}",
        ]
        if self.is_sampled:
            lines += [
                f"  Objects measured (sample)    : {self.sampled}"
                f"  ({100 * self.sampled / max(self.n_advertised, 1):.1f}%)",
                f"  Mean measured object         : "
                f"{_format_bytes(self.mean_bytes)}",
                f"  TOTAL DOWNLOAD (ESTIMATED)   : "
                f"{_format_bytes(self.download_bytes)}",
                "  method: sizes of a random sample read with S3 HEAD "
                "requests, no",
                "          payload transferred; total = mean x "
                f"{self.n_advertised} advertised.",
            ]
        else:
            lines += [
                f"  FILES TO DOWNLOAD            : {self.n_available}",
                f"  TOTAL DOWNLOAD               : "
                f"{_format_bytes(self.download_bytes)}",
                "  method: every object measured exactly with an S3 HEAD "
                "request.",
            ]
        lines += ["-" * 66,
            f"  Storage if kept gzipped      : {_format_bytes(self.gz_disk_bytes)}",
            f"  Storage if UNZIPPED (x{UNZIP_EXPANSION})   : "
            f"{_format_bytes(self.unzipped_bytes)}",
            "-" * 66,
            f"  smallest / median / largest  : {_format_bytes(good[0])} / "
            f"{_format_bytes(percentile(0.5))} / {_format_bytes(good[-1])}",
            f"  90th percentile              : {_format_bytes(percentile(0.9))}",
        ]
        if self.n_missing > 0 and not self.is_sampled:
            lines.append(f"  Advertised but unavailable   : {self.n_missing}")
        lines.append("-" * 66)
        lines.append("  If you cap file size (--max-size-mb):")
        for cap in (5, 10, 20, 50, 100, 200):
            count, total = self.files_under(cap)
            if count:
                expanded = _format_bytes(total * DISK_PER_DOWNLOAD * UNZIP_EXPANSION)
                lines.append(f"    <= {cap:4d} MB : {count:5d} files, "
                             f"{_format_bytes(total):>9}  (unzipped {expanded})")
        lines.append("-" * 66)
        lines.append("  10 largest:")
        for row in self.rows[:10]:
            lines.append(f"    {row['material_id']:<14} {row['formula_pretty']:<14} "
                         f"{row['nsites']:>4} sites {row['size_mb']:>8.1f} MB")
        lines.append(rule)
        return "\n".join(lines)


class MPDataFetcher:
    """
    A Materials Project chemical space and its charge-density dataset.

    Parameters
    ----------
    elements : sequence of str or str
        Elements spanning the space, e.g. ``["Pt", "Pd", "Ni"]``. A string is
        accepted in either ``"Pt-Pd-Ni"`` or ``"Pt Pd Ni"`` form.
    api_key : str, optional
        Materials Project API key. Falls back to ``$MP_API_KEY``, then a local
        ``.env``, then ``~/.env``.
    outdir : str or pathlib.Path, optional
        Where ``summary.*``, ``structures/``, ``manifest.*`` and the
        per-material ``mp-124/`` directories are written.
    compress : bool, optional
        Keep the downloaded CHGCARs gzipped (default). Poraquê reads them
        gzipped, so the only reason to turn this off is another tool that
        cannot — at roughly three times the storage.
    workers : int, optional
        Parallel S3 ``HEAD`` requests used while sizing.
    band_gap : tuple of float, optional
        ``(min, max)`` gap in eV. ``(0.0, 0.0)`` selects metals; passing
        ``None`` for either end leaves it open.
    crystal_system : str or sequence of str, optional
        Restrict to one or more crystal systems, e.g. ``"Cubic"``.
    is_stable : bool, optional
        Restrict to (or exclude) materials on the convex hull.
    num_sites : tuple of int, optional
        ``(min, max)`` sites per cell. The most effective filter for keeping a
        dataset trainable: cost grows with the FFT grid, which grows with the
        cell.
    exclude_deprecated : bool, optional
        Drop entries MP has marked deprecated. On by default.

    Notes
    -----
    Filters are applied where they are cheapest. ``band_gap``, ``num_sites``
    and ``is_stable`` are pushed into the API query so the server does the
    work; ``crystal_system`` is applied to the returned documents, because the
    symmetry it lives under is a sub-document rather than a queryable field.

    Examples
    --------
    >>> with MPDataFetcher(["Pt", "Pd", "Ni"], outdir="data/MP") as mp:  # doctest: +SKIP
    ...     print(mp.estimate())
    ...     mp.run(max_sites=8, max_size_mb=20)
    """

    def __init__(self, elements=None, api_key=None, outdir=".", compress=True,
                 workers=16, band_gap=None, crystal_system=None,
                 is_stable=None, num_sites=None, exclude_deprecated=True,
                 retries=4, retry_delay=2.0):
        if isinstance(elements, str):
            elements = elements.replace("-", " ").split()
        # No elements is not an error any more: it means *every* material the
        # index says has a charge density, which is the set this project wants
        # to size before deciding what subset to train on.
        self.elements = sorted(set(elements)) if elements else []
        self.retries = int(retries)
        self.retry_delay = float(retry_delay)

        self._api_key = api_key or load_api_key()

        self.outdir = Path(outdir).resolve()
        self.compress = bool(compress)
        self.workers = int(workers)

        self.band_gap = tuple(band_gap) if band_gap is not None else None
        self.num_sites = tuple(num_sites) if num_sites is not None else None
        self.is_stable = is_stable
        self.exclude_deprecated = bool(exclude_deprecated)
        if isinstance(crystal_system, str):
            crystal_system = [crystal_system]
        self.crystal_system = ({s.capitalize() for s in crystal_system}
                               if crystal_system else None)

        self._rester = None
        self._documents = None
        self._keys = None
        self._estimate = None
        self._searched_all = False
        self._fields = []

    # ------------------------------------------------------------------ #
    # Connection
    # ------------------------------------------------------------------ #
    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *exception):
        self.close()
        return False

    def _connect(self):
        """The live :class:`~mp_api.client.MPRester`, opened on first use."""
        if self._rester is None:
            from mp_api.client import MPRester

            self._rester = MPRester(self._api_key)
        return self._rester

    def close(self):
        """Close the API session, dropping the reference that holds the key."""
        if self._rester is not None:
            self._rester.session.close()
            self._rester = None

    def __repr__(self):
        count = len(self._documents) if self._documents is not None else "?"
        return (f"{type(self).__name__}({self.label}, "
                f"materials={count}, outdir='{self.outdir}')")

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def material_dir(self, identifier):
        """
        The directory one material's files live in: ``<outdir>/mp-124``.

        One directory per material, named by its id, holding a ``CHGCAR`` --
        the shape of a VASP run, which is the shape every other source in
        :mod:`poraque.data.sources` already reads. The flat
        ``chgcar/CHGCAR_mp-124.gz`` it replaces put the id in the filename and
        the whole archive in one directory, which meant nothing could be added
        beside a density later without inventing a second naming convention
        for it.
        """
        return self.outdir / str(identifier)

    @property
    def legacy_chgcar_dir(self):
        """
        The flat directory downloads used to go to, for resuming one.

        Nothing is written here any more. It is read by :meth:`_existing` and
        :meth:`load_manifest` so an archive fetched before the layout changed
        resumes instead of restarting.
        """
        return self.outdir / LEGACY_SUBDIRECTORY

    @property
    def structure_dir(self):
        """Directory the CIFs are written to."""
        return self.outdir / "structures"

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    @property
    def label(self):
        """How this selection is named in reports: a space, or the database."""
        return "-".join(self.elements) if self.elements else "all elements"

    @property
    def chemical_systems(self):
        """
        Every non-empty subsystem of the space — :math:`2^n - 1` of them.

        Querying the full system alone (``"Pt-Pd-Ni"``) returns only the
        ternaries, so each subset has to be listed for the query to span the
        whole space.
        """
        return ["-".join(sorted(subset))
                for size in range(1, len(self.elements) + 1)
                for subset in combinations(self.elements, size)]

    def _query_filters(self):
        """Server-side filters, as keyword arguments for the summary search."""
        filters = {}
        if self.band_gap is not None:
            filters["band_gap"] = self.band_gap
        if self.num_sites is not None:
            filters["num_sites"] = self.num_sites
        if self.is_stable is not None:
            filters["is_stable"] = bool(self.is_stable)
        return filters

    def search(self, refresh=False, fields=None):
        """
        Summary documents for every material in the space, cached.

        Parameters
        ----------
        refresh : bool, optional
            Re-query rather than reuse the cached result.
        fields : sequence of str, optional
            Which document fields to request; :data:`SUMMARY_FIELDS` by
            default. Pass :data:`ESTIMATE_FIELDS` when no geometry is needed —
            ``structure`` is the heaviest field there is, and an estimate never
            looks at one. A cached light result is re-queried automatically if
            something later asks for a field it does not carry, so this is an
            optimisation and never a trap.

        Returns
        -------
        list
            Summary documents, sorted for reproducible ordering.
        """
        fields = list(fields or SUMMARY_FIELDS)
        if self._documents is not None and not refresh:
            # `_fields` records what *this* object asked for. Empty means the
            # documents came from somewhere else -- injected by a caller or a
            # test -- and there is nothing to compare them against, so they are
            # taken as they are rather than second-guessed into a re-query.
            missing = set(fields) - set(self._fields) if self._fields else set()
            if not missing:
                return self._documents
            print(f"  re-querying: the cached documents were fetched without "
                  f"{sorted(missing)}")
            refresh = True

        rester = self._connect()
        filters = self._query_filters()

        if self.elements:
            systems = self.chemical_systems
            print(f"Searching {len(systems)} chemical systems in the "
                  f"{'-'.join(self.elements)} space ...")
            if filters:
                print(f"  filters: {filters}")
            documents = rester.materials.summary.search(
                chemsys=systems, fields=fields, **filters)
        else:
            # The whole database, narrowed server-side to the materials that
            # actually have a density. `has_props` is what makes this a
            # question the API answers rather than one answered by downloading
            # every summary document and discarding 90 % of them.
            print("Searching the whole database for materials with a charge "
                  "density ...")
            if filters:
                print(f"  filters: {filters}")
            documents, _ = with_retries(
                lambda: rester.materials.summary.search(
                    has_props=["charge_density"], fields=fields,
                    **filters),
                attempts=self.retries, base_delay=self.retry_delay,
                label="  ")

        kept = [d for d in documents if self._passes_local_filters(d)]
        if len(kept) != len(documents):
            print(f"  -> {len(documents) - len(kept)} dropped by "
                  f"crystal-system / deprecation filters")
        kept.sort(key=lambda d: (d.nelements, d.chemsys, d.formula_pretty,
                                 str(d.material_id)))
        self._searched_all = not self.elements
        print(f"  -> {len(kept)} materials found")
        self._documents = kept
        self._fields = fields
        # The resolved keys and sizes belong to the previous document set.
        self._keys = self._estimate = None
        return kept

    def _passes_local_filters(self, document):
        """Apply the filters the API query cannot express."""
        if self.exclude_deprecated and getattr(document, "deprecated", False):
            return False
        if self.crystal_system is not None:
            system = getattr(getattr(document, "symmetry", None),
                             "crystal_system", None)
            if str(system or "").capitalize() not in self.crystal_system:
                return False
        return True

    @staticmethod
    def _has_charge_density(document):
        """Whether the index advertises a charge density for this material."""
        properties = document.has_props
        if properties is None:
            return False
        if not isinstance(properties, dict):
            properties = dict(properties)
        return bool(properties.get("charge_density"))

    @property
    def materials(self):
        """Every material in the space (triggers :meth:`search`)."""
        return self.search()

    @property
    def with_charge_density(self):
        """The subset advertising a charge density."""
        return [d for d in self.search() if self._has_charge_density(d)]

    # ------------------------------------------------------------------ #
    # Object resolution and sizing
    # ------------------------------------------------------------------ #
    def _resolve_keys(self, refresh=False, identifiers=None):
        """
        ``{material_id: S3 key}``, resolved in a handful of API calls.

        Mirrors :meth:`MPRester.get_charge_density_from_material_id`, which
        issues two queries *per material*; batching them is what makes sizing a
        few-hundred-material space practical.

        Batched in chunks of :data:`ID_BATCH`, because the API refuses an id
        filter beyond a few thousand outright — *"List of material/molecule IDs
        provided is too long"* — which is what the whole-database query walks
        into on its first call. The suggested alternative, pulling every
        document and filtering locally, is the thing this method exists to
        avoid.

        Parameters
        ----------
        refresh : bool, optional
        identifiers : sequence of str, optional
            Resolve only these, rather than every material advertising a
            density. This is what makes a sampled estimate cheap: twenty
            resolutions instead of tens of thousands.
        """
        if self._keys is not None and not refresh and identifiers is None:
            return self._keys

        from emmet.core.mpid import AlphaID
        from mp_api.client.core.utils import validate_ids

        rester = self._connect()
        if identifiers is None:
            identifiers = [str(d.material_id) for d in self.with_charge_density]
        identifiers = [str(i) for i in identifiers]
        print(f"Resolving charge-density tasks for {len(identifiers)} "
              f"materials ...")

        candidates = {}
        for document in self._batched_search(
                rester.materials.search, "material_ids", identifiers,
                fields=["material_id", "calc_types"]):
            static = [str(task) for task, kind in (document["calc_types"] or {}).items()
                      if str(kind) in STATIC_CALC_TYPES]
            if static:
                candidates[str(document["material_id"])] = static

        tasks = sorted({task for group in candidates.values() for task in group})
        print(f"  -> {len(tasks)} candidate static tasks ...")
        task_docs = self._batched_search(
            rester.materials.tasks.search, "task_ids", tasks,
            fields=["task_id", "last_updated"])
        updated = {str(t["task_id"]): t["last_updated"] for t in task_docs}

        keys = {}
        for identifier, group in candidates.items():
            known = [task for task in group if task in updated]
            if not known:
                continue
            newest = max(known, key=lambda task: updated[task])
            alpha = AlphaID(validate_ids([newest])[0].split("-")[-1], prefix="mp")
            keys[identifier] = f"{CHGCAR_PREFIX}/{alpha.string}.json.gz"

        print(f"  -> resolved {len(keys)} objects")
        if len(identifiers) == len(self.with_charge_density):
            self._keys = keys
        return keys

    def _batched_search(self, search, argument, values, **kwargs):
        """
        Run an id-filtered search in chunks, retried, with progress.

        One call for a few hundred ids; a dozen for the whole database. The
        batch size is a fixed constant rather than adaptive because the limit
        is the server's and does not depend on anything this side knows.
        """
        values = list(values)
        results = []
        batches = range(0, len(values), ID_BATCH)
        for position, start in enumerate(batches, 1):
            chunk = values[start:start + ID_BATCH]
            if len(values) > ID_BATCH:
                print(f"    batch {position}/{len(batches)} "
                      f"({len(chunk)} ids) ...")
            found, _ = with_retries(
                lambda: search(**{argument: chunk}, **kwargs),
                attempts=self.retries, base_delay=self.retry_delay,
                label="    ")
            results.extend(found)
        return results

    def _head_sizes(self, keys):
        """Exact object sizes via unsigned S3 ``HEAD`` — no payload transferred."""
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        from botocore.exceptions import ClientError

        client = boto3.client("s3", config=Config(
            signature_version=UNSIGNED, max_pool_connections=self.workers + 4))

        def head(item):
            identifier, key = item
            try:
                response = client.head_object(Bucket=CHGCAR_BUCKET, Key=key)
                return identifier, response["ContentLength"]
            except ClientError:
                return identifier, -1       # advertised but absent

        print(f"Reading exact sizes ({len(keys)} HEAD requests) ...")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return dict(pool.map(head, keys.items()))

    def estimate(self, refresh=False, sample=None, seed=0):
        """
        Download size and storage projection. Downloads no density.

        Two methods, and the report says which one it used.

        **Exact** (``sample=None``): every object's size is read with an
        unsigned S3 ``HEAD``, which transfers no payload. One request per
        material, so a few-hundred-material space is a few seconds and the
        whole database — some tens of thousands of objects — is not.

        **Sampled** (``sample=N``): ``N`` materials are chosen at random, their
        objects HEADed, and the total taken as the sample mean times the number
        advertised. This is the practical method for sizing the full set, and
        it is what ``--sample`` exists for.

        The sample is drawn with a fixed ``seed`` so the same question twice
        gives the same answer; charge-density sizes are strongly right-tailed,
        so two different draws of twenty can differ by tens of percent and an
        estimate that moved every time it was asked would be useless for
        planning.

        Parameters
        ----------
        refresh : bool, optional
            Re-query rather than reuse the cached result.
        sample : int, optional
            Measure this many objects instead of all of them.
        seed : int, optional
            Sampling seed.

        Returns
        -------
        Estimate
        """
        if self._estimate is not None and not refresh:
            return self._estimate

        documents = self.search()
        advertised = [str(d.material_id) for d in self.with_charge_density]

        method = "exact"
        if sample and sample < len(advertised):
            # Sample the *materials*, then resolve only those. Resolving all of
            # them first and sampling the result would make a 20-object
            # estimate cost the same queries as an exact one over the whole
            # database -- which is the cost --sample exists to avoid, and which
            # the API refuses outright past a few thousand ids.
            method = "sampled"
            picked = random.Random(seed).sample(sorted(advertised), int(sample))
            print(f"Sampling {len(picked)} of {len(advertised)} materials "
                  f"(seed {seed}) and extrapolating ...")
            chosen = self._resolve_keys(refresh=refresh, identifiers=picked)
            resolvable = len(advertised)
        else:
            chosen = self._resolve_keys(refresh=refresh)
            resolvable = len(chosen)

        sizes = self._head_sizes(chosen)

        by_id = {str(d.material_id): d for d in documents}
        rows = []
        for identifier, key in chosen.items():
            document = by_id[identifier]
            rows.append({
                "material_id": identifier,
                "formula_pretty": document.formula_pretty,
                "chemsys": document.chemsys,
                "nsites": document.nsites,
                "volume": round(document.volume, 2),
                "s3_key": key,
                "size_bytes": sizes.get(identifier, -1),
                "size_mb": round(max(sizes.get(identifier, -1), 0) / 1e6, 2),
            })
        rows.sort(key=lambda row: -row["size_bytes"])

        self._estimate = Estimate(
            elements=self.elements,
            n_materials=len(documents),
            # The denominator the mean is multiplied by. In exact mode it is
            # the number of objects actually resolved; in sampled mode the
            # number the index advertises, since only a sample was resolved.
            n_advertised=resolvable,
            n_available=sum(1 for size in sizes.values() if size > 0),
            sizes=sizes,
            rows=rows,
            method=method,
            sampled=len(sizes),
        )
        return self._estimate

    def write_estimate(self, filename="chgcar_estimate.csv"):
        """Write the per-material size table."""
        estimate = self.estimate()
        self.outdir.mkdir(parents=True, exist_ok=True)
        path = self.outdir / filename
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(estimate.rows[0].keys()))
            writer.writeheader()
            writer.writerows(estimate.rows)
        print(f"  -> wrote {path}")
        return path

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def _flatten(self, document):
        """One summary document as a flat record for CSV and JSON."""
        symmetry = document.symmetry
        ordering = document.ordering
        return {
            "material_id": str(document.material_id),
            "formula_pretty": document.formula_pretty,
            "chemsys": document.chemsys,
            "nelements": document.nelements,
            "nsites": document.nsites,
            "spacegroup_symbol": getattr(symmetry, "symbol", None),
            "spacegroup_number": getattr(symmetry, "number", None),
            "crystal_system": str(getattr(symmetry, "crystal_system", "") or "") or None,
            "volume": document.volume,
            "density": document.density,
            "density_atomic": document.density_atomic,
            "energy_per_atom": document.energy_per_atom,
            "formation_energy_per_atom": document.formation_energy_per_atom,
            "energy_above_hull": document.energy_above_hull,
            "is_stable": document.is_stable,
            "band_gap": document.band_gap,
            "is_metal": document.is_metal,
            "is_magnetic": document.is_magnetic,
            "ordering": str(ordering) if ordering is not None else None,
            "total_magnetization": document.total_magnetization,
            "theoretical": document.theoretical,
            "has_charge_density": self._has_charge_density(document),
        }

    def write_summary(self, basename="summary"):
        """Write ``summary.csv`` and ``summary.json`` for the whole space."""
        records = [self._flatten(d) for d in self.search()]
        self.outdir.mkdir(parents=True, exist_ok=True)
        csv_path = self.outdir / f"{basename}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(records)
        with (self.outdir / f"{basename}.json").open("w") as handle:
            json.dump(records, handle, indent=2)
        print(f"  -> wrote {basename}.csv and {basename}.json "
              f"({len(records)} rows)")
        return csv_path

    def write_structures(self):
        """Write one CIF per material. Returns how many were written."""
        self.structure_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for document in self.search(fields=SUMMARY_FIELDS):
            if document.structure is not None:
                document.structure.to(filename=str(
                    self.structure_dir / f"{document.material_id}.cif"))
                written += 1
        print(f"  -> wrote {written} CIF files to {self.structure_dir.name}/")
        return written

    # ------------------------------------------------------------------ #
    # Download
    # ------------------------------------------------------------------ #
    def _chgcar_path(self, identifier, compress=None):
        """Where material ``identifier``'s CHGCAR belongs."""
        compress = self.compress if compress is None else compress
        suffix = ".gz" if compress else ""
        return self.material_dir(identifier) / f"{DENSITY_FILENAME}{suffix}"

    def _store_path(self, identifier):
        """Where material ``identifier``'s HDF5 field store belongs."""
        return self.material_dir(identifier) / STORE_FILENAME

    def _legacy_paths(self, identifier):
        """Every place a previous version of this class may have written it."""
        flat = self.legacy_chgcar_dir
        return (flat / f"CHGCAR_{identifier}.gz",
                flat / f"CHGCAR_{identifier}",
                flat / f"{identifier}.h5")

    def _existing(self, identifier):
        """
        Whichever variant of this material is already on disk, or ``None``.

        Every layout is consulted, not only the current one: gzipped or plain,
        ``CHGCAR`` or HDF5 store, in the material's own directory or in the
        flat ``chgcar/`` an older download used. A resume that only recognised
        the new spelling would re-fetch an entire archive to rename it.
        """
        candidates = (self._chgcar_path(identifier, compress=True),
                      self._chgcar_path(identifier, compress=False),
                      self._store_path(identifier),
                      *self._legacy_paths(identifier))
        for path in candidates:
            if path.exists() and path.stat().st_size > 0:
                return path
        return None

    def _manifest_name(self, path):
        """
        How a file is named in the manifest: relative to ``outdir``.

        The bare filename would do when every density had the material id in
        its name. Now they are all called ``CHGCAR`` and only the directory
        tells them apart, so the manifest records ``mp-124/CHGCAR.gz`` --
        which is also what makes a row locate its file.
        """
        try:
            return str(path.relative_to(self.outdir))
        except ValueError:                                      # pragma: no cover
            return path.name

    def download(self, limit=None, max_sites=None, max_size_mb=None,
                 compress=None, skip_existing=True, hdf5=False,
                 compression="gzip", compression_level=4):
        """
        Download the charge densities. Resumable; one failure never aborts the batch.

        Each density is written into **its own directory**, named by the
        material id::

            data/MP/mp-124/CHGCAR.gz
            data/MP/mp-81/CHGCAR.gz
            data/MP/mp-126/fields.h5        # --hdf5

        which is the shape of a VASP run and the shape
        :class:`~poraque.data.sources.BulkDensitySource` reads. The flat
        ``chgcar/CHGCAR_mp-124.gz`` layout this replaces is still read, and
        still resumed from, so an archive already on disk is not re-fetched.

        **Resume** is by manifest first and by file second. ``manifest.json``
        at the top of ``outdir`` records every entry already accounted for, so
        a run that is interrupted after 900 of 3000 objects re-reads that file
        and starts at 901 without stat-ing, re-resolving or re-heading the
        first 900. A file on disk with no manifest entry — a manifest lost, or
        a directory assembled by hand — is still honoured, so the two
        mechanisms overlap rather than depending on each other.

        **Rate limits and dropped connections** are retried with exponential
        backoff and jitter (:func:`with_retries`), controlled by the
        constructor's ``retries``/``retry_delay``. A permanent failure — an
        object that is not there — is recorded and skipped on the first
        attempt, because retrying it would only make the run slower without
        making it more complete.

        Parameters
        ----------
        limit : int, optional
            Download at most this many, smallest first.
        max_sites : int, optional
            Skip cells with more sites than this.
        max_size_mb : float, optional
            Skip objects larger than this, using the exact S3 sizes.
        compress : bool, optional
            Store gzipped; defaults to the instance's ``compress``. Ignored
            when ``hdf5`` is set, which has its own compression.
        skip_existing : bool, optional
            Leave already-downloaded entries alone, so an interrupted run
            resumes rather than restarting.
        hdf5 : bool, optional
            Convert each density to a Poraquê HDF5 field store
            (``mp-124/fields.h5``) instead of writing a ``CHGCAR``. The values
            are identical; what is gained is that the file is read without
            parsing millions of numbers, and what is lost is the PAW
            augmentation block, which no HDF5 layout here carries — so a store
            cannot seed a VASP ``ICHARG=1`` restart and a ``CHGCAR`` can.
        compression : str, optional
            HDF5 codec: ``"gzip"``, ``"lzf"`` or ``None``.
        compression_level : int, optional
            Gzip level.

        Returns
        -------
        list of dict
            The manifest, written to ``manifest.json`` **and**
            ``manifest.csv`` at the top of ``outdir`` after every file, so an
            interrupted run leaves a truthful record. Per entry: material id,
            formula, site count, grid dimensions, file (relative to ``outdir``,
            since every density is now called ``CHGCAR``), size, status,
            attempts, and the retrieval provenance.
        """
        compress = self.compress if compress is None else compress
        rester = self._connect()
        estimate = self.estimate()
        self.outdir.mkdir(parents=True, exist_ok=True)

        by_id = {str(d.material_id): d for d in self.search()}
        targets = [by_id[i] for i, size in estimate.sizes.items()
                   if size > 0 and i in by_id]
        targets.sort(key=lambda d: estimate.sizes[str(d.material_id)])

        if max_sites is not None:
            targets = [d for d in targets if d.nsites <= max_sites]
        if max_size_mb is not None:
            targets = [d for d in targets
                       if estimate.sizes[str(d.material_id)] <= max_size_mb * 1e6]
        if limit is not None:
            targets = targets[:limit]

        planned = sum(estimate.sizes[str(d.material_id)] for d in targets)
        on_disk = planned * DISK_PER_DOWNLOAD * (1 if compress else UNZIP_EXPANSION)
        store = "HDF5 stores" if hdf5 else "CHGCARs"
        print(f"Downloading {len(targets)} {store} "
              f"(~{_format_bytes(planned)} transfer, ~{_format_bytes(on_disk)} "
              f"on disk {'gzipped' if compress else 'unzipped'}) ...")

        done = self.load_manifest() if skip_existing else {}
        if done:
            print(f"  resuming: {len(done)} entries already in "
                  f"{MANIFEST_JSON}")
        provenance = retrieval_provenance()
        started = time.time()
        remaining = [d for d in targets
                     if not (skip_existing
                             and _is_settled(done.get(str(d.material_id))))]
        # Carry forward only the entries this run is *not* going to touch. A
        # row for a material about to be re-fetched -- a previous `failed` --
        # would otherwise survive beside the new one, and the manifest would
        # report the same material twice with two different verdicts.
        retrying = {str(d.material_id) for d in remaining}
        manifest = [row for identifier, row in done.items()
                    if identifier not in retrying]
        if len(remaining) != len(targets):
            print(f"  {len(targets) - len(remaining)} already done, "
                  f"{len(remaining)} to fetch")

        for position, document in enumerate(remaining, 1):
            identifier = str(document.material_id)
            label = (f"[{position}/{len(remaining)}] {identifier} "
                     f"{document.formula_pretty} ({document.nsites} sites)")

            found = self._existing(identifier) if skip_existing else None
            if found is not None:
                print(f"{label}: already present, skipping")
                manifest.append(self._manifest_row(identifier, document, found,
                                                   "cached",
                                                   provenance=provenance))
                self._write_manifest(manifest)
                continue

            path = (self._store_path(identifier) if hdf5
                    else self._chgcar_path(identifier, compress=compress))
            # The material's directory is created only once the download is
            # about to be attempted, so a filtered-out or failed entry does not
            # leave an empty `mp-124/` behind for a later scan to puzzle over.
            path.parent.mkdir(parents=True, exist_ok=True)
            clock = time.time()
            try:
                chgcar, attempts = with_retries(
                    lambda: rester.get_charge_density_from_material_id(identifier),
                    attempts=self.retries, base_delay=self.retry_delay,
                    label=f"{label}: ")
                if chgcar is None:
                    raise ValueError("the API returned no charge density")
                grid = self._store_density(chgcar, path, hdf5, compression,
                                           compression_level)
            except Exception as error:                          # noqa: BLE001
                # One material's failure must not cost the other 200. The
                # partial file goes, the reason is recorded, the batch carries
                # on -- and the manifest says `failed` rather than staying
                # silent about a hole in the dataset.
                print(f"{label}: FAILED - {type(error).__name__}: {error}")
                path.unlink(missing_ok=True)
                # An empty `mp-124/` left by a failure would be discovered as a
                # material with no density; `rmdir` refuses a non-empty one, so
                # a directory holding anything else survives untouched.
                with contextlib.suppress(OSError):
                    path.parent.rmdir()
                manifest.append(self._manifest_row(
                    identifier, document, None, "failed",
                    f"{type(error).__name__}: {error}", provenance=provenance))
                self._write_manifest(manifest)
                continue

            print(f"{label}: {'x'.join(str(n) for n in grid)}, "
                  f"{path.stat().st_size / 1e6:.1f} MB "
                  f"in {time.time() - clock:.1f}s"
                  f"{f' ({attempts} attempts)' if attempts > 1 else ''}")
            manifest.append(self._manifest_row(identifier, document, path,
                                               "downloaded", grid=grid,
                                               attempts=attempts,
                                               provenance=provenance))
            self._write_manifest(manifest)

        succeeded = sum(1 for row in manifest if _is_settled(row))
        # `.get`, because a row read back from a manifest written by an older
        # version -- or a `failed` row that never had a file -- carries only
        # what it carries. Losing the summary line to a KeyError at the end of
        # a three-hour run is not a trade worth making for strictness.
        total = sum(float(row.get("size_mb") or 0.0) for row in manifest)
        print(f"  -> {succeeded}/{len(targets)} on disk, {total:.0f} MB, "
              f"elapsed {(time.time() - started) / 60:.1f} min")
        failed = [row for row in manifest if row["status"] == "failed"]
        if failed:
            print(f"  -> {len(failed)} failed; re-running this command retries "
                  f"only those.")
        return manifest

    def _store_density(self, chgcar, path, hdf5, compression, level):
        """
        Write one downloaded density, and report its grid.

        Returns
        -------
        tuple of int
            ``(NGXF, NGYF, NGZF)``, which is the field the manifest records and
            the only thing here that has to be true whichever format was used.
        """
        if not hdf5:
            chgcar.write_file(str(path))
            return tuple(int(n) for n in chgcar.dim)

        # pymatgen holds rho*Omega on the grid, which is exactly the file
        # convention the store keeps, so this is a re-encoding and not a
        # conversion: no value changes, only how it is written down.
        from ..fields import ChargeDensity, FieldGrid
        from ..fields.hdf5 import write_fields

        values = np.asarray(chgcar.data["total"], dtype=float)
        structure = poscar_from_pymatgen(chgcar.structure)
        grid = FieldGrid(values.shape, structure.cell)
        field = ChargeDensity(values / grid.volume, grid, structure,
                              dtype="float64")
        write_fields(str(path), {"CHGCAR": field}, compression=compression,
                     level=level)
        return tuple(int(n) for n in values.shape)

    def load_manifest(self):
        """
        The JSON manifest of a previous run, keyed by material id.

        Returns
        -------
        dict
            Empty when there is none, or when it cannot be parsed — a corrupt
            manifest costs a re-download, and refusing to start would cost the
            whole archive.
        """
        path = self.outdir / MANIFEST_JSON
        if not path.exists():
            # An archive fetched before the layout changed kept its manifest
            # beside the densities. Reading it is what lets that download
            # resume rather than restart.
            path = self.legacy_chgcar_dir / MANIFEST_JSON
        if not path.exists():
            return {}
        try:
            with path.open() as handle:
                rows = json.load(handle)
        except (OSError, ValueError):
            print(f"  {path} is unreadable; starting without a resume point")
            return {}
        if isinstance(rows, dict):
            rows = rows.get("entries", [])
        return {str(row.get("material_id")): row for row in rows
                if row.get("material_id")}

    def _manifest_row(self, identifier, document, path, status, error="",
                      grid=None, attempts=1, provenance=None):
        """One manifest record: what was fetched, how big, and from where."""
        namer = self._manifest_name
        if grid is None and path is not None:
            grid = _peek_grid(path)
        return {
            "material_id": identifier,
            "formula_pretty": document.formula_pretty,
            "nsites": document.nsites,
            "file": namer(path) if path else "",
            "size_mb": round(path.stat().st_size / 1e6, 2) if path else 0.0,
            "compressed": bool(path and path.suffix in (".gz", ".h5")),
            "status": status,
            "error": error,
            "grid": list(grid) if grid else None,
            "attempts": int(attempts),
            **(provenance or {}),
        }

    def _write_manifest(self, manifest):
        """
        Rewrite both manifests from scratch, after every file.

        Rewritten rather than appended so an interrupted run never leaves a
        half-written final row, and written after *every* file rather than at
        the end so an interrupted run leaves a manifest at all — which is the
        thing the next run resumes from.
        """
        self.outdir.mkdir(parents=True, exist_ok=True)
        with (self.outdir / MANIFEST_JSON).open("w") as handle:
            json.dump(manifest, handle, indent=2)
        with (self.outdir / MANIFEST_FILENAME).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(manifest)

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    def _downloaded(self):
        """
        Every downloaded ``CHGCAR`` on disk, in either layout, sorted.

        One glob covers both: ``mp-124/CHGCAR.gz`` is where a download goes
        now and ``chgcar/CHGCAR_mp-124.gz`` is where it used to, and both are
        one level below ``outdir`` with a name beginning ``CHGCAR``. A machine
        part way through the change has some of each, and a storage command
        that saw only one of them would silently do half the job.
        """
        return sorted(path for path in
                      self.outdir.glob(f"*/{DENSITY_FILENAME}*")
                      if path.is_file())

    def decompress(self, keep_original=False, dry_run=False):
        """
        Expand the downloaded CHGCARs in place (``CHGCAR.gz`` -> ``CHGCAR``).

        .. warning::
           Poraquê reads gzipped volumetric files directly, so this is **not**
           a prerequisite for training — it costs roughly a threefold storage
           multiplier and buys nothing the pipeline uses. It is here for tools
           that cannot read ``.gz``.

        Parameters
        ----------
        keep_original : bool, optional
            Keep the ``.gz`` beside the expanded file. Doubles the peak
            requirement.
        dry_run : bool, optional
            Report the projected size and write nothing, so free space can be
            checked first.

        Returns
        -------
        dict
            ``{"n": files, "before": bytes, "after": bytes}``.
        """
        files = [path for path in self._downloaded()
                 if path.suffix == ".gz"]
        if not files:
            print(f"No gzipped CHGCARs under {self.outdir}")
            return {"n": 0, "before": 0, "after": 0}

        before = sum(path.stat().st_size for path in files)
        free = shutil.disk_usage(self.outdir).free
        projected = before * UNZIP_EXPANSION
        needed = projected + before if keep_original else projected

        print(f"{len(files)} gzipped CHGCARs, {_format_bytes(before)}")
        print(f"  projected unzipped : {_format_bytes(projected)} "
              f"(x{UNZIP_EXPANSION})")
        print(f"  free on volume     : {_format_bytes(free)}")

        if dry_run:
            print("  (dry run - nothing written)")
            return {"n": len(files), "before": before, "after": projected}

        extra = needed - (0 if keep_original else before)
        if extra > free * 0.95:
            hint = ("drop keep_original so the .gz files are removed as you go, "
                    if keep_original else "")
            raise OSError(
                f"Not enough free space: expanding needs ~{_format_bytes(extra)} "
                f"more, only {_format_bytes(free)} free. Free up space, {hint}"
                f"expand a subset, or leave the files gzipped - Poraquê reads "
                f"them as they are.")

        after = 0
        for position, archive in enumerate(files, 1):
            expanded = archive.with_suffix("")
            with gzip.open(archive, "rb") as source, expanded.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1 << 22)
            after += expanded.stat().st_size
            if not keep_original:
                archive.unlink()
            if position % 25 == 0 or position == len(files):
                print(f"  [{position}/{len(files)}] {_format_bytes(after)} written")
        print(f"  -> expanded {len(files)} files: {_format_bytes(before)} -> "
              f"{_format_bytes(after)} (x{after / before:.2f})")
        return {"n": len(files), "before": before, "after": after}

    def recompress(self, level=6):
        """Re-gzip expanded CHGCARs, reclaiming the space :meth:`decompress` cost."""
        files = [path for path in self._downloaded()
                 if path.suffix not in (".gz", ".h5")]
        if not files:
            print(f"No expanded CHGCARs under {self.outdir}")
            return {"n": 0, "before": 0, "after": 0}

        before = sum(path.stat().st_size for path in files)
        after = 0
        for position, plain in enumerate(files, 1):
            archive = plain.with_suffix(plain.suffix + ".gz")
            with plain.open("rb") as source, \
                    gzip.open(archive, "wb", compresslevel=level) as sink:
                shutil.copyfileobj(source, sink, length=1 << 22)
            after += archive.stat().st_size
            plain.unlink()
            if position % 25 == 0 or position == len(files):
                print(f"  [{position}/{len(files)}] {_format_bytes(after)} written")
        print(f"  -> recompressed {len(files)} files: {_format_bytes(before)} -> "
              f"{_format_bytes(after)}")
        return {"n": len(files), "before": before, "after": after}

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def dry_run(self, sample=None, seed=0):
        """
        Report what a fetch would transfer. **Writes nothing.**

        A true dry run: no ``summary.csv``, no ``summary.json``, no
        ``chgcar_estimate.csv``, no directories created. Sizing a space is a
        question, and asking it should not leave anything behind — least of
        all in whatever directory the command happened to be run from.

        Returns
        -------
        Estimate
        """
        # No geometry is written by a dry run, so none is fetched.
        self.search(fields=ESTIMATE_FIELDS)
        estimate = self.estimate(sample=sample, seed=seed)
        print("\n" + str(estimate) + "\n")
        count = (estimate.n_advertised if estimate.is_sampled
                 else estimate.n_available)
        print(f"  {count} file(s) to download, "
              f"{'~' if estimate.is_sampled else ''}"
              f"{_format_bytes(estimate.download_bytes)} total "
              f"({_format_bytes(estimate.gz_disk_bytes)} on disk gzipped).")
        print("  Dry run: nothing downloaded, nothing written.")
        return estimate

    def run(self, estimate_only=False, skip_chgcar=False, decompress=False,
            sample=None, seed=0, **download_kwargs):
        """
        The whole fetch: summary -> estimate -> structures -> CHGCARs.

        Parameters
        ----------
        estimate_only : bool, optional
            Report the size and stop, writing nothing at all --- see
            :meth:`dry_run`.
        skip_chgcar : bool, optional
            Write the metadata and structures only.
        decompress : bool, optional
            Expand the CHGCARs afterwards. Rarely wanted; see
            :meth:`decompress`.
        **download_kwargs
            Passed to :meth:`download`.

        Returns
        -------
        Estimate
        """
        if estimate_only:
            return self.dry_run(sample=sample, seed=seed)

        self.search()
        self.write_summary()
        estimate = self.estimate()
        self.write_estimate()
        print("\n" + str(estimate) + "\n")

        self.write_structures()
        if not skip_chgcar:
            self.download(**download_kwargs)
            if decompress:
                print()
                self.decompress()
        return estimate


def poscar_from_pymatgen(structure):
    """
    A pymatgen :class:`~pymatgen.core.Structure` as a Poraquê
    :class:`~poraque.fields.vasp.poscar.Poscar`.

    ``Poscar.from_structure`` wraps Poraquê's *own* structure class and reads
    ``.cell``/``.symbols``/``.counts`` off it; a pymatgen structure has a
    ``lattice`` and a list of sites instead, and nothing in between. This is
    the adapter, and it is here rather than in ``fields/`` because the
    Materials Project client is the only thing in the tree that hands out
    pymatgen objects.

    Sites are **grouped by species**, which VASP's format requires and which a
    pymatgen structure does not guarantee. Regrouping is safe here and would
    not be everywhere: the density grid is indexed by position in the cell and
    not by site order, so reordering the site list cannot move a value. The one
    thing that *is* per-site — the PAW augmentation block — is not carried into
    a store at all.

    Parameters
    ----------
    structure : pymatgen.core.Structure or Poscar
        A :class:`Poscar` is returned unchanged.

    Returns
    -------
    Poscar
    """
    from ..fields.vasp.poscar import Poscar

    # Already one of ours: an adapter's cheapest correct answer.
    if isinstance(structure, Poscar):
        return structure

    order, positions = [], []
    for site in structure:
        symbol = str(getattr(site, "specie", getattr(site, "species", ""))
                     ).split(":")[0].strip()
        symbol = getattr(getattr(site, "specie", None), "symbol", symbol)
        if not order or order[-1][0] != symbol:
            order.append([symbol, 0])
        order[-1][1] += 1
        positions.append(site.frac_coords)

    # Two blocks of the same species (A B A) would be written as three species
    # blocks, which is legal VASP but reads oddly; merge them.
    merged = {}
    for symbol, count in order:
        merged[symbol] = merged.get(symbol, 0) + count
    if len(merged) != len(order):
        indices = sorted(range(len(structure)),
                         key=lambda i: list(merged).index(
                             structure[i].specie.symbol))
        positions = [structure[i].frac_coords for i in indices]

    return Poscar(
        np.asarray(structure.lattice.matrix, dtype=float),
        list(merged), list(merged.values()),
        np.asarray(positions, dtype=float),
        comment=structure.composition.reduced_formula,
    )


def _is_settled(row):
    """Whether a manifest row means "do not fetch this again"."""
    return bool(row) and row.get("status") in ("downloaded", "cached")


def _peek_grid(path):
    """The grid dimensions of a stored density, from its header alone."""
    try:
        from ..ml.data import _peek_shape

        return tuple(_peek_shape(str(path)))
    except Exception:                                           # noqa: BLE001
        # A manifest without a grid is still a useful manifest; a download
        # that failed because reading the header back raised is not.
        return None


def load_api_key(explicit=None):
    """
    Resolve the Materials Project API key.

    Precedence: ``explicit``, then ``$MP_API_KEY``, then a ``.env`` in the
    working directory, then ``~/.env``. Keeping it in a ``.env`` is what stops
    it from reaching a committed config or a shell history.

    Parameters
    ----------
    explicit : str, optional
        A key supplied directly, which always wins.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        If no key is found anywhere, naming every place that was searched.
    """
    if explicit:
        return explicit

    key = os.getenv("MP_API_KEY")
    if not key:
        from dotenv import load_dotenv

        load_dotenv()
        load_dotenv(Path.home() / ".env")
        key = os.getenv("MP_API_KEY")

    if not key:
        raise ValueError(
            "No Materials Project API key. Pass api_key=..., or set MP_API_KEY "
            "in the environment, in a .env file beside the working directory, "
            "or in ~/.env. Get one at https://materialsproject.org/api."
        )
    return key


def build_parser():
    """Command-line interface for ``poraque-mp``."""
    parser = argparse.ArgumentParser(
        description="Fetch Materials Project charge densities for training.")
    parser.add_argument("--elements", nargs="+", default=None,
                        metavar="EL",
                        help="elements spanning the chemical space; omit (or "
                             "pass --all) to take every material in the "
                             "database that has a charge density")
    parser.add_argument("--all", action="store_true",
                        help="every material with a charge density, whatever "
                             "its composition. The search filters below still "
                             "apply, so this is also how a subset of the whole "
                             "database is selected")
    # Two spellings of one argument. The current directory is the default
    # because a command that writes hundreds of megabytes should put them
    # where it was run, not somewhere it picked.
    parser.add_argument("--outdir", "--output", dest="outdir", default=".",
                        metavar="DIR",
                        help="destination for summary.*, structures/, "
                             "manifest.* and one mp-<id>/ directory per "
                             "material (default: the current directory)")
    parser.add_argument("--estimate", action="store_true",
                        help="dry run: report the size on the console and "
                             "write nothing at all")
    parser.add_argument("--sample", type=int, metavar="N",
                        help="with --estimate, measure N randomly chosen "
                             "objects and extrapolate instead of measuring "
                             "every one. The practical way to size the whole "
                             "database; the report says which method it used")
    parser.add_argument("--seed", type=int, default=0,
                        help="sampling seed, so the same question twice gives "
                             "the same answer (default: 0)")
    parser.add_argument("--skip-chgcar", action="store_true",
                        help="write metadata and structures only")

    group = parser.add_argument_group("search filters")
    group.add_argument("--band-gap", nargs=2, type=float, metavar=("MIN", "MAX"),
                       help="band-gap window in eV; '0 0' selects metals")
    group.add_argument("--crystal-system", nargs="+", metavar="SYSTEM",
                       help="Cubic, Hexagonal, Tetragonal, ...")
    group.add_argument("--stable-only", action="store_true",
                       help="keep only materials on the convex hull")
    group.add_argument("--num-sites", nargs=2, type=int, metavar=("MIN", "MAX"),
                       help="sites-per-cell window, applied server-side")

    group = parser.add_argument_group("download limits")
    group.add_argument("--limit", type=int, help="download at most N, smallest first")
    group.add_argument("--max-sites", type=int, help="skip cells larger than this")
    group.add_argument("--max-size-mb", type=float,
                       help="skip objects larger than this")
    group.add_argument("--workers", type=int, default=16,
                       help="parallel S3 HEAD requests while sizing")
    group.add_argument("--retries", type=int, default=4,
                       help="attempts per object before giving up on it; a "
                            "rate limit or a dropped connection is retried "
                            "with exponential backoff and jitter (default: 4)")
    group.add_argument("--retry-delay", type=float, default=2.0,
                       help="seconds before the first retry, doubling "
                            "thereafter (default: 2)")
    group.add_argument("--restart", action="store_true",
                       help="ignore the manifest and any files already on "
                            "disk, and fetch everything again")

    group = parser.add_argument_group("storage")
    group.add_argument("--hdf5", action="store_true",
                       help="store each density as a chunked HDF5 field store "
                            "(mp-124.h5) instead of a CHGCAR. Same values, "
                            "read without parsing text; no PAW augmentation "
                            "block, so a store cannot seed a VASP ICHARG=1 "
                            "restart")
    group.add_argument("--compression", choices=("none", "gzip", "lzf"),
                       default="gzip",
                       help="HDF5 dataset filter, with --hdf5 (default: gzip). "
                            "Both codecs ship with h5py, so the result opens "
                            "anywhere h5py does")
    group.add_argument("--compression-level", type=int, default=4,
                       metavar="0-9",
                       help="gzip level; ignored by lzf (default: 4)")
    group.add_argument("--unzip", action="store_true",
                       help="store CHGCARs expanded (~3x space; Poraque reads "
                            "them gzipped, so this is rarely wanted)")
    group.add_argument("--decompress", action="store_true",
                       help="expand the CHGCARs already on disk, then exit")
    group.add_argument("--recompress", action="store_true",
                       help="re-gzip expanded CHGCARs on disk, then exit")
    group.add_argument("--dry-run", action="store_true",
                       help="with --decompress, only report the projected size")
    return parser


def main(argv=None):
    """Console entry point for ``poraque-mp``."""
    banner()
    args = build_parser().parse_args(argv)

    elements = None if args.all else args.elements
    if not elements and not args.all:
        # Neither an element list nor --all. Historically --elements was
        # required, so silently reading the omission as "the whole database"
        # would turn a forgotten flag into a multi-terabyte question.
        print("ERROR: pass --elements EL [EL ...] for a chemical space, or "
              "--all for every material with a charge density.",
              file=sys.stderr)
        return 1

    try:
        fetcher = MPDataFetcher(
            elements, outdir=args.outdir, compress=not args.unzip,
            workers=args.workers, band_gap=args.band_gap,
            crystal_system=args.crystal_system,
            is_stable=True if args.stable_only else None,
            num_sites=args.num_sites, retries=args.retries,
            retry_delay=args.retry_delay,
        )
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    with fetcher:
        if args.decompress:
            fetcher.decompress(dry_run=args.dry_run)
            return 0
        if args.recompress:
            fetcher.recompress()
            return 0
        fetcher.run(estimate_only=args.estimate, skip_chgcar=args.skip_chgcar,
                    sample=args.sample, seed=args.seed,
                    limit=args.limit, max_sites=args.max_sites,
                    max_size_mb=args.max_size_mb,
                    skip_existing=not args.restart, hdf5=args.hdf5,
                    compression=(None if args.compression == "none"
                                 else args.compression),
                    compression_level=args.compression_level)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
