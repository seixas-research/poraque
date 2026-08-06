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
    >>> fetcher = MPDataFetcher(["Ag", "Au", "Pt"], outdir="data/MP")   # doctest: +SKIP
    >>> print(fetcher.estimate())        # exact size, nothing transferred  # doctest: +SKIP
    >>> fetcher.run(max_size_mb=20)      # summary, structures, CHGCARs     # doctest: +SKIP

A chemical space is *every* material whose composition uses only the given
elements, across all stoichiometries and structures — the Ag-Au-Pt space
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

Both ratios were measured against the fully downloaded Au-Ag-Cu-Pd-Pt set
(85 objects: 1044.6 MB downloaded, 917.7 MB gzipped, 2.74 GB unzipped) rather
than guessed.

**There is no reason to unzip.** Poraquê reads gzipped volumetric files in
place (:mod:`poraque.fields.io.compressed`), so :meth:`decompress` exists only
for interoperating with tools that cannot, and costs roughly a threefold
storage multiplier when used.

Command line
------------
::

    poraque-mp --elements Ag Au Pt --estimate          # dry run, writes nothing
    poraque-mp --elements Ag Au Pt --outdir data/MP --max-size-mb 20
    poraque-mp --elements Si O --band-gap 0.5 6.0 --crystal-system Cubic

``--estimate`` is a **pure dry run**: it reports to the console and leaves no
file behind. Everything else writes into ``--outdir`` (``--output`` is accepted
too), which defaults to the current directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

# --- measured constants -----------------------------------------------------
# Calibrated on the Au-Ag-Cu-Pd-Pt set (85 charge densities); see the module
# docstring. Ratios, not fits: they convert one measured total into another.
DISK_PER_DOWNLOAD = 0.879  # gzipped-on-disk / bytes-downloaded
UNZIP_EXPANSION = 2.99     # unzipped / gzipped-on-disk

CHGCAR_BUCKET = "materialsproject-parsed"
CHGCAR_PREFIX = "chgcars"

#: Calculation types whose charge density MP publishes. A relaxation's density
#: is the last ionic step's and is not what the reported energy belongs to.
STATIC_CALC_TYPES = {"GGA Static", "GGA+U Static"}

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
                    "size_mb", "compressed", "status", "error"]

#: Written beside the CHGCARs; :mod:`poraque.data.mp_dataset` reads it back.
MANIFEST_FILENAME = "manifest.csv"


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
    """

    elements: list
    n_materials: int
    n_advertised: int
    n_available: int
    sizes: dict = field(repr=False, default_factory=dict)
    rows: list = field(repr=False, default_factory=list)

    @property
    def download_bytes(self):
        """Bytes that would cross the network."""
        return sum(size for size in self.sizes.values() if size > 0)

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
        """``(count, bytes)`` of the objects at or below a size cap."""
        kept = [s for s in self.sizes.values() if 0 < s <= max_size_mb * 1e6]
        return len(kept), sum(kept)

    def __str__(self):
        good = sorted(size for size in self.sizes.values() if size > 0)
        if not good:
            return "Estimate: no charge densities available"

        def percentile(fraction):
            return good[min(int(len(good) * fraction), len(good) - 1)]

        rule = "=" * 66
        lines = [
            rule,
            f"  CHGCAR ESTIMATE - {'-'.join(self.elements)} space",
            rule,
            f"  Materials in space           : {self.n_materials}",
            f"  Advertising a charge density : {self.n_advertised}",
            f"  FILES TO DOWNLOAD            : {self.n_available}",
            f"  TOTAL DOWNLOAD               : {_format_bytes(self.download_bytes)}",
            "-" * 66,
            f"  Storage if kept gzipped      : {_format_bytes(self.gz_disk_bytes)}",
            f"  Storage if UNZIPPED (x{UNZIP_EXPANSION})   : "
            f"{_format_bytes(self.unzipped_bytes)}",
            "-" * 66,
            f"  smallest / median / largest  : {_format_bytes(good[0])} / "
            f"{_format_bytes(percentile(0.5))} / {_format_bytes(good[-1])}",
            f"  90th percentile              : {_format_bytes(percentile(0.9))}",
        ]
        if self.n_missing:
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
        Elements spanning the space, e.g. ``["Ag", "Au", "Pt"]``. A string is
        accepted in either ``"Ag-Au-Pt"`` or ``"Ag Au Pt"`` form.
    api_key : str, optional
        Materials Project API key. Falls back to ``$MP_API_KEY``, then a local
        ``.env``, then ``~/.env``.
    outdir : str or pathlib.Path, optional
        Where ``summary.*``, ``structures/`` and ``chgcar/`` are written.
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
    >>> with MPDataFetcher(["Ag", "Au", "Pt"], outdir="data/MP") as mp:  # doctest: +SKIP
    ...     print(mp.estimate())
    ...     mp.run(max_sites=8, max_size_mb=20)
    """

    def __init__(self, elements, api_key=None, outdir=".", compress=True,
                 workers=16, band_gap=None, crystal_system=None,
                 is_stable=None, num_sites=None, exclude_deprecated=True):
        if isinstance(elements, str):
            elements = elements.replace("-", " ").split()
        if not elements:
            raise ValueError("at least one element is required.")
        self.elements = sorted(set(elements))

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
        return (f"{type(self).__name__}({'-'.join(self.elements)}, "
                f"materials={count}, outdir='{self.outdir}')")

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    @property
    def chgcar_dir(self):
        """Directory the CHGCARs are written to."""
        return self.outdir / "chgcar"

    @property
    def structure_dir(self):
        """Directory the CIFs are written to."""
        return self.outdir / "structures"

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    @property
    def chemical_systems(self):
        """
        Every non-empty subsystem of the space — :math:`2^n - 1` of them.

        Querying the full system alone (``"Ag-Au-Pt"``) returns only the
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

    def search(self, refresh=False):
        """
        Summary documents for every material in the space, cached.

        Parameters
        ----------
        refresh : bool, optional
            Re-query rather than reuse the cached result.

        Returns
        -------
        list
            Summary documents, sorted for reproducible ordering.
        """
        if self._documents is not None and not refresh:
            return self._documents

        rester = self._connect()
        systems = self.chemical_systems
        filters = self._query_filters()
        print(f"Searching {len(systems)} chemical systems in the "
              f"{'-'.join(self.elements)} space ...")
        if filters:
            print(f"  filters: {filters}")

        documents = rester.materials.summary.search(
            chemsys=systems, fields=SUMMARY_FIELDS, **filters)

        kept = [d for d in documents if self._passes_local_filters(d)]
        if len(kept) != len(documents):
            print(f"  -> {len(documents) - len(kept)} dropped by "
                  f"crystal-system / deprecation filters")
        kept.sort(key=lambda d: (d.nelements, d.chemsys, d.formula_pretty,
                                 str(d.material_id)))
        print(f"  -> {len(kept)} materials found")
        self._documents = kept
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
    def _resolve_keys(self, refresh=False):
        """
        ``{material_id: S3 key}``, resolved in two API calls for the whole set.

        Mirrors :meth:`MPRester.get_charge_density_from_material_id`, which
        issues two queries *per material*; batching them is what makes sizing a
        few-hundred-material space practical.
        """
        if self._keys is not None and not refresh:
            return self._keys

        from emmet.core.mpid import AlphaID
        from mp_api.client.core.utils import validate_ids

        rester = self._connect()
        identifiers = [str(d.material_id) for d in self.with_charge_density]
        print(f"Resolving charge-density tasks for {len(identifiers)} materials ...")

        material_docs = rester.materials.search(
            material_ids=identifiers, fields=["material_id", "calc_types"])
        candidates = {}
        for document in material_docs:
            static = [str(task) for task, kind in (document["calc_types"] or {}).items()
                      if str(kind) in STATIC_CALC_TYPES]
            if static:
                candidates[str(document["material_id"])] = static

        tasks = sorted({task for group in candidates.values() for task in group})
        print(f"  -> {len(tasks)} candidate static tasks ...")
        task_docs = rester.materials.tasks.search(
            task_ids=tasks, fields=["task_id", "last_updated"])
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
        self._keys = keys
        return keys

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

    def estimate(self, refresh=False):
        """
        Exact download size and storage projection. Downloads nothing.

        Returns
        -------
        Estimate
        """
        if self._estimate is not None and not refresh:
            return self._estimate

        documents = self.search()
        keys = self._resolve_keys(refresh=refresh)
        sizes = self._head_sizes(keys)

        by_id = {str(d.material_id): d for d in documents}
        rows = []
        for identifier, key in keys.items():
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
            n_advertised=len(self.with_charge_density),
            n_available=sum(1 for size in sizes.values() if size > 0),
            sizes=sizes,
            rows=rows,
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
        for document in self.search():
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
        return self.chgcar_dir / f"CHGCAR_{identifier}{'.gz' if compress else ''}"

    def _existing(self, identifier):
        """Whichever variant — gzipped or plain — is already on disk."""
        for compressed in (True, False):
            path = self._chgcar_path(identifier, compress=compressed)
            if path.exists() and path.stat().st_size > 0:
                return path
        return None

    def download(self, limit=None, max_sites=None, max_size_mb=None,
                 compress=None, skip_existing=True):
        """
        Download the charge densities. Resumable; one failure never aborts the batch.

        Parameters
        ----------
        limit : int, optional
            Download at most this many, smallest first.
        max_sites : int, optional
            Skip cells with more sites than this.
        max_size_mb : float, optional
            Skip objects larger than this, using the exact S3 sizes.
        compress : bool, optional
            Store gzipped; defaults to the instance's ``compress``.
        skip_existing : bool, optional
            Leave already-downloaded files alone, so an interrupted run
            resumes rather than restarting.

        Returns
        -------
        list of dict
            The manifest, also written to ``chgcar/manifest.csv`` after every
            file so an interrupted run leaves a truthful record.
        """
        compress = self.compress if compress is None else compress
        rester = self._connect()
        estimate = self.estimate()
        self.chgcar_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"Downloading {len(targets)} CHGCARs "
              f"(~{_format_bytes(planned)} transfer, ~{_format_bytes(on_disk)} "
              f"on disk {'gzipped' if compress else 'unzipped'}) ...")

        manifest, started = [], time.time()
        for position, document in enumerate(targets, 1):
            identifier = str(document.material_id)
            label = (f"[{position}/{len(targets)}] {identifier} "
                     f"{document.formula_pretty} ({document.nsites} sites)")

            found = self._existing(identifier) if skip_existing else None
            if found is not None:
                print(f"{label}: already present, skipping")
                manifest.append(self._manifest_row(identifier, document, found,
                                                   "cached"))
                self._write_manifest(manifest)
                continue

            path = self._chgcar_path(identifier, compress=compress)
            clock = time.time()
            try:
                chgcar = rester.get_charge_density_from_material_id(identifier)
                if chgcar is None:
                    raise ValueError("the API returned no charge density")
                chgcar.write_file(str(path))
            except Exception as error:                          # noqa: BLE001
                # One material's failure must not cost the other 200. The
                # partial file goes, the reason is recorded, the batch carries
                # on -- and the manifest says `failed` rather than staying
                # silent about a hole in the dataset.
                print(f"{label}: FAILED - {type(error).__name__}: {error}")
                path.unlink(missing_ok=True)
                manifest.append(self._manifest_row(
                    identifier, document, None, "failed",
                    f"{type(error).__name__}: {error}"))
                self._write_manifest(manifest)
                continue

            print(f"{label}: {path.stat().st_size / 1e6:.1f} MB "
                  f"in {time.time() - clock:.1f}s")
            manifest.append(self._manifest_row(identifier, document, path,
                                               "downloaded"))
            self._write_manifest(manifest)

        succeeded = sum(1 for row in manifest
                        if row["status"] in ("downloaded", "cached"))
        total = sum(row["size_mb"] for row in manifest)
        print(f"  -> {succeeded}/{len(targets)} on disk, {total:.0f} MB, "
              f"elapsed {(time.time() - started) / 60:.1f} min")
        return manifest

    @staticmethod
    def _manifest_row(identifier, document, path, status, error=""):
        """One manifest record."""
        return {
            "material_id": identifier,
            "formula_pretty": document.formula_pretty,
            "nsites": document.nsites,
            "file": path.name if path else "",
            "size_mb": round(path.stat().st_size / 1e6, 2) if path else 0.0,
            "compressed": bool(path and path.suffix == ".gz"),
            "status": status,
            "error": error,
        }

    def _write_manifest(self, manifest):
        """Rewrite ``chgcar/manifest.csv`` from scratch."""
        with (self.chgcar_dir / MANIFEST_FILENAME).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(manifest)

    # ------------------------------------------------------------------ #
    # Storage
    # ------------------------------------------------------------------ #
    def decompress(self, keep_original=False, dry_run=False):
        """
        Expand the downloaded CHGCARs in place (``CHGCAR_x.gz`` -> ``CHGCAR_x``).

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
        files = sorted(self.chgcar_dir.glob("CHGCAR_*.gz"))
        if not files:
            print(f"No gzipped CHGCARs in {self.chgcar_dir}")
            return {"n": 0, "before": 0, "after": 0}

        before = sum(path.stat().st_size for path in files)
        free = shutil.disk_usage(self.chgcar_dir).free
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
        files = [path for path in sorted(self.chgcar_dir.glob("CHGCAR_*"))
                 if path.suffix != ".gz"]
        if not files:
            print(f"No expanded CHGCARs in {self.chgcar_dir}")
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

    def disk_usage(self):
        """Bytes currently used by the downloaded CHGCARs, split by form."""
        gzipped = list(self.chgcar_dir.glob("CHGCAR_*.gz"))
        plain = [p for p in self.chgcar_dir.glob("CHGCAR_*") if p.suffix != ".gz"]
        return {
            "n_gz": len(gzipped),
            "bytes_gz": sum(p.stat().st_size for p in gzipped),
            "n_unzipped": len(plain),
            "bytes_unzipped": sum(p.stat().st_size for p in plain),
            "free": shutil.disk_usage(self.outdir).free,
        }

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def dry_run(self):
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
        self.search()
        estimate = self.estimate()
        print("\n" + str(estimate) + "\n")
        print(f"  {estimate.n_available} file(s) to download, "
              f"{_format_bytes(estimate.download_bytes)} total "
              f"({_format_bytes(estimate.gz_disk_bytes)} on disk gzipped).")
        print("  Dry run: nothing downloaded, nothing written.")
        return estimate

    def run(self, estimate_only=False, skip_chgcar=False, decompress=False,
            **download_kwargs):
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
            return self.dry_run()

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
    parser.add_argument("--elements", nargs="+", required=True,
                        metavar="EL", help="elements spanning the chemical space")
    # Two spellings of one argument. The current directory is the default
    # because a command that writes hundreds of megabytes should put them
    # where it was run, not somewhere it picked.
    parser.add_argument("--outdir", "--output", dest="outdir", default=".",
                        metavar="DIR",
                        help="destination for summary.*, structures/ and "
                             "chgcar/ (default: the current directory)")
    parser.add_argument("--estimate", action="store_true",
                        help="dry run: report the exact size on the console and "
                             "write nothing at all")
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

    group = parser.add_argument_group("storage")
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
    args = build_parser().parse_args(argv)

    try:
        fetcher = MPDataFetcher(
            args.elements, outdir=args.outdir, compress=not args.unzip,
            workers=args.workers, band_gap=args.band_gap,
            crystal_system=args.crystal_system,
            is_stable=True if args.stable_only else None,
            num_sites=args.num_sites,
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
                    limit=args.limit, max_sites=args.max_sites,
                    max_size_mb=args.max_size_mb)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
