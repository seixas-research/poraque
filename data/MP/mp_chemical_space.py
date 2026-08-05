#!/usr/bin/env python
"""MPChemicalSpace - query a Materials Project chemical space and fetch CHGCARs.

The chemical space is given to the constructor as a list of elements. Only
materials whose composition uses exclusively those elements are returned,
across all stoichiometries and structures.

    from mp_chemical_space import MPChemicalSpace

    space = MPChemicalSpace(["Si", "O", "H", "Al"])
    print(space.estimate())          # exact download size, nothing transferred
    space.write_summary()
    space.download(max_size_mb=20)   # resumable
    space.decompress()               # unzip CHGCARs in place

Size estimation is exact, not modelled: MP keeps charge densities as objects in
the public `materialsproject-parsed` S3 bucket, so the class resolves each
material's static-calculation task ID and issues an S3 HEAD request, reading
`Content-Length` without transferring any payload.

Storage is reported three ways, because they differ a lot:

    download   bytes pulled over the network (the gzipped S3 object)
    gz on disk bytes kept if CHGCARs stay gzipped   (x0.879 of download)
    unzipped   bytes kept if CHGCARs are expanded   (x2.99 of gz on disk)

Both ratios were measured against the fully downloaded Au-Ag-Cu-Pd-Pt set
(85 objects: 1044.6 MB downloaded, 917.7 MB gzipped, 2.74 GB unzipped), not
guessed. Unzipping is therefore roughly a 3x storage multiplier - check
`Estimate.unzipped_bytes` against free space before calling `decompress()`.

CLI:
    python mp_chemical_space.py --elements Si O H Al --estimate
    python mp_chemical_space.py --elements Si O H Al --max-size-mb 20
    python mp_chemical_space.py --elements Si O H Al --decompress
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

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from emmet.core.mpid import AlphaID
from mp_api.client import MPRester
from mp_api.client.core.utils import validate_ids

# --- measured constants -----------------------------------------------------
# Calibrated on the Au-Ag-Cu-Pd-Pt set (85 charge densities); see module docstring.
DISK_PER_DOWNLOAD = 0.879  # gzipped-on-disk / bytes-downloaded
UNZIP_EXPANSION = 2.99     # unzipped / gzipped-on-disk

CHGCAR_BUCKET = "materialsproject-parsed"
CHGCAR_PREFIX = "chgcars"
STATIC_CALC_TYPES = {"GGA Static", "GGA+U Static"}

SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "chemsys", "nelements", "nsites",
    "symmetry", "volume", "density", "density_atomic", "energy_per_atom",
    "formation_energy_per_atom", "energy_above_hull", "is_stable", "band_gap",
    "is_metal", "is_magnetic", "ordering", "total_magnetization",
    "theoretical", "deprecated", "has_props", "structure",
]

CSV_COLUMNS = [
    "material_id", "formula_pretty", "chemsys", "nelements", "nsites",
    "spacegroup_symbol", "spacegroup_number", "crystal_system", "volume",
    "density", "density_atomic", "energy_per_atom",
    "formation_energy_per_atom", "energy_above_hull", "is_stable", "band_gap",
    "is_metal", "is_magnetic", "ordering", "total_magnetization",
    "theoretical", "has_charge_density",
]


def _fmt(nbytes: float) -> str:
    """Human-readable byte count."""
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f} MB"
    return f"{nbytes / 1e3:.0f} kB"


@dataclass
class Estimate:
    """Exact download/storage figures for a chemical space's charge densities."""

    elements: list[str]
    n_materials: int
    n_advertised: int          # has_props.charge_density is true
    n_available: int           # object actually present in S3
    sizes: dict[str, int] = field(repr=False, default_factory=dict)
    rows: list[dict] = field(repr=False, default_factory=list)

    @property
    def download_bytes(self) -> int:
        return sum(s for s in self.sizes.values() if s > 0)

    @property
    def gz_disk_bytes(self) -> float:
        """Storage if CHGCARs are kept gzipped."""
        return self.download_bytes * DISK_PER_DOWNLOAD

    @property
    def unzipped_bytes(self) -> float:
        """Storage if CHGCARs are unzipped - roughly 3x the gzipped figure."""
        return self.gz_disk_bytes * UNZIP_EXPANSION

    @property
    def n_missing(self) -> int:
        """Advertised by MP but not actually fetchable."""
        return self.n_advertised - self.n_available

    def files_under(self, max_size_mb: float) -> tuple[int, int]:
        """(count, total bytes) of objects at or below a size cap."""
        kept = [s for s in self.sizes.values() if 0 < s <= max_size_mb * 1e6]
        return len(kept), sum(kept)

    def __str__(self) -> str:
        good = sorted(s for s in self.sizes.values() if s > 0)
        if not good:
            return "Estimate: no charge densities available"

        def pct(p):
            return good[min(int(len(good) * p), len(good) - 1)]

        L = ["=" * 66,
             f"  CHGCAR ESTIMATE - {'-'.join(self.elements)} space",
             "=" * 66,
             f"  Materials in space           : {self.n_materials}",
             f"  Advertising a charge density : {self.n_advertised}",
             f"  FILES TO DOWNLOAD            : {self.n_available}",
             f"  TOTAL DOWNLOAD               : {_fmt(self.download_bytes)}",
             "-" * 66,
             f"  Storage if kept gzipped      : {_fmt(self.gz_disk_bytes)}",
             f"  Storage if UNZIPPED (x{UNZIP_EXPANSION})   : {_fmt(self.unzipped_bytes)}",
             "-" * 66,
             f"  smallest / median / largest  : {_fmt(good[0])} / "
             f"{_fmt(pct(0.5))} / {_fmt(good[-1])}",
             f"  90th percentile              : {_fmt(pct(0.9))}",
             ]
        if self.n_missing:
            L.append(f"  Advertised but unavailable   : {self.n_missing}")
        L.append("-" * 66)
        L.append("  If you cap file size (--max-size-mb):")
        for cap in (5, 10, 20, 50, 100, 200):
            n, b = self.files_under(cap)
            if n:
                L.append(f"    <= {cap:4d} MB : {n:5d} files, {_fmt(b):>9}"
                         f"  (unzipped {_fmt(b * DISK_PER_DOWNLOAD * UNZIP_EXPANSION)})")
        L.append("-" * 66)
        L.append("  10 largest:")
        for r in self.rows[:10]:
            L.append(f"    {r['material_id']:<14} {r['formula_pretty']:<14} "
                     f"{r['nsites']:>4} sites {r['size_mb']:>8.1f} MB")
        L.append("=" * 66)
        return "\n".join(L)


class MPChemicalSpace:
    """A Materials Project chemical space and its charge-density dataset.

    Args:
        elements: elements spanning the space, e.g. ["Si", "O", "H", "Al"].
        api_key: MP API key. Falls back to $MP_API_KEY, then .env, then ~/.env.
        outdir: where results are written. Defaults to the current directory.
        compress: if True (default) CHGCARs are stored gzipped; if False they
            are written expanded, which costs about 3x the space.
        workers: parallel S3 HEAD requests used during estimation.
    """

    def __init__(self, elements, api_key=None, outdir=".", compress=True,
                 workers=16):
        if isinstance(elements, str):  # allow "Si-O-H-Al" or "Si O H Al"
            elements = elements.replace("-", " ").split()
        if not elements:
            raise ValueError("at least one element is required")
        self.elements = sorted(set(elements))

        if api_key is None:
            load_dotenv()
            load_dotenv(Path.home() / ".env")
            api_key = os.getenv("MP_API_KEY")
        if not api_key:
            raise ValueError(
                "No MP API key. Pass api_key=... or set MP_API_KEY in the "
                "environment, a local .env, or ~/.env.")
        self._api_key = api_key

        self.outdir = Path(outdir).resolve()
        self.compress = compress
        self.workers = workers

        self._mpr: MPRester | None = None
        self._docs: list | None = None
        self._keys: dict[str, str] | None = None
        self._estimate: Estimate | None = None

    # --- context manager / connection ---------------------------------------

    def __enter__(self):
        self._connect()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _connect(self) -> MPRester:
        if self._mpr is None:
            self._mpr = MPRester(self._api_key)
        return self._mpr

    def close(self) -> None:
        if self._mpr is not None:
            self._mpr.session.close()
            self._mpr = None

    def __repr__(self) -> str:
        n = len(self._docs) if self._docs is not None else "?"
        return (f"MPChemicalSpace({'-'.join(self.elements)}, "
                f"materials={n}, outdir='{self.outdir}')")

    # --- directories ---------------------------------------------------------

    @property
    def chgcar_dir(self) -> Path:
        return self.outdir / "chgcar"

    @property
    def structure_dir(self) -> Path:
        return self.outdir / "structures"

    # --- search --------------------------------------------------------------

    @property
    def chemical_systems(self) -> list[str]:
        """Every non-empty subsystem (2**n - 1).

        Querying the full system alone (e.g. "Al-H-O-Si") returns only the
        quaternary, so each subset must be listed to span the whole space.
        """
        return ["-".join(sorted(c))
                for n in range(1, len(self.elements) + 1)
                for c in combinations(self.elements, n)]

    def search(self, refresh: bool = False) -> list:
        """Summary documents for every material in the space (cached)."""
        if self._docs is not None and not refresh:
            return self._docs
        mpr = self._connect()
        systems = self.chemical_systems
        print(f"Searching {len(systems)} chemical systems in the "
              f"{'-'.join(self.elements)} space ...")
        docs = mpr.materials.summary.search(chemsys=systems, fields=SUMMARY_FIELDS)
        docs.sort(key=lambda d: (d.nelements, d.chemsys, d.formula_pretty,
                                 str(d.material_id)))
        print(f"  -> {len(docs)} materials found")
        self._docs = docs
        return docs

    @staticmethod
    def _has_cd(doc) -> bool:
        props = doc.has_props
        if props is None:
            return False
        if not isinstance(props, dict):
            props = dict(props)
        return bool(props.get("charge_density"))

    @property
    def materials(self) -> list:
        return self.search()

    @property
    def with_charge_density(self) -> list:
        return [d for d in self.search() if self._has_cd(d)]

    # --- charge-density object resolution ------------------------------------

    def _resolve_keys(self, refresh: bool = False) -> dict[str, str]:
        """material_id -> S3 key, batching the two underlying queries.

        Mirrors MPRester.get_charge_density_from_material_id, but resolves every
        material in two API calls instead of two per material.
        """
        if self._keys is not None and not refresh:
            return self._keys
        mpr = self._connect()
        docs = self.with_charge_density
        mids = [str(d.material_id) for d in docs]
        print(f"Resolving charge-density tasks for {len(mids)} materials ...")

        mat_docs = mpr.materials.search(material_ids=mids,
                                        fields=["material_id", "calc_types"])
        candidates: dict[str, list[str]] = {}
        for md in mat_docs:
            static = [str(t) for t, ct in (md["calc_types"] or {}).items()
                      if str(ct) in STATIC_CALC_TYPES]
            if static:
                candidates[str(md["material_id"])] = static

        all_tasks = sorted({t for ts in candidates.values() for t in ts})
        print(f"  -> {len(all_tasks)} candidate static tasks ...")
        task_docs = mpr.materials.tasks.search(task_ids=all_tasks,
                                               fields=["task_id", "last_updated"])
        updated = {str(t["task_id"]): t["last_updated"] for t in task_docs}

        keys: dict[str, str] = {}
        for mid, tasks in candidates.items():
            known = [t for t in tasks if t in updated]
            if not known:
                continue
            best = max(known, key=lambda t: updated[t])  # newest, as MPRester does
            alpha = AlphaID(validate_ids([best])[0].split("-")[-1], prefix="mp")
            keys[mid] = f"{CHGCAR_PREFIX}/{alpha.string}.json.gz"

        print(f"  -> resolved {len(keys)} objects")
        self._keys = keys
        return keys

    def _head_sizes(self, keys: dict[str, str]) -> dict[str, int]:
        """Exact object sizes via S3 HEAD - no payload transferred."""
        s3 = boto3.client("s3", config=Config(
            signature_version=UNSIGNED, max_pool_connections=self.workers + 4))

        def one(item):
            mid, key = item
            try:
                return mid, s3.head_object(Bucket=CHGCAR_BUCKET,
                                           Key=key)["ContentLength"]
            except ClientError:
                return mid, -1  # advertised but absent

        print(f"Reading exact sizes ({len(keys)} HEAD requests) ...")
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return dict(pool.map(one, keys.items()))

    # --- estimate ------------------------------------------------------------

    def estimate(self, refresh: bool = False) -> Estimate:
        """Exact download size and storage projection. Downloads nothing."""
        if self._estimate is not None and not refresh:
            return self._estimate

        docs = self.search()
        keys = self._resolve_keys(refresh=refresh)
        sizes = self._head_sizes(keys)

        by_id = {str(d.material_id): d for d in docs}
        rows = []
        for mid, key in keys.items():
            d = by_id[mid]
            rows.append({
                "material_id": mid,
                "formula_pretty": d.formula_pretty,
                "chemsys": d.chemsys,
                "nsites": d.nsites,
                "volume": round(d.volume, 2),
                "s3_key": key,
                "size_bytes": sizes.get(mid, -1),
                "size_mb": round(max(sizes.get(mid, -1), 0) / 1e6, 2),
            })
        rows.sort(key=lambda r: -r["size_bytes"])

        self._estimate = Estimate(
            elements=self.elements,
            n_materials=len(docs),
            n_advertised=len(self.with_charge_density),
            n_available=sum(1 for s in sizes.values() if s > 0),
            sizes=sizes,
            rows=rows,
        )
        return self._estimate

    def write_estimate(self, filename: str = "chgcar_estimate.csv") -> Path:
        """Write the per-material size table."""
        est = self.estimate()
        self.outdir.mkdir(parents=True, exist_ok=True)
        path = self.outdir / filename
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(est.rows[0].keys()))
            w.writeheader()
            w.writerows(est.rows)
        print(f"  -> wrote {path}")
        return path

    # --- metadata ------------------------------------------------------------

    def _flatten(self, doc) -> dict:
        sym = doc.symmetry
        return {
            "material_id": str(doc.material_id),
            "formula_pretty": doc.formula_pretty,
            "chemsys": doc.chemsys,
            "nelements": doc.nelements,
            "nsites": doc.nsites,
            "spacegroup_symbol": getattr(sym, "symbol", None),
            "spacegroup_number": getattr(sym, "number", None),
            "crystal_system": str(getattr(sym, "crystal_system", "") or "") or None,
            "volume": doc.volume,
            "density": doc.density,
            "density_atomic": doc.density_atomic,
            "energy_per_atom": doc.energy_per_atom,
            "formation_energy_per_atom": doc.formation_energy_per_atom,
            "energy_above_hull": doc.energy_above_hull,
            "is_stable": doc.is_stable,
            "band_gap": doc.band_gap,
            "is_metal": doc.is_metal,
            "is_magnetic": doc.is_magnetic,
            "ordering": str(doc.ordering) if doc.ordering is not None else None,
            "total_magnetization": doc.total_magnetization,
            "theoretical": doc.theoretical,
            "has_charge_density": self._has_cd(doc),
        }

    def write_summary(self, basename: str = "summary") -> Path:
        """Write summary.csv and summary.json."""
        records = [self._flatten(d) for d in self.search()]
        self.outdir.mkdir(parents=True, exist_ok=True)
        csv_path = self.outdir / f"{basename}.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(records)
        with (self.outdir / f"{basename}.json").open("w") as fh:
            json.dump(records, fh, indent=2)
        print(f"  -> wrote {basename}.csv and {basename}.json ({len(records)} rows)")
        return csv_path

    def write_structures(self) -> int:
        """Write one CIF per material."""
        self.structure_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for doc in self.search():
            if doc.structure is not None:
                doc.structure.to(filename=str(self.structure_dir /
                                              f"{doc.material_id}.cif"))
                n += 1
        print(f"  -> wrote {n} CIF files to {self.structure_dir.name}/")
        return n

    # --- download ------------------------------------------------------------

    def _chgcar_path(self, mid: str, compress: bool | None = None) -> Path:
        compress = self.compress if compress is None else compress
        suffix = ".gz" if compress else ""
        return self.chgcar_dir / f"CHGCAR_{mid}{suffix}"

    def _existing(self, mid: str) -> Path | None:
        """Return whichever variant (.gz or plain) is already on disk."""
        for c in (True, False):
            p = self._chgcar_path(mid, compress=c)
            if p.exists() and p.stat().st_size > 0:
                return p
        return None

    def download(self, limit=None, max_sites=None, max_size_mb=None,
                 compress=None, skip_existing=True) -> list[dict]:
        """Download CHGCARs. Resumable; one failure never aborts the batch.

        Args:
            limit: download at most this many.
            max_sites: skip cells with more than this many sites.
            max_size_mb: skip objects larger than this (uses exact S3 sizes).
            compress: store gzipped (default: the instance's `compress`).
            skip_existing: leave already-downloaded files alone.

        Returns the manifest as a list of dicts (also written to
        chgcar/manifest.csv after every file).
        """
        compress = self.compress if compress is None else compress
        mpr = self._connect()
        est = self.estimate()
        self.chgcar_dir.mkdir(parents=True, exist_ok=True)

        by_id = {str(d.material_id): d for d in self.search()}
        targets = [by_id[m] for m, s in est.sizes.items() if s > 0 and m in by_id]
        targets.sort(key=lambda d: est.sizes[str(d.material_id)])  # small first

        if max_sites is not None:
            targets = [d for d in targets if d.nsites <= max_sites]
        if max_size_mb is not None:
            targets = [d for d in targets
                       if est.sizes[str(d.material_id)] <= max_size_mb * 1e6]
        if limit is not None:
            targets = targets[:limit]

        planned = sum(est.sizes[str(d.material_id)] for d in targets)
        print(f"Downloading {len(targets)} CHGCARs (~{_fmt(planned)} transfer, "
              f"~{_fmt(planned * DISK_PER_DOWNLOAD * (1 if compress else UNZIP_EXPANSION))}"
              f" on disk {'gzipped' if compress else 'unzipped'}) ...")

        manifest, t0_all = [], time.time()
        for i, doc in enumerate(targets, 1):
            mid = str(doc.material_id)
            label = (f"[{i}/{len(targets)}] {mid} {doc.formula_pretty} "
                     f"({doc.nsites} sites)")

            found = self._existing(mid) if skip_existing else None
            if found is not None:
                print(f"{label}: already present, skipping")
                manifest.append(self._row(mid, doc, found, "cached"))
                self._write_manifest(manifest)
                continue

            path = self._chgcar_path(mid, compress=compress)
            t0 = time.time()
            try:
                chgcar = mpr.get_charge_density_from_material_id(mid)
                if chgcar is None:
                    raise ValueError("API returned no charge density")
                chgcar.write_file(str(path))
            except Exception as exc:
                print(f"{label}: FAILED - {type(exc).__name__}: {exc}")
                path.unlink(missing_ok=True)
                manifest.append(self._row(mid, doc, None, "failed",
                                          f"{type(exc).__name__}: {exc}"))
                self._write_manifest(manifest)
                continue

            print(f"{label}: {path.stat().st_size / 1e6:.1f} MB "
                  f"in {time.time() - t0:.1f}s")
            manifest.append(self._row(mid, doc, path, "downloaded"))
            self._write_manifest(manifest)

        ok = sum(1 for m in manifest if m["status"] in ("downloaded", "cached"))
        tot = sum(m["size_mb"] for m in manifest)
        print(f"  -> {ok}/{len(targets)} on disk, {tot:.0f} MB, "
              f"elapsed {(time.time() - t0_all) / 60:.1f} min")
        return manifest

    @staticmethod
    def _row(mid, doc, path, status, error="") -> dict:
        return {"material_id": mid, "formula_pretty": doc.formula_pretty,
                "nsites": doc.nsites,
                "file": path.name if path else "",
                "size_mb": round(path.stat().st_size / 1e6, 2) if path else 0.0,
                "compressed": bool(path and path.suffix == ".gz"),
                "status": status, "error": error}

    def _write_manifest(self, manifest: list[dict]) -> None:
        with (self.chgcar_dir / "manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["material_id", "formula_pretty",
                                               "nsites", "file", "size_mb",
                                               "compressed", "status", "error"])
            w.writeheader()
            w.writerows(manifest)

    # --- (de)compression ------------------------------------------------------

    def decompress(self, keep_original: bool = False, dry_run: bool = False) -> dict:
        """Unzip downloaded CHGCARs in place (CHGCAR_x.gz -> CHGCAR_x).

        Expands storage by roughly 3x. With dry_run=True nothing is written and
        the projected size is reported, so free space can be checked first.
        """
        files = sorted(self.chgcar_dir.glob("CHGCAR_*.gz"))
        if not files:
            print(f"No gzipped CHGCARs in {self.chgcar_dir}")
            return {"n": 0, "before": 0, "after": 0}

        before = sum(f.stat().st_size for f in files)
        free = shutil.disk_usage(self.chgcar_dir).free
        projected = before * UNZIP_EXPANSION
        # Unzipping adds the expanded copy; the .gz goes away unless kept.
        needed = projected if not keep_original else projected + before

        print(f"{len(files)} gzipped CHGCARs, {_fmt(before)}")
        print(f"  projected unzipped : {_fmt(projected)} (x{UNZIP_EXPANSION})")
        print(f"  additional space   : {_fmt(needed - (0 if keep_original else before))}")
        print(f"  free on volume     : {_fmt(free)}")

        if dry_run:
            print("  (dry run - nothing written)")
            return {"n": len(files), "before": before, "after": projected}

        extra = needed - (0 if keep_original else before)
        if extra > free * 0.95:
            hint = ("drop keep_original so the .gz files are removed as you go, "
                    if keep_original else "")
            raise OSError(
                f"Not enough free space: unzipping needs ~{_fmt(extra)} more, "
                f"only {_fmt(free)} free. Free up space, {hint}"
                f"unzip a subset, or leave the files gzipped "
                f"(Chgcar.from_file reads .gz directly).")

        after = 0
        for i, gz in enumerate(files, 1):
            out = gz.with_suffix("")  # strip .gz
            with gzip.open(gz, "rb") as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
            after += out.stat().st_size
            if not keep_original:
                gz.unlink()
            if i % 25 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {_fmt(after)} written")
        print(f"  -> unzipped {len(files)} files: {_fmt(before)} -> {_fmt(after)} "
              f"(x{after / before:.2f})")
        return {"n": len(files), "before": before, "after": after}

    def recompress(self, level: int = 6) -> dict:
        """Re-gzip unzipped CHGCARs (CHGCAR_x -> CHGCAR_x.gz), reclaiming space."""
        files = [f for f in sorted(self.chgcar_dir.glob("CHGCAR_*"))
                 if f.suffix != ".gz"]
        if not files:
            print(f"No unzipped CHGCARs in {self.chgcar_dir}")
            return {"n": 0, "before": 0, "after": 0}
        before = sum(f.stat().st_size for f in files)
        after = 0
        for i, raw in enumerate(files, 1):
            out = raw.with_suffix(raw.suffix + ".gz")
            with raw.open("rb") as src, gzip.open(out, "wb", compresslevel=level) as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
            after += out.stat().st_size
            raw.unlink()
            if i % 25 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] {_fmt(after)} written")
        print(f"  -> recompressed {len(files)} files: {_fmt(before)} -> {_fmt(after)}")
        return {"n": len(files), "before": before, "after": after}

    def disk_usage(self) -> dict:
        """Bytes currently used by downloaded CHGCARs, split by form."""
        gz = [f for f in self.chgcar_dir.glob("CHGCAR_*.gz")]
        raw = [f for f in self.chgcar_dir.glob("CHGCAR_*") if f.suffix != ".gz"]
        return {
            "n_gz": len(gz), "bytes_gz": sum(f.stat().st_size for f in gz),
            "n_unzipped": len(raw), "bytes_unzipped": sum(f.stat().st_size for f in raw),
            "free": shutil.disk_usage(self.outdir).free,
        }

    # --- convenience ---------------------------------------------------------

    def run(self, estimate_only=False, skip_chgcar=False, decompress=False,
            **download_kw) -> Estimate:
        """Full pipeline: summary -> estimate -> structures -> CHGCARs."""
        self.search()
        self.write_summary()
        est = self.estimate()
        self.write_estimate()
        print("\n" + str(est) + "\n")
        if estimate_only:
            print("Estimate only; nothing downloaded.")
            return est
        self.write_structures()
        if not skip_chgcar:
            self.download(**download_kw)
            if decompress:
                print()
                self.decompress()
        return est


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--elements", nargs="+", default=["Si", "O", "H", "Al"])
    p.add_argument("--outdir", default=str(Path(__file__).parent))
    p.add_argument("--estimate", action="store_true",
                   help="report exact size, download nothing")
    p.add_argument("--skip-chgcar", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-sites", type=int)
    p.add_argument("--max-size-mb", type=float)
    p.add_argument("--unzip", action="store_true",
                   help="store CHGCARs expanded instead of gzipped (~3x space)")
    p.add_argument("--decompress", action="store_true",
                   help="unzip CHGCARs already on disk, then exit")
    p.add_argument("--recompress", action="store_true",
                   help="re-gzip unzipped CHGCARs on disk, then exit")
    p.add_argument("--dry-run", action="store_true",
                   help="with --decompress, only report the projected size")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    try:
        space = MPChemicalSpace(args.elements, outdir=args.outdir,
                                compress=not args.unzip, workers=args.workers)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with space:
        if args.decompress:
            space.decompress(dry_run=args.dry_run)
            return 0
        if args.recompress:
            space.recompress()
            return 0
        space.run(estimate_only=args.estimate, skip_chgcar=args.skip_chgcar,
                  limit=args.limit, max_sites=args.max_sites,
                  max_size_mb=args.max_size_mb)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
