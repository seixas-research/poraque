# -*- coding: utf-8 -*-
# file: mp_dataset.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Train on Materials Project charge densities.

What arrives from the Materials Project is *one file per material* — a
compressed ``CHGCAR`` in a flat directory::

    data/MP/chgcar/
        CHGCAR_mp-124.gz        CHGCAR_mp-126.gz        CHGCAR_mp-81.gz
        manifest.csv

and nothing else: no ``POSCAR``, no ``INCAR``, no ``POTCAR``, no ``OUTCAR``, no
``TAUCAR``. That is three departures from the layout
:mod:`poraque.ml.data` expects, and this module closes each one.

**The files are compressed.** They are read in place, gzip and all, by
:mod:`poraque.fields.io.compressed`. Nothing is ever expanded on disk.

**There is no external potential.** A ``CHGCAR`` carries its own ``POSCAR`` in
its first lines, so the geometry is recoverable from the density alone, and
:class:`~poraque.fields.ExternalPotential` builds :math:`V_{\rm ext}` from it —
exactly as the ordinary pipeline does, which computes the potential rather than
reading it precisely so that training and inference see the same construction.

.. important::

   With no ``POTCAR`` there is no tabulated local pseudopotential, so the
   **Gaussian pseudo-ion model** is used rather than the exact ``model="potcar"``
   route. Measured against a reference VASP ``EXTCAR`` that model leaves a
   relative-:math:`L_2` residual of order 0.1 (see
   :mod:`poraque.fields.external`). The map being learned here is therefore
   *model potential* :math:`\to` *DFT density*, which is a well-defined and
   self-consistent learning problem — the same potential is built at inference
   time — but it is not the VASP :math:`V_{\rm ext}`. Anything comparing these
   models against ones trained on tabulated potentials has to say so.

**The valence charges are unknown.** :math:`Z^{\rm val}` comes from the
``POTCAR`` and there is none. It is recovered from the data instead, by
:func:`infer_valence_charges`: a ``CHGCAR`` integrates to the valence electron
count of its own cell, which gives one linear equation per material in the
per-element charges, and a chemical space of any breadth supplies more equations
than unknowns. On the Ag-Au-Pt set this returns 11, 11 and 10 to within
:math:`10^{-6}` — the POTCAR values, read off the densities.

**There is no** :math:`\tau` **and no energy.** ``chg2tau`` cannot be trained on
this data and :func:`available_tasks` says so rather than failing later;
``reference_energy`` is ``None`` throughout, which
:class:`~poraque.ml.data.FieldPairDataset` already treats as "not available"
rather than as zero.

Two entry points, one implementation:

:class:`MPChargeDensityDataset`
    Serves ``(V_ext, rho)`` pairs straight out of the download directory.
:func:`build_mp_cache`
    Writes the same pairs, spectrally downsampled, into the per-material
    directory layout the standard trainer reads — so ``poraque-train`` needs
    only to be pointed at it.

Both are thin specialisations. The layout itself is
:class:`~poraque.data.sources.BulkDensitySource`, one of the layouts
:class:`~poraque.data.dataset.MixedFieldDataset` reads, so a Materials Project
download can equally be *one entry in a list* of training paths beside local
VASP runs. What lives here is what is specific to the archive: how its files
are named, and how the valence charges its potentials need are recovered from
the densities.
"""

import os
from pathlib import Path

import numpy as np

from ..fields import ChargeDensity, FieldGrid
from ..fields.vasp.volumetric import read_structure_header
from ..ml.data import MaterialRecord
from .dataset import MixedFieldDataset

#: Prefix of a Materials Project charge-density file, as written by
#: :class:`~poraque.data.materials_project.MPDataFetcher`.
CHGCAR_PREFIX = "CHGCAR_"

#: Fields an MP download supplies. Everything else — ``TAUCAR``, ``OUTCAR``,
#: absolute energies — is absent by construction, and the pipeline is expected
#: to work around that rather than to demand it.
AVAILABLE_FIELDS = ("CHGCAR",)

#: Residual above which :func:`infer_valence_charges` refuses to round its
#: solution to integers, in electrons per material.
INTEGER_TOLERANCE = 0.05


def discover_mp_chgcars(root, pattern=CHGCAR_PREFIX):
    """
    Find the charge densities in a Materials Project download directory.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory holding ``CHGCAR_<material_id>[.gz]`` files — typically
        ``data/MP/chgcar``. A parent directory containing a ``chgcar/``
        subdirectory is accepted too, so ``data/MP`` also works.
    pattern : str, optional
        Filename prefix identifying a density.

    Returns
    -------
    list of MaterialRecord
        One record per file, identified by its material id and sorted for
        reproducible ordering. ``files`` holds only ``CHGCAR``: naming a file
        that is not there would turn a known absence into a crash on first
        access.

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist, or holds no matching file. The message
        names the directory that was searched — the usual cause is pointing at
        ``data/MP`` before anything has been downloaded.
    """
    root = Path(root)
    if root.is_dir() and (root / "chgcar").is_dir():
        root = root / "chgcar"
    if not root.is_dir():
        raise FileNotFoundError(f"No such directory: {root}")

    records = []
    for entry in sorted(os.listdir(root)):
        path = root / entry
        if not entry.startswith(pattern) or not path.is_file():
            continue
        from ..fields.io.compressed import strip_compression_suffix

        identifier = strip_compression_suffix(entry)[len(pattern):]
        records.append(MaterialRecord(identifier, str(root),
                                      files={"CHGCAR": str(path)}))

    if not records:
        raise FileNotFoundError(
            f"No {pattern}* files under {root}. Download some first: "
            f"poraque-mp --elements Ag Au Pt --outdir {root.parent}"
        )
    return records


def available_tasks(records=None):
    """
    Which regression tasks a Materials Project download can support.

    Returns
    -------
    list of str
        ``["ext2chg"]``. The Hohenberg-Kohn map is trainable because the
        external potential is *computed* from the structure the density itself
        carries. ``chg2tau`` is not, because MP publishes no kinetic energy
        density and no amount of pipeline work invents one.
    """
    return ["ext2chg"]


# ---------------------------------------------------------------------- #
# Valence charges from the densities themselves
# ---------------------------------------------------------------------- #
def _composition(structure):
    """``{element: count}`` for one structure, merging repeated species blocks."""
    composition = {}
    for element, count in zip(structure.elements, structure.counts):
        composition[element] = composition.get(element, 0) + int(count)
    return composition


def _electron_count(path):
    """
    Valence electrons in a ``CHGCAR``, from the density it holds.

    VASP stores :math:`\\rho\\,\\Omega`, so the mean of the grid block *is* the
    electron count and no volume or grid spacing enters — which is why this is
    exact rather than a quadrature estimate.
    """
    grid = FieldGrid.from_file(path)
    return float(ChargeDensity.read(path, grid=grid).integrate())


def infer_valence_charges(records, overrides=None, tolerance=INTEGER_TOLERANCE,
                          log=None):
    r"""
    Recover :math:`Z^{\rm val}` per element from the charge densities.

    A pseudopotential ``CHGCAR`` integrates to the valence electron count of
    its cell, so each material contributes one equation

    .. math::

        \sum_s n_s^{(m)}\, Z^{\rm val}_s \;=\; N^{(m)}_e ,

    linear in the per-element charges, with :math:`n_s^{(m)}` read from the
    structure header. A chemical space with more compositions than elements is
    therefore over-determined, and least squares recovers the charges the
    ``POTCAR`` would have stated.

    Only as many materials are read as the system needs: candidates are taken
    smallest-file-first and accepted only while they add rank, so a
    hundred-material space costs a handful of small reads rather than a full
    pass. The solution is rounded to integers when every residual is below
    ``tolerance``, since :math:`Z^{\rm val}` is an integer by construction and
    a value of 10.999998 in a log helps nobody.

    Parameters
    ----------
    records : sequence of MaterialRecord
        From :func:`discover_mp_chgcars`.
    overrides : dict, optional
        ``{element: charge}`` taken as given. An element covered here is
        removed from the system rather than fitted, so supplying one known
        charge can make an otherwise rank-deficient set solvable.
    tolerance : float, optional
        Largest residual, in electrons, that still permits rounding.
    log : callable, optional
        Receives one-line progress messages.

    Returns
    -------
    dict
        ``{element: valence charge}`` covering every element in ``records``.

    Raises
    ------
    ValueError
        If the compositions cannot determine every charge — for instance a
        dataset of a single binary AB, where any split of its electrons
        between A and B fits equally well. The message names the elements that
        remain undetermined and points at ``overrides``.
    """
    log = log or (lambda *_: None)
    overrides = {str(k).split("_")[0]: float(v)
                 for k, v in (overrides or {}).items()}

    # Header-only reads: a few hundred bytes each, whatever the grid.
    compositions = {}
    for record in records:
        compositions[record.identifier] = _composition(
            read_structure_header(record.files["CHGCAR"]))

    elements = sorted({element for composition in compositions.values()
                       for element in composition})
    unknown = [element for element in elements if element not in overrides]
    if not unknown:
        return {element: overrides[element] for element in elements}

    # Smallest first: rank is a property of the compositions, so there is no
    # reason to pay for a 200 MB density when a 1 MB one adds the same row.
    order = sorted(records,
                   key=lambda r: os.path.getsize(r.files["CHGCAR"]))

    rows, targets, used = [], [], []
    for record in order:
        composition = compositions[record.identifier]
        row = [composition.get(element, 0) for element in unknown]
        if not any(row):
            continue                    # fully covered by the overrides
        trial = np.array(rows + [row], dtype=float)
        if np.linalg.matrix_rank(trial) <= len(rows):
            continue                    # adds no information

        electrons = _electron_count(record.files["CHGCAR"])
        # Whatever the overrides already account for is not for the fit to find.
        known = sum(overrides.get(element, 0.0) * count
                    for element, count in composition.items())
        rows.append(row)
        targets.append(electrons - known)
        used.append(record.identifier)
        log(f"      {record.identifier}: {composition} -> "
            f"{electrons:.4f} electrons")
        if len(rows) == len(unknown):
            break

    matrix = np.array(rows, dtype=float).reshape(len(rows), len(unknown))
    if np.linalg.matrix_rank(matrix) < len(unknown):
        raise ValueError(
            f"The valence charges of {unknown} cannot be determined from these "
            f"{len(records)} materials: their compositions span only "
            f"{np.linalg.matrix_rank(matrix)} of {len(unknown)} independent "
            f"directions, so infinitely many charge assignments reproduce every "
            f"electron count equally well. Add materials of other "
            f"stoichiometries, or pass the known charges as overrides."
        )

    solution, *_ = np.linalg.lstsq(matrix, np.array(targets, dtype=float),
                                   rcond=None)
    residual = np.abs(matrix @ solution - np.array(targets))

    charges = dict(overrides)
    rounded = np.rint(solution)
    integral = np.max(np.abs(solution - rounded)) < tolerance
    for element, value, integer in zip(unknown, solution, rounded):
        charges[element] = float(integer if integral else value)

    log(f"      valence charges from {len(used)} densities: "
        + ", ".join(f"{e}={charges[e]:g}" for e in elements)
        + (f"  (max residual {residual.max():.2e} e)" if residual.size else ""))
    if not integral:
        log("      note: the solution is not integral, which a valence charge "
            "should be. Check that every file is a pseudopotential CHGCAR.")
    return charges


# ---------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------- #
class MPChargeDensityDataset(MixedFieldDataset):
    r"""
    ``(V_ext, rho)`` pairs served straight from a Materials Project download.

    A :class:`~poraque.data.dataset.MixedFieldDataset` pinned to one bulk
    archive: the external potential is synthesised from the structure each
    ``CHGCAR`` carries in its header, nothing but the density files is needed
    on disk, and they stay compressed.

    Use this when the download *is* the dataset. To train on it alongside local
    calculations, pass both directories to
    :class:`~poraque.data.dataset.MixedFieldDataset` instead — it detects each
    one and reads this same layout through the same code.

    Parameters
    ----------
    root : str or pathlib.Path
        The download directory, e.g. ``data/MP/chgcar`` (``data/MP`` also
        resolves).
    task : str or TaskSpec, optional
        Only ``"ext2chg"`` is available; see :func:`available_tasks`.
    resolution : int, optional
        Longest grid axis after spectral downsampling. ``None`` keeps the
        native MP grid, which is 48³ upwards and rarely what a first training
        run wants.
    charges : dict, optional
        ``{element: Z_val}``. Inferred from the densities by
        :func:`infer_valence_charges` when omitted.
    sigma : float or dict, optional
        Gaussian pseudo-ion width in Å. Defaults to
        :data:`poraque.fields.external.DEFAULT_SIGMA`, since no ``POTCAR`` core
        radius is available to derive one from.
    gaussian_blur : float, optional
        Width in Å of a blur applied to the finished potential.
    blur_method : {"spectral", "ndimage"}, optional
        How that blur is applied.
    spin : bool, optional
        ``False`` (default) reads the total density only — one channel,
        matching what the rest of the pipeline caches. ``True`` also loads the
        magnetisation block, giving a two-channel target. MP's static runs are
        spin-polarised, so both are available; for the non-magnetic metals that
        dominate the metallic chemical spaces the second channel is numerically
        zero and costs capacity for nothing.
    cache : bool, optional
        Keep decoded fields in memory. Worth enabling with ``resolution`` set,
        since the potential is otherwise recomputed on every epoch.
    **kwargs
        Passed to :class:`~poraque.data.dataset.MixedFieldDataset`.

    Examples
    --------
    >>> from poraque.data import MPChargeDensityDataset
    >>> data = MPChargeDensityDataset("data/MP/chgcar", resolution=32)  # doctest: +SKIP
    >>> sample = data[0]                                                # doctest: +SKIP
    >>> sample["input"].shape, sample["target"].shape                   # doctest: +SKIP
    (torch.Size([1, 32, 32, 32]), torch.Size([1, 32, 32, 32]))
    """

    def __init__(self, root, task="ext2chg", **kwargs):
        from ..ml.tasks import resolve_task

        task = resolve_task(task)
        if task.name not in available_tasks():
            raise ValueError(
                f"A Materials Project download cannot serve {task.name!r}: it "
                f"provides {list(AVAILABLE_FIELDS)} and this task needs "
                f"{list(task.required_files)}. MP publishes no kinetic energy "
                f"density, so only {available_tasks()} is trainable on it."
            )

        # Resolve `data/MP` to `data/MP/chgcar` before the source sees it, and
        # fail with a message naming the downloader when the directory is empty
        # -- the generic detector would only say it does not recognise it.
        if kwargs.get("materials") is None:
            root = os.path.dirname(
                discover_mp_chgcars(root)[0].files["CHGCAR"])

        kwargs.setdefault("format", "bulk")
        super().__init__(root, task, **kwargs)

    @property
    def charges(self):
        """``{element: Z_val}`` in force, inferred from the densities if needed."""
        return self.sources[0].charges


# ---------------------------------------------------------------------- #
# Cache construction
# ---------------------------------------------------------------------- #
def build_mp_cache(root, cache, resolution=32, charges=None, sigma=None,
                   gaussian_blur=None, blur_method="spectral", log=None,
                   limit=None):
    """
    Write an MP download out as the per-material layout the trainer reads.

    A thin wrapper over :func:`~poraque.data.cache.build_field_cache` that
    resolves ``data/MP`` to ``data/MP/chgcar`` and states the caveat the
    potential comes with. Each density becomes
    ``<cache>/<material_id>/{EXTCAR,CHGCAR}``, spectrally downsampled to
    ``resolution``, which is the same layout and the same reduction the
    calculation path produces — so everything downstream works on MP data with
    no further changes.

    Parameters
    ----------
    root : str or pathlib.Path
        The download directory.
    cache : str or pathlib.Path
        Destination.
    resolution : int, optional
        Longest grid axis after downsampling.
    charges : dict, optional
        ``{element: Z_val}``; inferred from the densities when omitted.
    sigma : float or dict, optional
        Gaussian pseudo-ion width in Å.
    gaussian_blur : float, optional
        Blur width in Å for the potential.
    blur_method : str, optional
        ``"spectral"`` or ``"ndimage"``.
    log : callable, optional
        Receives one line per material.
    limit : int, optional
        Build at most this many materials, smallest file first.

    Returns
    -------
    str
        The cache directory.
    """
    from .cache import build_field_cache

    emit = log or (lambda *_: None)
    records = discover_mp_chgcars(root)
    directory = os.path.dirname(records[0].files["CHGCAR"])

    charges = charges or infer_valence_charges(records, log=emit)
    emit("  valence charges: "
         + ", ".join(f"{element}={value:g}"
                     for element, value in sorted(charges.items())))
    emit("  NOTE: no POTCAR ships with an MP charge density, so V_ext is the "
         "Gaussian")
    emit("  pseudo-ion model rather than the tabulated VASP local potential. "
         "See")
    emit("  poraque.data.mp_dataset for what that does and does not license.")

    return build_field_cache(
        directory, cache, resolution=resolution, format="bulk",
        fields=("EXTCAR", "CHGCAR"), charges=charges, sigma=sigma,
        gaussian_blur=gaussian_blur, blur_method=blur_method, limit=limit,
        log=emit,
    )
