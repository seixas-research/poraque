# -*- coding: utf-8 -*-
# file: mp_dataset.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Train on Materials Project charge densities.

A ``poraque-mp`` download is **one directory per material, named by its id**,
each holding that material's density and the VASP inputs reconstructed beside
it::

    data/MP/
        manifest.json  manifest.csv
        mp-124/CHGCAR.gz  INCAR  KPOINTS  POSCAR  mp.json
        mp-126/fields.h5  INCAR  KPOINTS  POSCAR  mp.json    # --hdf5

That is the same shape as a VASP run tree and a prepared cache, so the generic
reader — :class:`~poraque.data.sources.CalculationSource` — handles it with no
special case, and a download can be one entry in ``data.data_paths`` beside
local calculations. What this module adds is what is specific to the archive:

**The files are compressed.** They are read in place, gzip and all, by
:mod:`poraque.fields.io.compressed`. Nothing is ever expanded on disk.

**There is no external potential and no** ``POTCAR``. A ``CHGCAR`` carries
its own structure in its first lines, so :math:`V_{\rm ext}` is built from the
density alone — exactly as the ordinary pipeline does, which computes the
potential rather than reading it so that training and inference see the same
construction. Without a pseudopotential library (``potcar_dir``) that
construction is the **Gaussian pseudo-ion model**, which differs from VASP's
tabulated local potential by a relative :math:`L_2` of order 0.1; the map
learned is then *model potential* :math:`\to` *DFT density*, self-consistent
but not the VASP :math:`V_{\rm ext}`. Anything comparing such a model with one
trained on tabulated potentials has to say so.

**The valence charges are unknown**, and are recovered from the densities by
:func:`~poraque.data.sources.infer_valence_charges`: a ``CHGCAR`` integrates
to the valence electron count of its own cell, one linear equation per material
in the per-element charges. On the Pt-Pd-Ni set this returns 11, 11 and 10 to
within :math:`10^{-6}` — the POTCAR values, read off the densities.

**There is no** :math:`\tau` **and no energy.** ``chg2tau`` cannot be trained
on this data and :func:`available_tasks` says so rather than failing later;
``reference_energy`` is ``None`` throughout, which
:class:`~poraque.ml.data.FieldPairDataset` already treats as "not available"
rather than as zero.

Two entry points, one implementation:

:class:`MPChargeDensityDataset`
    Serves ``(V_ext, rho)`` pairs straight out of the download directory.
:func:`build_mp_cache`
    Writes the same pairs, spectrally downsampled, into the per-material cache
    layout the standard trainer reads.

Both are thin specialisations of the generic machinery, kept for the message a
Materials Project user needs when a directory is empty: the one that names the
downloader.
"""

from pathlib import Path

from .dataset import MixedFieldDataset
from .sources import INTEGER_TOLERANCE, infer_valence_charges  # noqa: F401

#: Fields an MP download supplies. Everything else — ``TAUCAR``, ``OUTCAR``,
#: absolute energies — is absent by construction, and the pipeline is expected
#: to work around that rather than to demand it.
AVAILABLE_FIELDS = ("CHGCAR",)


def discover_mp_chgcars(root, pattern=""):
    """
    Find the charge densities in a Materials Project download directory.

    A download is one directory per material, named by its id, holding that
    material's ``CHGCAR`` — the same shape as every other dataset, so this is
    :func:`~poraque.data.sources.resolve_source` with a message that names the
    downloader when the directory is empty. The generic one would only say it
    found no materials, which is true and unhelpful in the one place where the
    cause is always "nothing has been downloaded yet".

    Parameters
    ----------
    root : str or pathlib.Path
        The download directory, e.g. ``data/MP``.
    pattern : str, optional
        Prefix filter on the material directories. Empty takes all of them.

    Returns
    -------
    list of MaterialRecord
        One record per material, sorted by identifier.

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist or holds no material.
    """
    from .sources import resolve_source

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No such directory: {root}")

    try:
        records = resolve_source(str(root), pattern=pattern).discover()
    except ValueError:
        records = []

    if not records:
        raise FileNotFoundError(
            f"No materials under {root}. A download is one directory per "
            f"material, each holding its CHGCAR. Download some first: "
            f"poraque-mp --elements Pt Pd Ni --outdir {root}"
        )
    return records


def available_tasks():
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
# Dataset
# ---------------------------------------------------------------------- #
class MPChargeDensityDataset(MixedFieldDataset):
    r"""
    ``(V_ext, rho)`` pairs served straight from a Materials Project download.

    A :class:`~poraque.data.dataset.MixedFieldDataset` pinned to one download
    directory: the external potential is synthesised from the structure each
    ``CHGCAR`` carries in its header, nothing but the density files is needed
    on disk, and they stay compressed.

    Use this when the download *is* the dataset. To train on it alongside local
    calculations, pass both directories to
    :class:`~poraque.data.dataset.MixedFieldDataset` instead — it reads this
    same layout through the same code.

    Parameters
    ----------
    root : str or pathlib.Path
        The download directory, e.g. ``data/MP``.
    task : str or TaskSpec, optional
        Only ``"ext2chg"`` is available; see :func:`available_tasks`.
    resolution : int, optional
        Longest grid axis after spectral downsampling. ``None`` keeps the
        native MP grid, which is 48³ upwards and rarely what a first training
        run wants.
    charges : dict, optional
        ``{element: Z_val}``. Inferred from the densities by
        :func:`~poraque.data.sources.infer_valence_charges` when omitted.
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
    >>> data = MPChargeDensityDataset("data/MP", resolution=32)        # doctest: +SKIP
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

        # Checked here, so an empty download fails with a message naming the
        # downloader rather than the generic "no materials under ...".
        if kwargs.get("materials") is None:
            discover_mp_chgcars(root)

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
    names the downloader when the directory is empty and states the caveat the
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
        root, cache, resolution=resolution,
        fields=("EXTCAR", "CHGCAR"), charges=charges, sigma=sigma,
        gaussian_blur=gaussian_blur, blur_method=blur_method, limit=limit,
        log=emit,
    )
