# -*- coding: utf-8 -*-
# file: sources.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
One interface over every layout training data arrives in.

Three shapes of directory show up in practice, and they have almost nothing in
common on disk::

    A. a DFT calculation            B. a bulk density archive     C. a prepared cache
    -------------------            ---------------------         -------------------
    runs/struct_000/               data/MP/                      cache/res32/
        POSCAR  INCAR  POTCAR          manifest.csv                  mp-124/
        CHGCAR  TAUCAR  OUTCAR         mp-124/CHGCAR.gz                  EXTCAR  CHGCAR
    runs/struct_001/                   mp-81/CHGCAR.gz               struct_000/
        ...                            mp-126/fields.h5                  EXTCAR  CHGCAR  TAUCAR

B and C look alike on purpose — one directory per material, named by it — and
the difference between them is the *content*, not the shape: an archive
publishes a density and nothing else, so its external potential has to be
built; a cache holds the fields already prepared, so nothing is computed. B
also reads loose ``CHGCAR_mp-124.gz`` files sitting directly in the root, for
an archive that arrives as a pile of files rather than a tree.

What they *do* have in common is what the operator needs: for each material, a
grid and up to three fields on it. That is the whole of this module's contract.
A :class:`MaterialSource` recognises one layout, enumerates its materials, says
which fields it can supply, and produces them — computing what is absent
wherever it honestly can and declining where it cannot.

The differences that remain are real physics, not plumbing:

===================  =========================  ==============================
Layout               :math:`V_{\rm ext}` from   :math:`\tau`
===================  =========================  ==============================
calculation          POSCAR + POTCAR tables,    read from ``TAUCAR`` when
                     **exact**                  the run wrote one
bulk densities       the CHGCAR's own header,   never — no archive publishes it
                     Gaussian model
prepared cache       read from ``EXTCAR``       read from ``TAUCAR`` if cached
===================  =========================  ==============================

:func:`detect_source` picks the class; :func:`resolve_source` builds an
instance, honouring an explicit ``format`` when the caller knows better than
the sniffer. :class:`~poraque.data.dataset.MixedFieldDataset` is what puts
several of them behind one PyTorch ``Dataset``.

Adding a layout means writing one class and calling :func:`register_source` —
nothing downstream changes.
"""

import os
import warnings
from abc import ABC, abstractmethod

import numpy as np

from ..fields import (
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    KineticEnergyDensity,
)
from ..ml.data import MaterialRecord

#: Field name -> the :class:`~poraque.fields.ScalarField` subclass reading it.
FIELD_CLASSES = {
    "EXTCAR": ExternalPotential,
    "CHGCAR": ChargeDensity,
    "TAUCAR": KineticEnergyDensity,
}

#: Filename prefix marking a standalone density in a bulk archive, matched
#: case-insensitively. Compression suffixes are stripped before matching.
BULK_PREFIXES = ("CHGCAR",)

#: Files whose presence marks a directory as a DFT calculation rather than an
#: archive of outputs.
#:
#: VASP's names, kept as the fallback so this constant still reads as it always
#: did; the live list comes from :func:`calculation_markers`, which asks the
#: registered readers. Hard-coding ``POSCAR``/``CONTCAR`` meant a Quantum
#: ESPRESSO or GPAW directory was not recognised as a calculation at all, so
#: :class:`CalculationSource` never even reached its reader — the one class a
#: new code is supposed to need.
CALCULATION_MARKERS = ("POSCAR", "CONTCAR")


def calculation_markers():
    """
    Structure filenames of every registered code, VASP's first.

    Each :class:`~poraque.fields.io.base.CalculationReader` already declares
    :attr:`structure_files`, which is exactly this information. Deriving the
    markers from the registry means registering a reader is genuinely all that
    a new code requires, rather than a reader *and* an edit here.

    Returns
    -------
    tuple of str
    """
    from ..fields.io import _READERS

    names = list(CALCULATION_MARKERS)
    for reader_class in _READERS.values():
        names.extend(name for name in reader_class.structure_files
                     if name not in names)
    return tuple(names)


def _is_density_file(entry, prefixes=BULK_PREFIXES):
    """
    Whether a filename names a standalone density.

    Two conditions, and the second is what keeps a metadata file out. Once the
    compression suffix is stripped, the name must *begin* with the prefix and
    have **nothing left that looks like an extension** — so ``CHGCAR_mp-124.gz``
    is a density and ``chgcar_estimate.csv``, sitting in the same download, is
    not. Matching on the prefix alone would hand the CSV to the volumetric
    parser and fail somewhere far less obvious.
    """
    from ..fields.io.compressed import strip_compression_suffix

    stem = strip_compression_suffix(entry)
    if not any(stem.upper().startswith(prefix) for prefix in prefixes):
        return False
    return not os.path.splitext(stem)[1]


def _lone_density(directory, prefixes=BULK_PREFIXES):
    """
    The single density file in a per-material directory, or ``None``.

    This is what distinguishes ``data/MP/mp-124/CHGCAR.gz`` — one material,
    named by its directory — from ``data/MP/chgcar/``, the flat archive that
    holds every material at once. **Exactly one** density is the test: a
    directory with two is not one material's, whatever it is, and guessing
    which of them to train on is not a decision to make silently.

    A calculation directory is excluded outright. It also holds one density,
    but it holds the *inputs* as well, and reading it here would build the
    external potential from a model when the ``POTCAR`` to do it exactly is
    sitting beside the file. An archive entry is *not* excluded, whatever
    input files ``poraque-mp`` reconstructed into it — those are for VASP, not
    for training, and only the density is recorded.

    Returns
    -------
    str or None
        Path to the density, in whichever storage layout the directory uses:
        a ``CHGCAR`` (compressed or not) or ``fields.h5::CHGCAR``.
    """
    if not os.path.isdir(directory) or _is_calculation_directory(directory):
        return None

    entries = [entry for entry in sorted(os.listdir(directory))
               if _is_density_file(entry, prefixes)
               and os.path.isfile(os.path.join(directory, entry))]
    if len(entries) == 1:
        return os.path.join(directory, entries[0])
    if entries:
        return None

    store = _hdf5_store(directory)
    if store is None:
        return None

    from ..fields.hdf5 import field_names, join_target

    try:
        available = set(field_names(store))
    except (OSError, ValueError):
        return None
    return join_target(store, "CHGCAR") if "CHGCAR" in available else None


def _hdf5_store(directory):
    """
    The path of a material's HDF5 field store, or ``None``.

    Any ``.h5``/``.hdf5`` in the directory counts, not only ``fields.h5``: a
    cache is often produced once and renamed, and refusing to read
    ``mp-124.h5`` because of its name would be a rule with no reason behind it.
    A directory with several is served by the first in sorted order, which is
    deterministic and is the only property that matters when it happens.
    """
    from ..fields.hdf5 import HDF5_SUFFIXES

    if not os.path.isdir(directory):
        return None
    for entry in sorted(os.listdir(directory)):
        if entry.lower().endswith(HDF5_SUFFIXES):
            return os.path.join(directory, entry)
    return None


def _is_archive_entry(directory):
    r"""
    Whether ``directory`` is one entry of a published density archive.

    ``poraque-mp`` reconstructs a ``POSCAR``, ``INCAR`` and ``KPOINTS`` beside
    each downloaded density, so the directory can be handed straight to VASP.
    That makes it indistinguishable *by its files* from a calculation
    directory — the marker is what tells them apart, and it is written by the
    thing that knows.

    The distinction is not bookkeeping. Read as a calculation, an archive entry
    loses two things at once: the valence charges
    :class:`BulkDensitySource` infers from the densities, which are the only
    source of :math:`Z^{\rm val}` when no ``POTCAR`` is present; and the
    reconstructed ``POSCAR`` becomes a *training input* rather than a
    convenience — and it was written from the density's own header, so a
    pipeline reading it would be taking the long way round to the same
    geometry while pretending it came from somewhere independent.
    """
    from .materials_project import ENTRY_MARKER

    return os.path.isfile(os.path.join(directory, ENTRY_MARKER))


def _is_calculation_directory(path):
    """
    Whether ``path`` holds the input files of a DFT run, in any code.

    An archive entry is excluded however many input files were reconstructed
    into it; see :func:`_is_archive_entry`.
    """
    if not any(os.path.exists(os.path.join(path, name))
               for name in calculation_markers()):
        return False
    return not _is_archive_entry(path)


def _subdirectories(root):
    """Immediate subdirectories of ``root``, sorted."""
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, entry) for entry in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, entry))]


# ---------------------------------------------------------------------- #
# The contract
# ---------------------------------------------------------------------- #
class MaterialSource(ABC):
    """
    One directory layout, and how to get fields out of it.

    Parameters
    ----------
    root : str or pathlib.Path
        The directory this instance is bound to.
    **options
        Layout-specific settings; unknown ones are ignored, so a single
        options dict can be handed to a mixture of sources without each caller
        having to know which keys apply where.

    Attributes
    ----------
    name : str
        Short identifier used in configs, logs and error messages.
    """

    name = "source"

    def __init__(self, root, **options):
        self.root = str(root)
        self.options = dict(options)

    def __repr__(self):
        return f"{type(self).__name__}({self.root!r})"

    # -- recognition ---------------------------------------------------- #
    @classmethod
    @abstractmethod
    def detect(cls, root):
        """Whether ``root`` is laid out the way this source expects."""

    # -- enumeration ---------------------------------------------------- #
    @abstractmethod
    def discover(self):
        """
        Every material under :attr:`root`.

        Returns
        -------
        list of MaterialRecord
            Sorted by identifier, each carrying ``source=self`` so a dataset
            built from several sources can dispatch per material.
        """

    def _record(self, identifier, directory, files):
        """One record, bound to this source."""
        return MaterialRecord(identifier, str(directory), files=dict(files),
                              source=self)

    # -- fields --------------------------------------------------------- #
    @abstractmethod
    def provides(self, record):
        """
        Field names this source can supply for ``record``.

        A field that is *computed* counts as provided — the external potential
        is never on disk in a calculation directory and is always available
        from one. A field that is simply absent does not, and that is what lets
        a task be skipped rather than crash halfway through an epoch.

        Returns
        -------
        tuple of str
        """

    @abstractmethod
    def reference_file(self, record):
        """
        The file whose header defines this material's grid and structure.

        Every field of one material is placed on this mesh, which is what makes
        an input and a target genuinely aligned rather than merely
        same-shaped.
        """

    def grid(self, record):
        """The shared :class:`~poraque.fields.FieldGrid` for ``record``."""
        return FieldGrid.from_file(self.reference_file(record))

    def shape(self, record):
        """
        The native grid shape, read from a header alone.

        Cheap by construction: bucketing a dataset by shape must not require
        decoding every field in it.
        """
        from ..ml.data import _peek_shape

        return tuple(_peek_shape(self.reference_file(record)))

    @abstractmethod
    def read(self, record, name, grid, spin=False):
        """
        Produce one field.

        Parameters
        ----------
        record : MaterialRecord
        name : str
            ``"EXTCAR"``, ``"CHGCAR"`` or ``"TAUCAR"``.
        grid : FieldGrid
            The shared mesh, imposed on the result rather than re-derived, so
            an inconsistency raises instead of passing silently.
        spin : bool, optional
            Read ``CHGCAR`` as a two-channel
            :class:`~poraque.fields.SpinDensity`.

        Returns
        -------
        ScalarField or SpinDensity
        """

    def is_spin_polarized(self, record):
        """Whether this material's density file carries a magnetisation block."""
        from ..fields import is_spin_polarized

        path = record.files.get("CHGCAR")
        return bool(path) and is_spin_polarized(path)

    def describe(self):
        """One line naming what this source is and where it points."""
        return f"{self.name}: {self.root}"

    # -- shared helpers ------------------------------------------------- #
    def _read_density(self, path, grid, spin):
        r"""
        Read a density file as one channel or as a spin pair.

        A dataset resolves to spin as a whole (see
        :func:`~poraque.data.cache._resolve_spin`), because one operator has
        one channel count. So a mixture is legal, and an ``ISPIN = 1`` member
        of a spin dataset is read as :math:`(\rho, m = 0)` rather than
        refused: that is a true statement about a non-magnetic calculation, and
        it is what makes a two-channel operator a strict generalisation of a
        one-channel one.

        :meth:`~poraque.fields.SpinDensity.read` still refuses a single-block
        file on its own, which is right for a *direct* call — there the zero
        magnetisation would be a claim the caller had not checked. Here it has
        been checked, one level up, against every material in the set.
        """
        if not spin:
            return ChargeDensity.read(path, grid=grid)

        from ..fields import SpinDensity, is_spin_polarized

        if is_spin_polarized(path):
            return SpinDensity.read(path, grid=grid)

        density = ChargeDensity.read(path, grid=grid)
        return SpinDensity(density.data, np.zeros_like(density.data),
                           density.grid, density.structure,
                           metadata={**density.metadata, "ispin": 1})

    def _blur(self, field):
        """Apply the configured Gaussian blur to a computed potential."""
        width = self.options.get("gaussian_blur")
        if not width:
            return field
        return field.smooth(width, self.options.get("blur_method", "spectral"))

    # -- POTCAR library -------------------------------------------------- #
    def library_potcar(self, structure):
        r"""
        The ``POTCAR`` for ``structure``, assembled from ``potcar_dir``.

        Data that carries a structure but no pseudopotentials — a Materials
        Project charge density, a run whose ``POTCAR`` was stripped — can still
        have the **exact** local potential reconstructed, provided the
        pseudopotentials it was computed with are available separately. That is
        what a library is for.

        Returns ``None``, rather than raising, when no library is configured or
        when it cannot serve this structure. The caller then falls back to the
        Gaussian pseudo-ion model, which is worse but not wrong; a hard failure
        here would turn a quality difference into an outage.

        Parameters
        ----------
        structure : Structure

        Returns
        -------
        Potcar or None
            Entries in the structure's species order, every one carrying a
            complete ``local part`` table.
        """
        directory = self.options.get("potcar_dir")
        if not directory:
            return None

        from ..fields.vasp.potcar import Potcar

        elements = tuple(dict.fromkeys(structure.elements))
        entries = [self._library_entry(element) for element in elements]
        if any(entry is None for entry in entries):
            return None
        return Potcar(entries)

    def _library_entry(self, element):
        """
        One species' library entry, or ``None`` with a warning.

        Cached and warned about **per element**, not per composition. A
        five-element space has thirty-one compositions and only five facts to
        report; warning per composition would bury the five in repetition.
        """
        cache = self.__dict__.setdefault("_potcar_cache", {})
        if element in cache:
            return cache[element]

        from ..fields.vasp.potcar import Potcar

        directory = self.options["potcar_dir"]
        entry = None
        try:
            entry = Potcar.from_library(directory, [element])[0]
        except (FileNotFoundError, ValueError, OSError) as error:
            warnings.warn(
                f"No usable {element!r} POTCAR in the library {directory}: "
                f"{error} Using the Gaussian pseudo-ion model for every "
                f"structure containing {element}.",
                RuntimeWarning, stacklevel=4,
            )

        # A truncated table parses without error but cannot be splined onto the
        # PSGMAX mesh, so gate on completeness rather than on presence.
        if entry is not None and not entry.has_local_table:
            warnings.warn(
                f"The {element!r} POTCAR in {directory} carries no complete "
                f"local-potential table; using the Gaussian pseudo-ion model "
                f"for every structure containing {element}.",
                RuntimeWarning, stacklevel=4,
            )
            entry = None

        cache[element] = entry
        return entry

    def potential_model(self):
        """
        Which construction this source uses for :math:`V_{\\rm ext}`.

        Two possible answers, ``"tabulated"`` and ``"gaussian"``, and the
        difference is the single most consequential property of a training set:
        they are different physical quantities, not different approximations to
        one. It is reported in the log, and it is what
        :class:`~poraque.data.dataset.MixedFieldDataset` compares across
        sources before deciding whether a mixture needs a warning.

        Returns
        -------
        str
        """
        return "tabulated" if self.options.get("potcar_dir") else "gaussian"


# ---------------------------------------------------------------------- #
# A: a DFT calculation directory
# ---------------------------------------------------------------------- #
class CalculationSource(MaterialSource):
    r"""
    Classic layout: one directory per material, holding a DFT run.

    Recognised by a ``POSCAR`` (or ``CONTCAR``) — either in ``root`` itself,
    which is then a single material, or in its subdirectories, which are then
    the dataset.

    The external potential is **computed** from the inputs, never read from any
    ``EXTCAR`` the directory happens to contain. That is deliberate: the
    training input has to be exactly what
    :class:`~poraque.fields.ExternalPotential` produces at inference time,
    when no such file exists. With a ``POTCAR`` present the tabulated local
    pseudopotential is used, which reproduces VASP's own field to a relative
    :math:`2\times10^{-5}`.

    ``TAUCAR`` is read when the run wrote one and reported absent when it did
    not, rather than demanded. A directory with a density and no kinetic energy
    density is a perfectly good ``ext2chg`` dataset, and refusing it would turn
    the commonest kind of archive into an error.

    Options
    -------
    code : str
        DFT code name, or ``"auto"`` to detect it per directory.
    pattern : str
        Keep only subdirectories whose name starts with this. The usual reason
        is a sibling directory of isolated-atom references that must not be
        trained on.
    potcar_dir : str
        A ``POTCAR`` library, used only for runs that have no ``POTCAR`` of
        their own — routinely stripped from archived calculations for licensing
        reasons. It restores the *exact* tabulated construction rather than
        dropping to a model form factor.
    charges : dict
        ``{element: Z_val}`` overriding the pseudopotential valence charges.
        The last resort when neither the run nor a library supplies them: it
        selects the Gaussian model. The same option means the same thing for
        :class:`BulkDensitySource`, so one mapping covers a mixture.
    sigma, gaussian_blur, blur_method
        Passed to the potential construction.
    """

    name = "vasp"

    @classmethod
    def detect(cls, root):
        if _is_calculation_directory(root):
            return True
        return any(_is_calculation_directory(child)
                   for child in _subdirectories(root))

    def _reader(self, directory):
        from ..fields.io import resolve_reader

        return resolve_reader(directory, self.options.get("code", "auto"))

    def discover(self):
        pattern = self.options.get("pattern") or ""
        if _is_calculation_directory(self.root):
            directories = [self.root]
        else:
            directories = [child for child in _subdirectories(self.root)
                           if os.path.basename(child).startswith(pattern)
                           and _is_calculation_directory(child)]

        records = []
        for directory in directories:
            reader = self._reader(directory)
            files = {}
            for kind, name in (("density", "CHGCAR"), ("kinetic", "TAUCAR")):
                path = reader.field_path(directory, kind)
                if os.path.exists(path):
                    files[name] = path
            if "CHGCAR" not in files:
                # No density, no material: every field of a run is placed on
                # the density's mesh, and there is nothing to learn without it.
                continue
            records.append(self._record(os.path.basename(os.path.normpath(directory)),
                                        directory, files))
        return records

    def provides(self, record):
        # EXTCAR is computed from the inputs, so it is always available here.
        return ("EXTCAR",) + tuple(name for name in ("CHGCAR", "TAUCAR")
                                   if name in record.files)

    def reference_file(self, record):
        return record.files["CHGCAR"]

    def potential_model(self):
        """
        ``"tabulated"`` when the runs, or a library, supply pseudopotentials.

        Checked against the discovered runs rather than assumed: a calculation
        archive normally carries its own ``POTCAR``\\ s and needs no library,
        but a stripped one falls back to the Gaussian model unless
        ``potcar_dir`` covers it.
        """
        if self.options.get("potcar_dir"):
            return "tabulated"
        records = self.discover()
        if records and all(os.path.exists(os.path.join(r.directory, "POTCAR"))
                           for r in records):
            return "tabulated"
        return "gaussian"

    def geometry(self, record):
        r"""
        The geometry :math:`V_{\rm ext}` must be built at.

        **The density's own header wins.** A volumetric file embeds the
        structure it was computed at, so for the purpose of pairing an input
        potential with a target density there is no more authoritative answer,
        and it is the same geometry :meth:`grid` already adopts. The
        directory's structure file is the fallback, used when the format
        carries no geometry (see
        :meth:`~poraque.fields.io.base.CalculationReader.read_field_structure`)
        or when the record has no density.

        Regression this exists for: ``structure_files`` is
        ``("POSCAR", "CONTCAR")`` and the first wins, which is correct for a
        static run and wrong for a **relaxation** -- there the ``POSCAR`` is the
        geometry the run started from while the ``CHGCAR`` holds the density at
        the geometry it ended at. One platinum slab differed by 0.12 Ang rms,
        which put a 2.5 % relative :math:`L^2` error into its external
        potential and made it the worst structure in the training set by a
        factor of 23, with nothing anywhere reporting a problem.

        Parameters
        ----------
        record : MaterialRecord

        Returns
        -------
        Structure
        """
        reader = self._reader(record.directory)
        density = record.files.get("CHGCAR")
        if density is not None:
            structure = reader.read_field_structure(density)
            if structure is not None:
                self._warn_on_geometry_drift(record, reader, structure)
                return structure

        # `reader.read_structure`, not `Poscar.from_file`: the reader is
        # already resolved for this directory and the neutral contract is what
        # makes a non-VASP run readable here at all. Parsing the structure file
        # with VASP's parser regardless of which code wrote it defeats the
        # abstraction one line after resolving it.
        return reader.read_structure(record.directory)

    #: Largest per-atom displacement, in Angstrom, between a density's own
    #: geometry and the directory's structure file that is written off as
    #: formatting. A ``CHGCAR`` header prints six decimals of a fractional
    #: coordinate, ~1e-5 Ang on a 10 Ang cell; anything at 1e-3 is real.
    GEOMETRY_TOLERANCE = 1e-3

    def _warn_on_geometry_drift(self, record, reader, structure):
        """
        Say so when the density and the structure file describe different
        geometries.

        The potential is built correctly either way -- that is what
        :meth:`geometry` decides -- but a directory whose input and output
        geometries disagree is not self-consistent, and the reason is usually
        that a relaxation moved the atoms and left the ``POSCAR`` behind. That
        is worth one line, because everything else about such a run looks
        entirely normal.
        """
        try:
            reference = reader.read_structure(record.directory)
        except (FileNotFoundError, NotImplementedError, OSError, ValueError):
            return

        try:
            cell = np.asarray(structure.cell, dtype=float)
            delta = (np.asarray(structure.scaled_positions, dtype=float)
                     - np.asarray(reference.scaled_positions, dtype=float))
        except (AttributeError, TypeError, ValueError):
            return
        if delta.size == 0 or delta.shape != (structure.natoms, 3):
            # Different atom counts are a different problem, and not one a
            # drift measurement can describe.
            return

        # Minimum image, so an atom reported at 0.0 in one file and 1.0 in the
        # other is not mistaken for a displacement of a whole lattice vector.
        drift = float(np.abs((delta - np.round(delta)) @ cell).max())
        if drift <= self.GEOMETRY_TOLERANCE:
            return

        log = self.options.get("log")
        if log:
            log(f"      {record.identifier}: the density's geometry differs "
                f"from "
                f"{os.path.basename(reader.structure_path(record.directory))} "
                f"by up to {drift:.3f} Ang -- a relaxation whose structure "
                f"file was left at the starting geometry. Building V_ext at "
                f"the density's geometry, which is the one it was computed "
                f"at.")

    def _external_potential(self, record, grid):
        r"""
        Build :math:`V_{\rm ext}` from the calculation's own inputs.

        A ``POTCAR`` in the directory is used as it stands, which is the exact
        tabulated route and the normal case. When the run has none -- routinely
        stripped from archived calculations for licensing -- the ``potcar_dir``
        library stands in for it, which keeps the *same* construction rather
        than silently dropping to a model form factor.

        Both routes are handed the geometry :meth:`geometry` resolves, so the
        potential is built at the positions the density was computed at.
        """
        reader = self._reader(record.directory)
        has_own = os.path.exists(os.path.join(record.directory, "POTCAR"))
        structure = self.geometry(record)

        if not has_own:
            potcar = self.library_potcar(structure)
            if potcar is not None:
                return ExternalPotential.from_potcar_tables(
                    structure, grid, potcar,
                    metadata={"code": reader.code,
                              "source": str(record.directory),
                              "derived_from": "density geometry + POTCAR library",
                              "potcar_dir": str(self.options["potcar_dir"])},
                )

        return ExternalPotential.from_calculation(
            record.directory, code=reader.code, grid=grid,
            sigma=self.options.get("sigma"),
            zval=self.options.get("charges"),
            structure=structure,
        )

    def read(self, record, name, grid, spin=False):
        if name == "EXTCAR":
            return self._blur(self._external_potential(record, grid))

        path = record.files.get(name)
        if path is None:
            raise FileNotFoundError(
                f"{record.identifier}: this calculation has no {name}. "
                f"It provides {list(self.provides(record))}."
            )
        if name == "CHGCAR":
            return self._read_density(path, grid, spin)
        return FIELD_CLASSES[name].read(path, grid=grid)


# ---------------------------------------------------------------------- #
# B: a bulk archive of standalone densities
# ---------------------------------------------------------------------- #
class BulkDensitySource(MaterialSource):
    r"""
    Bulk layout: standalone densities, one material each, nothing beside them.

    This is what a public archive ships — the Materials Project serves one
    ``CHGCAR`` per material and no inputs at all. Two arrangements are read,
    and a directory may hold both::

        data/MP/mp-124/CHGCAR.gz        one directory per material, named by it
        data/MP/mp-126/fields.h5        the same, stored as a field store
        archive/CHGCAR_mp-81.gz         standalone files, id in the filename

    The first is what :class:`~poraque.data.materials_project.MPDataFetcher`
    writes and is the layout to point a config at: a directory is what lets
    anything be added *beside* a density later, where a filename can only ever
    say one thing. The second is for an archive that arrives as loose files —
    unpacked from a tarball, or assembled by hand — which is a shape no amount
    of convention here controls.

    Compression is transparent, so nothing is ever expanded on disk.

    The structure comes from the density's own header — a ``CHGCAR`` carries
    its ``POSCAR`` in its first lines — and the external potential is built
    from it. The valence charges that needs are not in the archive either, and
    are recovered from the densities by
    :func:`~poraque.data.mp_dataset.infer_valence_charges`.

    .. important::

       With no ``POTCAR`` there is no tabulated local pseudopotential, so this
       source uses the **Gaussian pseudo-ion model**, whose residual against a
       reference VASP ``EXTCAR`` is of order 0.1 relative :math:`L_2`. A model
       trained here learns *model potential* :math:`\to` *DFT density*, which
       is self-consistent — inference builds the same potential — but is not
       the same map a :class:`CalculationSource` dataset teaches. Mixing the
       two in one training set mixes two definitions of the input field; see
       :class:`~poraque.data.dataset.MixedFieldDataset`, which says so out
       loud rather than letting it pass unnoticed.

    Options
    -------
    potcar_dir : str
        A ``POTCAR`` library. **This is what closes the gap above.** The
        archive supplies the structure and the library supplies the
        pseudopotentials, which together are everything the exact tabulated
        construction needs — so with it the caveat does not apply and the
        potential is VASP's own to a relative :math:`2\times10^{-5}`.
    charges : dict
        ``{element: Z_val}``. Taken from the library when one is configured,
        and otherwise inferred from the densities.
    prefixes : sequence of str
        Filename prefixes identifying a density.
    sigma, gaussian_blur, blur_method
        Passed to the Gaussian construction, which is only reached without a
        library.
    """

    name = "bulk"

    def __init__(self, root, **options):
        super().__init__(root, **options)
        self._charges = (dict(options["charges"])
                         if options.get("charges") else None)

    @staticmethod
    def _holds_flat_densities(root):
        """Whether ``root`` holds standalone density *files*."""
        return os.path.isdir(root) and any(
            _is_density_file(entry) and os.path.isfile(os.path.join(root, entry))
            for entry in os.listdir(root))

    @classmethod
    def _holds_material_directories(cls, root):
        """Whether ``root`` holds per-material *directories* of densities."""
        return any(_lone_density(child) for child in _subdirectories(root))

    @classmethod
    def _holds_densities(cls, root):
        """Whether ``root`` holds densities at all, in either arrangement."""
        return (cls._holds_flat_densities(root)
                or cls._holds_material_directories(root))

    @classmethod
    def detect(cls, root):
        if not os.path.isdir(root) or _is_calculation_directory(root):
            return False
        return cls._holds_densities(root)

    def discover(self):
        """
        Every material under the root, in whichever arrangement it uses.

        Both are scanned in one pass rather than one being chosen: an archive
        assembled by hand may well hold some of each, and a source that picked
        an arrangement and then ignored the other would drop materials without
        saying so. A material found both ways wins from its own directory,
        since that is the arrangement carrying the most information about it.
        """
        from ..fields.io.compressed import strip_compression_suffix

        prefixes = tuple(self.options.get("prefixes") or BULK_PREFIXES)
        records = {}
        for entry in sorted(os.listdir(self.root)):
            path = os.path.join(self.root, entry)
            if not os.path.isfile(path) or not _is_density_file(entry, prefixes):
                continue
            stem = strip_compression_suffix(entry)
            # `CHGCAR_mp-124` -> `mp-124`; a bare `CHGCAR` keeps its own name.
            identifier = stem.split("_", 1)[1] if "_" in stem else stem
            records[identifier] = self._record(identifier, self.root,
                                               {"CHGCAR": path})

        # `data/MP/mp-124/CHGCAR.gz`: the directory names the material, so the
        # file inside it does not have to, and a second field could be dropped
        # beside it tomorrow without renaming anything.
        for child in _subdirectories(self.root):
            density = _lone_density(child, prefixes)
            if density is None:
                continue
            identifier = os.path.basename(os.path.normpath(child))
            records[identifier] = self._record(identifier, child,
                                               {"CHGCAR": density})
        return [records[key] for key in sorted(records)]

    @property
    def charges(self):
        r"""
        ``{element: Z_val}`` in force.

        Three sources, in order:

        1. an explicit ``charges`` option;
        2. the ``ZVAL`` of each species in the ``potcar_dir`` library, which is
           where the number actually comes from;
        3. inference from the densities — a ``CHGCAR`` integrates to its cell's
           valence electron count, giving one linear equation per material in
           the per-element charges. This is what makes an archive usable with
           no pseudopotentials at all.

        Deferred rather than resolved in ``__init__``, because both (2) and (3)
        read files and a caller that only wants to enumerate the archive should
        never pay for that.
        """
        if self._charges is None:
            self._charges = self._resolve_charges()
        return self._charges

    def _resolve_charges(self):
        """Valence charges from the library where it has them, else inferred."""
        from .mp_dataset import infer_valence_charges

        records = self.discover()
        log = self.options.get("log")

        from_library = {}
        if self.options.get("potcar_dir"):
            from ..fields.vasp.volumetric import read_structure_header
            from ..fields.vasp.potcar import Potcar

            # Header reads only: a few hundred bytes per material.
            elements = sorted({element for record in records
                               for element in
                               read_structure_header(
                                   record.files["CHGCAR"]).elements})
            for element in elements:
                try:
                    entry = Potcar.from_library(
                        self.options["potcar_dir"], [element],
                        parse_tables=False)[0]
                except (FileNotFoundError, ValueError, OSError):
                    continue        # inference covers it below
                from_library[element] = float(entry.zval)

            if from_library and log:
                log("      valence charges from the POTCAR library: "
                    + ", ".join(f"{e}={z:g}"
                                for e, z in sorted(from_library.items())))

        try:
            return infer_valence_charges(records, overrides=from_library,
                                         log=log)
        except ValueError:
            # The compositions cannot determine what the library did not
            # supply. That is only fatal if something still needs a charge,
            # and the tabulated path does not -- so hand back what is known.
            if from_library:
                return from_library
            raise

    def provides(self, record):
        return ("EXTCAR", "CHGCAR")

    def reference_file(self, record):
        return record.files["CHGCAR"]

    def read(self, record, name, grid, spin=False):
        path = record.files["CHGCAR"]
        if name == "CHGCAR":
            return self._read_density(path, grid, spin)
        if name != "EXTCAR":
            raise FileNotFoundError(
                f"{record.identifier}: a bulk density archive holds only "
                f"{list(self.provides(record))}; it publishes no {name}. "
                f"Nothing can reconstruct one from a density alone."
            )
        return self._blur(self._external_potential(record, grid))

    def _external_potential(self, record, grid):
        r"""
        Build :math:`V_{\rm ext}` from the structure the density carries.

        Two constructions, and which one runs is the single most consequential
        thing about a model trained on this data:

        **With a ``potcar_dir``** the tabulated local pseudopotential is used —
        VASP's own ``POTION`` formula, reproducing a reference ``EXTCAR`` to a
        relative :math:`2\times10^{-5}`. The archive supplies the structure and
        the library supplies the pseudopotentials, which together are
        everything the exact construction needs.

        **Without one** the Gaussian pseudo-ion model is used, whose residual
        against that reference is of order :math:`0.1` relative :math:`L_2`.
        The map learned is then *model potential* :math:`\to` *DFT density* —
        well-posed and self-consistent, since inference builds the same
        potential, but not VASP's :math:`V_{\rm ext}`.
        """
        from ..fields.external import _widths_from_pseudopotentials
        from ..fields.vasp.volumetric import read_structure_header

        structure = read_structure_header(record.files["CHGCAR"])

        potcar = self.library_potcar(structure)
        if potcar is not None:
            return ExternalPotential.from_potcar_tables(
                structure, grid, potcar,
                metadata={"source": "bulk density archive", "code": "vasp",
                          "derived_from": "CHGCAR header + POTCAR library",
                          "potcar_dir": str(self.options["potcar_dir"])},
            )

        charges = self.charges
        missing = sorted({element for element in structure.elements
                          if element not in charges})
        if missing:
            raise ValueError(
                f"{record.identifier}: no valence charge for {missing}. Pass "
                f"charges={{'X': Z}} or potcar_dir=, or include materials "
                f"whose compositions determine them."
            )

        widths = _widths_from_pseudopotentials(
            structure, {}, self.options.get("sigma"), 0.5, "gaussian")
        return ExternalPotential.compute(
            structure, grid, charges, widths=widths, model="gaussian",
            metadata={"source": "bulk density archive", "code": "vasp",
                      "derived_from": "CHGCAR header"},
        )

    def describe(self):
        # "where it reaches" rather than a flat claim: a library that misses a
        # species falls back for that species alone, and the line must not
        # promise a construction some of the materials did not get.
        origin = (f"V_ext tabulated from {self.options['potcar_dir']} where it "
                  f"reaches, Gaussian otherwise"
                  if self.potential_model() == "tabulated"
                  else "V_ext from the Gaussian pseudo-ion model")
        return (f"{self.name}: {self.root} (structures from the CHGCAR "
                f"headers, {origin})")


# ---------------------------------------------------------------------- #
# C: a prepared cache
# ---------------------------------------------------------------------- #
class PreparedFieldsSource(MaterialSource):
    """
    Prepared layout: one directory per material, holding the fields themselves.

    This is what the cache builder writes and what
    :class:`~poraque.ml.data.FieldPairDataset` has always read — no inputs, no
    pseudopotentials, just ``EXTCAR``/``CHGCAR``/``TAUCAR`` already on the grid
    they will be trained on::

        cache/res32/mp-124/{EXTCAR,CHGCAR}
        cache/res32/struct_000/{EXTCAR,CHGCAR,TAUCAR}

    Nothing is computed. The distinction from :class:`CalculationSource` is the
    absence of a ``POSCAR``: a directory with one is a calculation whose
    potential must be rebuilt, a directory without one is a prepared field set
    to be read as it stands.

    **Either storage layout.** The same cache can keep its fields as
    ``CHGCAR``-format text files or as one chunked, optionally compressed
    ``fields.h5`` per material::

        cache/res32/mp-124/{EXTCAR,CHGCAR}          # storage: files
        cache/res32/mp-124/fields.h5                # storage: hdf5

    Only :meth:`discover` knows the difference, and only far enough to write
    ``…/fields.h5::CHGCAR`` into the record instead of ``…/CHGCAR``. Every
    reader below that point takes both spellings
    (:mod:`poraque.fields.hdf5`), so :meth:`read` is one method and not two,
    and a cache holding some materials in each layout works without anyone
    having intended it to.
    """

    name = "prepared"

    def potential_model(self):
        """
        ``"cached"`` --- whatever wrote the ``EXTCAR`` decided, and it is not
        recoverable from the file.
        """
        return "cached"

    @classmethod
    def detect(cls, root):
        """
        A prepared cache: per-material directories holding *prepared fields*.

        The rule is that a material's directory carries something more than a
        density. That is what separates this layout from a bulk archive, which
        since the per-material download layout has exactly the same shape:
        ``data/MP/mp-124/CHGCAR.gz`` and ``cache/res32/mp-124/CHGCAR`` are
        indistinguishable as directory trees, and only the contents say which
        is which.

        A cache always has more, whichever task it was built for — ``EXTCAR``
        with ``CHGCAR`` for ``ext2chg``, ``CHGCAR`` with ``TAUCAR`` for
        ``chg2tau`` — because :data:`~poraque.data.cache.CACHED_FIELDS` writes
        every field its source provides. An archive publishes the density and
        nothing else. So a directory of lone densities is declined here and
        falls through to :class:`BulkDensitySource`, which is the class that
        knows an external potential still has to be *built* for it.

        Getting this backwards is not a detection nicety: read as a cache, an
        MP download would offer ``CHGCAR`` alone, and every ``ext2chg`` run on
        it would fail for want of an input field that was never missing.
        """
        if not os.path.isdir(root) or _is_calculation_directory(root):
            return False
        for child in _subdirectories(root):
            if _is_calculation_directory(child):
                return False
            if set(cls._fields_in(child)) - {"CHGCAR"}:
                return True
        return False

    def discover(self):
        records = []
        for child in _subdirectories(self.root):
            files = self._fields_in(child)
            if "CHGCAR" in files:
                records.append(self._record(os.path.basename(child), child,
                                            files))
        return records

    @staticmethod
    def _fields_in(directory):
        """``{field: path}`` for one material, in whichever layout it uses."""
        from ..ml.data import prepared_fields

        return prepared_fields(directory, FIELD_CLASSES)

    def provides(self, record):
        return tuple(name for name in FIELD_CLASSES if name in record.files)

    def reference_file(self, record):
        return record.files["CHGCAR"]

    def read(self, record, name, grid, spin=False):
        path = record.files.get(name)
        if path is None:
            raise FileNotFoundError(
                f"{record.identifier}: {name} is not in {record.directory}. "
                f"It holds {list(self.provides(record))}."
            )
        if name == "CHGCAR":
            return self._read_density(path, grid, spin)
        return FIELD_CLASSES[name].read(path, grid=grid)


# ---------------------------------------------------------------------- #
# The factory
# ---------------------------------------------------------------------- #
#: Registered sources, in detection order.
_SOURCES = []

#: Other names for a format. ``mp`` is here because "a Materials Project
#: download" is what most people mean by a bulk density archive, and reaching
#: for that word in a config should not be an error.
FORMAT_ALIASES = {
    "mp": "bulk",
    "materials-project": "bulk",
    "materials_project": "bulk",
    "chgcar": "bulk",
    "calculation": "vasp",
    "cache": "prepared",
}


def register_source(source_class):
    """
    Register a source so :func:`detect_source` will consider it.

    Order matters and is registration order: the most specific layout must be
    asked first. A calculation directory contains a ``CHGCAR`` too, so
    :class:`CalculationSource` is asked before :class:`BulkDensitySource`, and
    the two would otherwise both claim it.

    Parameters
    ----------
    source_class : type
        A :class:`MaterialSource` subclass.

    Returns
    -------
    type
        ``source_class``, so this works as a decorator.
    """
    if not issubclass(source_class, MaterialSource):
        raise TypeError(f"{source_class!r} is not a MaterialSource subclass.")
    _SOURCES.append(source_class)
    return source_class


def available_formats():
    """Names accepted by ``format=`` , in detection order."""
    return [source.name for source in _SOURCES]


def detect_source(root):
    """
    Identify the layout of ``root``.

    Parameters
    ----------
    root : str or pathlib.Path

    Returns
    -------
    type
        The :class:`MaterialSource` subclass that recognises it.

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist.
    ValueError
        If nothing recognises it — the message lists what was looked for,
        because the answer is almost always one missing file.
    """
    root = str(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No such directory: {root}")

    for source_class in _SOURCES:
        if source_class.detect(root):
            return source_class

    raise ValueError(
        f"Cannot tell what kind of dataset {root!r} is. Expected one of: a "
        f"calculation directory or a directory of them (a POSCAR or CONTCAR "
        f"somewhere); a bulk archive of standalone CHGCAR files, compressed or "
        f"not; or a prepared cache of per-material EXTCAR/CHGCAR/TAUCAR "
        f"directories. Pass format= explicitly if the layout is one of these "
        f"under another name."
    )


def resolve_source(root, format="auto", **options):
    """
    Build the source for ``root``.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory to read.
    format : str, optional
        ``"auto"`` to detect, or one of :func:`available_formats` (or an alias
        from :data:`FORMAT_ALIASES`). An explicit name is checked against the
        directory and a mismatch raises, so a typo is not silently honoured
        into an empty dataset.
    **options
        Passed to the source; see each class.

    Returns
    -------
    MaterialSource
    """
    root = str(root)
    if format is None or str(format).lower() in ("auto", ""):
        return detect_source(root)(root, **options)

    key = str(format).lower()
    key = FORMAT_ALIASES.get(key, key)
    for source_class in _SOURCES:
        if source_class.name == key:
            if not source_class.detect(root):
                raise ValueError(
                    f"format={format!r} was requested for {root!r}, but that "
                    f"directory is not laid out as a {key} dataset. Detected: "
                    f"{detect_source(root).name!r}."
                )
            return source_class(root, **options)

    raise ValueError(
        f"Unknown data format {format!r}; available: {available_formats()} "
        f"(or 'auto')."
    )


def discover_records(sources, required=(), log=None):
    """
    Enumerate the materials of several sources, keeping identifiers unique.

    Two archives can easily both contain a ``struct_000``. Left alone, the
    collision would silently drop one material and corrupt every per-material
    report, so a duplicate is prefixed with its archive's directory name
    (``MP:mp-124``) and the run is told.

    Parameters
    ----------
    sources : sequence of MaterialSource
    required : sequence of str, optional
        Keep only materials supplying every one of these fields. This is a
        *per-material* filter, not a per-directory one: one calculation in an
        archive may have written a ``TAUCAR`` while its neighbour did not.
    log : callable, optional
        Receives one line per source.

    Returns
    -------
    list of MaterialRecord
        Sorted by identifier.
    """
    emit = log or (lambda *_: None)
    required = set(required)

    records, seen = [], set()
    for source in sources:
        found = source.discover()
        usable = [record for record in found
                  if required <= set(source.provides(record))]

        for record in usable:
            if record.identifier in seen:
                record.identifier = _disambiguate(record, seen, emit)
            seen.add(record.identifier)
        records.extend(usable)

        dropped = len(found) - len(usable)
        emit(f"  {source.describe()}: {len(usable)} material(s)"
             + (f", {dropped} without {'/'.join(sorted(required))}"
                if dropped else ""))

    records.sort(key=lambda record: record.identifier)
    return records


def _disambiguate(record, seen, emit):
    """Prefix a colliding identifier with its archive's directory name."""
    prefix = os.path.basename(os.path.normpath(record.source.root)) or "path"
    candidate = f"{prefix}:{record.identifier}"
    index = 2
    while candidate in seen:
        candidate = f"{prefix}{index}:{record.identifier}"
        index += 1
    emit(f"      note: two archives both call a material "
         f"{record.identifier!r}; this one is now {candidate!r}")
    return candidate


# Registration order is detection order; see register_source.
register_source(CalculationSource)
register_source(PreparedFieldsSource)
register_source(BulkDensitySource)
