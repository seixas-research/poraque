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


def _material_density(directory, prefixes=BULK_PREFIXES):
    r"""
    Path to this material's density, or ``None``.

    **A material is a directory with a density in it.** The density carries the
    grid every other field is placed on and the structure they are all built
    at, so nothing downstream can do anything without one; inputs beside it are
    a bonus, not the qualification. A directory holding a ``POSCAR`` and no
    ``CHGCAR`` is a calculation that has not run.

    Whichever storage the directory uses: a ``CHGCAR`` — compressed or not,
    named exactly or with the material id appended — or ``fields.h5::CHGCAR``.
    **Exactly one** is the test: a directory with two is not one material's,
    whatever it is, and choosing between them silently is not a decision to
    make on the reader's behalf.

    Returns
    -------
    str or None
    """
    if not os.path.isdir(directory):
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


def _is_calculation_directory(path):
    """
    Whether ``path`` holds the input files of a DFT run, in any code.

    No longer what identifies a *material* — :func:`_material_density` is —
    but still what says a material's inputs are there to build the external
    potential from, rather than only its density.
    """
    return any(os.path.exists(os.path.join(path, name))
               for name in calculation_markers())


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
        # Seeded from the explicit option, resolved lazily otherwise -- see
        # `charges`. On the base class because every source builds the same
        # potential from the same numbers.
        self._charges = (dict(options["charges"])
                         if options.get("charges") else None)
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
    @property
    def charges(self):
        r"""
        ``{element: Z_val}`` in force for this source.

        Three answers, in order, and the third is what makes the schema
        genuinely uniform:

        1. an explicit ``charges`` option;
        2. the ``ZVAL`` of each species in the ``potcar_dir`` library, which is
           where the number actually comes from;
        3. **inference from the densities** — a pseudopotential ``CHGCAR``
           integrates to its cell's valence electron count, giving one linear
           equation per material in the per-element charges, and a set with
           more compositions than elements over-determines them.

        (3) used to belong to :class:`BulkDensitySource` alone, which made the
        same directory answerable or not depending on how it was read: a folder
        of bare densities inferred its charges, and the identical folder with a
        ``POSCAR`` added raised ``No valence charge for ['Si']``. Nothing about
        the physics changed between those two, so nothing about the answer
        should have.

        Deferred rather than resolved in ``__init__``, because (2) and (3) read
        files and a caller that only wants to enumerate the source should never
        pay for that. (3) is reached only where a potential actually has to be
        *modelled*: with a ``POTCAR`` or a library entry the tabulated route
        needs no ``Z_val`` at all.
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
            from ..fields.vasp.potcar import Potcar
            from ..fields.vasp.volumetric import read_structure_header

            # Header reads only: a few hundred bytes per material.
            elements = sorted({element for record in records
                               for element in read_structure_header(
                                   self.reference_file(record)).elements})
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

    def potcar_coverage(self):
        r"""
        Which elements the configured library can actually serve.

        Returns ``None`` when no ``potcar_dir`` is configured — there is then
        nothing to have failed — and otherwise ``{element: bool}`` over the
        elements this source's own materials contain.

        **This is the difference between what was asked for and what was
        got.** :meth:`library_potcar` degrades gracefully when the library
        cannot be read, with a ``RuntimeWarning`` on stderr and an analytic
        pseudo-ion in place of the tabulated potential. That is defensible. What
        was not is that nothing downstream recorded it: a cache built during a
        filesystem outage — a compute node where ``/prj`` was not mounted,
        which is how this was found — is byte-identical in its fingerprint to
        one built from real pseudopotentials, is silently reused by every later
        run including on nodes where the library *is* mounted, and the one
        warning that would have said so appeared during the build that wrote
        it and never again.

        The answer is per element because the fallback is per element: a
        library can hold Pt and not Ag, and the run is then a mixture.

        Cheap by construction — a structure header per material (a few hundred
        bytes) and one library read per element, memoised on the source, which
        the build was going to pay for anyway.

        Returns
        -------
        dict or None
        """
        if not self.options.get("potcar_dir"):
            return None

        from ..fields.vasp.volumetric import read_structure_header

        elements = sorted({element for record in self.discover()
                           for element in read_structure_header(
                               self.reference_file(record)).elements})
        return {element: self._library_entry(element) is not None
                for element in elements}

    def potential_model(self):
        """
        Which construction this source uses for :math:`V_{\\rm ext}`.

        Two possible answers, ``"tabulated"`` and ``"gaussian"``, and the
        difference is the single most consequential property of a training set:
        they are different physical quantities, not different approximations to
        one. It is reported in the log, and it is what
        :class:`~poraque.data.dataset.MixedFieldDataset` compares across
        sources before deciding whether a mixture needs a warning.

        Answered from :meth:`potcar_coverage` rather than from the presence of
        a ``potcar_dir`` option: a library that cannot be read serves nothing,
        and reporting "tabulated" because one was *configured* is the same
        mistake the cache fingerprint was making.

        Returns
        -------
        str
        """
        coverage = self.potcar_coverage()
        if coverage is None:
            return "gaussian"
        return "tabulated" if all(coverage.values()) else "gaussian"


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
        """
        Whether ``root`` holds materials: one directly, or one per child.

        Keyed on the **density**, not on a ``POSCAR``. A directory of MP
        downloads and a directory of full VASP runs are then the same thing to
        the reader, which is the point of there being one source class — and
        the alternative, keying on the inputs, would refuse a density-only
        material that the class can perfectly well read.
        """
        if _material_density(root) is not None:
            return True
        return any(_material_density(child) is not None
                   for child in _subdirectories(root))

    def _reader(self, directory):
        from ..fields.io import resolve_reader

        return resolve_reader(directory, self.options.get("format", "auto"))

    def discover(self):
        pattern = self.options.get("pattern") or ""
        if _material_density(self.root) is not None:
            directories = [self.root]
        else:
            directories = [child for child in _subdirectories(self.root)
                           if os.path.basename(child).startswith(pattern)
                           and _material_density(child) is not None]

        records = []
        for directory in directories:
            reader = self._reader(directory)
            # The density first, from wherever it is -- a bare `CHGCAR`, a
            # compressed one, or a field store. `field_path` names the reader's
            # own convention and covers the common case; `_material_density`
            # covers the rest, including `mp-124/CHGCAR.gz` and `fields.h5`.
            density = reader.field_path(directory, "density")
            if not os.path.exists(density):
                density = _material_density(directory)
            if density is None:
                continue
            files = {"CHGCAR": density}

            # TAUCAR is read where the run wrote one and simply not offered
            # where it did not. A material with a density and no kinetic energy
            # density is a perfectly good `ext2chg` sample, and demanding one
            # would exclude every published archive.
            kinetic = reader.field_path(directory, "kinetic")
            if os.path.exists(kinetic):
                files["TAUCAR"] = kinetic

            records.append(
                self._record(os.path.basename(os.path.normpath(directory)),
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

        The runs' own files are consulted **first**, which is the order
        :meth:`_external_potential` actually uses: a directory holding a
        ``POTCAR`` is tabulated whether or not a library was configured or
        readable.
        """
        records = self.discover()
        if records and all(os.path.exists(os.path.join(r.directory, "POTCAR"))
                           for r in records):
            return "tabulated"
        return super().potential_model()

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

        With neither, the Gaussian pseudo-ion model is used, and the valence
        charges it needs come from :attr:`~MaterialSource.charges`: the
        explicit option, or **inferred from the densities**. Without that last
        step a stripped run directory raised where a directory of the same
        densities with the structure files deleted would have succeeded --
        which is not a difference the data justifies.

        Every route is handed the geometry :meth:`geometry` resolves, so the
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
            zval=self.options.get("charges") or self._modelled_charges(record),
            structure=structure,
        )

    def _modelled_charges(self, record):
        """
        Valence charges for the Gaussian fallback, or ``None``.

        ``None`` where the run carries its own ``POTCAR``: the tabulated route
        reads ``ZVAL`` from it and needs nothing from here, and inferring
        charges to hand it a number it will not use would parse every density
        in the set for nothing.

        Inference that cannot succeed is not an error *here* either --
        ``from_calculation`` raises a better message a moment later, naming the
        elements it could not place.
        """
        if os.path.exists(os.path.join(record.directory, "POTCAR")):
            return None
        try:
            return self.charges or None
        except (ValueError, OSError):
            return None

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
# The factory
# ---------------------------------------------------------------------- #
#: Values ``data.format`` accepts.
#:
#: ``"vasp"`` names the only DFT code whose files Poraquê reads, and ``"auto"``
#: works it out from what is in the directory. There is no third option and no
#: layout to choose: every dataset is a directory of per-material
#: subdirectories, and what a material's directory *holds* is read rather than
#: declared.
#:
#: Until 2026-08-31 this key chose between three *layouts* as well --
#: ``"bulk"`` for an archive of standalone densities and ``"prepared"`` for a
#: cache of already-built fields. Both are gone, and with them the question a
#: config had to answer about a directory it could simply be shown.
DATA_FORMATS = ("auto", "vasp")


def resolve_source(root, **options):
    """
    Build the source for ``root``.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory to read. It holds either one material's files directly, or
        one subdirectory per material.
    **options
        Passed to :class:`CalculationSource`; ``format`` selects the reader
        (``"vasp"``, or ``"auto"`` to detect it per directory).

    Returns
    -------
    CalculationSource

    Raises
    ------
    FileNotFoundError
        If ``root`` does not exist.
    ValueError
        If it holds no materials -- the message says what was looked for,
        because the answer is almost always one missing file.
    """
    root = str(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No such directory: {root}")

    fmt = str(options.get("format") or "auto").lower()
    if fmt not in DATA_FORMATS:
        raise ValueError(
            f"Unknown data format {options.get('format')!r}; "
            f"expected one of {list(DATA_FORMATS)}."
        )

    if not CalculationSource.detect(root):
        raise ValueError(
            f"No materials under {root!r}. A dataset directory holds one "
            f"subdirectory per material, each with that material's CHGCAR "
            f"(and optionally a TAUCAR and the run's inputs); a directory "
            f"holding one material's files directly is read as that single "
            f"material."
        )
    return CalculationSource(root, **options)


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

