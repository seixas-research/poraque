# -*- coding: utf-8 -*-
# file: base.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The code-agnostic ingestion interface.

Poraquê currently ingests VASP output, but the physics it extracts —
geometry, a plane-wave cutoff, valence charges, and three scalar fields on a
shared grid — is common to every plane-wave DFT code. This module defines the
contract a reader must satisfy so that **adding Quantum ESPRESSO or GPAW means
writing one class**, with no change to grids, fields, datasets or models.

The contract has four parts, deliberately kept small:

``read_structure``
    → :class:`~poraque.fields.structure.Structure` (Ångström, fractional
    coordinates, species-grouped).

``read_parameters``
    → :class:`CalculationParameters`. Cutoffs are normalized to **eV**, so a
    Quantum ESPRESSO reader converts its Rydberg ``ecutwfc`` here and nothing
    downstream needs to know.

``read_pseudopotentials``
    → ``{element: PseudopotentialInfo}``. Only the *valence charge* is truly
    required; it is what defines the ionic problem the valence electrons see,
    and every code records it (VASP ``ZVAL``, QE ``z_valence``, GPAW setup
    ``Nv``).

``read_field`` / ``write_field``
    → :class:`~poraque.fields.base.ScalarField` on a caller-supplied grid.

Everything else — grid construction, the external-potential model, the ML
dataset — consumes only these four and is therefore already code-agnostic.

Field kinds are referred to by the neutral names in :data:`FIELD_KINDS`
(``"external"``, ``"density"``, ``"kinetic"``) rather than by filename, since
``CHGCAR`` has no meaning outside VASP.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dataclass_field

from ..structure import element_of

#: Neutral names for the three scalar fields, mapped to filenames by each reader.
FIELD_KINDS = ("external", "density", "kinetic")


@dataclass
class PseudopotentialInfo:
    """
    The pseudopotential facts the external potential needs.

    Attributes
    ----------
    symbol : str
        Code-specific label, possibly decorated (``"Fe_pv"``, ``"O.pbe-rrkjus"``).
    element : str
        Bare chemical symbol.
    valence_charge : float
        Charge of the pseudo-ion in units of ``+e`` (VASP ``ZVAL``,
        QE ``z_valence``, GPAW ``Nv``). **Required.**
    core_radius : float or None
        Outermost pseudization radius in **Ångström**. Sets the natural
        softening length of the pseudo-ion model; ``None`` falls back to a
        default.
    recommended_cutoff : float or None
        Suggested plane-wave cutoff in **eV** (VASP ``ENMAX``).
    functional : str or None
        Exchange-correlation label the pseudopotential was generated with.
    """

    symbol: str
    element: str
    valence_charge: float
    core_radius: float = None
    recommended_cutoff: float = None
    functional: str = None


@dataclass
class CalculationParameters:
    """
    Run settings that determine the field grid.

    Attributes
    ----------
    cutoff : float or None
        Plane-wave cutoff for the wavefunctions, in **eV**.
    precision : str or None
        Grid-density setting (VASP ``PREC``); readers for codes without an
        equivalent leave it ``None``.
    grid_shape : tuple of int or None
        Explicit ``(N1, N2, N3)`` when the input file states it. Always wins
        over a cutoff-derived estimate.
    xc : str or None
        Exchange-correlation functional.
    extra : dict
        Remaining tags, verbatim, so nothing is silently lost.
    """

    cutoff: float = None
    precision: str = None
    grid_shape: tuple = None
    xc: str = None
    extra: dict = dataclass_field(default_factory=dict)


class CalculationReader(ABC):
    """
    Base class for per-code ingestion.

    Subclasses set :attr:`code` and :attr:`field_files` and implement the four
    abstract readers. Register the class with
    :func:`~poraque.fields.io.register_reader` to make it discoverable by name
    and by auto-detection.

    Attributes
    ----------
    code : str
        Short identifier, e.g. ``"vasp"``.
    field_files : dict
        ``{kind: filename}`` for the kinds in :data:`FIELD_KINDS`.
    structure_files : tuple of str
        Candidate structure filenames, in priority order.
    """

    code = "abstract"
    field_files = {}
    structure_files = ()

    # ------------------------------------------------------------------ #
    # Required interface
    # ------------------------------------------------------------------ #
    @abstractmethod
    def read_structure(self, directory):
        """
        Read the geometry.

        Returns
        -------
        Structure
        """

    @abstractmethod
    def read_parameters(self, directory):
        """
        Read the run settings that fix the grid.

        Returns
        -------
        CalculationParameters
        """

    @abstractmethod
    def read_pseudopotentials(self, directory):
        """
        Read per-element pseudopotential data.

        Returns
        -------
        dict
            ``{element: PseudopotentialInfo}``. May be empty when the code
            stores no pseudopotential file alongside the run; callers must then
            be given valence charges explicitly.
        """

    @abstractmethod
    def read_field(self, path, field_class, grid=None):
        """
        Read one volumetric file.

        Parameters
        ----------
        path : str
            File to read.
        field_class : type
            :class:`~poraque.fields.base.ScalarField` subclass to instantiate.
        grid : FieldGrid, optional
            Shared grid to impose; a mismatch must raise.

        Returns
        -------
        ScalarField
        """

    @abstractmethod
    def write_field(self, field, path, comment=None):
        """Write a :class:`~poraque.fields.base.ScalarField` in this code's format."""

    # ------------------------------------------------------------------ #
    # Optional interface
    # ------------------------------------------------------------------ #
    def read_field_structure(self, path):
        """
        The geometry a volumetric file carries in its own header, if any.

        Not abstract, because it is a property of the *format* rather than of
        the code: a ``CHGCAR`` and a Gaussian cube both embed the structure
        they were computed at, while a bare binary grid does not. The default
        answers ``None``, which means "this format carries no geometry, use
        :meth:`read_structure`".

        It exists because those two geometries can disagree. In a **relaxation**
        the structure file is what the run started from and the volumetric file
        is what it ended at, so a caller pairing a density with a potential
        built from the structure file would be pairing two different systems.
        The density's own header is the geometry the density was computed at,
        always, so it is the authority wherever it exists.

        Parameters
        ----------
        path : str
            A volumetric file written by this code.

        Returns
        -------
        Structure or None
        """
        return None

    # ------------------------------------------------------------------ #
    # Provided helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def detect(cls, directory):
        """
        Whether ``directory`` looks like a run of this code.

        The default heuristic requires one of :attr:`structure_files` to be
        present. Override for codes whose layout is ambiguous.

        Returns
        -------
        bool
        """
        return any(os.path.exists(os.path.join(directory, name))
                   for name in cls.structure_files)

    def field_path(self, directory, kind):
        """
        Path of the file holding a given field kind.

        Parameters
        ----------
        directory : str
            Calculation directory.
        kind : str
            One of :data:`FIELD_KINDS`.

        Returns
        -------
        str
        """
        if kind not in self.field_files:
            raise KeyError(
                f"{self.code} reader does not define a file for field kind "
                f"{kind!r}; known kinds: {sorted(self.field_files)}."
            )
        return os.path.join(directory, self.field_files[kind])

    def structure_path(self, directory):
        """
        Path of the first existing structure file.

        Returns
        -------
        str

        Raises
        ------
        FileNotFoundError
            If none of :attr:`structure_files` exists.
        """
        for name in self.structure_files:
            candidate = os.path.join(directory, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"No structure file {list(self.structure_files)} in {directory!r}."
        )

    def valence_charges(self, directory, overrides=None):
        """
        ``{element: valence_charge}`` for the run, with optional overrides.

        Parameters
        ----------
        directory : str
            Calculation directory.
        overrides : dict, optional
            ``{element: charge}`` taking precedence over the files.

        Returns
        -------
        dict
        """
        charges = {element: info.valence_charge
                   for element, info in self.read_pseudopotentials(directory).items()}
        if overrides:
            charges.update({element_of(k): float(v)
                            for k, v in overrides.items()})
        return charges

    def __repr__(self):
        return f"{type(self).__name__}(code={self.code!r})"
