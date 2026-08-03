# -*- coding: utf-8 -*-
# file: vasp.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
VASP implementation of the :class:`~poraque.fields.io.base.CalculationReader`
contract.

This is the reference implementation: it is what the Quantum ESPRESSO and GPAW
readers should be modelled on. All the format-specific parsing lives in
:mod:`poraque.fields.vasp`; this module only adapts it to the neutral
interface.
"""

import os

from ..vasp.incar import Incar
from ..vasp.poscar import Poscar
from ..vasp.potcar import Potcar
from ..vasp.volumetric import read_volumetric, write_volumetric
from .base import CalculationParameters, CalculationReader, PseudopotentialInfo


class VaspReader(CalculationReader):
    """
    Ingest a VASP calculation directory.

    Recognises ``POSCAR``/``CONTCAR`` for the geometry, ``INCAR`` for the
    cutoff and grid tags, ``POTCAR`` for valence charges, and the
    ``CHGCAR``-format volumetric files.
    """

    code = "vasp"
    structure_files = ("POSCAR", "CONTCAR")
    field_files = {
        "external": "EXTCAR",
        "density": "CHGCAR",
        "kinetic": "TAUCAR",
    }

    def read_structure(self, directory):
        """Read ``POSCAR`` (falling back to ``CONTCAR``)."""
        return Poscar.from_file(self.structure_path(directory))

    def read_parameters(self, directory):
        """
        Read ``INCAR``.

        ``ENCUT`` is already in eV, so no conversion is needed — unlike
        Quantum ESPRESSO, whose ``ecutwfc`` is in Rydberg.
        """
        path = os.path.join(directory, "INCAR")
        if not os.path.exists(path):
            return CalculationParameters()

        incar = Incar.from_file(path)
        return CalculationParameters(
            cutoff=incar.encut,
            precision=incar.prec,
            grid_shape=incar.fine_shape,
            xc=incar.get("GGA"),
            extra=dict(incar),
        )

    def read_pseudopotentials(self, directory):
        """
        Read ``POTCAR`` into ``{element: PseudopotentialInfo}``.

        Returns an empty mapping when no ``POTCAR`` is present, which callers
        must handle by supplying valence charges explicitly.
        """
        path = os.path.join(directory, "POTCAR")
        if not os.path.exists(path):
            return {}

        return {
            entry.element: PseudopotentialInfo(
                symbol=entry.symbol,
                element=entry.element,
                valence_charge=entry.zval,
                core_radius=entry.rcore_angstrom,
                recommended_cutoff=entry.enmax,
                functional=entry.functional,
            )
            for entry in Potcar.from_file(path)
        }

    def read_field(self, path, field_class, grid=None):
        """Read a ``CHGCAR``-format volumetric file."""
        return field_class.read(path, grid=grid)

    def write_field(self, field, path, comment=None):
        """Write a field in ``CHGCAR`` format."""
        return field.write(path, comment=comment)

    # ------------------------------------------------------------------ #
    # VASP-specific extras
    # ------------------------------------------------------------------ #
    def read_potcar(self, directory):
        """Return the raw :class:`~poraque.fields.vasp.Potcar`, or ``None``."""
        path = os.path.join(directory, "POTCAR")
        return Potcar.from_file(path) if os.path.exists(path) else None

    def read_incar(self, directory):
        """Return the raw :class:`~poraque.fields.vasp.Incar`, or ``None``."""
        path = os.path.join(directory, "INCAR")
        return Incar.from_file(path) if os.path.exists(path) else None

    @staticmethod
    def read_volumetric(path):
        """Low-level access to the volumetric parser."""
        return read_volumetric(path)

    @staticmethod
    def write_volumetric(path, structure, data, **kwargs):
        """Low-level access to the volumetric writer."""
        return write_volumetric(path, Poscar.from_structure(structure), data,
                                **kwargs)
