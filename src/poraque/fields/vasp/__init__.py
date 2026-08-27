# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Dependency-free readers/writers for the VASP file formats used by
:mod:`poraque.fields`, and the input decks the project asks VASP for
(:mod:`~poraque.fields.vasp.templates`): the one that writes a ``TAUCAR``, and
the ``ICHARG = 11`` family that reads a predicted ``CHGCAR`` back as a band
structure, a density of states or a total energy.
"""

from .incar import Incar
from .poscar import Poscar, symbol_to_z
from .potcar import Potcar, PotcarSingle
from .templates import (
    automatic_kpoints,
    band_structure_incar,
    dos_incar,
    fcc_band_path,
    kpoint_mesh_from_spacing,
    line_mode_kpoints,
    tau_incar,
    total_energy_incar,
    write_band_structure_deck,
    write_dos_deck,
    write_total_energy_deck,
)
from .volumetric import read_structure_header, read_volumetric, write_volumetric

__all__ = [
    "Incar",
    "Poscar",
    "Potcar",
    "PotcarSingle",
    "automatic_kpoints",
    "band_structure_incar",
    "dos_incar",
    "fcc_band_path",
    "kpoint_mesh_from_spacing",
    "line_mode_kpoints",
    "read_structure_header",
    "read_volumetric",
    "tau_incar",
    "total_energy_incar",
    "write_band_structure_deck",
    "write_dos_deck",
    "write_total_energy_deck",
    "write_volumetric",
    "symbol_to_z",
]
