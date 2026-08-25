# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Dependency-free readers/writers for the VASP file formats used by
:mod:`poraque.fields`, and the two input decks the project asks VASP for
(:mod:`~poraque.fields.vasp.templates`): the one that writes a ``TAUCAR``, and
the ``ICHARG = 11`` one that reads a predicted ``CHGCAR`` back.
"""

from .incar import Incar
from .poscar import Poscar, symbol_to_z
from .potcar import Potcar, PotcarSingle
from .templates import (
    band_structure_incar,
    fcc_band_path,
    line_mode_kpoints,
    tau_incar,
    write_band_structure_deck,
)
from .volumetric import read_structure_header, read_volumetric, write_volumetric

__all__ = [
    "Incar",
    "Poscar",
    "Potcar",
    "PotcarSingle",
    "band_structure_incar",
    "fcc_band_path",
    "line_mode_kpoints",
    "read_structure_header",
    "read_volumetric",
    "tau_incar",
    "write_band_structure_deck",
    "write_volumetric",
    "symbol_to_z",
]
