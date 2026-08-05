# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Dependency-free readers/writers for the VASP file formats used by
:mod:`poraque.fields`.
"""

from .incar import Incar
from .poscar import Poscar, symbol_to_z
from .potcar import Potcar, PotcarSingle
from .volumetric import read_structure_header, read_volumetric, write_volumetric

__all__ = [
    "Incar",
    "Poscar",
    "Potcar",
    "PotcarSingle",
    "read_structure_header",
    "read_volumetric",
    "write_volumetric",
    "symbol_to_z",
]
