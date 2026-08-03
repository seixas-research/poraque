# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
3D scalar fields on a shared real-space grid.

The organising idea of this package is that, for one material, *every* field —
the local external potential (``EXTCAR``), the valence charge density
(``CHGCAR``) and the kinetic energy density (``TAUCAR``) — is sampled on a
single :class:`FieldGrid` and serialized in a single (``CHGCAR``) format. That
invariant is what makes the fields directly comparable, composable, and usable
as aligned input/target pairs by the neural operators in :mod:`poraque.ml`.

Typical use::

    from poraque.fields import ExternalPotential, ChargeDensity, FieldGrid

    # One grid, shared by every field of this material.
    grid = FieldGrid.from_file("CHGCAR")          # or FieldGrid.from_encut(...)

    v_ext = ExternalPotential.from_vasp(".", grid=grid)
    v_ext.write("EXTCAR")

    rho = ChargeDensity.read("CHGCAR", grid=grid)
    print(v_ext.interaction_energy(rho), "eV")
"""

from .base import ScalarField
from .constants import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    COULOMB_CONSTANT_EV_ANGSTROM,
    EV_TO_HARTREE,
    HARTREE_TO_EV,
    HBAR2_OVER_2M_EV_ANGSTROM2,
)
from .density import (
    ChargeDensity,
    KineticEnergyDensity,
    spectral_gradient,
    thomas_fermi_tau,
    von_weizsacker_tau,
)
from .external import ExternalPotential
from .grid import FieldGrid, fft_friendly_size

__all__ = [
    "ChargeDensity",
    "ExternalPotential",
    "FieldGrid",
    "KineticEnergyDensity",
    "ScalarField",
    "fft_friendly_size",
    "spectral_gradient",
    "thomas_fermi_tau",
    "von_weizsacker_tau",
    "ANGSTROM_TO_BOHR",
    "BOHR_TO_ANGSTROM",
    "COULOMB_CONSTANT_EV_ANGSTROM",
    "EV_TO_HARTREE",
    "HARTREE_TO_EV",
    "HBAR2_OVER_2M_EV_ANGSTROM2",
]
