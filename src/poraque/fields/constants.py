# -*- coding: utf-8 -*-
# file: constants.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Physical constants for the VASP-facing :mod:`poraque.fields` stack.

Poraquê works in the **VASP convention** — Ångström for lengths, electronvolt
for energies — because every quantity it reads or writes (``POSCAR``,
``INCAR``, ``POTCAR``, ``CHGCAR``) is expressed that way. Conversion factors to
Hartree atomic units are kept because the analytic functionals (Thomas-Fermi,
von Weizsäcker) have their natural form there and convert internally.
"""

#: Bohr radius in Ångström (CODATA 2018).
BOHR_TO_ANGSTROM = 0.529177210903
#: Ångström in Bohr.
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

#: Hartree energy in electronvolt (CODATA 2018).
HARTREE_TO_EV = 27.211386245988
#: Electronvolt in Hartree.
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV

#: Coulomb constant ``e^2 / (4 pi eps0)`` in eV·Å.
#: Equals ``HARTREE_TO_EV * BOHR_TO_ANGSTROM``.
COULOMB_CONSTANT_EV_ANGSTROM = HARTREE_TO_EV * BOHR_TO_ANGSTROM  # 14.399645 eV.A

#: ``hbar^2 / (2 m_e)`` in eV·Å².  Converts a plane-wave cutoff (eV) into a
#: maximum wavevector: ``G_cut = sqrt(ENCUT / HBAR2_OVER_2M)`` in Å⁻¹.
HBAR2_OVER_2M_EV_ANGSTROM2 = 0.5 * HARTREE_TO_EV * BOHR_TO_ANGSTROM ** 2  # 3.80998 eV.A^2

#: Thomas-Fermi constant ``(3/10) (3 pi^2)^(2/3)`` in Hartree atomic units.
#: ``tau_TF[rho] = C_TF * rho^(5/3)``.
C_TF = 2.871234000188191
