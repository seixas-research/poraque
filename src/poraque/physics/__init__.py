# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Energy functionals evaluated on predicted scalar fields.

Where :mod:`poraque.ml.physics` provides *differentiable* (PyTorch) operators
used inside the training loop, this package provides the plain NumPy
counterparts used *after* prediction: given :math:`\\rho`, :math:`\\tau` and
:math:`V_{\\rm ext}` on a shared grid, it integrates them into the energy
components of the Kohn-Sham total-energy expression.

The split is deliberate. Training needs gradients and runs on whatever device
holds the model; energy evaluation needs neither, and forcing it through torch
would make :class:`~poraque.calculator.Poraque` depend on a GPU context to
report a number.

    from poraque.physics import EnergyCalculator, ewald_energy
"""

from .energy import (
    XC_FUNCTIONALS,
    EnergyComponents,
    EnergyCalculator,
    ewald_energy,
    hartree_energy,
    hartree_potential,
    lda_exchange_energy,
    pbe_correlation_energy,
    pbe_exchange_energy,
    pw92_correlation_energy,
    xc_energy,
)

__all__ = [
    "XC_FUNCTIONALS",
    "EnergyComponents",
    "EnergyCalculator",
    "ewald_energy",
    "hartree_energy",
    "hartree_potential",
    "lda_exchange_energy",
    "pbe_correlation_energy",
    "pbe_exchange_energy",
    "pw92_correlation_energy",
    "xc_energy",
]
