# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Analysis of predicted fields.

Where :mod:`poraque.physics` turns fields into energies, this package turns
them into *per-atom* quantities: population analysis, partial charges, and the
charge-conservation checks that must pass before any of it means anything.

    from poraque.analysis import partial_charges, verify_total_charge

Nothing here is learned. Every quantity is a deterministic functional of a
density that has already been predicted, so the error in a partial charge is
inherited from the density and is never smaller than it.
"""

from .charges import (
    ChargeCheck,
    PartialCharges,
    PARTITION_METHODS,
    atomic_radial_profile,
    bader_charges,
    hirshfeld_charges,
    partial_charges,
    verify_total_charge,
    voronoi_charges,
)

__all__ = [
    "ChargeCheck",
    "PartialCharges",
    "PARTITION_METHODS",
    "atomic_radial_profile",
    "bader_charges",
    "hirshfeld_charges",
    "partial_charges",
    "verify_total_charge",
    "voronoi_charges",
]
