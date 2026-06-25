# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Modular pseudopotential handling for Poraquê.

The public entry points are:

* :func:`resolve_pseudopotentials` — turn a user specification (``"auto"``, a
  ``{symbol: spec}`` mapping, or pre-built objects) into one
  :class:`LocalPseudopotential` per element, and
* :func:`build_pseudopotential_potential` — assemble the valence external
  potential on the grid and report the total number of valence electrons.

This keeps the core-valence separation in one place so the engines never need
to know whether they are running an all-electron or a pseudopotential
calculation.
"""

import numpy as np
from ase.data import chemical_symbols

from .base import LocalPseudopotential, VALENCE_ELECTRONS, valence_electrons
from .local import GaussianCorePP, SoftCoulombPP, default_pseudopotential
from .io import read_pseudopotential

__all__ = [
    "LocalPseudopotential",
    "SoftCoulombPP",
    "GaussianCorePP",
    "VALENCE_ELECTRONS",
    "valence_electrons",
    "default_pseudopotential",
    "read_pseudopotential",
    "resolve_pseudopotentials",
    "build_pseudopotential_potential",
]


def _resolve_one(symbol, spec, atomic_number=None):
    """Resolve a single element's pseudopotential specification."""
    if isinstance(spec, LocalPseudopotential):
        return spec
    if spec is None or (isinstance(spec, str) and spec.lower() == "auto"):
        return default_pseudopotential(symbol, atomic_number)
    if isinstance(spec, str):
        return read_pseudopotential(spec)
    if isinstance(spec, dict):
        kwargs = dict(spec)
        kwargs.setdefault("symbol", symbol)
        kwargs.setdefault("z_valence", valence_electrons(symbol, atomic_number))
        return SoftCoulombPP(**kwargs)
    raise TypeError(f"Unsupported pseudopotential spec for {symbol!r}: {spec!r}")


def resolve_pseudopotentials(system, pseudopotentials):
    """
    Resolve a pseudopotential specification into one object per element.

    Parameters
    ----------
    system : System
        Provides ``atomic_numbers`` (used to look up chemical symbols).
    pseudopotentials : "auto" or dict or LocalPseudopotential
        ``"auto"`` builds a default local pseudopotential for every element. A
        ``dict`` maps chemical symbols to per-element specifications (a built
        :class:`LocalPseudopotential`, a path to a pseudopotential file, a
        keyword ``dict`` for :class:`SoftCoulombPP`, or ``"auto"``). A single
        :class:`LocalPseudopotential` is applied to all atoms.

    Returns
    -------
    dict
        Mapping ``{chemical_symbol: LocalPseudopotential}``.
    """
    symbols = {chemical_symbols[z]: int(z) for z in system.atomic_numbers}

    if isinstance(pseudopotentials, LocalPseudopotential):
        return {sym: pseudopotentials for sym in symbols}

    resolved = {}
    for sym, z in symbols.items():
        if isinstance(pseudopotentials, dict):
            if sym not in pseudopotentials:
                raise KeyError(f"No pseudopotential supplied for element {sym!r}.")
            spec = pseudopotentials[sym]
        else:  # "auto" / None
            spec = "auto"
        resolved[sym] = _resolve_one(sym, spec, z)
    return resolved


def build_pseudopotential_potential(grid, system, pseudopotentials, mic=True):
    """
    Build the valence external potential and count valence electrons.

    Parameters
    ----------
    grid : Grid
        Real-space grid.
    system : System
        Atomic structure.
    pseudopotentials : "auto" or dict or LocalPseudopotential
        See :func:`resolve_pseudopotentials`.
    mic : bool, optional
        Use the minimum-image convention for periodic cells.

    Returns
    -------
    tuple
        ``(v_ext, n_valence)`` — the local pseudopotential summed over all ions
        on the grid (Hartree) and the total number of valence electrons.
    """
    table = resolve_pseudopotentials(system, pseudopotentials)
    v_ext = np.zeros(grid.shape)
    n_valence = 0.0
    for z, pos in zip(system.atomic_numbers, system.positions):
        pp = table[chemical_symbols[int(z)]]
        v_ext += pp.local_potential(grid, pos, mic=mic)
        n_valence += pp.z_valence
    return v_ext, n_valence
