# -*- coding: utf-8 -*-
# file: base.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Core-valence separation and the local-pseudopotential base class.

Pseudopotentials replace the singular all-electron nuclear potential and the
tightly bound core electrons by a smooth *effective* potential acting only on
the chemically active **valence** electrons. This module provides

* :data:`VALENCE_ELECTRONS` — a simple, transparent core-valence prescription
  (the number of valence electrons kept explicitly for each element), and
* :class:`LocalPseudopotential` — the abstract base every local pseudopotential
  derives from. A local pseudopotential is fully specified by a radial function
  :math:`v_\\text{loc}(r)` that is evaluated on the real-space grid with the
  minimum-image convention so that it is consistent with periodic boundary
  conditions.
"""

import numpy as np


# A pragmatic, element-by-element core-valence split. The value is the number
# of valence electrons treated explicitly; the remaining ``Z - n_valence``
# electrons are folded into the (frozen) core described by the pseudopotential.
VALENCE_ELECTRONS = {
    "H": 1, "He": 2,
    "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 1, "Ca": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7, "Kr": 8,
}


def valence_electrons(symbol, atomic_number=None):
    """
    Number of valence electrons kept explicitly for an element.

    Parameters
    ----------
    symbol : str
        Chemical symbol (e.g. ``"Si"``).
    atomic_number : int, optional
        Fallback used when ``symbol`` is not in :data:`VALENCE_ELECTRONS`; in
        that (all-electron) case the full nuclear charge is returned.

    Returns
    -------
    int
        Valence-electron count ``Z_valence``.
    """
    if symbol in VALENCE_ELECTRONS:
        return VALENCE_ELECTRONS[symbol]
    if atomic_number is not None:
        return int(atomic_number)
    raise KeyError(f"No valence prescription for element {symbol!r}.")


class LocalPseudopotential:
    """
    Base class for local (angular-momentum-independent) pseudopotentials.

    A concrete pseudopotential implements :meth:`radial_potential`, the
    spherically symmetric local potential :math:`v_\\text{loc}(r)` (Hartree) for
    a single ion of valence charge :attr:`z_valence`. :meth:`local_potential`
    then maps that radial form onto a :class:`~poraque.core.Grid` for an ion at
    a given position, honouring the minimum-image convention for periodic cells.

    Parameters
    ----------
    symbol : str
        Chemical symbol of the element.
    z_valence : float
        Number of valence electrons (the effective ionic charge seen far from
        the core).
    """

    def __init__(self, symbol, z_valence):
        self.symbol = symbol
        self.z_valence = float(z_valence)

    def radial_potential(self, r):
        """Local potential ``v_loc(r)`` (Hartree) as a function of radius."""
        raise NotImplementedError

    def local_potential(self, grid, position, mic=True):
        """
        Evaluate ``v_loc(|r - R|)`` on the grid for an ion at ``position``.

        Parameters
        ----------
        grid : Grid
            Real-space grid providing :meth:`~poraque.core.Grid.get_xyz`.
        position : array_like
            Cartesian ion coordinate (Bohr).
        mic : bool, optional
            Use the minimum-image convention for periodic boundary conditions.

        Returns
        -------
        numpy.ndarray
            The local pseudopotential sampled on the grid.
        """
        coords = grid.get_xyz()
        dr = coords - np.asarray(position, dtype=float)
        if mic:
            cell = np.asarray(grid.cell, dtype=float)
            s = dr @ np.linalg.inv(cell)
            s -= np.round(s)
            dr = s @ cell
        r = np.sqrt(np.sum(dr**2, axis=-1))
        return self.radial_potential(r)

    def __repr__(self):
        return f"{type(self).__name__}(symbol={self.symbol!r}, z_valence={self.z_valence})"
