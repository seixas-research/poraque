# -*- coding: utf-8 -*-
# file: local.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Built-in analytic local pseudopotentials."""

import numpy as np
from scipy.special import erf

from .base import LocalPseudopotential, valence_electrons


class SoftCoulombPP(LocalPseudopotential):
    r"""
    Regularized (soft-Coulomb) local pseudopotential.

    .. math::

        v_\text{loc}(r) = -\frac{Z_v}{\sqrt{r^2 + r_c^2}}

    The valence charge :math:`Z_v` sets the long-range :math:`-Z_v/r` tail seen
    by the valence electrons, while the core radius :math:`r_c` smooths the
    singularity so the potential is representable on a finite real-space grid.

    Parameters
    ----------
    symbol : str
        Chemical symbol.
    z_valence : float
        Valence charge.
    rc : float, optional
        Core (softening) radius in Bohr.
    """

    def __init__(self, symbol, z_valence, rc=0.8):
        super().__init__(symbol, z_valence)
        self.rc = float(rc)

    def radial_potential(self, r):
        return -self.z_valence / np.sqrt(r**2 + self.rc**2)


class GaussianCorePP(LocalPseudopotential):
    r"""
    Error-function-screened local pseudopotential.

    .. math::

        v_\text{loc}(r) = -\frac{Z_v}{r}\,\operatorname{erf}\!\left(\frac{r}{\sqrt{2}\,r_c}\right)

    This is the smooth local potential produced by a normalized Gaussian core
    charge of width ``rc``. It reproduces the correct ``-Z_v/r`` Coulomb tail
    while remaining finite (``-Z_v sqrt(2/pi)/rc``) at the origin, which makes it
    a slightly more physical local prescription than :class:`SoftCoulombPP`.

    Parameters
    ----------
    symbol : str
        Chemical symbol.
    z_valence : float
        Valence charge.
    rc : float, optional
        Gaussian core width in Bohr.
    """

    def __init__(self, symbol, z_valence, rc=0.5):
        super().__init__(symbol, z_valence)
        self.rc = float(rc)

    def radial_potential(self, r):
        r_safe = np.maximum(r, 1e-12)
        v = -self.z_valence * erf(r_safe / (np.sqrt(2.0) * self.rc)) / r_safe
        # erf(x)/x -> 2/sqrt(pi) as x->0; fix the regularized points.
        v0 = -self.z_valence * np.sqrt(2.0 / np.pi) / self.rc
        return np.where(r < 1e-12, v0, v)


def default_pseudopotential(symbol, atomic_number=None, rc=0.8):
    """
    Build a sensible built-in local pseudopotential for an element.

    Uses the :data:`~poraque.pseudopotentials.base.VALENCE_ELECTRONS`
    prescription to set the valence charge and returns a :class:`SoftCoulombPP`.

    Parameters
    ----------
    symbol : str
        Chemical symbol.
    atomic_number : int, optional
        Used as the all-electron fallback when the element has no tabulated
        valence count.
    rc : float, optional
        Core radius in Bohr.
    """
    zv = valence_electrons(symbol, atomic_number)
    return SoftCoulombPP(symbol, zv, rc=rc)
