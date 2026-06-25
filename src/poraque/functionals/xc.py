# -*- coding: utf-8 -*-
# file: xc.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Local-density approximation exchange-correlation functionals.

The functionals here follow the shared :class:`~poraque.functionals.base.Functional`
interface (``energy`` and ``potential``) so they can be reused unchanged by both
the OF-DFT engine and the KS-DFT engine.
"""

import numpy as np

from .base import Functional

# Small density floor to keep n^p and 1/n stable in vacuum regions.
_DENS_FLOOR = 1e-30


class DiracExchange(Functional):
    r"""
    Dirac (LDA / Slater) exchange functional.

    .. math::

        E_x[n] = -C_x \int n(\mathbf{r})^{4/3}\, d\mathbf{r}, \qquad
        C_x = \frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}

    with potential :math:`v_x = -\frac{4}{3} C_x\, n^{1/3}`.
    """

    def __init__(self):
        super().__init__("Exchange (Dirac)")
        self.C_x = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)

    def energy(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        return -self.C_x * backend.integrate(n ** (4.0 / 3.0), grid)

    def potential(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        return -(4.0 / 3.0) * self.C_x * n ** (1.0 / 3.0)


class PW92Correlation(Functional):
    """
    Perdew-Wang (1992) LDA correlation functional (spin-unpolarized).

    Implements the parametrization of the correlation energy per electron
    ``eps_c(rs)`` and its corresponding potential via
    ``v_c = eps_c - (rs/3) d eps_c / d rs``.
    """

    # Parameters for the paramagnetic (unpolarized) correlation energy.
    _A = 0.031091
    _alpha1 = 0.21370
    _beta1 = 7.5957
    _beta2 = 3.5876
    _beta3 = 1.6382
    _beta4 = 0.49294

    def __init__(self):
        super().__init__("Correlation (PW92)")

    def _rs(self, n):
        n = np.maximum(n, _DENS_FLOOR)
        return (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)

    def _eps_and_deps(self, rs):
        """Return ``eps_c(rs)`` and ``d eps_c / d rs``."""
        A, a1 = self._A, self._alpha1
        b1, b2, b3, b4 = self._beta1, self._beta2, self._beta3, self._beta4

        sqrt_rs = np.sqrt(rs)
        # Q = 2A (b1 rs^1/2 + b2 rs + b3 rs^3/2 + b4 rs^2)
        Q = 2.0 * A * (b1 * sqrt_rs + b2 * rs + b3 * rs * sqrt_rs + b4 * rs**2)
        # dQ/drs
        dQ = 2.0 * A * (0.5 * b1 / sqrt_rs + b2 + 1.5 * b3 * sqrt_rs + 2.0 * b4 * rs)

        log_term = np.log1p(1.0 / Q)
        eps = -2.0 * A * (1.0 + a1 * rs) * log_term

        # d eps / drs
        d_log = -(1.0 / (Q * (Q + 1.0))) * dQ  # d/drs log(1 + 1/Q)
        deps = -2.0 * A * (a1 * log_term + (1.0 + a1 * rs) * d_log)
        return eps, deps

    def energy(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        rs = self._rs(n)
        eps, _ = self._eps_and_deps(rs)
        return backend.integrate(n * eps, grid)

    def potential(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        rs = self._rs(n)
        eps, deps = self._eps_and_deps(rs)
        # v_c = eps - (rs/3) d eps/d rs
        v_c = eps - (rs / 3.0) * deps
        # Zero out the (numerically meaningless) vacuum contribution.
        return np.where(density.data > _DENS_FLOOR, v_c, 0.0)


class LDA(Functional):
    """
    Convenience exchange-correlation functional combining Dirac exchange and
    PW92 correlation.

    Parameters
    ----------
    correlation : bool, optional
        Include PW92 correlation (default ``True``). When ``False`` this is a
        pure exchange-only LDA.
    """

    def __init__(self, correlation=True):
        super().__init__("XC (LDA)")
        self._x = DiracExchange()
        self._c = PW92Correlation() if correlation else None

    def energy(self, density, system, grid, backend):
        e = self._x.energy(density, system, grid, backend)
        if self._c is not None:
            e += self._c.energy(density, system, grid, backend)
        return e

    def potential(self, density, system, grid, backend):
        v = self._x.potential(density, system, grid, backend)
        if self._c is not None:
            v = v + self._c.potential(density, system, grid, backend)
        return v
