# -*- coding: utf-8 -*-
# file: kinetic.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Kinetic energy density functionals (KEDFs) for OF-DFT.

This module provides the explicit (semi-)local KEDFs used by the orbital-free
engine: the Thomas-Fermi functional, the von Weizsäcker functional, and their
combination (TF + lambda * vW).
"""

import numpy as np

from .base import Functional


class ThomasFermi(Functional):
    r"""
    Thomas-Fermi kinetic energy functional.

    .. math::

        T_{TF}[n] = C_{TF} \int n(\mathbf{r})^{5/3}\, d\mathbf{r}, \qquad
        C_{TF} = \frac{3}{10}(3\pi^2)^{2/3}

    Parameters
    ----------
    coeff : float, optional
        Overall scaling of the functional (default ``1.0``). Used by TFvW
        as a Pauli-term prefactor.
    """

    def __init__(self, coeff=1.0):
        super().__init__("Thomas-Fermi")
        self.coeff = coeff
        self.C_TF = 0.3 * (3 * np.pi**2) ** (2 / 3)

    def energy(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        e_dens = self.coeff * self.C_TF * n ** (5 / 3)
        return backend.integrate(e_dens, grid)

    def potential(self, density, system, grid, backend):
        n = np.maximum(density.data, 0.0)
        return self.coeff * (5 / 3) * self.C_TF * n ** (2 / 3)


class VonWeizsaecker(Functional):
    r"""
    von Weizsäcker kinetic energy functional.

    .. math::

        T_{vW}[n] = \frac{\lambda}{8} \int \frac{|\nabla n|^2}{n}\, d\mathbf{r}
                  = -\frac{\lambda}{2} \int \sqrt{n}\, \nabla^2 \sqrt{n}\, d\mathbf{r}

    Parameters
    ----------
    lambda_ : float, optional
        Mixing/prefactor :math:`\lambda` (default ``1.0`` = full vW).
    laplacian : {"fd", "fft"}, optional
        Discretization of :math:`\nabla^2\sqrt{n}`. ``"fd"`` (default) uses the
        backend's local finite-difference stencil; ``"fft"`` uses the spectral
        reciprocal-space Laplacian, which is more accurate for well-resolved,
        band-limited periodic densities. **Note:** the spectral operator is
        globally nonlocal, so it spreads the von Weizsäcker potential across the
        whole cell and breaks the short-rangedness that subsystem/embedding
        (FDE) nonadditive potentials rely on; keep ``"fd"`` for embedding and
        coarse molecular grids. See ``plan/claude_report.md``.
    """

    def __init__(self, lambda_=1.0, laplacian="fd"):
        super().__init__("von Weizsäcker")
        self.lambda_ = lambda_
        if laplacian not in ("fd", "fft"):
            raise ValueError(f"laplacian must be 'fd' or 'fft', got {laplacian!r}")
        self.laplacian = laplacian

    def _laplacian(self, field, grid, backend):
        if self.laplacian == "fft":
            return backend.laplacian_fft(field, grid)
        return backend.laplacian(field, grid)

    def energy(self, density, system, grid, backend):
        sqrt_n = np.sqrt(np.maximum(density.data, 0.0))
        lap_sqrt_n = self._laplacian(sqrt_n, grid, backend)
        return -0.5 * self.lambda_ * backend.integrate(sqrt_n * lap_sqrt_n, grid)

    def potential(self, density, system, grid, backend):
        sqrt_n = np.sqrt(np.maximum(density.data, 0.0))
        safe_sqrt_n = np.where(sqrt_n > 1e-12, sqrt_n, 1e-12)
        lap_sqrt_n = self._laplacian(sqrt_n, grid, backend)
        return -0.5 * self.lambda_ * lap_sqrt_n / safe_sqrt_n


class TFvW(Functional):
    r"""
    Thomas-Fermi-von Weizsäcker kinetic energy functional.

    .. math::

        T[n] = c_{TF}\, T_{TF}[n] + \lambda\, T_{vW}[n]

    Common choices are ``lambda_vw = 1`` (full gradient correction),
    ``lambda_vw = 1/9`` (second-order gradient expansion), and the pure vW
    limit ``c_tf = 0, lambda_vw = 1`` (exact for one- and two-electron
    densities).

    Parameters
    ----------
    lambda_vw : float, optional
        Prefactor of the von Weizsäcker term (default ``1.0``).
    c_tf : float, optional
        Prefactor of the Thomas-Fermi term (default ``1.0``).
    laplacian : {"fd", "fft"}, optional
        Laplacian discretization for the von Weizsäcker term (default ``"fd"``).
        See :class:`VonWeizsaecker`.
    """

    def __init__(self, lambda_vw=1.0, c_tf=1.0, laplacian="fd"):
        super().__init__("Thomas-Fermi-von Weizsäcker")
        self.lambda_vw = lambda_vw
        self.c_tf = c_tf
        self._tf = ThomasFermi(coeff=c_tf)
        self._vw = VonWeizsaecker(lambda_=lambda_vw, laplacian=laplacian)

    def energy(self, density, system, grid, backend):
        return (self._tf.energy(density, system, grid, backend)
                + self._vw.energy(density, system, grid, backend))

    def potential(self, density, system, grid, backend):
        return (self._tf.potential(density, system, grid, backend)
                + self._vw.potential(density, system, grid, backend))
