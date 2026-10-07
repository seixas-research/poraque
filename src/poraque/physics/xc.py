# -*- coding: utf-8 -*-
# file: xc.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Pointwise spin-polarized LDA and PBE, for the grid and for the PAW spheres.

:mod:`poraque.physics.energy` integrates the unpolarized functionals over a
plane-wave grid. The PAW total energy (:mod:`poraque.physics.paw`) needs the
same functionals pointwise --- on a radial grid times an angular one inside each
augmentation sphere, and on the plane-wave grid with a magnetization --- so they
are written here once, as functions of the spin densities and their gradients,
in Hartree atomic units:

* :func:`lda_xc` --- Slater exchange plus Perdew--Wang 1992 correlation with
  its full spin interpolation;
* :func:`pbe_xc` --- PBE exchange (spin-scaled) and correlation
  (:math:`\phi`-scaled), returned split into its LDA part and the gradient
  correction, as VASP accumulates them.

Every function returns an energy **per volume**, :math:`e = n\varepsilon`, in
Hartree/Bohr³. Densities below :data:`FLOOR` contribute nothing.

The functions take NumPy arrays or torch tensors and compute in whichever they
are handed, with the same expressions: the PAW energy is evaluated in NumPy
from files, and differentiated in torch for the forces
(:mod:`poraque.ml.energy`). Nothing here imports torch.

References
----------
J. P. Perdew and Y. Wang, *Phys. Rev. B* **45**, 13244 (1992); J. P. Perdew,
K. Burke and M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996); A. H.
MacDonald and S. H. Vosko, *J. Phys. C* **12**, 2977 (1979) for the
relativistic exchange factor.
"""

import numpy as np


class _Ops:
    """The handful of array operations used here, for NumPy or torch."""

    def __init__(self, module, is_torch):
        self.m = module
        self.torch = is_torch

    def asarray(self, x):
        if self.torch:
            return x
        return np.asarray(x, dtype=float)

    def where(self, condition, a, b):
        return self.m.where(condition, a, b)

    def clip(self, x, low=None, high=None):
        return self.m.clip(x, low, high)

    def __getattr__(self, name):
        return getattr(self.m, name)


def _ops(*values):
    for value in values:
        if type(value).__module__.startswith("torch"):
            import torch

            return _Ops(torch, True)
    return _Ops(np, False)

#: Densities (e/Bohr³) below this contribute no energy.
FLOOR = 1e-30

#: PBE constants. beta and kappa as in the original paper.
PBE_KAPPA = 0.804
PBE_BETA = 0.06672455060314922
PBE_MU = PBE_BETA * np.pi ** 2 / 3.0
PBE_GAMMA = (1.0 - np.log(2.0)) / np.pi ** 2

#: Speed of light in atomic units, for the relativistic exchange factor.
SPEED_OF_LIGHT = 137.035999

#: PW92 G(r_s) parameters: (A, alpha1, beta1, beta2, beta3, beta4).
_PW92_PARA = (0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
_PW92_FERRO = (0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
_PW92_ALPHA = (0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
_F2 = 1.709921     # f''(0) of the spin interpolation


def _g(rs, parameters):
    xp = _ops(rs)
    a, alpha1, beta1, beta2, beta3, beta4 = parameters
    root = xp.sqrt(rs)
    denominator = 2.0 * a * (beta1 * root + beta2 * rs + beta3 * rs * root
                             + beta4 * rs * rs)
    return -2.0 * a * (1.0 + alpha1 * rs) * xp.log1p(1.0 / denominator)


def _f_zeta(zeta):
    return (((1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0)
            / (2.0 ** (4.0 / 3.0) - 2.0))


def pw92_epsilon(n, zeta=0.0):
    """
    PW92 correlation energy per electron (Hartree), spin-interpolated.

    Parameters
    ----------
    n : numpy.ndarray
        Total density, e/Bohr³ (positive).
    zeta : numpy.ndarray or float
        Spin polarization ``(n_up - n_dn)/n``.
    """
    xp = _ops(n, zeta)
    rs = (3.0 / (4.0 * np.pi * n)) ** (1.0 / 3.0)
    paramagnetic = _g(rs, _PW92_PARA)
    if isinstance(zeta, float) and zeta == 0.0:
        return paramagnetic
    del xp
    ferro = _g(rs, _PW92_FERRO)
    alpha = -_g(rs, _PW92_ALPHA)
    f = _f_zeta(zeta)
    z4 = zeta ** 4
    return (paramagnetic + alpha * f / _F2 * (1.0 - z4)
            + (ferro - paramagnetic) * f * z4)


def _exchange_unpolarized(n, relativistic=False):
    """Slater exchange energy per volume of an unpolarized density n."""
    xp = _ops(n)
    e = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * n ** (4.0 / 3.0)
    if relativistic:
        beta = (3.0 * np.pi ** 2 * n) ** (1.0 / 3.0) / SPEED_OF_LIGHT
        small = beta < 1e-4
        safe = xp.where(small, xp.ones_like(beta), beta)
        root = xp.sqrt(1.0 + safe * safe)
        term = (safe * root - xp.arcsinh(safe)) / (safe * safe)
        factor = xp.where(small, 1.0 - (2.0 / 3.0) * beta ** 2,
                          1.0 - 1.5 * term ** 2)
        e = e * factor
    return e


def lda_xc(n_up, n_dn, relativistic=False):
    """
    LDA exchange-correlation energy per volume (Hartree/Bohr³).

    Parameters
    ----------
    n_up, n_dn : numpy.ndarray
        Spin densities, e/Bohr³.
    relativistic : bool
        Apply the MacDonald--Vosko factor to exchange.
    """
    xp = _ops(n_up, n_dn)
    n_up = xp.clip(xp.asarray(n_up), 0.0, None)
    n_dn = xp.clip(xp.asarray(n_dn), 0.0, None)
    n = n_up + n_dn
    active = n > FLOOR
    safe = xp.where(active, n, xp.ones_like(n))
    zeta = xp.where(active, (n_up - n_dn) / safe, xp.zeros_like(n))
    # Short of full polarization: (1 - zeta)^(1/3) has no derivative at 1.
    zeta = xp.clip(zeta, -1.0 + 1e-12, 1.0 - 1e-12)
    exchange = 0.5 * (_exchange_unpolarized(2.0 * n_up, relativistic)
                      + _exchange_unpolarized(2.0 * n_dn, relativistic))
    correlation = safe * pw92_epsilon(safe, zeta)
    return xp.where(active, exchange + correlation, xp.zeros_like(n))


def pbe_xc(n_up, n_dn, grad_up, grad_dn, grad_total, relativistic=False):
    """
    PBE energy per volume (Hartree/Bohr³), split as VASP accumulates it.

    Parameters
    ----------
    n_up, n_dn : numpy.ndarray
        Spin densities, e/Bohr³.
    grad_up, grad_dn, grad_total : numpy.ndarray
        ``|grad n_up|``, ``|grad n_dn|`` and ``|grad (n_up + n_dn)|``,
        e/Bohr⁴.
    relativistic : bool
        Relativistic factor on the LDA part of exchange (only).

    Returns
    -------
    tuple of numpy.ndarray
        ``(lda, gradient_correction)``; PBE is their sum.
    """
    xp = _ops(n_up, n_dn, grad_up, grad_dn, grad_total)
    n_up = xp.clip(xp.asarray(n_up), 0.0, None)
    n_dn = xp.clip(xp.asarray(n_dn), 0.0, None)
    lda = lda_xc(n_up, n_dn, relativistic=relativistic)

    def exchange_correction(n_sigma, grad_sigma):
        # PBE exchange of 2 n_sigma, halved: the spin-scaling relation.
        n2 = 2.0 * n_sigma
        active = n2 > FLOOR
        safe = xp.where(active, n2, xp.ones_like(n2))
        k_f = (3.0 * np.pi ** 2 * safe) ** (1.0 / 3.0)
        s2 = (2.0 * grad_sigma / (2.0 * k_f * safe)) ** 2
        enhancement = PBE_KAPPA - PBE_KAPPA / (1.0 + PBE_MU * s2 / PBE_KAPPA)
        e_x = _exchange_unpolarized(safe)
        return xp.where(active, 0.5 * e_x * enhancement, xp.zeros_like(n2))

    correction = (exchange_correction(n_up, xp.asarray(grad_up))
                  + exchange_correction(n_dn, xp.asarray(grad_dn)))

    n = n_up + n_dn
    active = n > FLOOR
    safe = xp.where(active, n, xp.ones_like(n))
    zeta = xp.clip(xp.where(active, (n_up - n_dn) / safe, xp.zeros_like(n)),
                   -1.0 + 1e-12, 1.0 - 1e-12)
    epsilon_c = pw92_epsilon(safe, zeta)
    phi = 0.5 * ((1.0 + zeta) ** (2.0 / 3.0) + (1.0 - zeta) ** (2.0 / 3.0))
    k_f = (3.0 * np.pi ** 2 * safe) ** (1.0 / 3.0)
    k_s = xp.sqrt(4.0 * k_f / np.pi)
    t2 = (xp.asarray(grad_total) / (2.0 * phi * k_s * safe)) ** 2
    phi3 = phi ** 3
    denominator = xp.expm1(-epsilon_c / (PBE_GAMMA * phi3))
    a = (PBE_BETA / PBE_GAMMA) / xp.clip(denominator, 1e-300, None)
    at2 = a * t2
    h = PBE_GAMMA * phi3 * xp.log1p((PBE_BETA / PBE_GAMMA) * t2
                                    * (1.0 + at2) / (1.0 + at2 + at2 * at2))
    correction = correction + xp.where(active, safe * h, xp.zeros_like(n))
    return lda, correction
