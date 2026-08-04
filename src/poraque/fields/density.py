# -*- coding: utf-8 -*-
# file: density.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Charge density (``CHGCAR``) and kinetic energy density (``TAUCAR``).

Both classes are the companions of
:class:`~poraque.fields.ExternalPotential` on the *same*
:class:`~poraque.fields.FieldGrid`. They are fully functional for **reading and
writing** — which is all the machine-learning pipeline in :mod:`poraque.ml`
needs, since ``CHGCAR`` and ``TAUCAR`` are produced by a plane-wave DFT code
and read from disk. Solving for these fields is outside the scope of this
package, so the ``compute()`` constructors raise
:class:`NotImplementedError`.
"""

import numpy as np

from .base import ScalarField
from .constants import C_TF, HARTREE_TO_EV, BOHR_TO_ANGSTROM


class ChargeDensity(ScalarField):
    r"""
    Valence electron density :math:`\rho(\mathbf{r})` in electrons/Å³.

    Notes
    -----
    VASP writes ``CHGCAR`` as :math:`\rho(\mathbf{r})\,\Omega` — the density
    multiplied by the cell volume — so :attr:`volume_scaled` is true and the
    conversion happens transparently in
    :meth:`~poraque.fields.base.ScalarField.read` and
    :meth:`~poraque.fields.base.ScalarField.write`. Consequently
    :meth:`~poraque.fields.base.ScalarField.integrate` returns the electron
    count directly.
    """

    name = "valence charge density"
    default_filename = "CHGCAR"
    unit = "e/Ang^3"
    volume_scaled = True

    def electron_count(self):
        """Number of valence electrons in the cell."""
        return self.integrate()

    @classmethod
    def compute(cls, *args, **kwargs):
        """
        Solve for the density self-consistently.

        Raises
        ------
        NotImplementedError
            Always. Poraquê consumes densities from a plane-wave DFT code
            rather than computing them; read one with
            :meth:`~poraque.fields.base.ScalarField.read`.
        """
        raise NotImplementedError(
            "ChargeDensity is read, not computed: use "
            "ChargeDensity.read(path, grid=shared_grid)."
        )


class KineticEnergyDensity(ScalarField):
    r"""
    Positive-definite kinetic energy density :math:`\tau(\mathbf{r})` in eV/Å³.

    Notes
    -----
    Unlike ``CHGCAR``, no single ``TAUCAR`` convention is fixed by VASP itself.
    Poraquê adopts the ``CHGCAR`` one — values multiplied by the cell volume —
    so that ``EXTCAR``, ``CHGCAR`` and ``TAUCAR`` differ only in *what* they
    store, never in *how*. Subclass and flip :attr:`volume_scaled` if your
    generator writes raw values.
    """

    name = "kinetic energy density"
    default_filename = "TAUCAR"
    unit = "eV/Ang^3"
    volume_scaled = True

    def kinetic_energy(self):
        """Total kinetic energy in eV."""
        return self.integrate()

    @classmethod
    def compute(cls, *args, **kwargs):
        """
        Evaluate :math:`\\tau` from orbitals or from a KEDF.

        Raises
        ------
        NotImplementedError
            Always. Read a reference ``TAUCAR``, use
            :func:`thomas_fermi_tau` / :func:`von_weizsacker_tau` for the
            orbital-free approximants, or predict one with a trained
            ``chg2tau`` operator.
        """
        raise NotImplementedError(
            "KineticEnergyDensity is read or predicted, not computed: use "
            "KineticEnergyDensity.read(path, grid=shared_grid), an analytic "
            "KEDF, or a trained chg2tau operator."
        )


# ---------------------------------------------------------------------- #
# Orbital-free kinetic energy densities
#
# These closed-form functionals of the density are the physical anchors of the
# PI-FNO programme (see docs/notes/pi_fno.md): tau_vW is a rigorous *lower* bound on
# the exact tau, and tau_TF its high-density limit, so together they bracket
# any prediction a network makes for TAUCAR.
# ---------------------------------------------------------------------- #
#: eV/Å³ per Hartree/Bohr³ — converts atomic-unit energy densities to VASP units.
_HA_PER_BOHR3_TO_EV_PER_ANG3 = HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3


def thomas_fermi_tau(density):
    r"""
    Thomas-Fermi kinetic energy density,
    :math:`\tau_{\rm TF} = C_{\rm TF}\,\rho^{5/3}`.

    Parameters
    ----------
    density : ChargeDensity or array_like
        Electron density in electrons/Å³.

    Returns
    -------
    numpy.ndarray
        :math:`\tau_{\rm TF}` in eV/Å³.
    """
    rho = np.asarray(density, dtype=float)
    rho_bohr = rho * BOHR_TO_ANGSTROM ** 3          # e/Bohr^3
    tau_atomic = C_TF * np.clip(rho_bohr, 0.0, None) ** (5.0 / 3.0)
    return tau_atomic * _HA_PER_BOHR3_TO_EV_PER_ANG3


def von_weizsacker_tau(density, grid, epsilon=1e-12):
    r"""
    von Weizsäcker kinetic energy density,
    :math:`\tau_{\rm vW} = |\nabla\rho|^2 / (8\rho)`.

    This is the exact :math:`\tau` for a one-orbital system and a rigorous
    lower bound in general, which makes ``tau >= tau_vW`` a usable hard
    physical constraint on any learned ``CHGCAR -> TAUCAR`` map.

    Parameters
    ----------
    density : ChargeDensity or array_like
        Electron density in electrons/Å³.
    grid : FieldGrid
        Mesh supplying the reciprocal vectors used for the spectral gradient.
    epsilon : float, optional
        Floor on :math:`\rho` guarding the division.

    Returns
    -------
    numpy.ndarray
        :math:`\tau_{\rm vW}` in eV/Å³.
    """
    rho = np.asarray(density, dtype=float)
    rho_bohr = rho * BOHR_TO_ANGSTROM ** 3
    gradient = spectral_gradient(rho_bohr, grid, length_unit="bohr")
    grad_squared = sum(component ** 2 for component in gradient)
    tau_atomic = grad_squared / (8.0 * np.clip(rho_bohr, epsilon, None))
    return tau_atomic * _HA_PER_BOHR3_TO_EV_PER_ANG3


def spectral_gradient(field, grid, length_unit="angstrom"):
    r"""
    Gradient of a periodic field via FFT, :math:`\nabla f \to i\mathbf{G}\hat f`.

    Spectral differentiation is exact for band-limited periodic fields, which
    is precisely the class of functions a plane-wave DFT grid carries — a
    finite-difference stencil would introduce an error the physics losses would
    then have to absorb.

    Parameters
    ----------
    field : array_like
        Real array of shape ``grid.shape``.
    grid : FieldGrid
        Mesh supplying the reciprocal vectors (Å⁻¹).
    length_unit : {"angstrom", "bohr"}, optional
        Unit of the differentiation variable in the returned derivative.

    Returns
    -------
    tuple of numpy.ndarray
        ``(df/dx, df/dy, df/dz)``.
    """
    field_g = np.fft.fftn(np.asarray(field, dtype=float))
    scale = 1.0 if length_unit == "angstrom" else BOHR_TO_ANGSTROM
    return tuple(
        np.real(np.fft.ifftn(1j * component * scale * field_g))
        for component in grid.get_g_vectors()
    )
