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

    def normalized(self, n_electrons, clip_negative=True):
        r"""
        Rescale so the density integrates to exactly ``n_electrons``.

        A Kohn-Sham density integrates to the valence count the
        pseudopotentials fix; that number is an input to a DFT calculation, not
        an output of one. A *predicted* density has no such guarantee, and the
        error it carries is not benign: every electrostatic term is at least
        linear in :math:`\rho` and the Hartree energy is quadratic, so a 1 %
        drift moves a total energy by tens of eV — orders of magnitude more
        than the differences the energy is wanted for. Worse, the drift varies
        from structure to structure, so it does not cancel in a difference.

        Rescaling by a single global factor is the minimal repair: it restores
        the one exactly-known integral property without touching the shape of
        the field, which is what the operator was actually trained to predict.

        Parameters
        ----------
        n_electrons : float
            Target valence electron count, :math:`\sum_s N_s Z^{\rm val}_s`.
        clip_negative : bool, optional
            First clip small negative values to zero. Fourier-truncated
            densities ring slightly negative in the interstitial, which is
            unphysical and makes :math:`\rho^{4/3}` in the exchange energy
            ill-defined. On by default.

        Returns
        -------
        ChargeDensity
            A new field; ``self`` is unchanged.

        Raises
        ------
        ValueError
            If the density integrates to zero, leaving nothing to rescale.
        """
        values = np.asarray(self.data, dtype=float)
        if clip_negative:
            values = np.clip(values, 0.0, None)

        current = float(self.grid.integrate(values))
        if abs(current) < 1e-30:
            raise ValueError(
                "The density integrates to zero, so it cannot be normalized "
                "to a finite electron count."
            )

        metadata = dict(self.metadata or {})
        metadata["electron_count_before_normalization"] = current
        metadata["electron_count"] = float(n_electrons)
        return type(self)(values * (float(n_electrons) / current), self.grid,
                          self.structure, metadata=metadata)

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
    A ``TAUCAR`` written by VASP 6.6.1 with ``LTAU = .TRUE.`` differs from a
    ``CHGCAR`` in **both** of the ways a volumetric file can, and Poraquê once
    guessed wrong about each. Neither guess was detectable by inspection —
    which is the whole reason :mod:`poraque.data.validation` exists — so both
    are recorded here with the measurement that settled them, on the Pt data of
    2026-08-27 (31 bulk cells of 32 atoms, plus the isolated atom).

    **The values are not multiplied by the cell volume.** ``CHGCAR`` stores
    :math:`\rho\Omega`; ``TAUCAR`` stores :math:`\tau`. Read as though it
    were volume-scaled, :math:`\int\tau` came out :math:`9.2\times10^{-4}`
    of the Thomas-Fermi estimate in a 499.7 Å³ cell and
    :math:`5.5\times10^{-4}` in a 1000 Å³ one — the error tracking
    :math:`\Omega` exactly, which a unit confusion could not do.

    **The two blocks of a spin-polarised file are the two spin channels**,
    :math:`\tau_\uparrow` and :math:`\tau_\downarrow`, *not* the
    total/magnetisation pair a ``CHGCAR`` uses. So the total is their sum, and
    reading the first block alone loses half of :math:`\tau`. In the nearly
    unpolarised bulk cells the second block equals the first to
    :math:`3\times10^{-5}`, which no magnetisation is; in the isolated atom,
    genuinely polarised at 2 μ_B, it is 0.774 of it and still everywhere
    positive.

    Together the two corrections move :math:`\int\tau / \int\tau_{\rm TF}`
    from 0.46 to 0.92 in bulk and from 0.55 to 0.97 in the atom, and take the
    von Weizsäcker violation rate — a *theorem*, so the decisive test — from
    71 % of the atom's significant points to exactly zero in every system.
    """

    name = "kinetic energy density"
    default_filename = "TAUCAR"
    unit = "eV/Ang^3"
    volume_scaled = False
    reads_all_blocks = True

    @classmethod
    def combine_blocks(cls, raw, extra):
        r"""
        Sum the spin channels of a ``TAUCAR`` into the total :math:`\tau`.

        A one-block file is already the total: that is both an ``ISPIN = 1``
        run and every ``TAUCAR`` Poraquê writes itself, so a cache round trip
        does not double anything.

        Parameters
        ----------
        raw : numpy.ndarray
            First grid block — :math:`\tau_\uparrow` when there is a second.
        extra : sequence of numpy.ndarray
            The remaining blocks; one of them for ``ISPIN = 2``.

        Returns
        -------
        numpy.ndarray
        """
        if not extra:
            return raw
        total = np.array(raw, dtype=float, copy=True)
        for block in extra:
            total += block
        return total

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
