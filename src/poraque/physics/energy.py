# -*- coding: utf-8 -*-
# file: energy.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Total-energy components from the three predicted fields.

Poraquê predicts :math:`\rho(\mathbf r)` and :math:`\tau(\mathbf r)` and
computes :math:`V_{\rm ext}(\mathbf r)` analytically. This module turns that
triple into energies by integrating the Kohn-Sham total-energy expression on
the shared grid,

.. math::

    E = \underbrace{\int\tau\,d^3r}_{T_{\rm s}}
      + \underbrace{\int\rho V_{\rm ext}\,d^3r + E_{\alpha Z}}_{E_{\rm ext}}
      + \underbrace{\tfrac12\int\rho v_{\rm H}\,d^3r}_{E_{\rm H}}
      + E_{\rm xc}[\rho]
      + E_{\rm Ewald},

with every term in eV.

Conventions
-----------
**The** :math:`\mathbf G = 0` **bookkeeping is the subtle part.** Each of the
three electrostatic terms — electron-ion, Hartree, ion-ion — diverges at
:math:`\mathbf G = 0`, and the divergences cancel exactly in a neutral cell.
The standard plane-wave treatment, which Poraquê follows, drops
:math:`\mathbf G = 0` from all three and adds back the finite remainder of the
electron-ion term:

.. math::

    E_{\alpha Z} = \frac{N_{\rm elec}}{\Omega}\sum_s N_s\,
                   \mathrm{PSCORE}_s ,
    \qquad
    \mathrm{PSCORE}_s = \lim_{q\to0} v^{s}_{\rm short}(q),

which is VASP's ``alpha Z`` term. It is read straight from the ``POTCAR``
tables when they are available, and is otherwise reported as ``None`` rather
than silently assumed to be zero — it is a per-atom constant of order eV, so
quietly omitting it would corrupt any comparison between cells of different
composition.

**Signs.** :math:`V_{\rm ext}` is the potential energy *of an electron* (eV,
negative near the ions) and :math:`\rho` is the positive electron number
density, so :math:`\int\rho V_{\rm ext}` is directly the interaction energy.

What this is not
----------------
.. warning::

   These are **pseudo-valence** energies. ``CHGCAR`` holds the valence pseudo
   density and ``TAUCAR`` the valence pseudo kinetic energy density, so the
   PAW one-centre terms — the difference between the pseudo and all-electron
   descriptions inside the augmentation spheres — are absent entirely. The
   absolute number will therefore **not** reproduce a VASP total energy, and no
   amount of tuning the terms here will make it.

   What survives is the *variation* with geometry, since the missing one-centre
   terms are dominated by a per-atom constant. Use this for energy
   **differences** between structures of the same composition, and calibrate
   against reference DFT before trusting the scale. :attr:`EnergyComponents.total`
   carries this caveat in its own docstring for the same reason.
"""

from dataclasses import dataclass

import numpy as np

from ..fields.constants import (
    BOHR_TO_ANGSTROM,
    COULOMB_CONSTANT_EV_ANGSTROM,
    HARTREE_TO_EV,
)

#: eV per Hartree, per Å³ per Bohr³ — converts an atomic-unit energy *density*.
_HA_PER_BOHR3_TO_EV_PER_ANG3 = HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3

#: Dirac exchange coefficient :math:`-\frac34(3/\pi)^{1/3}`, Hartree atomic units.
_DIRAC_X = -0.75 * (3.0 / np.pi) ** (1.0 / 3.0)

# Perdew-Wang 1992 correlation, unpolarized branch (PRB 45, 13244).
_PW92 = dict(A=0.031091, alpha1=0.21370,
             beta1=7.5957, beta2=3.5876, beta3=1.6382, beta4=0.49294)

# PBE (PRL 77, 3865). kappa is fixed by the Lieb-Oxford bound, mu and beta by
# the linear response of the uniform gas; mu = beta * pi^2 / 3.
_PBE_KAPPA = 0.804
_PBE_BETA = 0.06672455060314922
_PBE_MU = _PBE_BETA * np.pi ** 2 / 3.0
#: :math:`\gamma = (1 - \ln 2)/\pi^2`.
_PBE_GAMMA = (1.0 - np.log(2.0)) / np.pi ** 2

#: Exchange-correlation approximations accepted by :func:`xc_energy`.
#: ``"pbe"`` is the default: the reference calculations use ``PAW_PBE``
#: pseudopotentials with ``LEXCH = PE``, so the fields being integrated are
#: PBE quantities.
XC_FUNCTIONALS = ("pbe", "lda", "pbe-x", "lda-x", "x-only", "none")


# ===================================================================== #
# Electrostatics of the electrons
# ===================================================================== #
def hartree_potential(density, grid):
    r"""
    Hartree potential of an electron density.

    Solves :math:`\nabla^2 v_{\rm H} = -4\pi e^2\rho` in reciprocal space,

    .. math:: v_{\rm H}(\mathbf G) = \frac{4\pi e^2\rho(\mathbf G)}{G^2},
              \qquad v_{\rm H}(\mathbf G = 0) = 0 .

    Dropping :math:`\mathbf G = 0` is the neutralizing-background convention,
    the same one :class:`~poraque.fields.ExternalPotential` uses, so the two
    potentials are directly addable.

    Parameters
    ----------
    density : ChargeDensity or array_like
        Electron density in e/Å³.
    grid : FieldGrid or array_like
        Shared mesh, or the ``(3, 3)`` lattice vectors in Å — in which case a
        grid is built from them and the density's own shape. The second form
        exists so the solver can be called without first constructing a
        :class:`~poraque.fields.FieldGrid`.

    Returns
    -------
    numpy.ndarray
        Potential energy of an electron in eV, shape ``grid.shape``. Positive
        (repulsive), as it must be.

    Notes
    -----
    The factor is :math:`4\pi e^2` with :math:`e^2` in eV·Å
    (:data:`~poraque.fields.constants.COULOMB_CONSTANT_EV_ANGSTROM`), so a
    density in e/Å³ and a cell in Å give a potential in **eV** with no further
    conversion. Working in Hartree atomic units and converting afterwards
    would be the usual source of a factor of two here.
    """
    values = np.asarray(density, dtype=float)
    grid = _as_grid(grid, values.shape)
    _check_shape(values, grid, "density")

    g2 = grid.get_g2()
    kernel = np.zeros_like(g2)
    nonzero = g2 > 1e-12
    kernel[nonzero] = 1.0 / g2[nonzero]

    transformed = np.fft.fftn(values)
    return np.real(np.fft.ifftn(
        4.0 * np.pi * COULOMB_CONSTANT_EV_ANGSTROM * kernel * transformed
    ))


def hartree_energy(density, grid):
    r"""
    Hartree (classical electron-electron) energy,
    :math:`\tfrac12\int\rho\,v_{\rm H}\,d^3r`.

    Parameters
    ----------
    density : ChargeDensity or array_like
        Electron density in e/Å³.
    grid : FieldGrid
        Shared mesh.

    Returns
    -------
    float
        Energy in eV, non-negative.

    Notes
    -----
    A uniform density gives exactly zero: with :math:`\mathbf G = 0` removed,
    the electrons are neutralized by a uniform background of the same charge,
    and a uniform system has no self-interaction left. That is the correct
    result in this convention, not a numerical accident.
    """
    values = np.asarray(density, dtype=float)
    return 0.5 * grid.integrate(values * hartree_potential(values, grid))


# ===================================================================== #
# Exchange and correlation
# ===================================================================== #
def lda_exchange_energy(density, grid):
    r"""
    Dirac (local density approximation) exchange energy.

    .. math:: E_{\rm x} = -\frac34\left(\frac3\pi\right)^{1/3}
              \int\rho^{4/3}\,d^3r .

    Parameters
    ----------
    density : ChargeDensity or array_like
        Electron density in e/Å³.
    grid : FieldGrid
        Shared mesh.

    Returns
    -------
    float
        Energy in eV, negative.

    Notes
    -----
    The density is clipped at zero first. A *predicted* field, or a reference
    one that has been band-limited, can undershoot slightly into negative
    values from Gibbs ringing; :math:`\rho^{4/3}` of a negative number is not
    real, and the physical content of those points is "no electrons here".
    """
    rho_bohr = _clipped_bohr_density(density, grid)
    epsilon = _DIRAC_X * rho_bohr ** (4.0 / 3.0)          # Hartree/Bohr^3
    return grid.integrate(epsilon * _HA_PER_BOHR3_TO_EV_PER_ANG3)


def pw92_correlation_energy(density, grid):
    r"""
    Perdew-Wang 1992 correlation energy, unpolarized branch.

    .. math::

        \varepsilon_{\rm c}(r_s) = -2A(1+\alpha_1 r_s)\,
        \ln\!\left[1 + \frac{1}{2A\left(\beta_1 r_s^{1/2}
        + \beta_2 r_s + \beta_3 r_s^{3/2} + \beta_4 r_s^{2}\right)}\right]

    with :math:`r_s = (3/4\pi\rho)^{1/3}` and
    :math:`E_{\rm c} = \int\rho\,\varepsilon_{\rm c}\,d^3r`.

    Parameters
    ----------
    density : ChargeDensity or array_like
        Electron density in e/Å³.
    grid : FieldGrid
        Shared mesh.

    Returns
    -------
    float
        Energy in eV, negative.

    References
    ----------
    J. P. Perdew and Y. Wang, *Phys. Rev. B* **45**, 13244 (1992).
    """
    rho_bohr = _clipped_bohr_density(density, grid)

    # r_s diverges as rho -> 0. The floor keeps it finite; the accompanying
    # `rho * epsilon_c` vanishes there anyway, so the choice of floor does not
    # affect the integral, only the arithmetic on the way to it.
    safe = np.clip(rho_bohr, 1e-30, None)
    epsilon_c = _pw92_epsilon(safe)
    return grid.integrate(rho_bohr * epsilon_c * _HA_PER_BOHR3_TO_EV_PER_ANG3)


def _pw92_epsilon(density_bohr):
    r"""
    PW92 correlation energy *per electron*, Hartree.

    Shared by :func:`pw92_correlation_energy` and
    :func:`pbe_correlation_energy` — PBE's :math:`H` is a correction added to
    exactly this quantity, so the two must not drift apart.

    Parameters
    ----------
    density_bohr : numpy.ndarray
        Density in e/Bohr³, already floored away from zero.
    """
    r_s = (3.0 / (4.0 * np.pi * density_bohr)) ** (1.0 / 3.0)
    root = np.sqrt(r_s)

    p = _PW92
    denominator = 2.0 * p["A"] * (p["beta1"] * root + p["beta2"] * r_s
                                  + p["beta3"] * r_s * root
                                  + p["beta4"] * r_s * r_s)
    return (-2.0 * p["A"] * (1.0 + p["alpha1"] * r_s)
            * np.log1p(1.0 / denominator))


def pbe_exchange_energy(density, grid):
    r"""
    PBE (generalized gradient) exchange energy.

    .. math::

        E_{\rm x}^{\rm PBE} = \int \rho\,\varepsilon_{\rm x}^{\rm LDA}(\rho)\,
        F_{\rm x}(s)\,d^3r ,
        \qquad
        F_{\rm x}(s) = 1 + \kappa - \frac{\kappa}{1 + \mu s^2/\kappa} ,

    with the reduced gradient :math:`s = |\nabla\rho|/(2k_{\rm F}\rho)` and
    :math:`k_{\rm F} = (3\pi^2\rho)^{1/3}`. :math:`\kappa = 0.804` is fixed by
    the Lieb-Oxford bound and :math:`\mu = \beta\pi^2/3` by the linear response
    of the uniform gas — PBE has no fitted parameters.

    Parameters
    ----------
    density : ChargeDensity or array_like
        Electron density in e/Å³.
    grid : FieldGrid
        Shared mesh, supplying the reciprocal vectors for the gradient.

    Returns
    -------
    float
        Energy in eV, negative.

    Notes
    -----
    :math:`F_{\rm x}` is bounded by :math:`1 + \kappa`, so the integrand stays
    finite where :math:`\rho \to 0` even though :math:`s` diverges there: the
    prefactor :math:`\rho^{4/3}` vanishes faster.

    References
    ----------
    J. P. Perdew, K. Burke and M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865
    (1996).
    """
    rho, gradient = _density_and_gradient(density, grid)
    s2 = _reduced_gradient_squared(rho, gradient)

    enhancement = 1.0 + _PBE_KAPPA - _PBE_KAPPA / (1.0 + _PBE_MU * s2 / _PBE_KAPPA)
    epsilon = _DIRAC_X * rho ** (4.0 / 3.0) * enhancement
    return grid.integrate(epsilon * _HA_PER_BOHR3_TO_EV_PER_ANG3)


def pbe_correlation_energy(density, grid):
    r"""
    PBE (generalized gradient) correlation energy.

    .. math::

        E_{\rm c}^{\rm PBE} = \int\rho\left[
          \varepsilon_{\rm c}^{\rm PW92}(r_s) + H(r_s, t)\right] d^3r ,

    .. math::

        H = \gamma\ln\!\left[1 + \frac{\beta}{\gamma}t^2\,
            \frac{1 + At^2}{1 + At^2 + A^2t^4}\right],
        \qquad
        A = \frac{\beta}{\gamma}
            \left[e^{-\varepsilon_{\rm c}/\gamma} - 1\right]^{-1},

    with :math:`t = |\nabla\rho|/(2k_s\rho)` and
    :math:`k_s = \sqrt{4k_{\rm F}/\pi}`. The spin-scaling factor
    :math:`\phi` is 1 throughout: Poraquê's fields are spin-unpolarized.

    Parameters
    ----------
    density : ChargeDensity or array_like
        Electron density in e/Å³.
    grid : FieldGrid
        Shared mesh.

    Returns
    -------
    float
        Energy in eV, negative.

    Notes
    -----
    :math:`H \to 0` as :math:`\nabla\rho \to 0`, so PBE correlation reduces
    exactly to PW92 on a uniform density — the same limit that makes PBE
    exchange reduce to Dirac. Both are checked in the test-suite.

    References
    ----------
    J. P. Perdew, K. Burke and M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865
    (1996).
    """
    rho, gradient = _density_and_gradient(density, grid)
    safe = np.clip(rho, 1e-30, None)

    epsilon_c = _pw92_epsilon(safe)

    k_f = (3.0 * np.pi ** 2 * safe) ** (1.0 / 3.0)
    k_s = np.sqrt(4.0 * k_f / np.pi)
    t2 = gradient / (2.0 * k_s * safe) ** 2

    # A = (beta/gamma) / (exp(-eps_c/gamma) - 1). eps_c is negative, so the
    # exponential exceeds 1 and the denominator is positive; expm1 keeps it
    # accurate where eps_c is small and the difference would cancel.
    denominator = np.expm1(-epsilon_c / _PBE_GAMMA)
    a_coefficient = (_PBE_BETA / _PBE_GAMMA) / np.clip(denominator, 1e-30, None)

    at2 = a_coefficient * t2
    ratio = (1.0 + at2) / (1.0 + at2 + at2 * at2)
    h = _PBE_GAMMA * np.log1p((_PBE_BETA / _PBE_GAMMA) * t2 * ratio)

    return grid.integrate(rho * (epsilon_c + h) * _HA_PER_BOHR3_TO_EV_PER_ANG3)


def xc_energy(density, grid, functional="pbe"):
    r"""
    Exchange-correlation energy.

    Parameters
    ----------
    density : ChargeDensity or array_like
        Electron density in e/Å³.
    grid : FieldGrid
        Shared mesh.
    functional : str, optional
        One of :data:`XC_FUNCTIONALS`:

        ``"pbe"``
            PBE exchange + PBE correlation (**default**).
        ``"lda"``
            Dirac exchange + PW92 correlation.
        ``"pbe-x"``
            PBE exchange alone.
        ``"lda-x"`` / ``"x-only"``
            Dirac exchange alone.
        ``"none"``
            Zero, for isolating the other terms.

    Returns
    -------
    float
        Energy in eV.

    Raises
    ------
    ValueError
        On an unrecognized name, listing the valid ones. Falling back to a
        default would silently answer a different question than the one asked.

    Notes
    -----
    PBE is the default because it matches the reference data: the calculations
    Poraquê ingests use ``PAW_PBE`` pseudopotentials with ``LEXCH = PE``, so
    :math:`\rho` and :math:`\tau` are PBE quantities. Evaluating an LDA
    :math:`E_{\rm xc}` on a PBE density is not a PBE energy and not an LDA one
    either. Change this only to match a differently generated dataset.

    On the reference Au supercells the two differ by :math:`-0.92` eV/atom
    (0.65 % of :math:`E_{\rm xc}`), so the choice is not cosmetic.

    .. warning::

       PBE is semilocal and needs :math:`\nabla\rho`. On a *predicted* field
       that gradient carries the network's noise, and the enhancement factor
       amplifies it; on a band-limited grid it also aliases wherever the
       density has sharp core peaks. The extra physics is only worth having
       once the density is accurate enough for its gradient to mean something
       — check both against a reference before trusting the difference.
    """
    name = str(functional).lower()
    if name == "none":
        return 0.0
    if name in ("lda-x", "x-only"):
        return lda_exchange_energy(density, grid)
    if name == "pbe-x":
        return pbe_exchange_energy(density, grid)
    if name == "lda":
        return (lda_exchange_energy(density, grid)
                + pw92_correlation_energy(density, grid))
    if name == "pbe":
        return (pbe_exchange_energy(density, grid)
                + pbe_correlation_energy(density, grid))
    raise ValueError(
        f"Unknown functional {functional!r}; expected one of "
        f"{', '.join(XC_FUNCTIONALS)}."
    )


# ===================================================================== #
# Electrostatics of the ions
# ===================================================================== #
def ewald_energy(structure, charges, accuracy=1e-12):
    r"""
    Ewald energy of the pseudo-ions in a neutralizing background.

    .. math::

        E = \frac{e^2}{2}\sideset{}{'}\sum_{i,j,\mathbf R}
              \frac{q_iq_j\,\mathrm{erfc}(\eta|\mathbf r_{ij}+\mathbf R|)}
                   {|\mathbf r_{ij}+\mathbf R|}
          + \frac{2\pi e^2}{\Omega}\sum_{\mathbf G\neq0}
              \frac{e^{-G^2/4\eta^2}}{G^2}|S(\mathbf G)|^2
          - \frac{\eta e^2}{\sqrt\pi}\sum_i q_i^2
          - \frac{\pi e^2}{2\eta^2\Omega}\Bigl(\sum_i q_i\Bigr)^2 .

    Parameters
    ----------
    structure : Structure
        Geometry; only the cell and the Cartesian positions are used.
    charges : dict or array_like
        ``{element: Z_val}`` pseudo-ion charges in units of ``+e``, or one
        charge per atom. Element keys are matched against the *bare* element
        name, so ``Au`` covers a ``Au_pv`` POTCAR.
    accuracy : float, optional
        Target relative truncation error; sets both cutoffs.

    Returns
    -------
    float
        Energy in eV.

    Notes
    -----
    The last term is the interaction of the ions with the compensating
    background. It is not optional: without it the result depends on the
    arbitrary splitting parameter :math:`\eta` instead of being invariant
    under it, which is the cheapest available test that the implementation is
    right.
    """
    cell = np.asarray(structure.cell, dtype=float)
    positions = np.asarray(structure.positions, dtype=float)
    volume = float(abs(np.linalg.det(cell)))
    q = _per_atom_charges(structure, charges)
    natoms = len(q)

    k_e = COULOMB_CONSTANT_EV_ANGSTROM

    # eta balances the work between the two sums; the result is independent of
    # it, so this choice is about cost, not correctness.
    eta = np.sqrt(np.pi) * (natoms / volume ** 2) ** (1.0 / 6.0)
    span = np.sqrt(-np.log(accuracy))
    r_cut, g_cut = span / eta, 2.0 * eta * span

    # ---- real space ------------------------------------------------- #
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    repeats = _shell_counts(cell, r_cut)
    lattice = _lattice_points(repeats) @ cell

    delta = positions[:, None, :] - positions[None, :, :]        # (N, N, 3)
    real_sum = 0.0
    for shift in lattice:
        distance = np.linalg.norm(delta + shift, axis=-1)
        # Skip the i == j self term in the home cell only; the same pair in an
        # image cell is a genuine interaction.
        mask = distance > 1e-8
        if not mask.any():
            continue
        pair = q[:, None] * q[None, :]
        real_sum += np.sum(np.where(
            mask & (distance < r_cut),
            pair * _erfc(eta * distance) / np.where(mask, distance, 1.0),
            0.0,
        ))
    real_sum *= 0.5 * k_e

    # ---- reciprocal space ------------------------------------------- #
    g_repeats = _shell_counts(reciprocal, g_cut)
    g_vectors = _lattice_points(g_repeats) @ reciprocal
    g2 = np.sum(g_vectors ** 2, axis=1)
    keep = (g2 > 1e-12) & (g2 < g_cut ** 2)
    g_vectors, g2 = g_vectors[keep], g2[keep]

    phase = np.exp(1j * (g_vectors @ positions.T))               # (G, N)
    structure_factor = phase @ q
    reciprocal_sum = (2.0 * np.pi * k_e / volume) * np.sum(
        np.exp(-g2 / (4.0 * eta ** 2)) / g2 * np.abs(structure_factor) ** 2
    )

    # ---- self and background ---------------------------------------- #
    self_term = -k_e * eta / np.sqrt(np.pi) * np.sum(q ** 2)
    background = -k_e * np.pi / (2.0 * eta ** 2 * volume) * np.sum(q) ** 2

    return float(real_sum + reciprocal_sum + self_term + background)


def alpha_z_energy(structure, pscore, n_electrons):
    r"""
    Finite :math:`\mathbf G = 0` remainder of the electron-ion energy.

    .. math:: E_{\alpha Z} = \frac{N_{\rm elec}}{\Omega}
              \sum_s N_s\,\mathrm{PSCORE}_s

    Parameters
    ----------
    structure : Structure
        Geometry, for the species counts and the cell volume.
    pscore : dict
        ``{element: PSCORE}`` in eV·Å³, from
        :attr:`~poraque.fields.vasp.potcar.PotcarSingle.pscore`.
    n_electrons : float
        Valence electron count in the cell.

    Returns
    -------
    float
        Energy in eV.

    Raises
    ------
    KeyError
        If a species present in ``structure`` has no ``PSCORE``. Defaulting to
        zero would look like a working calculation while silently dropping a
        term of order eV per atom.
    """
    volume = float(abs(np.linalg.det(np.asarray(structure.cell, dtype=float))))
    total = 0.0
    for symbol, atom_slice in structure.species_slices():
        element = symbol.split("_")[0].split(".")[0]
        if element not in pscore:
            raise KeyError(
                f"No PSCORE for {element!r}. Read the POTCAR with "
                f"parse_tables=True, or pass alpha_z=False to skip the term "
                f"knowingly."
            )
        count = atom_slice.stop - atom_slice.start
        total += count * float(pscore[element])
    return float(n_electrons * total / volume)


# ===================================================================== #
# Assembled result
# ===================================================================== #
@dataclass
class EnergyComponents:
    """
    Decomposition of the total energy, all in eV.

    Attributes
    ----------
    kinetic : float
        :math:`T_{\\rm s} = \\int\\tau\\,d^3r`, the non-interacting kinetic
        energy, integrated straight from ``TAUCAR``.
    external : float
        :math:`\\int\\rho V_{\\rm ext}\\,d^3r`, the :math:`\\mathbf G \\neq 0`
        electron-ion energy.
    alpha_z : float or None
        Finite :math:`\\mathbf G = 0` remainder of the electron-ion term.
        ``None`` when the ``POTCAR`` tables were unavailable — in which case
        :attr:`total` is missing a term and says so.
    hartree : float
        Classical electron-electron repulsion.
    xc : float
        Exchange-correlation.
    ewald : float or None
        Ion-ion electrostatics; ``None`` when no charges were supplied.
    n_electrons : float
        :math:`\\int\\rho\\,d^3r`, carried along as a sanity check: it should
        equal the total ``ZVAL`` to a few parts in :math:`10^{4}`. A
        predicted density that has drifted off it invalidates every
        electrostatic term below.
    nominal_electrons : float or None
        :math:`\\sum_s N_s Z^{\\rm val}_s`, the count the pseudopotentials fix.
        ``None`` when no valence charges were supplied. Compare against
        :attr:`n_electrons` through :attr:`electron_drift`.
    reference : float or None
        :math:`E_{\\rm ref} = \\sum_i E_{\\rm iso}(Z_i)`, the sum of
        isolated-atom energies for this composition. ``None`` when no
        :class:`~poraque.physics.ReferenceEnergies` covering the structure was
        supplied, in which case :attr:`cohesive` is ``None`` too.
    functional : str
        Which exchange-correlation approximation was used.
    """

    kinetic: float
    external: float
    hartree: float
    xc: float
    alpha_z: float = None
    ewald: float = None
    n_electrons: float = None
    nominal_electrons: float = None
    reference: float = None
    natoms: int = None
    functional: str = "pbe"

    @property
    def electron_drift(self):
        """
        Relative electron-count error of the density, or ``None``.

        ``(n_electrons - nominal_electrons) / nominal_electrons``. Anything
        above ~1e-3 means the electrostatic terms are being integrated against
        a density that does not hold the right amount of charge, and the
        energy should not be trusted at the accuracy of an energy difference.
        """
        if self.nominal_electrons in (None, 0.0) or self.n_electrons is None:
            return None
        return (self.n_electrons - self.nominal_electrons) / self.nominal_electrons

    @property
    def electronic(self):
        """Sum of the electron-only terms (everything but :attr:`ewald`)."""
        return (self.kinetic + self.external + self.hartree + self.xc
                + (self.alpha_z or 0.0))

    @property
    def potential(self):
        """
        Potential energy: everything that is not kinetic.

        External (including :attr:`alpha_z`), Hartree, exchange-correlation and
        ion-ion.
        """
        return (self.external + self.hartree + self.xc
                + (self.alpha_z or 0.0) + (self.ewald or 0.0))

    @property
    def total(self):
        """
        Total energy, :attr:`kinetic` + :attr:`potential`.

        .. warning::

           A **pseudo-valence** energy: the PAW one-centre terms are absent, so
           this does not reproduce a VASP total energy. Its geometry
           *dependence* is the usable part. See :attr:`missing` for terms that
           were skipped outright.
        """
        return self.kinetic + self.potential

    @property
    def cohesive(self):
        r"""
        :math:`\Delta E = E_{\rm total} - E_{\rm ref}`, or ``None``.

        The energy released on assembling this cell from isolated atoms. This
        is the number to quote and to compare against a reference calculation:
        :attr:`total` carries a per-atom offset of order :math:`10^3` eV that
        belongs to the pseudopotential and the missing PAW one-centre terms,
        and subtracting the isolated-atom energies removes exactly that part.

        ``None`` when no reference energies were available — reporting zero
        would claim the atoms are infinitely unbound.

        Notes
        -----
        Referencing changes nothing at fixed composition: two structures with
        the same formula share :math:`E_{\rm ref}`, so it cancels in
        :math:`\Delta E_1 - \Delta E_2` exactly. Its value is in comparisons
        *across* compositions, where it does not cancel and without which the
        comparison is undefined. See :mod:`poraque.physics.reference`.
        """
        if self.reference is None:
            return None
        return self.total - self.reference

    @property
    def cohesive_per_atom(self):
        """
        :attr:`cohesive` divided by the atom count, or ``None``.

        Set by :meth:`EnergyCalculator.compute`, which knows the structure.
        """
        if self.cohesive is None or not self.natoms:
            return None
        return self.cohesive / self.natoms

    @property
    def missing(self):
        """
        Names of the terms that were not computed, as a tuple.

        Empty means every component this module knows about was evaluated —
        which still does not include the PAW one-centre terms, since Poraquê
        never sees them.
        """
        absent = []
        if self.alpha_z is None:
            absent.append("alpha_z")
        if self.ewald is None:
            absent.append("ewald")
        return tuple(absent)

    def as_dict(self):
        """Plain ``dict`` of the components and the derived totals."""
        return {
            "kinetic": self.kinetic,
            "external": self.external,
            "alpha_z": self.alpha_z,
            "hartree": self.hartree,
            "xc": self.xc,
            "ewald": self.ewald,
            "potential": self.potential,
            "total": self.total,
            "n_electrons": self.n_electrons,
            "nominal_electrons": self.nominal_electrons,
            "electron_drift": self.electron_drift,
            "reference": self.reference,
            "cohesive": self.cohesive,
            "cohesive_per_atom": self.cohesive_per_atom,
            "natoms": self.natoms,
            "functional": self.functional,
            "missing": list(self.missing),
        }

    def __str__(self):
        rows = [
            ("kinetic      T_s", self.kinetic),
            ("external     E_ext", self.external),
            ("alpha Z", self.alpha_z),
            ("Hartree      E_H", self.hartree),
            (f"xc ({self.functional})", self.xc),
            ("Ewald", self.ewald),
        ]
        lines = [f"  {label:<22s} {'' if value is None else f'{value:16.6f}'}"
                 f"{'  (not computed)' if value is None else ' eV'}"
                 for label, value in rows]
        lines.append("  " + "-" * 40)
        lines.append(f"  {'potential':<22s} {self.potential:16.6f} eV")
        lines.append(f"  {'TOTAL':<22s} {self.total:16.6f} eV")
        if self.reference is not None:
            lines.append(f"  {'- reference E_ref':<22s} "
                         f"{self.reference:16.6f} eV")
            lines.append(f"  {'= COHESIVE dE':<22s} "
                         f"{self.cohesive:16.6f} eV")
            per_atom = self.cohesive_per_atom
            if per_atom is not None:
                lines.append(f"  {'':<22s} {per_atom:16.6f} eV/atom")
        if self.n_electrons is not None:
            lines.append(f"  {'electrons':<22s} {self.n_electrons:16.6f}")
        drift = self.electron_drift
        if drift is not None:
            lines.append(f"  {'  nominal':<22s} {self.nominal_electrons:16.6f}"
                         f"   (drift {drift:+.3e})")
        if self.missing:
            lines.append(f"  incomplete: missing {', '.join(self.missing)}")
        return "\n".join(lines)


class EnergyCalculator:
    r"""
    Integrate predicted fields into energy components.

    Parameters
    ----------
    grid : FieldGrid
        The shared mesh all three fields live on.
    structure : Structure, optional
        Geometry. Required for :func:`ewald_energy` and
        :func:`alpha_z_energy`; without it those terms are reported as ``None``
        rather than assumed zero.
    charges : dict, optional
        ``{element: Z_val}``. Usually taken from
        ``potential.metadata["charges"]``.
    pscore : dict, optional
        ``{element: PSCORE}`` in eV·Å³, for the :math:`\mathbf G = 0`
        remainder.
    functional : str, optional
        Exchange-correlation approximation, passed to :func:`xc_energy`.
        Defaults to ``"pbe"``, matching the reference data.
    references : ReferenceEnergies, optional
        Isolated-atom energies, enabling
        :attr:`EnergyComponents.cohesive`. Without them the decomposition
        still reports :attr:`~EnergyComponents.total`; the cohesive energy is
        reported as ``None`` rather than as the total, since an unreferenced
        total is not a cohesive energy.

    Examples
    --------
    >>> calculator = EnergyCalculator(grid, structure, charges)   # doctest: +SKIP
    >>> components = calculator.compute(density, tau, potential)  # doctest: +SKIP
    >>> print(components)                                          # doctest: +SKIP
    """

    def __init__(self, grid, structure=None, charges=None, pscore=None,
                 functional="pbe", references=None):
        self.grid = grid
        self.structure = structure
        self.charges = dict(charges) if charges else None
        self.pscore = dict(pscore) if pscore else None
        self.functional = functional
        self.references = references

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_potential(cls, potential, pscore=None, functional="pbe"):
        """
        Build from an :class:`~poraque.fields.ExternalPotential`.

        The potential already carries the grid, the structure and the valence
        charges it was built from, so this is the constructor to prefer: it
        cannot disagree with the field it will be integrated against.

        Parameters
        ----------
        potential : ExternalPotential
        pscore : dict, optional
            ``{element: PSCORE}``; also read from
            ``potential.metadata["pscore"]`` when present.
        functional : str, optional

        Returns
        -------
        EnergyCalculator
        """
        metadata = getattr(potential, "metadata", {}) or {}
        return cls(
            grid=potential.grid,
            structure=potential.structure,
            charges=metadata.get("charges"),
            pscore=pscore if pscore is not None else metadata.get("pscore"),
            functional=functional,
        )

    # ------------------------------------------------------------------ #
    # Energies
    # ------------------------------------------------------------------ #
    def kinetic_energy(self, tau):
        r"""
        :math:`T_{\rm s} = \int\tau\,d^3r` in eV.

        Parameters
        ----------
        tau : KineticEnergyDensity or array_like
            Kinetic energy density in eV/Å³.
        """
        values = np.asarray(tau, dtype=float)
        _check_shape(values, self.grid, "tau")
        return self.grid.integrate(values)

    def external_energy(self, density, potential):
        r"""
        :math:`\int\rho V_{\rm ext}\,d^3r` in eV, excluding
        :math:`\mathbf G = 0`.
        """
        rho = np.asarray(density, dtype=float)
        v = np.asarray(potential, dtype=float)
        _check_shape(rho, self.grid, "density")
        _check_shape(v, self.grid, "potential")
        return self.grid.integrate(rho * v)

    def hartree_energy(self, density):
        """Classical electron-electron repulsion in eV."""
        return hartree_energy(density, self.grid)

    def xc_energy(self, density):
        """Exchange-correlation energy in eV."""
        return xc_energy(density, self.grid, functional=self.functional)

    def ewald_energy(self):
        """
        Ion-ion energy in eV, or ``None`` without a structure and charges.
        """
        if self.structure is None or not self.charges:
            return None
        return ewald_energy(self.structure, self.charges)

    @property
    def nominal_electrons(self):
        r"""
        Valence electron count implied by the pseudopotentials,
        :math:`\sum_s N_s Z^{\rm val}_s`, or ``None`` without charges.

        This is ``NELECT``: a fixed property of the cell and the ``POTCAR``,
        not something a calculation converges to. It is the count the
        :math:`\mathbf G = 0` bookkeeping must use — see :meth:`compute`.
        """
        if self.structure is None or not self.charges:
            return None
        return float(np.sum(_per_atom_charges(self.structure, self.charges)))

    def compute(self, density, tau, potential):
        r"""
        Every component at once.

        Parameters
        ----------
        density : ChargeDensity or array_like
            :math:`\rho` in e/Å³.
        tau : KineticEnergyDensity or array_like
            :math:`\tau` in eV/Å³.
        potential : ExternalPotential or array_like
            :math:`V_{\rm ext}` in eV.

        Returns
        -------
        EnergyComponents

        Notes
        -----
        :math:`E_{\alpha Z}` is scaled by the **nominal** valence count
        :attr:`nominal_electrons`, not by :math:`\int\rho\,d^3r`. The two agree
        for a reference density but not for a predicted one, and the
        distinction is not cosmetic: the prefactor multiplies a quantity of
        order :math:`10^3` eV, so a density carrying a 0.1 % electron-count
        drift would move :math:`E_{\alpha Z}` by a couple of eV — larger than
        the energy differences this calculator exists to resolve, and varying
        from structure to structure in a way that does not cancel. The
        measured integral is still reported as
        :attr:`EnergyComponents.n_electrons`, where it belongs: as a
        diagnostic of the prediction, not as a factor inside it.
        """
        rho = np.asarray(density, dtype=float)
        n_electrons = self.grid.integrate(rho)
        nominal = self.nominal_electrons

        alpha_z = None
        if self.structure is not None and self.pscore:
            alpha_z = alpha_z_energy(
                self.structure, self.pscore,
                nominal if nominal is not None else n_electrons)

        return EnergyComponents(
            kinetic=self.kinetic_energy(tau),
            external=self.external_energy(rho, potential),
            hartree=self.hartree_energy(rho),
            xc=self.xc_energy(rho),
            alpha_z=alpha_z,
            ewald=self.ewald_energy(),
            n_electrons=n_electrons,
            nominal_electrons=nominal,
            reference=self.reference_energy(),
            natoms=(None if self.structure is None else self.structure.natoms),
            functional=self.functional,
        )

    def reference_energy(self):
        r"""
        :math:`E_{\rm ref} = \sum_i E_{\rm iso}(Z_i)` in eV, or ``None``.

        ``None`` when no references were supplied *or* when they do not cover
        every species present. A partial sum is not returned: it would be an
        energy quietly missing whole atoms, and the resulting "cohesive
        energy" would look reasonable while being wrong by electron-volts per
        uncovered atom.
        """
        if self.structure is None or self.references is None:
            return None
        if not self.references.covers(self.structure):
            return None
        return self.references.total_for(self.structure)

    def potential_energy(self, density, tau, potential):
        """
        Potential energy in eV — everything that is not kinetic.

        Convenience wrapper around ``compute(...).potential``.
        """
        return self.compute(density, tau, potential).potential

    def total_energy(self, density, tau, potential):
        """
        Total energy in eV.

        Convenience wrapper around ``compute(...).total``. Read the caveat on
        :attr:`EnergyComponents.total` before comparing it to a DFT number.
        """
        return self.compute(density, tau, potential).total

    def __repr__(self):
        return (f"EnergyCalculator(shape={self.grid.shape}, "
                f"functional={self.functional!r}, "
                f"ewald={'yes' if self.charges else 'no'}, "
                f"alpha_z={'yes' if self.pscore else 'no'})")


# ===================================================================== #
# Helpers
# ===================================================================== #
def _as_grid(grid, shape):
    """
    Accept a :class:`FieldGrid` or bare lattice vectors.

    Anything carrying a ``get_g2`` is already a grid; a ``(3, 3)`` array is
    lattice vectors, and the mesh shape then has to come from the field being
    solved for.
    """
    if hasattr(grid, "get_g2"):
        return grid

    from ..fields.grid import FieldGrid

    cell = np.asarray(grid, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(
            f"Expected a FieldGrid or (3, 3) lattice vectors, got an array of "
            f"shape {cell.shape}."
        )
    return FieldGrid(tuple(shape), cell)


def _check_shape(values, grid, label):
    if values.shape != tuple(grid.shape):
        raise ValueError(
            f"{label} has shape {values.shape} but the grid is {tuple(grid.shape)}."
        )


def _clipped_bohr_density(density, grid):
    """Density in e/Bohr³, floored at zero, shape-checked against ``grid``."""
    values = np.asarray(density, dtype=float)
    _check_shape(values, grid, "density")
    return np.clip(values, 0.0, None) * BOHR_TO_ANGSTROM ** 3


def _density_and_gradient(density, grid):
    r"""
    ``(rho, |grad rho|^2)`` in atomic units, for the semilocal functionals.

    The gradient is taken of the *unclipped* field. Clipping first would put a
    kink at every point where a band-limited density rings below zero, and
    spectral differentiation of a kink rings far worse than the undershoot it
    was meant to remove. The clipped density is used only in the algebra,
    where :math:`\rho^{4/3}` and the denominators need it non-negative.
    """
    from ..fields.density import spectral_gradient

    values = np.asarray(density, dtype=float)
    _check_shape(values, grid, "density")

    raw_bohr = values * BOHR_TO_ANGSTROM ** 3
    components = spectral_gradient(raw_bohr, grid, length_unit="bohr")
    gradient_squared = sum(component ** 2 for component in components)
    return np.clip(raw_bohr, 0.0, None), gradient_squared


def _reduced_gradient_squared(density_bohr, gradient_squared):
    r"""
    :math:`s^2 = |\nabla\rho|^2/(2k_{\rm F}\rho)^2`, the PBE exchange variable.

    Diverges as :math:`\rho \to 0`, which is physical: the enhancement factor
    saturates at :math:`1 + \kappa` there while the :math:`\rho^{4/3}`
    prefactor vanishes, so the product is well behaved. The floor only keeps
    the intermediate arithmetic finite.
    """
    safe = np.clip(density_bohr, 1e-30, None)
    k_f = (3.0 * np.pi ** 2 * safe) ** (1.0 / 3.0)
    return gradient_squared / (2.0 * k_f * safe) ** 2


def _per_atom_charges(structure, charges):
    """``(natoms,)`` charge array from a per-element map or a per-atom list."""
    if isinstance(charges, dict):
        lookup = {str(k).split("_")[0].split(".")[0]: float(v)
                  for k, v in charges.items()}
        out = np.empty(structure.natoms, dtype=float)
        for symbol, atom_slice in structure.species_slices():
            element = symbol.split("_")[0].split(".")[0]
            if element not in lookup:
                raise KeyError(f"No charge given for species {element!r}.")
            out[atom_slice] = lookup[element]
        return out

    out = np.asarray(charges, dtype=float).ravel()
    if out.size != structure.natoms:
        raise ValueError(
            f"{out.size} charges for {structure.natoms} atoms."
        )
    return out


def _shell_counts(vectors, cutoff):
    """
    Repeats along each axis needed to cover a sphere of radius ``cutoff``.

    Uses the perpendicular width of each lattice plane, not the vector length:
    on a strongly skewed cell the two differ by a large factor and the naive
    choice silently truncates the sum.
    """
    vectors = np.asarray(vectors, dtype=float)
    volume = abs(np.linalg.det(vectors))
    counts = []
    for axis in range(3):
        other = vectors[[i for i in range(3) if i != axis]]
        area = np.linalg.norm(np.cross(other[0], other[1]))
        width = volume / area
        counts.append(int(np.ceil(cutoff / width)))
    return counts


def _lattice_points(counts):
    """``(M, 3)`` integer triples spanning ``[-n, n]`` on each axis."""
    ranges = [np.arange(-n, n + 1) for n in counts]
    mesh = np.meshgrid(*ranges, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1).astype(float)


def _erfc(x):
    """``erfc`` from SciPy when available, else via :func:`math.erfc`."""
    try:
        from scipy.special import erfc
    except ImportError:                                   # pragma: no cover
        import math
        return np.vectorize(math.erfc)(x)
    return erfc(x)
