# -*- coding: utf-8 -*-
# file: forces.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Hellmann-Feynman forces from a predicted density.

The total energy of :mod:`poraque.physics.energy` depends on the ionic
positions in two different ways. Two terms depend on them *explicitly*:

.. math::

    E_{\rm ext} = \int\rho\,V_{\rm ext}[\{\mathbf R\}]\,d^3r ,
    \qquad
    E_{\rm Ewald}[\{\mathbf R\}] ,

while :math:`T_{\rm s}`, :math:`E_{\rm H}` and :math:`E_{\rm xc}` depend on
them only *through* :math:`\rho` and :math:`\tau`. The Hellmann-Feynman
theorem says the second group contributes nothing:

.. math::

    \mathbf F_I = -\frac{\partial E}{\partial\mathbf R_I}
                = -\int\rho\,
                  \frac{\partial V_{\rm ext}}{\partial\mathbf R_I}\,d^3r
                  - \frac{\partial E_{\rm Ewald}}{\partial\mathbf R_I} ,

because :math:`\delta E/\delta\rho` is constant at the variational minimum and
:math:`\int\delta\rho = 0`. Both surviving terms are analytic — no
differentiation through the network, and no finite differences on a fixed grid,
which would be dominated by the grid's own discontinuity as an atom crosses
between voxels.

.. warning::

   **The predicted density is not variational.** Poraquê's :math:`\rho` comes
   from an operator, not from minimising this functional, so the term the
   theorem discards is not exactly zero. What is computed here is the exact
   Hellmann-Feynman force *of the predicted density*, which is the standard
   construction and the one the theorem is named for — but it is not the exact
   gradient of :meth:`~poraque.calculator.Poraque.get_potential_energy`, and
   the two differ by the neglected Pulay-like term. The size of that term is
   set by how far :math:`\rho` sits from the true ground-state density; see
   :func:`force_consistency_error` for a way to bound it in practice.

Sign convention: :math:`V_{\rm ext}` is the potential energy of an electron
(negative near a nucleus) and :math:`\rho` is the positive electron number
density, matching :mod:`poraque.physics.energy`.
"""

import numpy as np

from ..fields.constants import COULOMB_CONSTANT_EV_ANGSTROM
from ..fields.structure import element_of
from .energy import _erfc, _lattice_points, _per_atom_charges, _shell_counts


def local_potential_forces(density, structure, grid, potcar):
    r"""
    Electron-ion Hellmann-Feynman force, :math:`-\int\rho\,\partial_{\mathbf R}
    V_{\rm ext}`.

    Evaluated in reciprocal space, where the position dependence of
    :math:`V_{\rm ext}` sits entirely in the structure factor and the
    derivative is exact:

    .. math::

        V_{\rm ext}(\mathbf G) = \frac1\Omega\sum_s w_s(G)
            \sum_{I\in s} e^{-i\mathbf G\cdot\mathbf R_I} ,
        \qquad
        \frac{\partial}{\partial\mathbf R_I}e^{-i\mathbf G\cdot\mathbf R_I}
            = -i\mathbf G\,e^{-i\mathbf G\cdot\mathbf R_I} ,

    with :math:`w_s(G) = v^s_{\rm short}(G) - 4\pi Z^{\rm val}_s e^2/G^2` the
    same tabulated form factor
    :meth:`~poraque.fields.ExternalPotential.from_potcar_tables` builds the
    potential from, truncated at the same ``PSGMAX``. Using one form factor for
    both means the force cannot drift away from the potential it differentiates.

    Parameters
    ----------
    density : ChargeDensity or array_like
        :math:`\rho` in e/Å³, on ``grid``.
    structure : Structure
        Geometry.
    grid : FieldGrid
        Shared mesh.
    potcar : Potcar
        Read with ``parse_tables=True``.

    Returns
    -------
    numpy.ndarray
        ``(natoms, 3)`` forces in eV/Å.
    """
    from scipy.interpolate import CubicSpline

    rho = np.asarray(density, dtype=float)
    if rho.shape != tuple(grid.shape):
        raise ValueError(
            f"density has shape {rho.shape} but the grid is {tuple(grid.shape)}."
        )

    entries = {entry.element: entry for entry in potcar}
    g2 = grid.get_g2()
    magnitude = np.sqrt(g2)
    gx, gy, gz = grid.get_g_vectors()
    inverse_g2 = grid.get_inverse_g2(g2)
    nonzero = inverse_g2 > 0

    # rho(G) with the same normalisation the potential uses: V(r) is built as
    # ifftn(v_G) * npoints, so v_G are coefficients of exp(+i G.r) and the
    # matching density transform is the plain forward FFT.
    rho_g = np.fft.fftn(rho)

    forces = np.zeros((structure.natoms, 3), dtype=float)
    prefactor = grid.volume / grid.npoints / grid.volume    # = 1 / npoints

    for symbol, atom_slice in structure.species_slices():
        element = element_of(symbol)
        entry = entries.get(element)
        if entry is None or entry.local_potential is None:
            raise ValueError(
                f"No tabulated local potential for {element!r}. Read the "
                f"POTCAR with parse_tables=True."
            )

        spline = CubicSpline(entry.local_q_grid, entry.local_potential,
                             bc_type="natural", extrapolate=False)
        limit = entry.psgmax - entry.psgmax / entry.NPSPTS
        inside = nonzero & (magnitude < limit)

        form = np.zeros_like(g2)
        form[inside] = (spline(magnitude[inside])
                        - 4.0 * np.pi * entry.zval
                        * COULOMB_CONSTANT_EV_ANGSTROM * inverse_g2[inside])

        # conj(rho(G)) * w(G), shared by every atom of this species.
        common = np.conj(rho_g) * form

        scaled = np.atleast_2d(structure.scaled_positions[atom_slice])
        m1, m2, m3 = grid.fft_frequencies()
        for offset, position in enumerate(scaled):
            phase = (np.exp(-2j * np.pi * m1 * position[0])[:, None, None]
                     * np.exp(-2j * np.pi * m2 * position[1])[None, :, None]
                     * np.exp(-2j * np.pi * m3 * position[2])[None, None, :])
            # F = -dE/dR = -(1/N) sum_G conj(rho) w (-i G) e^{-iG.R}
            #            = (1/N) sum_G Re[ i G conj(rho) w e^{-iG.R} ]
            weighted = common * phase
            forces[atom_slice.start + offset] = prefactor * np.array([
                float(np.sum(np.real(1j * gx * weighted))),
                float(np.sum(np.real(1j * gy * weighted))),
                float(np.sum(np.real(1j * gz * weighted))),
            ])

    return forces


def ewald_forces(structure, charges, accuracy=1e-12):
    r"""
    Ion-ion force, the analytic gradient of :func:`~poraque.physics.energy.ewald_energy`.

    Both lattice sums are differentiated in closed form. The self-energy and
    the background term carry no position dependence and so contribute nothing.

    Parameters
    ----------
    structure : Structure
        Geometry.
    charges : dict or array_like
        ``{element: Z_val}`` or one charge per atom.
    accuracy : float, optional
        Target relative truncation error; sets both cutoffs, exactly as in the
        energy so the two stay consistent.

    Returns
    -------
    numpy.ndarray
        ``(natoms, 3)`` forces in eV/Å.
    """
    cell = np.asarray(structure.cell, dtype=float)
    positions = np.asarray(structure.positions, dtype=float)
    volume = float(abs(np.linalg.det(cell)))
    q = _per_atom_charges(structure, charges)
    natoms = len(q)

    k_e = COULOMB_CONSTANT_EV_ANGSTROM
    eta = np.sqrt(np.pi) * (natoms / volume ** 2) ** (1.0 / 6.0)
    span = np.sqrt(-np.log(accuracy))
    r_cut, g_cut = span / eta, 2.0 * eta * span

    forces = np.zeros((natoms, 3), dtype=float)

    # ---- real space ----------------------------------------------------- #
    delta = positions[:, None, :] - positions[None, :, :]        # r_i - r_j
    lattice = _lattice_points(_shell_counts(cell, r_cut)) @ cell
    pair = q[:, None] * q[None, :]

    for shift in lattice:
        separation = delta + shift
        distance = np.linalg.norm(separation, axis=-1)
        mask = (distance > 1e-8) & (distance < r_cut)
        if not mask.any():
            continue
        safe = np.where(mask, distance, 1.0)
        # -d/dd [erfc(eta d)/d] = erfc(eta d)/d^2 + 2 eta exp(-eta^2 d^2)/(sqrt(pi) d)
        radial = np.where(
            mask,
            (_erfc(eta * safe) / safe ** 2
             + 2.0 * eta / np.sqrt(np.pi) * np.exp(-(eta * safe) ** 2) / safe),
            0.0,
        )
        weight = k_e * pair * radial / safe
        forces += np.sum(weight[:, :, None] * separation, axis=1)

    # ---- reciprocal space ------------------------------------------------ #
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    g_vectors = _lattice_points(_shell_counts(reciprocal, g_cut)) @ reciprocal
    g2 = np.sum(g_vectors ** 2, axis=1)
    keep = (g2 > 1e-12) & (g2 < g_cut ** 2)
    g_vectors, g2 = g_vectors[keep], g2[keep]

    phase = np.exp(1j * (g_vectors @ positions.T))               # (G, N)
    structure_factor = phase @ q                                 # (G,)
    kernel = np.exp(-g2 / (4.0 * eta ** 2)) / g2                 # (G,)

    # dE/dr_i = (2 pi k_e / V) sum_G f(G) * 2 q_i G Im[e^{-iG.r_i} S*]
    # note phase[:, i] = e^{+iG.r_i}, so e^{-iG.r_i} = conj(phase[:, i])
    contribution = np.imag(np.conj(phase) * structure_factor[:, None])  # (G, N)
    gradient = (4.0 * np.pi * k_e / volume) * (
        (kernel[:, None] * contribution).T @ g_vectors)          # (N, 3)
    forces -= q[:, None] * gradient

    return forces


def hellmann_feynman_forces(density, structure, grid, potcar=None,
                            charges=None, accuracy=1e-12):
    r"""
    Total force on every ion: electron-ion plus ion-ion.

    Parameters
    ----------
    density : ChargeDensity or array_like
        :math:`\rho` in e/Å³.
    structure : Structure
        Geometry.
    grid : FieldGrid
        Shared mesh.
    potcar : Potcar, optional
        Supplies the local form factors. Without it the electron-ion term
        cannot be evaluated and only the Ewald force is returned — which is
        not a force on a real system, so this raises instead unless
        ``charges`` alone was asked for.
    charges : dict, optional
        ``{element: Z_val}``. Taken from ``potcar`` when omitted.
    accuracy : float, optional
        Ewald truncation target.

    Returns
    -------
    numpy.ndarray
        ``(natoms, 3)`` forces in eV/Å.

    Raises
    ------
    ValueError
        Without a ``POTCAR``: the electron-ion term is not optional, and
        returning the ion-ion force alone would be a large, plausible-looking,
        entirely wrong number.
    """
    if potcar is None:
        raise ValueError(
            "Hellmann-Feynman forces need the tabulated local potential from a "
            "POTCAR: the electron-ion term is the same order as the ion-ion "
            "one and cancels most of it. Returning the Ewald force alone would "
            "be wrong by hundreds of percent, so it is not offered."
        )

    if charges is None:
        charges = {entry.element: entry.zval for entry in potcar}

    return (local_potential_forces(density, structure, grid, potcar)
            + ewald_forces(structure, charges, accuracy=accuracy))


def force_consistency_error(forces):
    r"""
    Net force on the cell, which translation invariance requires to vanish.

    :math:`\sum_I\mathbf F_I = 0` holds for any periodic system by Newton's
    third law, independently of whether the density is any good. It therefore
    tests the *implementation* rather than the physics: a non-zero sum means a
    sign, a convention or a truncation is wrong, while a zero sum says nothing
    about whether :math:`\rho` was accurate.

    Parameters
    ----------
    forces : array_like
        ``(natoms, 3)`` in eV/Å.

    Returns
    -------
    float
        :math:`|\sum_I\mathbf F_I|` in eV/Å.
    """
    return float(np.linalg.norm(np.sum(np.asarray(forces, dtype=float), axis=0)))
