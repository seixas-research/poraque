# -*- coding: utf-8 -*-
# file: engine.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .core import Density, Result, State
from .functionals import Hartree, LDA


class OFDFTEngine:
    """
    Engine for Orbital-Free DFT calculations.

    Performs direct minimization of the total energy with respect to the
    positivity-preserving variable ``w = sqrt(n)`` under the electron-number
    constraint ``integral(w^2) = N``. A backtracking line search keeps the
    energy monotonically decreasing for stability.
    """

    def __init__(self, system, grid, functionals, backend, settings):
        self.system = system
        self.grid = grid
        self.functionals = functionals
        self.backend = backend
        self.settings = settings

    def compute_total_energy(self, density):
        """Compute the total energy and its per-functional components."""
        total_e = 0.0
        components = {}
        for func in self.functionals:
            e = func.energy(density, self.system, self.grid, self.backend)
            components[func.name] = e
            total_e += e
        return total_e, components

    def compute_effective_potential(self, density):
        """Compute the effective potential ``v_eff = sum_i v_i``."""
        v_eff = np.zeros(self.grid.shape)
        for func in self.functionals:
            v_eff += func.potential(density, self.system, self.grid, self.backend)
        return v_eff

    def _normalize_w(self, w):
        """Rescale ``w`` so that ``integral(w^2) == N``."""
        norm_sq = self.backend.integrate(w**2, self.grid)
        return w * np.sqrt(self.system.electrons / norm_sq)

    def _energy_of_w(self, w):
        """Energy and components for a given (already normalized) ``w``."""
        density = Density(self.grid, w**2)
        return self.compute_total_energy(density)

    def _inner(self, a, b):
        """Grid inner product ``<a, b> = integral(a b) dV``."""
        return self.backend.integrate(a * b, self.grid)

    def _line_search(self, w, direction, energy, step):
        """
        Backtracking line search along ``direction`` from ``w``.

        Uses the normalization retraction ``w' = normalize(w + alpha d)`` and
        accepts the first step that does not raise the energy.

        Returns
        -------
        tuple
            ``(accepted, w_new, energy_new, components_new, used_step)``.
        """
        trial_step = step
        for _ in range(self.settings.max_line_search):
            w_trial = self._normalize_w(w + trial_step * direction)
            e_trial, comp_trial = self._energy_of_w(w_trial)
            if e_trial <= energy:
                return True, w_trial, e_trial, comp_trial, trial_step
            trial_step *= 0.5
        return False, w, energy, None, trial_step

    def run(self, initial_density):
        """
        Minimize the energy with a projected conjugate-gradient method.

        The energy is minimized over ``w = sqrt(n)`` constrained to the sphere
        ``integral(w^2) = N``. Search directions are built with the
        Polak-Ribière(+) formula, projected onto the tangent space at the
        current point, and followed with a backtracking line search using a
        normalization retraction. The method restarts to steepest descent
        periodically and whenever a conjugate direction stops being a descent
        direction or its line search fails.
        """
        density = initial_density
        density.normalize(self.system.electrons)

        w = self._normalize_w(np.sqrt(np.maximum(density.data, 0.0)))
        step = self.settings.mixing
        max_step = 50.0 * self.settings.mixing

        history = {"energy": [], "residual": [], "mu": [], "step": []}
        converged = False

        energy, components = self._energy_of_w(w)
        g_prev = None
        d_prev = None
        i = 0
        for i in range(self.settings.max_iter):
            density.data = w**2
            v_eff = self.compute_effective_potential(density)

            # Chemical potential (Lagrange multiplier) projecting onto the N-surface.
            mu = self._inner(w**2, v_eff) / self.system.electrons
            # Gradient of E w.r.t. w: dE/dw = 2 w v_eff, already tangent (<g, w> = 0).
            g = 2 * w * (v_eff - mu)

            residual = self.backend.norm(g) * np.sqrt(self.grid.volume_element)
            history["energy"].append(energy)
            history["residual"].append(residual)
            history["mu"].append(mu)

            if residual < self.settings.tolerance:
                converged = True
                history["step"].append(step)
                break

            # Build the conjugate-gradient search direction.
            restart = (d_prev is None) or (i % self.settings.cg_restart == 0)
            if restart:
                direction = -g
            else:
                # Polak-Ribière(+) coefficient.
                beta = self._inner(g, g - g_prev) / self._inner(g_prev, g_prev)
                beta = max(beta, 0.0)
                direction = -g + beta * d_prev
                # Project onto the tangent space at w.
                direction = direction - (self._inner(direction, w) / self._inner(w, w)) * w
                # Fall back to steepest descent if not a descent direction.
                if self._inner(direction, g) > 0.0:
                    direction = -g

            accepted, w_new, e_new, comp_new, used_step = self._line_search(
                w, direction, energy, step
            )

            # If a conjugate direction's line search fails, restart with -g.
            if not accepted and not restart:
                accepted, w_new, e_new, comp_new, used_step = self._line_search(
                    w, -g, energy, step
                )
                direction = -g

            history["step"].append(used_step)

            if not accepted:
                # No downhill step found: stationary point reached.
                converged = True
                break

            w = w_new
            energy, components = e_new, comp_new
            g_prev = g
            d_prev = direction
            # Adaptively grow the trial step after a success.
            step = min(used_step * 1.5, max_step)

        density.data = w**2
        return Result(
            energy=energy,
            components=components,
            density=density,
            converged=converged,
            iterations=i + 1,
            history=history,
        )


def aufbau_occupations(n_states, n_electrons, max_occ=2.0):
    """
    Fill ``n_states`` levels with ``n_electrons`` electrons (Aufbau order).

    Parameters
    ----------
    n_states : int
        Number of available orbitals (assumed energy-ordered).
    n_electrons : float
        Total electron count to distribute.
    max_occ : float, optional
        Maximum occupation per orbital (2.0 for spin-degenerate closed shells).

    Returns
    -------
    numpy.ndarray
        Occupation numbers, one per orbital.
    """
    occ = np.zeros(n_states)
    remaining = float(n_electrons)
    for i in range(n_states):
        occ[i] = min(max_occ, remaining)
        remaining -= occ[i]
        if remaining <= 0:
            break
    return occ


class KSDFTEngine:
    """
    Kohn-Sham DFT engine on the real-space grid.

    The Kohn-Sham Hamiltonian is

    .. math::

        \\hat{H} = -\\tfrac{1}{2}\\nabla^2 + v_{ext}(\\mathbf{r})
                   + v_H[n](\\mathbf{r}) + v_{xc}[n](\\mathbf{r})

    with the kinetic operator applied spectrally (FFT) and the local potential
    applied as a diagonal multiplication. Occupied orbitals are obtained with a
    matrix-free Hermitian eigensolver, and the density is updated with linear
    mixing inside a fixed-point SCF loop.

    Parameters
    ----------
    system : System
        Atomic structure (provides the electron count).
    grid : Grid
        Real-space grid.
    v_ext : numpy.ndarray
        External (electron-nucleus) potential on the grid, in Hartree.
    backend : Backend
        Numerical backend.
    settings : SolverSettings
        SCF configuration.
    xc : Functional, optional
        Exchange-correlation functional (default :class:`~poraque.functionals.LDA`).
    hartree : bool, optional
        Include the Hartree term (default ``True``).
    n_extra_states : int, optional
        Extra unoccupied states to compute for eigensolver robustness.
    """

    def __init__(self, system, grid, v_ext, backend, settings,
                 xc=None, hartree=True, n_extra_states=2):
        self.system = system
        self.grid = grid
        self.v_ext = np.asarray(v_ext, dtype=float)
        self.backend = backend
        self.settings = settings
        self.xc = xc if xc is not None else LDA()
        self.hartree = Hartree() if hartree else None
        self.n_extra_states = n_extra_states

        self.n_occupied = int(np.ceil(self.system.electrons / 2.0))
        self.n_states = self.n_occupied + n_extra_states
        self._g2 = grid.get_g2()
        self.dV = grid.volume_element

    def _apply_kinetic(self, psi):
        """Apply -1/2 nabla^2 spectrally to a grid-shaped wavefunction."""
        psi_g = np.fft.fftn(psi)
        return 0.5 * np.real(np.fft.ifftn(self._g2 * psi_g))

    def _hamiltonian_operator(self, v_eff):
        """Build a matrix-free Hermitian LinearOperator for H = T + v_eff."""
        shape = self.grid.shape
        n = self.grid.N

        def matvec(x):
            psi = x.reshape(shape)
            hpsi = self._apply_kinetic(psi) + v_eff * psi
            return hpsi.ravel()

        return LinearOperator((n, n), matvec=matvec, dtype=float)

    def _effective_potential(self, density):
        v = np.array(self.v_ext, dtype=float)
        if self.hartree is not None:
            v = v + self.hartree.potential(density, self.system, self.grid, self.backend)
        v = v + self.xc.potential(density, self.system, self.grid, self.backend)
        return v

    def _solve_orbitals(self, v_eff):
        """Return the lowest eigenvalues and grid-normalized orbitals."""
        H = self._hamiltonian_operator(v_eff)
        k = min(self.n_states, self.grid.N - 2)
        # Shift-invert-free: smallest algebraic eigenvalues.
        eigvals, eigvecs = eigsh(H, k=k, which="SA")
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]
        # Grid normalization: integral |psi|^2 dV = 1  ->  divide l2-vectors by sqrt(dV).
        orbitals = np.array([
            eigvecs[:, i].reshape(self.grid.shape) / np.sqrt(self.dV)
            for i in range(eigvecs.shape[1])
        ])
        return eigvals, orbitals

    def _total_energy(self, density, eigvals, occupations):
        """KS total energy via the band-energy decomposition."""
        e_band = float(np.sum(occupations * eigvals[: len(occupations)]))
        e_ext = self.backend.integrate(self.v_ext * density.data, self.grid)

        e_h = 0.0
        v_h_n = 0.0
        if self.hartree is not None:
            e_h = self.hartree.energy(density, self.system, self.grid, self.backend)
            v_h = self.hartree.potential(density, self.system, self.grid, self.backend)
            v_h_n = self.backend.integrate(v_h * density.data, self.grid)

        e_xc = self.xc.energy(density, self.system, self.grid, self.backend)
        v_xc = self.xc.potential(density, self.system, self.grid, self.backend)
        v_xc_n = self.backend.integrate(v_xc * density.data, self.grid)

        # E_tot = E_band - E_H + (E_xc - integral(v_xc n))
        # (the external and kinetic pieces are folded into E_band via v_eff).
        e_total = e_band - v_h_n + e_h + e_xc - v_xc_n
        components = {
            "Band": e_band,
            "External": e_ext,
            "Hartree": e_h,
            "XC": e_xc,
        }
        return e_total, components

    def run(self, initial_density=None):
        """Run the SCF loop and return a :class:`Result` with the KS state."""
        if initial_density is None:
            n0 = self.system.electrons / self.grid.volume
            density = Density(self.grid, np.full(self.grid.shape, n0))
        else:
            density = initial_density
        density.normalize(self.system.electrons)

        history = {"energy": [], "density_residual": []}
        converged = False
        energy = np.nan
        eigvals = np.zeros(self.n_states)
        occupations = aufbau_occupations(self.n_states, self.system.electrons)
        orbitals = None

        i = 0
        for i in range(self.settings.max_iter):
            v_eff = self._effective_potential(density)
            eigvals, orbitals = self._solve_orbitals(v_eff)
            occupations = aufbau_occupations(len(eigvals), self.system.electrons)

            new_data = np.zeros(self.grid.shape)
            for f, psi in zip(occupations, orbitals):
                if f > 0:
                    new_data += f * np.abs(psi) ** 2
            new_density = Density(self.grid, new_data)
            new_density.normalize(self.system.electrons)

            residual = self.backend.norm(new_density.data - density.data) * np.sqrt(self.dV)
            energy, components = self._total_energy(new_density, eigvals, occupations)
            history["energy"].append(energy)
            history["density_residual"].append(residual)

            # Linear density mixing.
            mixed = (1 - self.settings.mixing) * density.data + self.settings.mixing * new_density.data
            density = Density(self.grid, mixed)
            density.normalize(self.system.electrons)

            if residual < self.settings.tolerance:
                converged = True
                break

        state = State(self.grid, orbitals, occupations[: len(orbitals)]) if orbitals is not None else None
        if state is not None:
            state.eigenvalues = eigvals
        return Result(
            energy=energy,
            components=components,
            density=density,
            state=state,
            converged=converged,
            iterations=i + 1,
            history=history,
        )
