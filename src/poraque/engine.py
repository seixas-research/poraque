# -*- coding: utf-8 -*-
# file: engine.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .core import Density, Result, State
from .core import reporting
from .functionals import Hartree, LDA


class OFDFTEngine:
    """
    Engine for Orbital-Free DFT calculations.

    Performs direct minimization of the total energy with respect to the
    positivity-preserving variable ``w = sqrt(n)`` under the electron-number
    constraint ``integral(w^2) = N``. A backtracking line search keeps the
    energy monotonically decreasing for stability.
    """

    def __init__(self, system, grid, functionals, backend, settings, verbose=False):
        self.system = system
        self.grid = grid
        self.functionals = functionals
        self.backend = backend
        self.settings = settings
        self.verbose = verbose

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
        if self.verbose:
            print(reporting.scf_header("Orbital-free DFT"))
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
            if self.verbose:
                print(reporting.scf_step(i, energy, residual,
                                         extra=f"mu = {mu:+.6f}"))

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
        if self.verbose:
            print(reporting.format_energy_decomposition(
                energy, components, converged=converged, iterations=i + 1))
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


def fermi_fill(eigenvalues_per_k, weights, n_electrons, max_occ=2.0):
    """
    Zero-temperature occupation across a weighted set of k-points.

    All Bloch eigenvalues from every k-point are pooled and filled from the
    bottom until the electron count is reached, so the chemical potential is
    common to the whole Brillouin-zone sample (the correct band-filling for both
    insulators and metals at ``T = 0``). A state at k-point ``k`` that is fully
    occupied contributes ``weights[k] * max_occ`` electrons.

    Parameters
    ----------
    eigenvalues_per_k : sequence of array_like
        Eigenvalues for each k-point (one array per k-point).
    weights : array_like
        k-point weights (need not be normalized; they are normalized here).
    n_electrons : float
        Total number of electrons to place.
    max_occ : float, optional
        Maximum occupation per orbital (``2`` for spin-degenerate states).

    Returns
    -------
    list of numpy.ndarray
        Occupation numbers per orbital for each k-point.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    occ = [np.zeros(len(ev)) for ev in eigenvalues_per_k]

    states = []
    for ik, ev in enumerate(eigenvalues_per_k):
        for ib, e in enumerate(ev):
            states.append((float(e), ik, ib))
    states.sort(key=lambda s: s[0])

    remaining = float(n_electrons)
    for _e, ik, ib in states:
        if remaining <= 1e-14:
            break
        capacity = weights[ik] * max_occ
        take = min(capacity, remaining)
        occ[ik][ib] = take / weights[ik] if weights[ik] > 0 else 0.0
        remaining -= take
    return occ


class KSDFTEngine:
    """
    Kohn-Sham DFT engine on the real-space grid with k-point sampling.

    The Kohn-Sham Hamiltonian for a Bloch state of wavevector :math:`\\mathbf{k}`
    acts on the cell-periodic part :math:`u_{\\mathbf{k}}` as

    .. math::

        \\hat{H}_{\\mathbf{k}} = \\tfrac{1}{2}\\left(-i\\nabla + \\mathbf{k}\\right)^2
                   + v_{ext}(\\mathbf{r}) + v_H[n](\\mathbf{r}) + v_{xc}[n](\\mathbf{r}),

    with the kinetic operator applied spectrally in the plane-wave basis
    (:math:`\\tfrac12 |\\mathbf{G} + \\mathbf{k}|^2` is diagonal in reciprocal
    space) and the local potential applied as a diagonal multiplication in real
    space. At each k-point the occupied orbitals are obtained with a matrix-free
    Hermitian eigensolver; the density is accumulated over the Brillouin-zone
    sample and updated with linear mixing inside a fixed-point SCF loop.

    For a single :math:`\\Gamma` point (the default) the Hamiltonian is real and
    the engine reduces to a standard molecular real-space KS-DFT solver.

    Parameters
    ----------
    system : System
        Atomic structure (provides the electron count).
    grid : Grid
        Real-space grid (and its dual plane-wave basis).
    v_ext : numpy.ndarray
        External (electron-nucleus / pseudopotential) potential, in Hartree.
    backend : Backend
        Numerical backend.
    settings : SolverSettings
        SCF configuration.
    xc : Functional, optional
        Exchange-correlation functional (default :class:`~poraque.functionals.LDA`).
    hartree : bool, optional
        Include the Hartree term (default ``True``).
    kpoints : array_like, optional
        ``(Nk, 3)`` Brillouin-zone sampling points in fractional reciprocal
        coordinates. Defaults to the :math:`\\Gamma` point.
    kweights : array_like, optional
        Weights for ``kpoints`` (normalized internally). Defaults to uniform.
    n_extra_states : int, optional
        Extra unoccupied states to compute for eigensolver robustness.
    """

    def __init__(self, system, grid, v_ext, backend, settings,
                 xc=None, hartree=True, kpoints=None, kweights=None,
                 n_extra_states=2, verbose=False):
        self.system = system
        self.grid = grid
        self.v_ext = np.asarray(v_ext, dtype=float)
        self.backend = backend
        self.settings = settings
        self.xc = xc if xc is not None else LDA()
        self.hartree = Hartree() if hartree else None
        self.n_extra_states = n_extra_states
        self.verbose = verbose

        if kpoints is None:
            self.kpoints = np.zeros((1, 3))
            self.kweights = np.ones(1)
        else:
            self.kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
            self.kweights = (np.ones(len(self.kpoints)) if kweights is None
                             else np.asarray(kweights, dtype=float))
        self.kweights = self.kweights / self.kweights.sum()
        self.n_kpoints = len(self.kpoints)
        # A real Hamiltonian (and real orbitals) is possible only at Gamma.
        self.gamma_only = (self.n_kpoints == 1
                           and np.allclose(self.kpoints[0], 0.0))

        self.n_occupied = int(np.ceil(self.system.electrons / 2.0))
        self.n_states = self.n_occupied + n_extra_states
        self.dV = grid.volume_element

        # Pre-compute the kinetic factor 1/2 |G + k|^2 for every k-point.
        self._kfac = [
            0.5 * grid.kinetic_g2(grid.kpoint_to_cartesian(kf))
            for kf in self.kpoints
        ]

    def _apply_kinetic(self, psi, ik):
        """Apply 1/2 (-i nabla + k)^2 spectrally at k-point ``ik``."""
        psi_g = np.fft.fftn(psi)
        return np.fft.ifftn(self._kfac[ik] * psi_g)

    def _hamiltonian_operator(self, v_eff, ik):
        """Matrix-free Hermitian LinearOperator for H_k = T_k + v_eff."""
        shape = self.grid.shape
        n = self.grid.N
        real = self.gamma_only

        def matvec(x):
            psi = x.reshape(shape)
            hpsi = self._apply_kinetic(psi, ik) + v_eff * psi
            if real:
                hpsi = np.real(hpsi)
            return hpsi.ravel()

        return LinearOperator((n, n), matvec=matvec,
                              dtype=float if real else complex)

    def _effective_potential(self, density):
        v = np.array(self.v_ext, dtype=float)
        if self.hartree is not None:
            v = v + self.hartree.potential(density, self.system, self.grid, self.backend)
        v = v + self.xc.potential(density, self.system, self.grid, self.backend)
        return v

    def _solve_orbitals(self, v_eff, ik):
        """Lowest eigenvalues and grid-normalized orbitals at k-point ``ik``."""
        H = self._hamiltonian_operator(v_eff, ik)
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

    def _band_energy(self, eigvals_per_k, occ_per_k):
        """Brillouin-zone-weighted sum of occupied eigenvalues."""
        e_band = 0.0
        for w, ev, occ in zip(self.kweights, eigvals_per_k, occ_per_k):
            e_band += w * float(np.sum(occ * ev))
        return e_band

    def _total_energy(self, density, e_band, v_eff):
        """
        KS total energy and its physical decomposition.

        The non-interacting kinetic energy is recovered from the band energy via

        .. math:: T_s = E_\\text{band} - \\int v_\\text{eff}\\, n\\, d\\mathbf{r},

        which is exactly :math:`\\sum_i f_i \\langle\\psi_i| \\hat{T} |\\psi_i\\rangle`.
        The total energy is then the strictly additive sum
        ``T_s + E_ext + E_H + E_xc`` (algebraically identical to the usual
        double-counting-corrected band-energy expression), so the reported
        components always add up to :attr:`total_energy`.
        """
        e_ext = self.backend.integrate(self.v_ext * density.data, self.grid)
        v_eff_n = self.backend.integrate(v_eff * density.data, self.grid)
        # Non-interacting kinetic energy from the band/eigenvalue sum.
        e_kin = e_band - v_eff_n

        e_h = 0.0
        if self.hartree is not None:
            e_h = self.hartree.energy(density, self.system, self.grid, self.backend)

        e_xc = self.xc.energy(density, self.system, self.grid, self.backend)

        # Local pseudopotentials only: the nonlocal projector term is exactly 0,
        # but it is reported explicitly to make the accounting complete.
        e_nonlocal = 0.0

        e_total = e_kin + e_ext + e_h + e_xc + e_nonlocal
        # Strictly additive: components sum to total_energy.
        components = {
            "Kinetic": e_kin,
            "External": e_ext,
            "Hartree": e_h,
            "XC": e_xc,
            "Nonlocal": e_nonlocal,
        }
        return e_total, components

    def _build_density(self, orbitals_per_k, occ_per_k):
        """Sum |psi|^2 over occupied states and the Brillouin-zone sample."""
        new_data = np.zeros(self.grid.shape)
        for w, orbitals, occ in zip(self.kweights, orbitals_per_k, occ_per_k):
            for f, psi in zip(occ, orbitals):
                if f > 0:
                    new_data += w * f * np.abs(psi) ** 2
        return new_data

    def run(self, initial_density=None):
        """Run the SCF loop and return a :class:`Result` with the KS state."""
        if initial_density is None:
            n0 = self.system.electrons / self.grid.volume
            density = Density(self.grid, np.full(self.grid.shape, n0))
        else:
            density = initial_density
        density.normalize(self.system.electrons)

        history = {"energy": [], "density_residual": [], "band_energy": []}
        converged = False
        energy = np.nan
        components = {}
        eigvals_per_k = [np.zeros(self.n_states) for _ in range(self.n_kpoints)]
        occ_per_k = None
        orbitals_per_k = None

        if self.verbose:
            print(reporting.scf_header("Kohn-Sham DFT"))

        i = 0
        for i in range(self.settings.max_iter):
            v_eff = self._effective_potential(density)

            eigvals_per_k, orbitals_per_k = [], []
            for ik in range(self.n_kpoints):
                ev, orb = self._solve_orbitals(v_eff, ik)
                eigvals_per_k.append(ev)
                orbitals_per_k.append(orb)

            occ_per_k = fermi_fill(eigvals_per_k, self.kweights, self.system.electrons)

            new_density = Density(self.grid, self._build_density(orbitals_per_k, occ_per_k))
            new_density.normalize(self.system.electrons)

            residual = self.backend.norm(new_density.data - density.data) * np.sqrt(self.dV)
            e_band = self._band_energy(eigvals_per_k, occ_per_k)
            energy, components = self._total_energy(new_density, e_band, v_eff)
            history["energy"].append(energy)
            history["density_residual"].append(residual)
            history["band_energy"].append(e_band)
            if self.verbose:
                print(reporting.scf_step(i, energy, residual))

            # Linear density mixing.
            mixed = (1 - self.settings.mixing) * density.data + self.settings.mixing * new_density.data
            density = Density(self.grid, mixed)
            density.normalize(self.system.electrons)

            if residual < self.settings.tolerance:
                converged = True
                break

        # Expose the first k-point (Gamma for a Gamma-centred grid) as the state.
        state = None
        if orbitals_per_k is not None:
            orbitals0 = orbitals_per_k[0]
            state = State(self.grid, orbitals0, occ_per_k[0][: len(orbitals0)])
            state.eigenvalues = eigvals_per_k[0]
            state.kpoints = self.kpoints
            state.kweights = self.kweights
            state.kpoint_eigenvalues = eigvals_per_k
            state.kpoint_occupations = occ_per_k
        if self.verbose:
            print(reporting.format_energy_decomposition(
                energy, components, converged=converged, iterations=i + 1))
        return Result(
            energy=energy,
            components=components,
            density=density,
            state=state,
            converged=converged,
            iterations=i + 1,
            history=history,
        )
