"""Harris–Foulkes energies for periodic, local, spin-unpolarized Hamiltonians.

Units are eV, Angstrom and electrons/Angstrom**3. The FFT collocation basis
contains the entire mesh (no spherical cutoff); convergence requires varying
mesh and cell. This is a Gamma-only local-pseudopotential backend, not PAW:
nonlocal projectors, partial cores and augmentation energies are absent.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

from ..fields import FieldGrid
from ..fields.constants import (BOHR_TO_ANGSTROM, HARTREE_TO_EV,
                                HBAR2_OVER_2M_EV_ANGSTROM2)
from .energy import hartree_potential
from .xc import lda_xc


@dataclass(frozen=True)
class HarrisResult:
    """Energy decomposition; eigenvalues include spin via occupations only."""

    total: float
    band: float
    hartree: float
    xc: float
    hartree_double_counting: float
    xc_double_counting: float
    ion_ion: float
    electrons: float
    eigenvalues: np.ndarray
    occupations: np.ndarray
    output_density: np.ndarray
    residual: float


def integrate(values: np.ndarray, grid: FieldGrid) -> float:
    """Periodic rectangular quadrature with accurate float64 summation."""
    array = np.asarray(values, dtype=np.float64)
    if array.shape != grid.shape or not np.isfinite(array).all():
        raise ValueError('Integration requires a finite array on the shared grid')
    return math.fsum(array.ravel()) * grid.volume / array.size


def lda_terms(density: np.ndarray, grid: FieldGrid) -> tuple[float, np.ndarray]:
    """PW92 LDA energy and its exact discrete functional derivative.

    Autograd differentiates the existing XC implementation, avoiding inconsistent
    finite differences in dilute vacuum regions. Density stays fixed throughout.
    """
    n = torch.tensor(density * BOHR_TO_ANGSTROM**3,
                     dtype=torch.float64, requires_grad=True)
    e = lda_xc(n / 2, n / 2)
    derivative, = torch.autograd.grad(e.sum(), n)
    return (integrate(e.detach().numpy() * HARTREE_TO_EV /
                      BOHR_TO_ANGSTROM**3, grid),
            derivative.detach().numpy() * HARTREE_TO_EV)


class HarrisEvaluator:
    """One diagonalization in V_ext + V_H[rho_in] + V_xc[rho_in].

    Only LDA is supported. ``ion_ion`` must use the same ionic model and G=0
    convention as ``external``. No density update or SCF step occurs here.
    Input densities must be nonnegative and normalized to ``electrons``;
    this evaluator never clips or rescales model predictions silently.
    """

    def __init__(self, grid: FieldGrid, external: np.ndarray, electrons: float,
                 ion_ion: float = 0.0, tolerance: float = 1e-8) -> None:
        self.grid = grid
        self.external = np.array(external, dtype=np.float64, copy=True)
        integrate(self.external, grid)
        if not np.isfinite(electrons) or not 0 < electrons <= 2 * np.prod(grid.shape):
            raise ValueError('Electron count exceeds the spin-degenerate basis capacity')
        if not np.isfinite(ion_ion) or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError('Invalid ion energy or eigensolver tolerance')
        self.electrons, self.ion_ion, self.tolerance = electrons, ion_ion, tolerance
        self.kinetic = HBAR2_OVER_2M_EV_ANGSTROM2 * sum(
            component**2 for component in grid.get_g_vectors())
        inverse = np.ix_(*[(-np.arange(length)) % length for length in grid.shape])
        self.real_hamiltonian = np.allclose(self.kinetic, self.kinetic[inverse],
                                           rtol=1e-14, atol=1e-14)

    def evaluate(self, density: np.ndarray) -> HarrisResult:
        """Evaluate E_band - integral(n V_H) + E_H - integral(n V_xc) + E_xc + E_II.

        The orbital output density is diagnostic only. All double counting
        terms use the supplied input density, never the orbital density.
        Occupations are zero-temperature, spin-degenerate (up to two).
        """
        n = np.array(density, dtype=np.float64, copy=True)
        count = integrate(n, self.grid)
        if np.any(n < 0) or not np.isclose(count, self.electrons, rtol=1e-6, atol=1e-8):
            raise ValueError('Density must be nonnegative and have the required electron count')
        vh = hartree_potential(n, self.grid)
        exc, vxc = lda_terms(n, self.grid)
        potential = self.external + vh + vxc
        size = n.size

        def apply(vector: np.ndarray) -> np.ndarray:
            psi = vector.reshape(self.grid.shape)
            # Keep complex components: on skew cells the even-mesh Nyquist
            # modes need not form inversion pairs in this FFT basis.
            kinetic_psi = np.fft.ifftn(self.kinetic * np.fft.fftn(psi))
            if self.real_hamiltonian:
                kinetic_psi = kinetic_psi.real
            return (kinetic_psi + potential * psi).ravel()

        dtype = np.float64 if self.real_hamiltonian else np.complex128
        hamiltonian = LinearOperator((size, size), matvec=apply, dtype=dtype)
        occupied = int(np.ceil(self.electrons / 2))
        if size <= 128 or occupied >= size - 1:
            if size > 2048:
                raise ValueError('Near-full basis occupation would require a large dense solve')
            matrix = np.column_stack([apply(column) for column in np.eye(size)])
            eigenvalues, vectors = np.linalg.eigh(matrix)
            eigenvalues, vectors = eigenvalues[:occupied], vectors[:, :occupied]
        else:
            eigenvalues, vectors = eigsh(hamiltonian, k=occupied, which='SA',
                                        tol=self.tolerance,
                                        v0=np.random.default_rng(0).normal(size=size))
            order = np.argsort(eigenvalues)
            eigenvalues, vectors = eigenvalues[order], vectors[:, order]
        residual = max(np.linalg.norm(apply(vectors[:, i]) - value * vectors[:, i])
                       for i, value in enumerate(eigenvalues))
        if residual > max(1e-6, 100 * self.tolerance):
            raise RuntimeError(f'Eigensolver residual too large: {residual} eV')
        occupations = np.clip(self.electrons - 2 * np.arange(occupied), 0, 2)
        band = float(eigenvalues @ occupations)
        hdc = integrate(n * vh, self.grid)
        xdc = integrate(n * vxc, self.grid)
        eh = 0.5 * hdc
        output = ((np.abs(vectors)**2) @ occupations).reshape(n.shape) * size / self.grid.volume
        return HarrisResult(band - hdc + eh - xdc + exc + self.ion_ion,
                            band, eh, exc, hdc, xdc, self.ion_ion, count,
                            eigenvalues, occupations, output, float(residual))
