"""Cutoff plane-wave Bloch Hamiltonians and fixed-density Harris energies.

Scalar norm-conserving pseudopotentials use identity overlap. PAW/USPP must
not be passed as NC data: their generalized overlap and augmentation are absent.
All public energies are eV, lengths Angstrom, and k points fractional reciprocal.
"""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from scipy.special import erfc, expit, xlogy

from ..fields import FieldGrid
from ..fields.vasp.incar import Incar
from ..fields.vasp.kpoints import Kpoints
from ..fields.constants import HBAR2_OVER_2M_EV_ANGSTROM2 as KINETIC
from .energy import hartree_potential
from .harris import integrate, lda_terms
from .pseudopotentials import NonlocalPotential


@dataclass(frozen=True)
class ElectronicSettings:
    """Supported INCAR controls. NELM/EDIFF remain recorded SCF metadata.

    EDIFF is not an eigensolver residual tolerance. The dense Hermitian solver
    is validated independently. SIGMA=0 selects zero-temperature filling;
    ISMEAR=-1 uses Fermi-Dirac and 0 uses Gaussian broadening.
    """

    encut: float
    nbands: int | None = None
    ismear: int = 0
    sigma: float = 0.0
    ediff: float = 1e-4
    nelm: int = 60
    ispin: int = 1
    functional: str = 'lda'
    lsorbit: bool = False
    noncollinear: bool = False
    saxis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def spinor(self) -> bool:
        """SOC implies noncollinear two-component wavefunctions."""
        return self.lsorbit or self.noncollinear

    @property
    def magnetic(self) -> bool:
        """Use explicit spin states (occupancy <=1), rather than spin degeneracy."""
        return self.ispin == 2 or self.spinor

    @classmethod
    def from_incar(cls, incar: Incar) -> 'ElectronicSettings':
        """Reject requested physics the backend cannot honor, rather than ignore it."""
        def number(key: str, default: float | None = None) -> float:
            if key not in incar:
                if default is None:
                    raise ValueError(f'INCAR requires {key}')
                return default
            values = incar.numbers(key)
            if len(values) != 1 or not np.isfinite(values[0]):
                raise ValueError(f'Invalid scalar INCAR tag {key}')
            return values[0]

        def integer(key: str, default: int) -> int:
            value = number(key, default)
            if value != int(value):
                raise ValueError(f'{key} must be an integer')
            return int(value)

        gga = str(incar.get('GGA', '')).strip().upper()
        if (any(incar.get_bool(key, False) for key in ('LHFCALC', 'LDAU')) or
                gga not in ('', 'PE') or
                str(incar.get('METAGGA', '')).strip()):
            raise ValueError('Only LDA or PBE (GGA=PE) is supported, without hybrid/U/meta-GGA')
        if 'NUPDOWN' in incar or incar.get_bool('LSPIRAL', False):
            raise ValueError('Fixed spin-moment constraints and spin spirals are unsupported')
        axis = tuple(incar.numbers('SAXIS')) if 'SAXIS' in incar else (0., 0., 1.)
        result = cls(number('ENCUT'), integer('NBANDS', 0) if 'NBANDS' in incar else None,
                     integer('ISMEAR', 1), number('SIGMA', 0.2),
                     number('EDIFF', 1e-4), integer('NELM', 60),
                     integer('ISPIN', 1), 'pbe' if gga == 'PE' else 'lda',
                     incar.get_bool('LSORBIT', False), incar.get_bool('LNONCOLLINEAR', False), axis)
        result.validate()
        return result

    def validate(self) -> None:
        """Validate values also for settings constructed without an INCAR file."""
        if (not np.isfinite([self.encut, self.sigma, self.ediff]).all() or
                self.encut <= 0 or self.sigma < 0 or self.ediff <= 0 or self.nelm < 1 or
                self.ismear not in (-1, 0) or
                self.ispin not in (1, 2) or self.functional not in ('lda', 'pbe') or
                len(self.saxis) != 3 or not np.isfinite(self.saxis).all() or
                np.linalg.norm(self.saxis) == 0 or
                (self.nbands is not None and (self.nbands < 1 or int(self.nbands) != self.nbands))):
            raise ValueError('Require ENCUT>0, NBANDS>0, ISMEAR=-1/0, SIGMA>=0 and valid convergence tags')


@dataclass(frozen=True)
class KState:
    """Eigenpairs and integer reciprocal basis indices for one Bloch wavevector."""

    eigenvalues: np.ndarray
    vectors: np.ndarray
    indices: np.ndarray
    residual: float


class BlochHamiltonian:
    """H_GG'(k)=delta_GG' hbar²|G+k|²/2m + Veff(G-G') + (B D B†)_GG'.

    The plane-wave sphere is selected separately at every k using ENCUT.
    Grid differences must lie below Nyquist; undersampled potentials fail.
    Dense diagonalization deliberately targets small electronic-structure jobs;
    ``max_basis`` bounds memory before allocating matrices.
    """

    def __init__(self, grid: FieldGrid, effective: np.ndarray, encut: float,
                 nonlocal_potential: NonlocalPotential, max_basis: int = 2048) -> None:
        integrate(effective, grid)
        if not np.isfinite(encut) or encut <= 0 or max_basis < 1:
            raise ValueError('Invalid cutoff or matrix memory bound')
        if not np.allclose(nonlocal_potential.structure.cell, grid.cell):
            raise ValueError('Nonlocal potential/grid cell mismatch')
        self.grid, self.encut, self.nonlocal_potential = grid, encut, nonlocal_potential
        self.kinetic_prefactor = getattr(nonlocal_potential,'kinetic_prefactor',KINETIC)
        self.fourier = np.fft.fftn(effective) / np.prod(grid.shape)
        self.max_basis = max_basis

    def basis(self, kpoint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Enumerate all reciprocal vectors inside the |G+k| energy sphere."""
        k = np.asarray(kpoint, dtype=float)
        if k.shape != (3,) or not np.isfinite(k).all():
            raise ValueError('Require a finite fractional three-vector k')
        # Equivalent k points are reduced before enumerating the basis.
        k = (k + 0.5) % 1 - 0.5
        reciprocal = self.grid.reciprocal_cell
        radius = np.sqrt(self.encut / self.kinetic_prefactor)
        bounds = radius * np.linalg.norm(np.linalg.inv(reciprocal), axis=0)
        low, high = np.ceil(-bounds - k).astype(int), np.floor(bounds - k).astype(int)
        if math.prod((high - low + 1).clip(min=0)) > 5_000_000:
            raise ValueError('Plane-wave candidate set exceeds safety limit')
        indices = np.array(list(product(*(range(a, b + 1) for a, b in zip(low, high)))), dtype=int)
        if indices.size == 0:
            raise ValueError('ENCUT gives an empty basis at this k point')
        q = (indices + k) @ reciprocal
        active = self.kinetic_prefactor * np.sum(q*q, axis=1) <= self.encut * (1 + 1e-12)
        indices, q = indices[active], q[active]
        if not len(indices) or len(indices) > self.max_basis:
            raise ValueError(f'Plane-wave basis size {len(indices)} outside 1..{self.max_basis}')
        if np.any(2 * np.ptp(indices, axis=0) >= np.asarray(self.grid.shape)):
            raise ValueError('Density/potential FFT mesh aliases plane-wave differences; increase NGXF/YF/ZF')
        return indices, q

    def matrix(self, kpoint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the full complex Hermitian matrix, including nonlocal terms."""
        indices, q = self.basis(kpoint)
        difference = indices[:, None, :] - indices[None, :, :]
        address = difference % np.asarray(self.grid.shape)
        matrix = self.fourier[tuple(address[..., axis] for axis in range(3))].copy()
        matrix[np.diag_indices(len(q))] += self.kinetic_prefactor * np.sum(q*q, axis=1)
        matrix += self.nonlocal_potential.matrix(q)
        if not np.allclose(matrix, matrix.conj().T, atol=1e-9, rtol=1e-12):
            raise ValueError('Non-Hermitian Hamiltonian')
        return matrix, indices

    def solve(self, kpoint: np.ndarray, nbands: int) -> KState:
        """Lowest NBANDS eigenpairs with an explicit residual check."""
        matrix, indices = self.matrix(kpoint)
        if not 0 < nbands <= len(matrix):
            raise ValueError(f'NBANDS={nbands} exceeds basis dimension {len(matrix)}')
        from ..hpc.runtime import try_solve
        native=try_solve(matrix,None,nbands)
        if native is None:
            values, vectors = eigh(matrix, subset_by_index=(0, nbands - 1), driver='evr')
        else:
            values,vectors=native.eigenvalues,native.vectors
        residual = float(np.max(np.linalg.norm(matrix @ vectors - vectors * values, axis=0)))
        if not np.isfinite(values).all() or residual > 1e-7:
            raise RuntimeError(f'Bloch eigenpair residual {residual:g} eV')
        return KState(values, vectors, indices, residual)


def occupations(eigenvalues: np.ndarray, weights: np.ndarray, electrons: float,
                ismear: int, sigma: float, degeneracy: int = 2) -> tuple[np.ndarray, float, float]:
    """Global chemical potential, occupations [0,2], and T*S correction in eV.

    Electron count is sum_k w_k sum_n f_kn. Zero-temperature degenerate states
    share a common fractional filling, including across different k points.
    At finite width require a negligible occupation in the highest computed
    band, so truncating the band set cannot silently corrupt the Fermi level.
    """
    e, w = np.asarray(eigenvalues), np.asarray(weights)
    if (e.ndim != 2 or w.shape != (len(e),) or not np.isfinite(e).all() or
            not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1) or
            degeneracy not in (1, 2) or
            not np.isfinite(electrons) or not 0 < electrons <= degeneracy * e.shape[1] or
            ismear not in (-1, 0) or not np.isfinite(sigma) or sigma < 0):
        raise ValueError('Invalid spectrum, weights, electron count or smearing')
    if sigma == 0:
        f = np.zeros_like(e)
        order = np.argsort(e.ravel())
        remaining, offset, mu = electrons, 0, float(e.min())
        while offset < len(order) and remaining > 1e-12:
            energy = e.ravel()[order[offset]]
            end = offset + 1
            while end < len(order) and abs(e.ravel()[order[end]] - energy) < 1e-9:
                end += 1
            group = order[offset:end]
            capacity = degeneracy * np.sum(w[group // e.shape[1]])
            if capacity > 0:
                fraction = min(1.0, remaining / capacity)
                f.ravel()[group] = degeneracy * fraction
                remaining -= capacity * fraction
                mu = float(energy)
            offset = end
        return f, mu, 0.0
    fraction = (lambda mu: expit((mu - e) / sigma)) if ismear == -1 else (
        lambda mu: 0.5 * erfc((e - mu) / sigma))
    mu = brentq(lambda value: float(np.sum(w[:, None] * degeneracy * fraction(value)) - electrons),
                float(e.min() - 100 * sigma - 1), float(e.max() + 100 * sigma + 1), xtol=1e-12)
    p = fraction(mu)
    if np.any(p[w > 0, -1] > 1e-7):
        raise ValueError('Highest band remains occupied; increase NBANDS or ENCUT')
    if ismear == -1:
        ts = -degeneracy * sigma * np.sum(w[:, None] * (xlogy(p, p) + xlogy(1-p, 1-p)))
    else:
        ts = degeneracy/2 * sigma / np.sqrt(np.pi) * np.sum(w[:, None] * np.exp(-((e-mu)/sigma)**2))
    return degeneracy*p, float(mu), float(ts)


@dataclass(frozen=True)
class BlochHarrisResult:
    """Internal/free Harris energies, BZ eigenpairs, density and decomposition."""

    total: float
    free_energy: float
    band: float
    hartree: float
    xc: float
    xc_double_counting: float
    ion_ion: float
    entropy_term: float
    fermi: float
    eigenvalues: np.ndarray
    occupations: np.ndarray
    weights: np.ndarray
    output_density: np.ndarray
    max_residual: float
    output_magnetization: np.ndarray | None = None
    spin_mode: str = 'scalar'


class BlochHarrisEvaluator:
    """Construct H(k) from one fixed input density, including NC nonlocal terms."""

    def __init__(self, grid: FieldGrid, external: np.ndarray, electrons: float,
                 ion_ion: float, settings: ElectronicSettings,
                 nonlocal_potential: NonlocalPotential) -> None:
        settings.validate()
        integrate(external, grid)
        if not np.isfinite([electrons, ion_ion]).all() or electrons <= 0:
            raise ValueError('Invalid electron count or ionic energy')
        self.grid, self.external, self.electrons = grid, np.array(external, copy=True), electrons
        self.ion_ion, self.settings, self.nonlocal_potential = ion_ion, settings, nonlocal_potential

    def hamiltonian(self, density: np.ndarray) -> tuple[BlochHamiltonian, float, float, float]:
        """Build potentials and double counting exclusively from rho_in."""
        if self.settings.magnetic:
            from .magnetic_bloch import MagneticHarrisEvaluator
            return MagneticHarrisEvaluator(self).hamiltonian(density)
        n = np.asarray(density, dtype=float)
        if np.any(n < 0) or not np.isclose(integrate(n, self.grid), self.electrons, rtol=1e-6):
            raise ValueError('Require nonnegative density normalized to NELECT')
        vh = hartree_potential(n, self.grid)
        if self.settings.functional == 'pbe':
            from .spin_xc import SpinDensityInput, spin_xc
            xc = spin_xc(SpinDensityInput(n, np.zeros_like(n)), self.grid, 'pbe')
            exc, vxc = xc.energy, xc.scalar
        else:
            exc, vxc = lda_terms(n, self.grid)
        h = BlochHamiltonian(self.grid, self.external + vh + vxc,
                             self.settings.encut, self.nonlocal_potential)
        return h, 0.5 * integrate(n * vh, self.grid), exc, integrate(n * vxc, self.grid)

    def evaluate(self, density: np.ndarray, kpoints: Kpoints) -> BlochHarrisResult:
        """BZ-weighted band sum - E_H - integral(n V_xc) + E_xc + E_II.

        A density-independent norm-conserving V_nl is already in the band sum;
        it adds no Hartree/XC double-counting term. No SCF update is performed.
        """
        if kpoints.weights is None:
            raise ValueError('Line-mode paths cannot be used for total-energy integration')
        if self.settings.magnetic:
            from .magnetic_bloch import MagneticHarrisEvaluator
            return MagneticHarrisEvaluator(self).evaluate(density, kpoints)
        h, eh, exc, xdc = self.hamiltonian(density)
        minimum = min(len(h.basis(point)[0]) for point in kpoints.points)
        nbands = self.settings.nbands or min(minimum, math.ceil(self.electrons / 2) + 4)
        states = [h.solve(point, nbands) for point in kpoints.points]
        values = np.array([state.eigenvalues for state in states])
        f, mu, ts = occupations(values, kpoints.weights, self.electrons,
                                 self.settings.ismear, self.settings.sigma)
        band = math.fsum((kpoints.weights[:, None] * f * values).ravel())
        total = math.fsum([band, -eh, -xdc, exc, self.ion_ion])
        output = np.zeros(self.grid.shape)
        for state, weight, filling in zip(states, kpoints.weights, f):
            for column, occupation in zip(state.vectors.T, filling):
                coefficients = np.zeros(self.grid.shape, dtype=complex)
                address = state.indices % np.asarray(self.grid.shape)
                coefficients[tuple(address.T)] = column
                psi = np.fft.ifftn(coefficients) * np.prod(self.grid.shape) / np.sqrt(self.grid.volume)
                output += weight * occupation * np.abs(psi)**2
        return BlochHarrisResult(total, total-ts, band, eh, exc, xdc, self.ion_ion, ts,
                                 mu, values, f, kpoints.weights.copy(), output,
                                 max(state.residual for state in states))

    def bands(self, density: np.ndarray, path: Kpoints, nbands: int | None = None) -> dict:
        """JSON-compatible raw eigenvalues and path coordinates; no path Fermi fit."""
        if path.weights is not None:
            raise ValueError('Band output requires a line-mode path')
        if self.settings.magnetic:
            from .magnetic_bloch import MagneticHarrisEvaluator
            return MagneticHarrisEvaluator(self).bands(density, path, nbands)
        h, _, _, _ = self.hamiltonian(density)
        count = nbands or self.settings.nbands or math.ceil(self.electrons / 2) + 4
        states = [h.solve(point, count) for point in path.points]
        return {'kpoints_fractional': path.points.tolist(),
                'distance_inverse_angstrom': path.distances(self.grid.cell).tolist(),
                'eigenvalues_ev': [state.eigenvalues.tolist() for state in states],
                'segment_starts': list(path.segment_starts), 'labels': list(path.labels),
                'max_residual_ev': max(state.residual for state in states)}
