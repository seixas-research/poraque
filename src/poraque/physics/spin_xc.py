"""Periodic spin LDA/PBE energies and variational discrete XC potentials.

Density matrix convention: R=(n I + m.sigma)/2; m is in electrons/A^3,
not multiplied by mu_B. Collinear input is (n,m_z), noncollinear input is
(n,m_x,m_y,m_z). Noncollinear PBE uses the locally collinear extension:
PBE[n_+,n_-,grad n_+,grad n_-], n_±=(n±|m|)/2. It omits transverse spin
stiffness and spin-current terms; it is not a unique general noncollinear GGA.
"""
from dataclasses import dataclass

import numpy as np
import torch

from ..fields import FieldGrid
from ..fields.constants import BOHR_TO_ANGSTROM as A0, HARTREE_TO_EV
from .harris import integrate
from .xc import lda_xc, pbe_xc


def density_gradients(density: np.ndarray, grid: FieldGrid) -> np.ndarray:
    """Cartesian spectral gradient, shape (3,Nx,Ny,Nz), in e/A^4.

    Row lattice vectors support skew cells. Real even-grid Nyquist derivatives
    use the real skew-adjoint FFT operator (the self-inverse Nyquist component
    has zero derivative). The same operator is used in the energy and its
    autograd adjoint, preserving discrete integration by parts.
    """
    integrate(density, grid)
    transformed = np.fft.fftn(density)
    return np.stack([np.fft.ifftn(1j*g*transformed).real for g in grid.get_g_vectors()])


def _gradient(density: torch.Tensor, grid: FieldGrid) -> torch.Tensor:
    """Torch counterpart; differentiating it includes the GGA divergence term."""
    transformed = torch.fft.fftn(density)
    return torch.stack([torch.fft.ifftn(1j * torch.as_tensor(g) * transformed).real
                        for g in grid.get_g_vectors()])


@dataclass(frozen=True)
class SpinDensityInput:
    """Fixed charge density and scalar/vector magnetization in Cartesian axes."""

    charge: np.ndarray
    magnetization: np.ndarray

    def validate(self, grid: FieldGrid, electrons: float | None = None) -> None:
        """Require Hermitian positive semidefinite R: n>=|m|, n>=0."""
        count = integrate(self.charge, grid)
        m = np.asarray(self.magnetization)
        if m.shape not in (grid.shape, (3, *grid.shape)) or not np.isfinite(m).all():
            raise ValueError('Magnetization requires scalar m_z or three Cartesian components')
        magnitude = np.abs(m) if m.ndim == 3 else np.linalg.norm(m, axis=0)
        if np.any(self.charge < 0) or np.any(magnitude > self.charge + 1e-12):
            raise ValueError('Spin density matrix is not positive semidefinite: require n>=|m|')
        if electrons is not None and not np.isclose(count, electrons, rtol=1e-6, atol=1e-8):
            raise ValueError('Spin charge density is not normalized to NELECT')

    def matrix(self) -> np.ndarray:
        """Return R_ab(r), with R_up,down=(m_x-i*m_y)/2."""
        m = np.asarray(self.magnetization)
        if m.ndim == 3:
            mx, my, mz = np.zeros_like(m), np.zeros_like(m), m
        else:
            mx, my, mz = m
        return np.array([[self.charge+mz, mx-1j*my],
                         [mx+1j*my, self.charge-mz]]) / 2


@dataclass(frozen=True)
class SpinXC:
    """E_xc and V_xc=v I+B.sigma; double counting is integral(n v+m.B)."""

    energy: float
    scalar: np.ndarray
    magnetic: np.ndarray
    double_counting: float


def spin_xc(density: SpinDensityInput, grid: FieldGrid, functional: str = 'lda',
            a0: float = A0, hartree_ev: float = HARTREE_TO_EV) -> SpinXC:
    r"""Differentiate the complete discrete spin LDA/PBE energy.

    v_sigma = partial e/partial n_sigma - div(partial e/partial grad n_sigma).
    Autograd through the FFT gradient supplies the divergence automatically.
    Hence integral Tr(R V_xc)=integral(n v+m.B) already includes all gradient
    derivatives: adding a separate gradient double-counting term would count
    them twice. Atomic-unit density and gradient conversions occur here once.
    """
    density.validate(grid)
    if functional not in ('lda', 'pbe'):
        raise ValueError('XC must be lda or pbe')
    n = torch.tensor(density.charge*a0**3, dtype=torch.float64, requires_grad=True)
    m = torch.tensor(density.magnetization*a0**3, dtype=torch.float64, requires_grad=True)
    longitudinal = m if m.ndim == 3 else torch.linalg.vector_norm(m, dim=0)
    up, down = (n+longitudinal)/2, (n-longitudinal)/2
    if functional == 'pbe':
        gradients = [torch.linalg.vector_norm(_gradient(value, grid)*a0, dim=0)
                     for value in (up, down, n)]
        local, correction = pbe_xc(up, down, *gradients)
        energy_density = local + correction
    else:
        energy_density = lda_xc(up, down)
    dn, dm = torch.autograd.grad(energy_density.sum(), (n, m))
    scalar, magnetic = dn.detach().numpy()*hartree_ev, dm.detach().numpy()*hartree_ev
    energy = integrate(energy_density.detach().numpy()*hartree_ev/a0**3, grid)
    contraction = density.magnetization*magnetic
    if contraction.ndim == 4:
        contraction = contraction.sum(axis=0)
    if not np.isfinite(scalar).all() or not np.isfinite(magnetic).all():
        raise FloatingPointError('Nonfinite spin XC functional derivative')
    return SpinXC(energy, scalar, magnetic,
                  integrate(density.charge*scalar+contraction, grid))
