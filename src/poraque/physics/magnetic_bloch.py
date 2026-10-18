"""Collinear and spinor Bloch diagonalization with optional PAW overlap.

Basis ordering for spinors is [all up plane waves, all down plane waves].
Occupations are <=1 per explicit spin state; there is no extra factor of two.
"""
import math
import numpy as np
from scipy.linalg import eigh, block_diag

from .bloch import BlochHamiltonian, KState, BlochHarrisResult, occupations
from .spin_xc import SpinDensityInput, spin_xc
from .energy import hartree_potential
from .harris import integrate


class SpinorHamiltonian(BlochHamiltonian):
    """H=[[H0+Bz, Bx-iBy], [Bx+iBy,H0-Bz]] + Vnl_SOC.

    For PAW, S_spinor=diag(S,S). The generalized eigenproblem is solved with
    Hermitian-definite LAPACK, and the residual Hc-epsilon Sc is checked.
    """

    def __init__(self, grid, scalar, magnetic, encut, nonlocal_potential, soc=False):
        super().__init__(grid, scalar, encut, nonlocal_potential, max_basis=1024)
        if np.shape(magnetic) != (3, *grid.shape):
            raise ValueError('Spinor potential requires Bx,By,Bz on the grid')
        self.magnetic_fourier = np.fft.fftn(magnetic, axes=(-3, -2, -1))/np.prod(grid.shape)
        self.soc = soc

    def matrix(self, kpoint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the 2N x 2N spinor Hamiltonian, including spin-flip projectors."""
        indices, q = self.basis(kpoint)
        address = (indices[:, None, :] - indices[None, :, :]) % self.grid.shape
        select = tuple(address[..., i] for i in range(3))
        scalar = self.fourier[select].copy()
        scalar[np.diag_indices(len(q))] += self.kinetic_prefactor*np.sum(q*q, axis=1)
        bx, by, bz = [value[select] for value in self.magnetic_fourier]
        matrix = np.block([[scalar+bz, bx-1j*by], [bx+1j*by, scalar-bz]])
        matrix += self.nonlocal_potential.spinor_matrix(q, soc=self.soc)
        if not np.allclose(matrix, matrix.conj().T, atol=1e-9):
            raise ValueError('Non-Hermitian spinor Hamiltonian')
        return matrix, indices

    def solve(self, kpoint: np.ndarray, nbands: int) -> KState:
        matrix, indices = self.matrix(kpoint)
        k = (np.asarray(kpoint)+.5) % 1-.5
        q = (indices+k) @ self.grid.reciprocal_cell
        overlap = (block_diag(self.nonlocal_potential.overlap(q), self.nonlocal_potential.overlap(q))
                   if hasattr(self.nonlocal_potential, 'overlap') else np.eye(len(matrix)))
        return generalized_solve(matrix, overlap, indices, nbands)


class GeneralizedBlochHamiltonian(BlochHamiltonian):
    """Scalar/collinear PAW Hamiltonian with S=I+P Q P†."""

    def solve(self, kpoint: np.ndarray, nbands: int) -> KState:
        matrix, indices = self.matrix(kpoint)
        k = (np.asarray(kpoint)+.5) % 1-.5
        q = (indices+k) @ self.grid.reciprocal_cell
        overlap = self.nonlocal_potential.overlap(q)
        return generalized_solve(matrix, overlap, indices, nbands)


def generalized_solve(matrix: np.ndarray, overlap: np.ndarray, indices: np.ndarray,
                      nbands: int) -> KState:
    """Solve Hc=eSc, require positive-definite S, check residual and S-normality."""
    if not 0 < nbands <= len(matrix):
        raise ValueError(f'NBANDS={nbands} exceeds Hamiltonian dimension {len(matrix)}')
    if not np.allclose(overlap, overlap.conj().T, atol=1e-10):
        raise ValueError('Non-Hermitian PAW overlap')
    from ..hpc.runtime import try_solve
    native=try_solve(matrix,overlap,nbands)
    if native is None:
        values, vectors = eigh(matrix, overlap, subset_by_index=(0, nbands-1), driver='gvx')
    else:
        values,vectors=native.eigenvalues,native.vectors
    residual = float(np.max(np.linalg.norm(matrix@vectors-(overlap@vectors)*values, axis=0)))
    if (not np.isfinite(values).all() or residual > 1e-7 or
            not np.allclose(vectors.conj().T@overlap@vectors, np.eye(nbands), atol=1e-9)):
        raise RuntimeError(f'Generalized Bloch eigenproblem residual {residual:g} eV')
    return KState(values, vectors, indices, residual)


def orbital_fields(state: KState, grid, spinor: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-state smooth charge and magnetization, before PAW compensation."""
    size = len(state.indices)
    fields = []
    for components in ([state.vectors[:size], state.vectors[size:]] if spinor else [state.vectors]):
        array = np.zeros((*grid.shape, components.shape[1]), dtype=complex)
        address = state.indices % np.array(grid.shape)
        array[tuple(address.T)] = components
        fields.append(np.fft.ifftn(array, axes=(0,1,2))*np.prod(grid.shape)/np.sqrt(grid.volume))
    up = fields[0]
    if not spinor:
        return np.abs(up)**2, np.zeros((3,*up.shape))
    down = fields[1]
    cross = up*down.conj()
    return np.abs(up)**2+np.abs(down)**2, np.array([2*cross.real,-2*cross.imag,np.abs(up)**2-np.abs(down)**2])


class MagneticHarrisEvaluator:
    """Spin-aware NC reference path; official PAW uses PawHarrisEvaluator instead."""

    def __init__(self, parent):
        self.parent = parent

    def hamiltonian(self, density: SpinDensityInput):
        p = self.parent
        if not isinstance(density, SpinDensityInput):
            raise ValueError('Magnetic calculations require SpinDensityInput from ext2chg')
        density.validate(p.grid, p.electrons)
        xc = spin_xc(density, p.grid, p.settings.functional)
        vh = hartree_potential(density.charge, p.grid)
        scalar = p.external+vh+xc.scalar
        if p.settings.spinor:
            m = xc.magnetic if xc.magnetic.ndim == 4 else np.array([
                np.zeros(p.grid.shape),np.zeros(p.grid.shape),xc.magnetic])
            if p.settings.lsorbit and not any(s.total_j is not None for s in p.nonlocal_potential.datasets.values()):
                raise ValueError('SOC requires fully relativistic nonlocal data')
            h = (SpinorHamiltonian(p.grid,scalar,m,p.settings.encut,p.nonlocal_potential,p.settings.lsorbit),)
        else:
            if xc.magnetic.ndim != 3:
                raise ValueError('Collinear ISPIN=2 requires scalar m_z')
            h = tuple(BlochHamiltonian(p.grid,scalar+sign*xc.magnetic,p.settings.encut,p.nonlocal_potential)
                      for sign in (1,-1))
        return h, .5*integrate(density.charge*vh,p.grid), xc.energy, xc.double_counting

    def evaluate(self, density: SpinDensityInput, kpoints):
        p = self.parent
        hs, eh, exc, xdc = self.hamiltonian(density)
        states, values, f, mu, ts = solve_mesh(hs,kpoints,p.settings,p.electrons,p.settings.spinor)
        charge, magnetization = output_fields(states,f,kpoints.weights,p.grid,p.settings.spinor)
        band = math.fsum((values*f*kpoints.weights.reshape((-1,)+(1,)*(values.ndim-1))).ravel())
        total = math.fsum([band,-eh,-xdc,exc,p.ion_ion])
        return BlochHarrisResult(total,total-ts,band,eh,exc,xdc,p.ion_ion,ts,mu,values,f,
            kpoints.weights.copy(),charge,max(s.residual for row in states for s in row),magnetization,
            'spinor' if p.settings.spinor else 'collinear')

    def bands(self,density,path,nbands=None):
        hs,_,_,_ = self.hamiltonian(density)
        return band_path(hs,path,self.parent.settings,self.parent.electrons,self.parent.grid,nbands)


def solve_mesh(hs, kpoints, settings, electrons: float, spinor: bool, degeneracy: int = 1):
    """Separate collinear solves, one common Fermi level, occupancy <=1."""
    if kpoints.weights is None:
        raise ValueError('Line-mode paths are not a BZ quadrature')
    minimum = min(len(h.basis(k)[0])*(2 if spinor else 1) for h in hs for k in kpoints.points)
    nbands = settings.nbands or min(minimum, math.ceil(electrons/(degeneracy*len(hs)))+4)
    states = [[h.solve(k,nbands) for h in hs] for k in kpoints.points]
    values = np.array([[s.eigenvalues for s in row] for row in states])
    flat = values.reshape(len(values),-1)
    order = np.argsort(flat,axis=1)
    sorted_values = np.take_along_axis(flat,order,axis=1)
    sorted_f,mu,ts = occupations(sorted_values,kpoints.weights,electrons,settings.ismear,settings.sigma,degeneracy=degeneracy)
    f = np.take_along_axis(sorted_f,np.argsort(order,axis=1),axis=1).reshape(values.shape)
    if settings.sigma > 0 and np.any(f[kpoints.weights>0,:,-1] > 1e-7):
        raise ValueError('Highest spin-channel band remains occupied; increase NBANDS')
    if len(hs)==1:
        return states,values[:,0],f[:,0],mu,ts
    return states,values,f,mu,ts


def output_fields(states, filling, weights, grid, spinor: bool):
    """Smooth density diagnostic; PAW adds compensation from orbital records."""
    charge,magnetization = np.zeros(grid.shape),np.zeros((3,*grid.shape))
    for ik,(row,weight) in enumerate(zip(states,weights)):
        for spin,state in enumerate(row):
            n,m = orbital_fields(state,grid,spinor)
            f = filling[ik] if len(row)==1 else filling[ik,spin]
            contribution = np.einsum('...b,b->...',n,f)*weight
            charge += contribution
            if spinor:
                magnetization += np.einsum('c...b,b->c...',m,f)*weight
            else:
                magnetization[2] += (1 if spin==0 else -1)*contribution
    return charge,magnetization


def band_path(hs,path,settings,electrons,grid,nbands=None):
    """Structured explicit-spin bands (Nk,2,Nbands) or spinor bands (Nk,Nbands)."""
    if path.weights is not None:
        raise ValueError('Band output requires a line-mode path')
    count = nbands or settings.nbands or math.ceil(electrons/(1 if settings.spinor else 2))+4
    states = [[h.solve(k,count) for h in hs] for k in path.points]
    values = np.array([[s.eigenvalues for s in row] for row in states])
    if len(hs)==1:
        values = values[:,0]
    return {'kpoints_fractional':path.points.tolist(),'distance_inverse_angstrom':path.distances(grid.cell).tolist(),
        'eigenvalues_ev':values.tolist(),'spin_mode':'spinor' if settings.spinor else ('collinear' if len(hs)==2 else 'scalar'),
        'spin_channels':['up','down'] if len(hs)==2 else None,
        'segment_starts':list(path.segment_starts),'labels':list(path.labels),
        'max_residual_ev':max(s.residual for row in states for s in row)}
