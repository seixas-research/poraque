"""Variational PAW one-center derivatives, real-harmonic projectors and SOC.

Only UserPawDataset instances can initialize this backend. No UPF/NC path
is used. The model's PAW LM records are independent fixed-density inputs; no
reference-atom occupancy fallback is allowed for Harris evaluation.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn
from scipy.linalg import block_diag

from ..fields import FieldGrid
from ..fields.vasp.user_paw import UserPawDataset
from ..ml.energy import OneCenterTorch, _radial_gradient
from .paw import (OneCenterEnergy, _vasp_ylm, AUTOA, HARTREE, ENERGY_DENSITY,
                  FELECT, SQRT4PI, radial_hartree_potential, radial_gradient)
from .xc import lda_xc, pbe_xc
from .spin_orbit import l_dot_s
from .pseudopotentials import sph_harm_y


class PawOneCenterFunctional(OneCenterTorch):
    """Full angular-grid AE-minus-PS PAW correction and exact discrete derivative.

    Includes Dbar, Hartree multipoles, compensation charges, frozen-core XC,
    and non-spherical LDA/PBE. Noncollinear PBE is locally collinear, using
    eigen-densities of the local 2x2 density matrix; transverse stiffness is
    outside this approximation. All derivatives include radial/angular GGA.
    """

    def __init__(self, numpy_energy: OneCenterEnergy) -> None:
        super().__init__(numpy_energy)
        self.functional = numpy_energy.functional

    def xc(self, total: torch.Tensor, magnetization: torch.Tensor | None, core: torch.Tensor) -> torch.Tensor:
        r2 = self.r*self.r
        n = torch.einsum('pl,alr->apr',self.ylm,total)/r2+core
        gradn = (_radial_gradient(n,self.r)[:,:,None,:]*self.directions[None,:,:,None]
                 +torch.einsum('pls,alr->apsr',self.ylm_gradient,total)/(r2*self.r))
        if magnetization is None:
            longitudinal, gradm = torch.zeros_like(n), torch.zeros_like(gradn)
        elif magnetization.ndim == 3:
            longitudinal = torch.einsum('pl,alr->apr',self.ylm,magnetization)/r2
            gradm = (_radial_gradient(longitudinal,self.r)[:,:,None,:]*self.directions[None,:,:,None]
                     +torch.einsum('pls,alr->apsr',self.ylm_gradient,magnetization)/(r2*self.r))
        else:
            vector = torch.einsum('pl,aclr->acpr',self.ylm,magnetization)/r2
            gradient = (_radial_gradient(vector,self.r)[:,:,:,None,:]*self.directions[None,None,:,:,None]
                        +torch.einsum('pls,aclr->acpsr',self.ylm_gradient,magnetization)/(r2*self.r))
            longitudinal = torch.linalg.vector_norm(vector,dim=1)
            denominator = torch.where(longitudinal>0,longitudinal,torch.ones_like(longitudinal))
            gradm = torch.sum(vector[:,:,:,None,:]*gradient,dim=1)/denominator[:,:,None,:]
        up,down = (n+longitudinal)/2,(n-longitudinal)/2
        # The energy and its derivative use the same floor; this retains the
        # existing VASP-style spin-PBE dilute-density regularization.
        if magnetization is not None:
            up,down = torch.clamp(up,min=1e-7),torch.clamp(down,min=1e-7)
        if self.functional == 'lda':
            e = lda_xc(up*AUTOA**3,down*AUTOA**3)
        else:
            def norm(value):
                return torch.linalg.vector_norm(value,dim=2)*AUTOA**4
            local,corr = pbe_xc(up*AUTOA**3,down*AUTOA**3,
                norm((gradn+gradm)/2),norm((gradn-gradm)/2),norm(gradn))
            e = local+corr
        return torch.einsum('p,apr,r->a',self.angular_weights,e*ENERGY_DENSITY,r2*self.weights)

    def forward(self, records: torch.Tensor) -> torch.Tensor:
        """Records (atoms,1/2/4,K) are charge, optionally mz or mx,my,mz."""
        total = records[:,0]
        ae,ps = self.multipoles(total)
        ae_m=ps_m=None
        if records.shape[1] == 2:
            ae_m,ps_m = self.multipoles(records[:,1])
        elif records.shape[1] == 4:
            shape=(len(records),3,ae.shape[1],ae.shape[2])
            ae_m,ps_m = self.multipoles(records[:,1:].reshape(-1,self.record_length))
            ae_m,ps_m=ae_m.reshape(shape),ps_m.reshape(shape)
        elif records.shape[1] != 1:
            raise ValueError('PAW occupancy sets must contain charge, charge/mz, or charge/mx/my/mz')
        return (total@self.linear+self.hartree(ae)-self.hartree(ps)
                +self.xc(ae,ae_m,self.core_ae)-self.xc(ps,ps_m,self.core_ps)-self.core_xc)


def angular_data(channels: list[int]) -> tuple[np.ndarray, np.ndarray, dict[int,np.ndarray]]:
    """Gaunt map R_K=Tr(D G_K) and complex-to-VASP-real harmonic transforms."""
    from ..fields.vasp.augmentation import record_layout
    lmax=max(channels)
    ct,weights=np.polynomial.legendre.leggauss(4*lmax+3)
    phi=2*np.pi*np.arange(8*lmax+6)/(8*lmax+6)
    z,p=np.meshgrid(ct,phi,indexing='ij')
    directions=np.stack([np.sqrt(1-z*z)*np.cos(p),np.sqrt(1-z*z)*np.sin(p),z],axis=-1).reshape(-1,3)
    w=np.repeat(weights,len(phi))*2*np.pi/len(phi)
    harmonics=_vasp_ylm(directions,2*lmax)
    offsets=np.r_[0,np.cumsum([2*ell+1 for ell in channels])]
    gaunt=np.zeros((len(record_layout(channels)),offsets[-1],offsets[-1]))
    for k,(a,b,L,M) in enumerate(record_layout(channels)):
        block=np.einsum('p,pi,pj,p->ij',w,harmonics[channels[a]],harmonics[channels[b]],harmonics[L][:,M])
        gaunt[k,offsets[a]:offsets[a+1],offsets[b]:offsets[b+1]]=block
        if a!=b:
            gaunt[k,offsets[b]:offsets[b+1],offsets[a]:offsets[a+1]]=block.T
    transforms={}
    theta=np.arccos(directions[:,2])
    azimuth=np.arctan2(directions[:,1],directions[:,0])
    for ell in set(channels):
        complex_y=np.stack([sph_harm_y(ell,m,theta,azimuth) for m in range(-ell,ell+1)],axis=1)
        transforms[ell]=complex_y.conj().T@(w[:,None]*harmonics[ell])
    return gaunt,offsets,transforms


@dataclass
class PawAtom:
    """Official radial data and variational atom-centered Hamiltonian terms."""

    dataset: UserPawDataset
    functional: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset,UserPawDataset):
            raise TypeError('PAW Hamiltonian requires an official local VASP dataset')
        self.dataset.validate_source()
        allowed={'pbe':('PE',),'lda':('CA','PW')}
        if self.dataset.setup.parameters.get('LEXCH') not in allowed.get(self.functional,()):
            raise ValueError('Official PAW dataset XC does not match the requested functional')
        self.setup=self.dataset.setup
        self.oc=OneCenterEnergy(self.setup,lasph=True,functional=self.functional)
        self.energy=PawOneCenterFunctional(self.oc)
        self.gaunt,self.offsets,self.real_transforms=angular_data(self.setup.channels)
        self.projector_splines=[]
        for ell,values in zip(self.setup.channels,self.setup.projector_reciprocal):
            step=self.setup.projector_qmax/len(values)
            # Odd/even continuation about q=0, retaining the native table scale.
            x=np.arange(-1,len(values))*step
            y=np.r_[(-1)**ell*values[1],values]
            self.projector_splines.append(CubicSpline(x,y,bc_type='natural'))
        self.qmatrix=np.zeros((self.offsets[-1],self.offsets[-1]))
        for i,li in enumerate(self.setup.channels):
            for j,lj in enumerate(self.setup.channels):
                if li==lj:
                    self.qmatrix[self.offsets[i]:self.offsets[i+1],self.offsets[j]:self.offsets[j+1]]=(
                        np.eye(2*li+1)*self.oc.qpaw[i,j,0])

    def projectors(self,q: np.ndarray,position: np.ndarray,volume: float) -> np.ndarray:
        """<G+k|p_ai> from official reciprocal tables, with real VASP harmonics.

        Tables already contain the radial Fourier normalization; multiplying
        by another 4*pi or converting the radius to Bohr would be incorrect.
        """
        radius=np.linalg.norm(q,axis=1)
        if np.any(radius>=self.setup.projector_qmax*.97):
            raise ValueError('ENCUT reaches the end of official projector tables; choose a suitable dataset')
        direction=np.divide(q,radius[:,None],out=np.zeros_like(q),where=radius[:,None]>0)
        direction[radius==0]=[0,0,1]
        y=_vasp_ylm(direction,max(self.setup.channels))
        columns=[(-1j)**ell*spline(radius)[:,None]*y[ell]
                 for ell,spline in zip(self.setup.channels,self.projector_splines)]
        return np.concatenate(columns,axis=1)*np.exp(-1j*q@position)[:,None]/np.sqrt(volume)

    def derivative(self,records: np.ndarray) -> tuple[float,np.ndarray]:
        """Exact derivative of the chosen discrete one-center functional wrt R_K."""
        value=torch.tensor(records[None],dtype=torch.float64,requires_grad=True)
        energy=self.energy(value).sum()
        gradient,=torch.autograd.grad(energy,value)
        if not torch.isfinite(energy) or not torch.isfinite(gradient).all():
            raise FloatingPointError('Invalid PAW occupancy functional/derivative')
        return float(energy.detach()),gradient.detach().numpy()[0]

    def compensation(self,grid: FieldGrid,position: np.ndarray) -> np.ndarray:
        """d n_hat(r)/d R_K, on the same periodic FFT mesh as rho_in."""
        g=np.stack(grid.get_g_vectors(),axis=-1).reshape(-1,3)
        radius=np.linalg.norm(g,axis=1)
        directions=np.divide(g,radius[:,None],out=np.zeros_like(g),where=radius[:,None]>0)
        directions[radius==0]=[0,0,1]
        y=_vasp_ylm(directions,self.oc.lmax)
        radial=[]
        for ell in range(self.oc.lmax+1):
            values=np.empty(len(radius))
            for start in range(0,len(radius),256):
                stop=min(start+256,len(radius))
                values[start:stop]=np.sum(spherical_jn(ell,radius[start:stop,None]*self.oc.r)
                    *self.oc.aug[ell]*self.oc.weights,axis=1)
            # The defining monopole moment is exactly one. Preserve it at
            # G=0 instead of propagating radial quadrature drift into Tr(SD).
            values[radius==0]=1.0 if ell==0 else 0.0
            radial.append(values)
        phase=np.exp(-1j*g@position)/grid.volume
        result=[]
        for a,b,ell,m in self.oc.layout:
            coefficient=4*np.pi*(-1j)**ell*radial[ell]*y[ell][:,m]*phase*self.oc.qpaw[a,b,ell]
            result.append(np.fft.ifftn(coefficient.reshape(grid.shape)).real*np.prod(grid.shape))
        return np.array(result)

    def soc(self,records: np.ndarray) -> np.ndarray:
        r"""PAW AE radial ZORA SOC: int u_i xi(r) u_j dr times L.S.

        xi = hbar²/[2(m c)²] * (1-V/(2 m c²))^-2 * (dV/dr)/r.
        V is the absolute spherical AE potential (nucleus+core+valence+XC),
        reconstructed for rho_in. This is the spherical on-site PAW SOC
        approximation, not an empirical atomic-number scaling.
        """
        value=torch.tensor(records[None],dtype=torch.float64)
        ae,ps=self.energy.multipoles(value[:,0])
        ae=ae.detach().requires_grad_(True)
        magnetization=None
        if records.shape[0]==2:
            magnetization=self.energy.multipoles(value[:,1])[0]
        elif records.shape[0]==4:
            magnetization=self.energy.multipoles(value[:,1:].reshape(-1,self.energy.record_length))[0]
            magnetization=magnetization.reshape(1,3,*ae.shape[1:])
        exc=self.energy.xc(ae,magnetization,self.energy.core_ae).sum()
        grad,=torch.autograd.grad(exc,ae)
        w=self.oc.weights
        vxc=np.divide(grad.detach().numpy()[0,0],w*SQRT4PI,
                      out=np.zeros_like(w),where=w>0)
        r=self.oc.r
        core_record=self.setup.tables['core charge-density']
        hartree=radial_hartree_potential(0,r,ae.detach().numpy()[0,0]+core_record)/SQRT4PI
        z=self.dataset.potcar[0].atomic_number
        potential=hartree-FELECT*z/r+vxc
        c=137.035999
        xi=(AUTOA**2/(2*c*c))*radial_gradient(potential,r)/r/(1-potential/(2*HARTREE*c*c))**2
        size=self.offsets[-1]
        result=np.zeros((2*size,2*size),dtype=complex)
        for a,ell in enumerate(self.setup.channels):
            if ell==0:
                continue
            t=self.real_transforms[ell]
            rotation=block_diag(t,t)
            angular=rotation.conj().T@l_dot_s(ell)@rotation
            ia=np.r_[np.arange(self.offsets[a],self.offsets[a+1]),size+np.arange(self.offsets[a],self.offsets[a+1])]
            for b,lb in enumerate(self.setup.channels):
                if lb!=ell:
                    continue
                ib=np.r_[np.arange(self.offsets[b],self.offsets[b+1]),size+np.arange(self.offsets[b],self.offsets[b+1])]
                strength=np.sum(self.oc.ae[a]*self.oc.ae[b]*xi*w)
                result[np.ix_(ia,ib)]+=strength*angular
        return result
