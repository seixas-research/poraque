"""Official VASP PAW generalized Bloch/Harris solver on ext2chg input fields.

The band term is evaluated from Hc=epsilon Sc. Both H and S use the official
PAW projectors. The smooth density includes compensation (CHGCAR convention).
The one-center term is E1[R_in]-R_in.dE1/dR_in; adding E1 alone to a band
energy would double count its Hamiltonian contribution.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import block_diag

from ..fields import ExternalPotential, FieldGrid
from ..fields.structure import Structure
from ..fields.vasp.user_paw import UserPawDataset
from ..fields.vasp.kpoints import Kpoints
from ..fields.constants import COULOMB_CONSTANT_EV_ANGSTROM
from .bloch import ElectronicSettings, BlochHarrisResult
from .magnetic_bloch import (GeneralizedBlochHamiltonian, SpinorHamiltonian,
                             solve_mesh, output_fields, band_path)
from .spin_xc import SpinDensityInput, spin_xc
from .paw import (AUTOA,HARTREE,FELECT,atomic_energy,_reciprocal_table)
from .paw_variational import PawAtom
from .energy import hartree_potential,ewald_energy
from .harris import integrate


def official_local_potential(structure: Structure, grid: FieldGrid,
                             datasets: dict[str,UserPawDataset]) -> ExternalPotential:
    """Official local Fourier tables with VASP's Coulomb constant and zero mode."""
    g=np.stack(grid.get_g_vectors(),axis=-1)
    magnitude=np.linalg.norm(g,axis=-1)
    coefficients=np.zeros(grid.shape,dtype=complex)
    for element,count in zip(structure.elements,structure.counts):
        setup=datasets[element].setup
        table=setup.reciprocal['local part']
        qmax,values=table[0],table[1:]
        spline=_reciprocal_table(values,qmax)
        active=(magnitude>0)&(magnitude<qmax*(1-1/len(values)))
        transformed=np.zeros(grid.shape)
        transformed[active]=spline(magnitude[active])-4*np.pi*setup.zval*FELECT/magnitude[active]**2
        positions=structure.positions[np.repeat(structure.elements,structure.counts)==element]
        factor=sum(np.exp(-1j*g@position) for position in positions)
        coefficients+=transformed*factor/grid.volume
    values=np.fft.ifftn(coefficients).real*np.prod(grid.shape)
    return ExternalPotential(values,grid,structure,metadata={'model':'official-VASP-PAW'})


@dataclass(frozen=True)
class PawHarrisInput:
    """One prediction: CHGCAR-convention charge/magnetization and atom LM records.

    records[atom] shape is (sets,K): sets=1 (charge), 2 (charge,mz) or
    4 (charge,mx,my,mz). No atom/reference-occupancy fallback is supplied.
    """

    density: SpinDensityInput
    records: tuple[np.ndarray,...]


class PawNonlocalPotential:
    """Prepared density-dependent PAW strengths and overlap at all k points."""

    kinetic_prefactor=.5*HARTREE*AUTOA**2

    def __init__(self,structure: Structure,atoms: list[PawAtom],strengths: list[np.ndarray],
                 soc: list[np.ndarray],channel: int | None = None) -> None:
        self.structure,self.atoms,self.strengths,self.soc=structure,atoms,strengths,soc
        self.channel=channel

    def overlap(self,q: np.ndarray) -> np.ndarray:
        """S_GG'=delta_GG'+sum_a <G|p_i> Q_ij <p_j|G'>."""
        result=np.eye(len(q),dtype=complex)
        for atom,position in zip(self.atoms,self.structure.positions):
            b=atom.projectors(q,position,self.structure.volume)
            result+=b@atom.qmatrix@b.conj().T
        return result

    def matrix(self,q: np.ndarray) -> np.ndarray:
        """Scalar/collinear projector Hamiltonian, including compensation derivatives."""
        result=np.zeros((len(q),len(q)),dtype=complex)
        for atom,position,strength in zip(self.atoms,self.structure.positions,self.strengths):
            b=atom.projectors(q,position,self.structure.volume)
            d=strength[0] if self.channel is None else strength[0]+self.channel*strength[3]
            result+=b@d@b.conj().T
        return result

    def spinor_matrix(self,q: np.ndarray,soc: bool = True) -> np.ndarray:
        """PAW spin-flip D blocks plus the on-site radial L.S matrix.

        Vnl=diag(B,B) [[D0+Dz,Dx-iDy],[Dx+iDy,D0-Dz]] diag(B,B)†
             +diag(B,B) D_SOC diag(B,B)†.
        The overlap remains spin diagonal; all spatial/spin phases are retained.
        """
        result=np.zeros((2*len(q),2*len(q)),dtype=complex)
        for atom,position,d,so in zip(self.atoms,self.structure.positions,self.strengths,self.soc):
            b=atom.projectors(q,position,self.structure.volume)
            projector=block_diag(b,b)
            matrix=np.block([[d[0]+d[3],d[1]-1j*d[2]],[d[1]+1j*d[2],d[0]-d[3]]])
            if soc:
                matrix=matrix+so
            result+=projector@matrix@projector.conj().T
        return result


class PawHarrisEvaluator:
    """Fixed-density official PAW LDA/PBE, collinear and SOC spinor evaluation.

    Includes the full angular one-center functional, frozen-core XC, PAW
    compensation derivatives, alpha-Z/Ewald/atomic offsets and generalized S.
    Numerical radial/FFT/angular/cutoff approximations must be converged; this
    implementation has not been certified to reproduce VASP SOC total energies.
    """

    def __init__(self,structure: Structure,grid: FieldGrid,electrons: float,
                 settings: ElectronicSettings,datasets: dict[str,UserPawDataset]) -> None:
        settings.validate()
        self.structure,self.grid,self.electrons,self.settings=structure,grid,electrons,settings
        if not np.isfinite(electrons) or electrons<=0:
            raise ValueError('Invalid NELECT')
        self.datasets=datasets
        self.species=list(np.repeat(structure.elements,structure.counts))
        for dataset in datasets.values():
            error=dataset.setup.error_table
            if error is not None and not error[0][0]<=settings.encut<=error[0][-1]:
                raise ValueError('ENCUT lies outside the official PAW basis-error table; extrapolation is disabled')
        models={element:PawAtom(datasets[element],settings.functional) for element in structure.elements}
        self.atoms=[models[element] for element in self.species]
        bytes_required=sum(len(atom.oc.layout)*int(np.prod(grid.shape))*8 for atom in self.atoms)
        if bytes_required>1_000_000_000:
            raise ValueError('PAW compensation cache exceeds 1 GB; a streamed backend is required')
        self.compensation=[atom.compensation(grid,position) for atom,position in zip(self.atoms,structure.positions)]
        self.external=official_local_potential(structure,grid,datasets).data
        self.core=self._core_density()
        setups={element:dataset.setup for element,dataset in datasets.items()}
        charges={element:setup.zval for element,setup in setups.items()}
        self.ion_ion=ewald_energy(structure,charges)*FELECT/COULOMB_CONSTANT_EV_ANGSTROM
        ztotal=sum(atom.setup.zval for atom in self.atoms)
        self.alpha_z=sum(atom.setup.reciprocal['local part'][1] for atom in self.atoms)*ztotal/grid.volume
        self.atomic=atomic_energy(setups,structure,settings.encut)
        self.last_terms={}

    def _core_density(self) -> np.ndarray:
        g=np.stack(self.grid.get_g_vectors(),axis=-1)
        magnitude=np.linalg.norm(g,axis=-1)
        coefficients=np.zeros(self.grid.shape,dtype=complex)
        for atom,position in zip(self.atoms,self.structure.positions):
            values=atom.setup.reciprocal.get('core charge-density (partial)')
            if values is None:
                continue
            qmax=atom.setup.reciprocal['local part'][0]
            spline=_reciprocal_table(values,qmax)
            active=magnitude<qmax*(1-3/len(values))
            coefficients+=np.where(active,spline(np.where(active,magnitude,0)),0)*np.exp(-1j*g@position)/self.grid.volume
        return np.fft.ifftn(coefficients).real*np.prod(self.grid.shape)

    def _prepare(self,prediction: PawHarrisInput) -> tuple[tuple, float, float, float, float]:
        if not isinstance(prediction,PawHarrisInput):
            raise TypeError('Official PAW Harris requires ext2chg density and predicted occupancy records')
        density=prediction.density
        density.validate(self.grid,self.electrons)
        if len(prediction.records)!=len(self.atoms):
            raise ValueError('One predicted PAW occupancy array per atom is required')
        expected=4 if self.settings.spinor else (2 if self.settings.ispin==2 else 1)
        m=density.magnetization
        if not self.settings.magnetic and np.any(np.asarray(m)!=0):
            raise ValueError('Scalar ISPIN=1 requires zero magnetization')
        if self.settings.spinor and m.ndim!=4:
            raise ValueError('Spinors require Cartesian mx,my,mz input')
        if not self.settings.spinor and m.ndim!=3:
            raise ValueError('Collinear calculations require scalar mz input')
        full=SpinDensityInput(density.charge+self.core,m)
        xc=spin_xc(full,self.grid,self.settings.functional,a0=AUTOA,hartree_ev=HARTREE)
        vh=hartree_potential(density.charge,self.grid)*FELECT/COULOMB_CONSTANT_EV_ANGSTROM
        scalar=self.external+vh+xc.scalar
        magnetic=(xc.magnetic if xc.magnetic.ndim==4 else
                  np.array([np.zeros(self.grid.shape),np.zeros(self.grid.shape),xc.magnetic]))
        eh=.5*integrate(density.charge*vh,self.grid)
        contraction=m*xc.magnetic
        if contraction.ndim==4:
            contraction=contraction.sum(axis=0)
        xdc=integrate(density.charge*xc.scalar+contraction,self.grid)
        strengths,soc=[],[]
        one_energy,one_dc=0.,0.
        for atom,records,comp in zip(self.atoms,prediction.records,self.compensation):
            records=np.asarray(records,dtype=float)
            if records.shape!=(expected,len(atom.oc.layout)) or not np.isfinite(records).all():
                raise ValueError('Predicted PAW occupancy shape/sets do not match the official POTCAR')
            energy,derivative=atom.derivative(records)
            one_energy+=energy
            one_dc+=float(np.sum(records*derivative))
            gradient=np.zeros((4,len(atom.oc.layout)))
            gradient[0]=derivative[0]
            if expected==2:
                gradient[3]=derivative[1]
            elif expected==4:
                gradient[1:]=derivative[1:]
            volume_element=self.grid.volume/np.prod(self.grid.shape)
            gradient+=np.einsum('kxyz,cxyz->ck',comp,np.concatenate([scalar[None],magnetic]),optimize=True)*volume_element
            strengths.append(np.einsum('ck,kij->cij',gradient,atom.gaunt,optimize=True))
            soc.append(atom.soc(records) if self.settings.lsorbit else np.zeros((2*atom.offsets[-1],)*2))
        potential=PawNonlocalPotential(self.structure,self.atoms,strengths,soc)
        if self.settings.spinor:
            hs=(SpinorHamiltonian(self.grid,scalar,magnetic,self.settings.encut,potential,self.settings.lsorbit),)
        elif self.settings.ispin==2:
            hs=tuple(GeneralizedBlochHamiltonian(self.grid,scalar+sign*magnetic[2],self.settings.encut,
                PawNonlocalPotential(self.structure,self.atoms,strengths,soc,sign)) for sign in (1,-1))
        else:
            hs=(GeneralizedBlochHamiltonian(self.grid,scalar,self.settings.encut,potential),)
        self.last_terms={'one_center_energy':one_energy,'one_center_double_counting':one_dc,
                         'alpha_z':self.alpha_z,'atomic':self.atomic,'hartree':eh,'xc':xc.energy,
                         'xc_double_counting':xdc,'ion_ion':self.ion_ion}
        return hs,eh,xc.energy,xdc,one_energy-one_dc

    def hamiltonian(self,prediction: PawHarrisInput) -> tuple[tuple, float, float, float, float]:
        """Prepare the density-dependent PAW H and S operators without SCF updates."""
        return self._prepare(prediction)

    def evaluate(self,prediction: PawHarrisInput,kpoints: Kpoints) -> BlochHarrisResult:
        r"""E=band-EH-int(nT*vxc+m.Bxc)+Exc+E1-R.dE1+E_alphaZ+E_Ewald+E_atom.

        SOC is already in the eigenvalues. It is a frozen one-body operator
        for rho_in and is not added again as a standalone total-energy term.
        """
        hs,eh,exc,xdc,one_correction=self._prepare(prediction)
        degeneracy=1 if self.settings.magnetic else 2
        states,values,f,mu,ts=solve_mesh(hs,kpoints,self.settings,self.electrons,self.settings.spinor,degeneracy)
        w=kpoints.weights.reshape((-1,)+(1,)*(values.ndim-1))
        band=math.fsum((w*values*f).ravel())
        total=math.fsum([band,-eh,-xdc,exc,one_correction,self.alpha_z,self.ion_ion,self.atomic])
        output,mag=output_fields(states,f,kpoints.weights,self.grid,self.settings.spinor)
        if not self.settings.magnetic:
            mag[:]=0
        # Reconstruct orbital PAW LM records to add compensation to the smooth
        # output density. This diagnostic does not enter input double counting.
        out_records=[np.zeros((4,len(atom.oc.layout))) for atom in self.atoms]
        for ik,(row,weight,k) in enumerate(zip(states,kpoints.weights,kpoints.points)):
            for spin,state in enumerate(row):
                q=(state.indices+(k+.5)%1-.5)@self.grid.reciprocal_cell
                filling=f[ik] if len(row)==1 else f[ik,spin]
                for ia,(atom,position) in enumerate(zip(self.atoms,self.structure.positions)):
                    b=atom.projectors(q,position,self.grid.volume)
                    n=len(state.indices)
                    u=b.conj().T@state.vectors[:n]
                    d=b.conj().T@state.vectors[n:] if self.settings.spinor else None
                    uu=np.einsum('ib,b,jb->ij',u,filling,u.conj())*weight
                    if d is not None:
                        dd=np.einsum('ib,b,jb->ij',d,filling,d.conj())*weight
                        ud=np.einsum('ib,b,jb->ij',u,filling,d.conj())*weight
                        components=[uu+dd,ud+ud.conj().T,1j*(ud-ud.conj().T),uu-dd]
                    elif self.settings.ispin==2:
                        components=[uu,np.zeros_like(uu),np.zeros_like(uu),(1 if spin==0 else -1)*uu]
                    else:
                        components=[uu,np.zeros_like(uu),np.zeros_like(uu),np.zeros_like(uu)]
                    out_records[ia]+=np.einsum('cij,kji->ck',np.array(components),atom.gaunt).real
        for records,comp in zip(out_records,self.compensation):
            fields=np.einsum('ck,kxyz->cxyz',records,comp)
            output+=fields[0]
            mag+=fields[1:]
        if not np.isclose(integrate(output,self.grid),self.electrons,atol=1e-6,rtol=1e-6):
            raise RuntimeError('PAW overlap/output compensation electron count mismatch')
        return BlochHarrisResult(total,total-ts,band,eh,exc,xdc,self.ion_ion,ts,mu,values,f,
            kpoints.weights.copy(),output,max(s.residual for row in states for s in row),mag,
            'spinor' if self.settings.spinor else ('collinear' if self.settings.ispin==2 else 'scalar'))

    def bands(self,prediction: PawHarrisInput,path: Kpoints,nbands: int | None = None) -> dict:
        hs,*_=self._prepare(prediction)
        return band_path(hs,path,self.settings,self.electrons,self.grid,nbands)
