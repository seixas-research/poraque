"""Variational spin/GGA, angular SOC and official PAW regression checks."""
from pathlib import Path
import os

import numpy as np
import pytest

from poraque.fields import FieldGrid
from poraque.fields.vasp import Incar,Kpoints
from poraque.fields.structure import Structure
from poraque.fields.vasp.user_paw import UserPawDataset as OfficialPawDataset
from poraque.physics.spin_xc import SpinDensityInput,spin_xc,density_gradients
from poraque.physics.spin_orbit import l_dot_s,j_projector
from poraque.physics.harris import integrate,lda_terms
from poraque.physics.bloch import ElectronicSettings


def test_settings_spin_pbe_soc():
    settings=ElectronicSettings.from_incar(Incar.from_string(
        'ENCUT=20; ISMEAR=-1; ISPIN=2; GGA=PE; LSORBIT=.TRUE.; SAXIS=1 1 0'))
    assert settings.spinor and settings.magnetic and settings.functional=='pbe'
    assert settings.saxis==(1,1,0)
    for tag in ('ISPIN=3','SAXIS=0 0 0','GGA=PS','NUPDOWN=2'):
        with pytest.raises(ValueError):
            ElectronicSettings.from_incar(Incar.from_string('ENCUT=20; ISMEAR=0; '+tag))


def test_spectral_gradient_skew_cell():
    grid=FieldGrid((7,7,7),np.array([[4.,0,0],[.7,5.,0],[.2,.4,6.]]))
    indices=np.indices(grid.shape)
    wave=np.array([1,2,-1])
    phase=2*np.pi*sum(wave[i]*indices[i]/grid.shape[i] for i in range(3))
    g=wave@grid.reciprocal_cell
    actual=density_gradients(np.sin(phase),grid)
    assert np.allclose(actual,g[:,None,None,None]*np.cos(phase),atol=1e-12)


@pytest.mark.parametrize('functional',['lda','pbe'])
@pytest.mark.parametrize('vector',[False,True])
def test_spin_xc_variational(functional,vector):
    grid=FieldGrid((5,5,5),np.eye(3)*5)
    x=np.indices(grid.shape)[0]
    n=.08+.01*np.cos(2*np.pi*x/5)
    m=.02+.003*np.sin(2*np.pi*x/5)
    if vector:
        m=np.array([m/2,m/3,m])
    density=SpinDensityInput(n,m)
    xc=spin_xc(density,grid,functional)
    rng=np.random.default_rng(8)
    dn=rng.normal(size=n.shape)
    dm=rng.normal(size=m.shape)
    step=1e-7
    plus=spin_xc(SpinDensityInput(n+step*dn,m+step*dm),grid,functional).energy
    minus=spin_xc(SpinDensityInput(n-step*dn,m-step*dm),grid,functional).energy
    contraction=xc.magnetic*dm
    if vector:
        contraction=contraction.sum(axis=0)
    assert (plus-minus)/(2*step)==pytest.approx(integrate(xc.scalar*dn+contraction,grid),rel=2e-6)
    product=m*xc.magnetic
    if vector:
        product=product.sum(axis=0)
    assert xc.double_counting==pytest.approx(integrate(n*xc.scalar+product,grid))


def test_pbe_limits_rotation_and_vacuum():
    grid=FieldGrid((5,5,5),np.eye(3)*5)
    n=np.full(grid.shape,.04)
    zero=np.zeros((3,*grid.shape))
    pbe=spin_xc(SpinDensityInput(n,zero),grid,'pbe')
    lda,_=lda_terms(n,grid)
    assert pbe.energy==pytest.approx(lda,abs=1e-10)
    assert np.all(pbe.magnetic==0)
    z=np.zeros_like(zero)
    z[2]=.01
    x=np.zeros_like(zero)
    x[0]=.01
    a,b=spin_xc(SpinDensityInput(n,z),grid,'pbe'),spin_xc(SpinDensityInput(n,x),grid,'pbe')
    assert a.energy==pytest.approx(b.energy)
    assert np.allclose(a.magnetic[2],b.magnetic[0])
    vacuum=spin_xc(SpinDensityInput(np.zeros(grid.shape),zero),grid,'pbe')
    assert vacuum.energy==0 and np.isfinite(vacuum.scalar).all()


@pytest.mark.parametrize('ell',[1,2,3])
def test_l_dot_s_and_j_manifolds(ell):
    ls=l_dot_s(ell)
    values=np.linalg.eigvalsh(ls)
    assert np.allclose(values[:2*ell],-(ell+1)/2)
    assert np.allclose(values[2*ell:],ell/2)
    plus,minus=j_projector(ell,ell+.5),j_projector(ell,ell-.5)
    assert np.allclose(plus@minus,0)
    assert np.allclose(plus+minus,np.eye(len(ls)))
    assert np.linalg.norm(ls[:2*ell+1,2*ell+1:])>0


@pytest.fixture(scope='module')
def official_h():
    source=os.environ.get('PORAQUE_TEST_POTCAR_H')
    if not source:
        pytest.skip('Supply a licensed external POTCAR with PORAQUE_TEST_POTCAR_H')
    from poraque.physics.paw_variational import PawAtom
    dataset=OfficialPawDataset.load(source,'pbe')
    return dataset,PawAtom(dataset,'pbe')


def test_official_provenance_and_no_lda_fallback(official_h):
    dataset,atom=official_h
    assert dataset.path==Path(os.environ['PORAQUE_TEST_POTCAR_H']).resolve()
    assert len(dataset.sha256)==64
    assert len(atom.setup.projector_reciprocal)==len(atom.setup.channels)
    assert atom.setup.augmentation_charges.shape==atom.setup.d0.shape
    with pytest.raises(ValueError,match='does not match'):
        OfficialPawDataset.load(dataset.path,'lda')
    with pytest.raises(ValueError,match='file'):
        OfficialPawDataset.load(dataset.path.parent,'pbe')
    with pytest.raises(ValueError,match='explicit'):
        OfficialPawDataset.load(None,'pbe')


def test_paw_record_derivative_and_gaunt(official_h):
    _,atom=official_h
    records=atom.oc.atomic_record()[None]
    energy,gradient=atom.derivative(records)
    rng=np.random.default_rng(7)
    delta=rng.normal(size=records.shape)*.01
    step=1e-5
    plus=atom.derivative(records+step*delta)[0]
    minus=atom.derivative(records-step*delta)[0]
    assert (plus-minus)/(2*step)==pytest.approx(np.sum(gradient*delta),rel=1e-5,abs=1e-6)
    assert np.isfinite(energy)
    matrix=np.eye(atom.offsets[-1])*.2
    direction=np.random.default_rng(9).normal(size=matrix.shape)
    direction=(direction+direction.T)/2
    operator=np.einsum('k,kij->ij',gradient[0],atom.gaunt)
    assert np.trace(operator@direction)==pytest.approx(
        gradient[0]@np.einsum('ij,kji->k',direction,atom.gaunt))


def test_paw_soc_hermiticity_offdiagonal_and_compensation(official_h):
    _,atom=official_h
    records=atom.oc.atomic_record()[None]
    soc=atom.soc(records)
    assert np.allclose(soc,soc.conj().T,atol=1e-10)
    n=atom.offsets[-1]
    assert np.linalg.norm(soc[:n,n:])>0
    grid=FieldGrid((9,9,9),np.eye(3)*5)
    comp=atom.compensation(grid,np.array([.2,.3,.4]))
    projected=np.einsum('k,kij->ij',np.array([integrate(v,grid) for v in comp]),atom.gaunt)
    assert np.allclose(projected,atom.qmatrix,atol=1e-9)


def test_official_spinor_generalized_kramers(official_h):
    from poraque.physics.paw_bloch import PawHarrisEvaluator,PawHarrisInput
    dataset,atom=official_h
    structure=Structure(np.eye(3)*5,['H'],[1],[[.2,.3,.4]])
    grid=FieldGrid((9,9,9),structure.cell)
    settings=ElectronicSettings(20,nbands=6,lsorbit=True,functional='pbe')
    evaluator=PawHarrisEvaluator(structure,grid,1,settings,{'H':dataset})
    records=np.zeros((4,len(atom.oc.layout)))
    records[0]=atom.oc.atomic_record()
    density=SpinDensityInput(np.full(grid.shape,1/grid.volume),np.zeros((3,*grid.shape)))
    prediction=PawHarrisInput(density,(records,))
    hs,*_=evaluator.hamiltonian(prediction)
    state=hs[0].solve(np.zeros(3),6)
    assert np.allclose(state.eigenvalues[::2],state.eigenvalues[1::2],atol=1e-9)
    matrix,indices=hs[0].matrix(np.zeros(3))
    q=indices@grid.reciprocal_cell
    s=evaluator.last_terms
    assert np.isfinite(s['one_center_energy'])
    assert matrix.shape==(2*len(indices),)*2
    assert state.residual<1e-10
    overlap=hs[0].nonlocal_potential.overlap(q)
    from scipy.linalg import block_diag
    assert np.allclose(state.vectors.conj().T@block_diag(overlap,overlap)@state.vectors,np.eye(6),atol=1e-10)
    mesh=Kpoints.from_string('Gamma\n0\nGamma\n1 1 1',grid.cell)
    result=evaluator.evaluate(prediction,mesh)
    assert integrate(result.output_density,grid)==pytest.approx(1,abs=1e-10)
    correction=s['one_center_energy']-s['one_center_double_counting']+s['alpha_z']+s['atomic']
    assert result.total==pytest.approx(result.band-result.hartree-result.xc_double_counting+
                                       result.xc+result.ion_ion+correction)


def test_official_heavy_element_soc():
    source=os.environ.get('PORAQUE_TEST_POTCAR_BI')
    if not source:
        pytest.skip('Supply a licensed external Bi POTCAR with PORAQUE_TEST_POTCAR_BI')
    from poraque.physics.paw_variational import PawAtom
    atom=PawAtom(OfficialPawDataset.load(source,'pbe'),'pbe')
    records=atom.oc.atomic_record()[None]
    soc=atom.soc(records)
    n=atom.offsets[-1]
    assert np.isfinite(soc).all()
    assert np.allclose(soc,soc.conj().T,atol=1e-9)
    assert np.linalg.norm(soc[:n,n:])>1e-3


def test_scalar_collinear_spinor_zero_soc_equivalence(official_h):
    from poraque.physics.paw_bloch import PawHarrisEvaluator,PawHarrisInput
    dataset,atom=official_h
    structure=Structure(np.eye(3)*5,['H'],[1],[[.2,.3,.4]])
    grid=FieldGrid((9,9,9),structure.cell)
    mesh=Kpoints.from_string('G\n0\nGamma\n1 1 1',grid.cell)
    energies=[]
    for mode in ['scalar','collinear','spinor']:
        spinor=mode=='spinor'
        settings=ElectronicSettings(20,nbands=4,ispin=2 if mode=='collinear' else 1,
                                     noncollinear=spinor,functional='pbe')
        m=np.zeros((3,*grid.shape)) if spinor else np.zeros(grid.shape)
        rec=np.zeros((4 if spinor else (2 if mode=='collinear' else 1),len(atom.oc.layout)))
        rec[0]=atom.oc.atomic_record()
        evaluator=PawHarrisEvaluator(structure,grid,1,settings,{'H':dataset})
        result=evaluator.evaluate(PawHarrisInput(SpinDensityInput(np.full(grid.shape,1/grid.volume),m),(rec,)),mesh)
        energies.append(result.total)
        assert np.max(result.occupations)<=(2 if mode=='scalar' else 1)
    assert np.allclose(energies,energies[0],atol=1e-9)


def test_four_channel_ext2chg_decoding(official_h):
    from poraque.fields import ExternalPotential
    from poraque.fields.vector_spin import VectorSpinDensity
    from poraque.ml.training import FieldOperator
    _,atom=official_h
    structure=Structure(np.eye(3)*5,['H'],[1],[[.2,.3,.4]])
    grid=FieldGrid((9,9,9),structure.cell)
    field=ExternalPotential(np.zeros(grid.shape),grid,structure)
    operator=FieldOperator('ext2chg',width=4,modes=2,n_layers=1,
        use_coordinates=False,out_channels=4,site_layouts={1:atom.setup.channels},
        readout_g_max=2.,device='cpu')
    prediction=operator.predict(field)
    assert isinstance(prediction,VectorSpinDensity)
    assert prediction.data.shape==(4,*grid.shape)
    assert np.array_equal(prediction.data[0],prediction.total)
    assert np.array_equal(prediction.data[1:],prediction.magnetization)
