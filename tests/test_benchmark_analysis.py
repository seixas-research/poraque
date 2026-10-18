"""Energy definitions/convergence/error statistics; no POTCAR fixtures."""
import numpy as np
import pytest
from poraque.benchmarks.analysis import read_vasp_reference,energy_difference,summarize,stationarity_fit


def test_energy_conventions_and_convergence(tmp_path):
    (tmp_path/'OUTCAR').write_text('vasp.6.6.1\nIteration 1( 4)\n'
        'aborting loop because EDIFF is reached\nfree energy TOTEN = -3.000000 eV\n'
        'energy without entropy = -2.900000 energy(sigma->0) = -2.950000\n'
        'General timing and accounting informations for this job\n')
    (tmp_path/'OSZICAR').write_text('DAV: 4 -3.0\n 1 F= -3.0 E0= -2.95\n')
    reference=read_vasp_reference(tmp_path)
    assert reference.converged and reference.finished and reference.version=='6.6.1'
    assert reference.free_ev==-3 and reference.internal_ev==-2.9 and reference.sigma_zero_ev==-2.95
    assert reference.electronic_steps==4
    # An earlier converged ionic step cannot bless a later failed SCF loop.
    with (tmp_path/'OUTCAR').open('a') as f:
        f.write('Iteration 2( 99)\nfree energy TOTEN = -4.0 eV\n')
    assert not read_vasp_reference(tmp_path).converged


def test_oszicar_only_is_not_convergence_proof(tmp_path):
    (tmp_path/'OSZICAR').write_text('DAV: 100 -3.0\n 1 F= -3.0D+00 E0= -2.95\n')
    reference=read_vasp_reference(tmp_path)
    assert reference.free_ev==-3 and reference.internal_ev is None
    assert not reference.converged and not reference.finished


def test_errors_and_quadratic_fit():
    error=energy_difference(-2,-2.1,2)
    assert error['delta_ev']==pytest.approx(.1)
    assert error['absolute_error_mev_per_atom']==pytest.approx(50)
    summary=summarize([{'status':'ok','comparison':error},{'status':'failed'}])
    assert summary['valid']==1 and summary['failed']==1
    x=np.array([0.,.125,.25,.5,1.])
    result=stationarity_fit(x,-2+.3*x+.8*x*x)
    assert result['linear_coefficient_ev']==pytest.approx(.3)
    assert result['quadratic_coefficient_ev']==pytest.approx(.8)
    with pytest.raises(ValueError):
        energy_difference(np.nan,0,1)


def test_missing_runtime_potcar_is_required(tmp_path):
    from poraque.fields.vasp.user_paw import UserPawDataset
    with pytest.raises(ValueError,match='explicit'):
        UserPawDataset.load(None,'pbe')
    with pytest.raises(ValueError,match='file'):
        UserPawDataset.load(tmp_path,'pbe')
