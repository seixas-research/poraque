"""Accuracy gates and scaling arithmetic must reject misleading benchmark claims."""
import numpy as np
import pytest
from poraque.benchmarks.performance import (assert_accuracy, add_efficiencies,
    harris_energy, make_problem, summarize_samples, STAGES)


def test_absolute_energy_gate_even_with_large_offset():
    values = np.array([[1.,2.,3.]])
    assert_accuracy(values,values+1e-10,1e6,1e6+1e-9)
    with pytest.raises(AssertionError,match='Accuracy gate'):
        assert_accuracy(values,values,1e6,1e6+1e-7)
    with pytest.raises(AssertionError,match='Accuracy gate'):
        assert_accuracy(values,values+2e-8,0.,0.)
    with pytest.raises(AssertionError,match='Nonfinite'):
        assert_accuracy(values,values*np.nan,0.,0.)


def test_global_k_filling_and_harris_correction():
    problem = dict(electrons=2.,nk=2,correction=-3.)
    # Both lowest occupied states happen to be at the same k; equal local
    # filling per k would give the incorrect band energy 10 instead of 1.
    values = np.array([[0.,1.],[10.,11.]])
    assert harris_energy(values,problem)==-2.


def test_scaling_needs_matching_native_reference():
    rows = [dict(mode='strong',n=16,nk=4,python_threads=1,ranks=1,
                 threads=1,ranks_per_k=1,c_total_s=8.),
            dict(mode='strong',n=16,nk=4,python_threads=1,ranks=2,
                 threads=2,ranks_per_k=1,c_total_s=3.),
            dict(mode='weak',n=16,nk=4,python_threads=1,ranks=1,
                 threads=2,ranks_per_k=1,c_total_s=8.),
            dict(mode='weak',n=16,nk=16,python_threads=1,ranks=4,
                 threads=2,ranks_per_k=1,c_total_s=10.),
            dict(mode='strong',n=32,nk=4,python_threads=1,ranks=2,
                 threads=1,ranks_per_k=1,c_total_s=3.)]
    add_efficiencies(rows)
    assert rows[1]['parallel_efficiency']==pytest.approx(2/3)
    assert rows[3]['weak_efficiency']==.8
    assert rows[4]['parallel_efficiency'] is None


def test_statistics_retain_variance_and_reject_bad_samples():
    stats = summarize_samples([{s:1. for s in STAGES},{s:3. for s in STAGES}])
    assert stats['total_s']==2.
    assert stats['total_std_s']==pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError):
        summarize_samples([{s:np.nan for s in STAGES}])


def test_reproducible_positive_overlap_and_no_fft_aliasing():
    p = make_problem(7,3,projectors=4)
    assert np.linalg.eigvalsh(p['overlap']).min()>0
    assert p['electrons']<=2*p['n']
    assert np.array_equal(p['beta'],make_problem(7,3,projectors=4)['beta'])
    with pytest.raises(ValueError,match='aliasing'):
        make_problem(64,2,grid_size=2)


def test_scoped_production_hooks_restore_after_failure():
    from poraque.benchmarks.hooks import production_profile
    original = np.fft.fftn
    with pytest.raises(RuntimeError):
        with production_profile() as counters:
            np.fft.fftn(np.ones((2,2,2)))
            assert counters['fft']>0
            raise RuntimeError('intentional')
    assert np.fft.fftn is original
