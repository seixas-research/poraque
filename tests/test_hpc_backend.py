"""Opt-in MPI test: provide a built library; no development libraries required in CI."""
from pathlib import Path
import json
import os
import subprocess
import sys
import pytest


def test_hybrid_mpi_backend():
    library=os.environ.get('PORAQUE_HPC_TEST_LIBRARY')
    if not library:
        pytest.skip('Set PORAQUE_HPC_TEST_LIBRARY to run real MPI/ScaLAPACK/FFTW tests')
    worker=Path(__file__).with_name('hpc_worker.py')
    completed=subprocess.run(['mpiexec','-n','4',sys.executable,str(worker),library],
        check=True,capture_output=True,text=True,timeout=90,
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
    rows=[json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
    assert len(rows)==2
    assert {row['group'] for row in rows}=={0,1}
    assert all(row['eigen_error']<1e-10 and row['fft_error']<1e-10 for row in rows)


def test_validated_profile_report(tmp_path):
    """Exercise k pools and distributed solves, then check reported scopes/errors."""
    library=os.environ.get('PORAQUE_HPC_TEST_LIBRARY')
    if not library:
        pytest.skip('Set PORAQUE_HPC_TEST_LIBRARY for the real native profile integration test')
    import csv
    output=tmp_path/'profile.csv'
    subprocess.run([sys.executable,'-m','poraque.benchmarks.performance',
        '--library',library,'--sizes','13','--ranks','1','2',
        '--ranks-per-k','1','2','--nk','2','--threads','1',
        '--mode','both','--warmups','0','--repeats','1','--output',str(output)],
        check=True,capture_output=True,text=True,timeout=120)
    with output.open() as handle:
        rows=list(csv.DictReader(handle))
    assert len(rows)==6
    assert all(float(r['eigen_error_ev'])<1e-8 and float(r['harris_error_ev'])<1e-8 for r in rows)
    assert all(float(r['c_internal_solve_root_s'])>=float(r['c_internal_pzheevd_s'])>0 for r in rows)
    assert all(float(r['total_speedup'])==pytest.approx(
        float(r['python_total_s'])/float(r['c_total_s'])) for r in rows)
    assert all(r['parallel_efficiency'] for r in rows if r['mode']=='strong')
    assert len(json.loads(output.with_suffix('.json').read_text())['runs'])==8
