"""MPI subprocess validation using generated matrices, never potential data."""
import argparse
import json
import numpy as np
from scipy.linalg import eigh
from poraque.hpc import ParallelEigensolver

parser=argparse.ArgumentParser()
parser.add_argument('library')
parser.add_argument('--ranks-per-k',type=int,default=2)
args=parser.parse_args()
with ParallelEigensolver(args.library,ranks_per_k=args.ranks_per_k,block=2,threads=2) as solver:
    rng=np.random.default_rng(38+solver.group)
    worst_eigen=worst_fft=worst_residual=0.
    for n in [1,3,13]:
        h=s=None
        if solver.rank==0:
            x=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
            h=x+x.conj().T
            x=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
            s=x.conj().T@x+np.eye(n)
            original=h.copy()
        result=solver.solve(h,s)
        if solver.rank==0:
            assert np.array_equal(h,original)
            error=float(np.max(np.abs(result.eigenvalues-eigh(h,s,eigvals_only=True))))
            worst_eigen=max(worst_eigen,error)
            worst_residual=max(worst_residual,result.max_residual)
            assert error<1e-10
    for s in [None, np.diag([-1.,1.,1.])]:
        h=np.eye(3) if solver.rank==0 else None
        overlap=s if solver.rank==0 else None
        if s is None:
            result=solver.solve(h)
            if solver.rank==0:
                assert np.allclose(result.eigenvalues,1)
        else:
            try:
                solver.solve(h,overlap)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Indefinite overlap must fail collectively')
    # Invalid root shape also propagates without a deadlock.
    try:
        solver.solve(np.ones((2,3)) if solver.rank==0 else None)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid root shape must fail collectively')
    # Arbitrary complex projector couplings exercise the same matrix algebra
    # as off-diagonal spinor/SOC blocks, without proprietary potential data.
    if solver.rank==0:
        local=np.eye(9,dtype=complex)
        kinetic=np.arange(9,dtype=float)
        beta=rng.normal(size=(9,4))+1j*rng.normal(size=(9,4))
        d=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
        d=d+d.conj().T
        built=solver.assemble(local,kinetic,beta,d)
        assert np.allclose(built,local+np.diag(kinetic)+beta@d@beta.conj().T,atol=1e-12)
        # m=0 must support purely local potentials without null-sized crashes.
        assert np.allclose(solver.assemble(local,kinetic,beta[:,:0],d[:0,:0]),local+np.diag(kinetic))
    profile=solver.profile(reset=True)
    assert profile['solve_root']>=profile['solve_local']>=profile['pzheevd']>0
    assert profile['assemble']>0
    assert all(value==0 for value in solver.profile().values())
    # Original-communicator collectives span independent k groups, exactly
    # one root contributes each group's data.
    total=solver.reduce(np.array([1. if solver.rank==0 else 0.]),sum=True)
    assert total[0]==solver.groups
    solver.barrier()
    if solver.rank==0:
        values=rng.normal(size=(5,4,3))+1j*rng.normal(size=(5,4,3))
        transform=solver.fft(values)
        worst_fft=float(np.max(np.abs(transform-np.fft.fftn(values))))
        assert np.allclose(solver.fft(transform,True),values,atol=1e-12)
        print(json.dumps({'group':solver.group,'groups':solver.groups,'size':solver.size,
            'eigen_error':worst_eigen,'eigen_residual':worst_residual,'fft_error':worst_fft}),flush=True)
