"""Scoped, opt-in Python timing hooks for the production PAW workflow.

Hooks are process-global while active: use only in an otherwise idle benchmark
worker. They restore original functions even on errors. There is no profiling
overhead in ordinary solver runs. Inclusive stages may overlap and must not be
summed (the k-point loop includes construction and diagonalization).
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import wraps
from time import perf_counter
from typing import Callable, Iterator
from unittest.mock import patch


def _timed(function: Callable, name: str, counters: dict[str, float]) -> Callable:
    """Accumulate elapsed wall time, including allocation and dispatch overhead."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        start = perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            counters[name] = counters.get(name,0.)+(perf_counter()-start)
    return wrapped


@contextmanager
def production_profile() -> Iterator[dict[str, float]]:
    """Time forward/inverse FFTs, H/S construction, solve APIs and k-point loop.

    Forward FFT totals include Hartree/XC/local-potential transforms; inverse
    transforms include output diagnostics. PyTorch's internal GGA FFTs are not
    intercepted. Native solve timing includes its residual checks in production.
    """
    import numpy as np
    from poraque.physics import bloch, magnetic_bloch, paw_bloch
    from poraque.hpc.binding import ParallelEigensolver

    counters: dict[str,float] = {}
    targets = [(np.fft,'fftn','fft'),(np.fft,'ifftn','inverse_fft'),
               (bloch.BlochHamiltonian,'matrix','construction'),
               (magnetic_bloch.SpinorHamiltonian,'matrix','construction'),
               (paw_bloch.PawNonlocalPotential,'overlap','overlap_construction'),
               (bloch,'eigh','diagonalization'),
               (magnetic_bloch,'eigh','diagonalization'),
               (ParallelEigensolver,'solve','diagonalization'),
               (paw_bloch,'solve_mesh','k_loop'),
               (paw_bloch.PawHarrisEvaluator,'_prepare','input_terms')]
    with ExitStack() as stack:
        for owner,attribute,name in targets:
            original = getattr(owner,attribute)
            stack.enter_context(patch.object(owner,attribute,_timed(original,name,counters)))
        yield counters
