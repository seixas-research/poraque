"""Explicit single-process opt-in for the existing Python solver loop.

Set PORAQUE_DIAGONALIZER_LIBRARY to a built library path. Distributed jobs use
ParallelEigensolver directly, assigning k tasks by group/groups; this adapter
refuses multi-rank launches rather than repeating the entire mesh on every rank.
"""
import atexit
import os

from .binding import ParallelEigensolver, EigenResult

_solver = None


def try_solve(matrix, overlap, nbands: int) -> EigenResult | None:
    """Use the optional native backend, or return None when it was not requested."""
    global _solver
    library = os.environ.get('PORAQUE_DIAGONALIZER_LIBRARY')
    if not library:
        return None
    if _solver is None:
        _solver = ParallelEigensolver(library,threads=int(os.environ.get('PORAQUE_HPC_THREADS','1')))
        if _solver.size!=1 or _solver.groups!=1:
            _solver.close()
            _solver=None
            raise RuntimeError('Automatic solver adapter is single-process; use the collective MPI k-group API')
        atexit.register(_solver.close)
    return _solver.solve(matrix,overlap,nbands)
