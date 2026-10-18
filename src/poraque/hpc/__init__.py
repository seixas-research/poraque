"""Optional explicit MPI/ScaLAPACK backend; importing does not initialize MPI."""
from .binding import ParallelEigensolver, EigenResult

__all__ = ['ParallelEigensolver', 'EigenResult']
