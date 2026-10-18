"""ctypes interface to the optional LP64 MPI/ScaLAPACK/FFTW library.

Construct and call on the MPI main thread, collectively on all communicator
ranks. No mpi4py dependency is required; an externally owned communicator can
be supplied as its Fortran handle (comm.py2f()). MPI is finalized only if this
library initialized it, at explicit shutdown or normal interpreter exit after
all contexts have been closed. MPI worker ranks may pass None for root inputs.
"""
from __future__ import annotations

import atexit
import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np

_REGISTERED = set()


def _pointer(array: np.ndarray | None) -> ct.c_void_p:
    return ct.c_void_p() if array is None else ct.c_void_p(array.ctypes.data)


@dataclass(frozen=True)
class EigenResult:
    """Lowest requested eigenpairs on a k-group root, with sampled diagnostics."""

    eigenvalues: np.ndarray
    vectors: np.ndarray
    max_residual: float
    overlap_error: float


class ParallelEigensolver:
    """Independent MPI k groups, each with a near-square BLACS rank grid.

    The root convenience API scatters a Python-built dense matrix and gathers
    eigenvectors, so root memory is O(N²). Advanced C callers can bypass that
    staging with pq_local_shape/pq_solve_local and distributed matrix assembly.
    ScaLAPACK pzheevd computes the full spectrum; nbands slices its result and
    does not reduce its O(N³) work. Generalized PAW Hc=epsilon Sc is supported.
    """

    def __init__(self, library: str | Path, ranks_per_k: int = 1, block: int = 32,
                 threads: int = 1, communicator_f: int = -1) -> None:
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('MPI backend must be created on the main thread')
        if min(ranks_per_k, block, threads)<1:
            raise ValueError('Ranks, block size and threads must be positive')
        self.library = ct.CDLL(str(Path(library).expanduser().resolve(strict=True)))
        self.library.pq_error_string.argtypes = [ct.c_int]
        self.library.pq_error_string.restype = ct.c_char_p
        self.library.pq_context_create.argtypes = [ct.c_int]*4+[ct.POINTER(ct.c_void_p)]
        self.library.pq_context_info.argtypes = [ct.c_void_p]+[ct.POINTER(ct.c_int)]*4
        self.library.pq_consensus.argtypes = [ct.c_void_p,ct.c_int]
        self.library.pq_solve_root.argtypes = [ct.c_void_p,ct.c_int]+[ct.c_void_p]*4
        self.library.pq_fft3.argtypes = [ct.c_void_p]+[ct.c_int]*3+[ct.c_void_p]*2+[ct.c_int]
        self.library.pq_profile.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_int]
        self.library.pq_reduce_base.argtypes = [ct.c_void_p,ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int]
        self.library.pq_barrier_base.argtypes = [ct.c_void_p]
        self.library.pq_assemble.argtypes = [ct.c_void_p,ct.c_int,ct.c_int]+[ct.c_void_p]*5
        self.library.pq_context_destroy.argtypes = [ct.c_void_p]
        self.library.pq_shutdown.argtypes = []
        self.context = ct.c_void_p()
        self._check(self.library.pq_context_create(communicator_f,ranks_per_k,block,threads,ct.byref(self.context)))
        info = [ct.c_int() for _ in range(4)]
        self._check(self.library.pq_context_info(self.context,*[ct.byref(value) for value in info]))
        self.group, self.groups, self.rank, self.size = [value.value for value in info]
        path=str(Path(library).resolve())
        if path not in _REGISTERED:
            atexit.register(self.library.pq_shutdown)
            _REGISTERED.add(path)

    def _check(self, status: int) -> None:
        if status:
            raise RuntimeError(self.library.pq_error_string(status).decode())

    def _main(self) -> None:
        if not self.context or threading.current_thread() is not threading.main_thread():
            raise RuntimeError('Use an open MPI context from its main thread')

    def solve(self, hamiltonian: np.ndarray | None, overlap: np.ndarray | None = None,
              nbands: int | None = None, verify: bool = True,
              diagnostics: bool = True) -> EigenResult | None:
        """Collective complex Hermitian solve; only group root supplies matrices.

        NumPy inputs are never modified. Positive-definiteness of the PAW overlap
        is checked by distributed Cholesky; failures propagate collectively.
        Residual checks sample up to eight low and four high requested states.
        ``diagnostics=False, verify=False`` skips residual products for timed
        repetitions only, after a separate validated solve of the same input.
        """
        self._main()
        if verify and not diagnostics:
            raise ValueError('Verification requires diagnostics')
        h=s=w=z=None
        error=None
        n=0
        if self.rank==0:
            try:
                h=np.array(hamiltonian,dtype=np.complex128,order='F',copy=True)
                if h.ndim!=2 or h.shape[0]!=h.shape[1] or not len(h) or not np.isfinite(h).all():
                    raise ValueError('Require a finite square Hamiltonian')
                n=len(h)
                if not np.allclose(h,h.conj().T,rtol=1e-12,atol=1e-10):
                    raise ValueError('Hamiltonian is not Hermitian')
                if overlap is not None:
                    s=np.array(overlap,dtype=np.complex128,order='F',copy=True)
                    if s.shape!=h.shape or not np.isfinite(s).all() or not np.allclose(s,s.conj().T,rtol=1e-12,atol=1e-10):
                        raise ValueError('Overlap must be finite, Hermitian and match H')
                if nbands is not None and (int(nbands)!=nbands or not 1<=nbands<=n):
                    raise ValueError('Invalid requested band count')
                w=np.empty(n,dtype=np.float64)
                z=np.empty((n,n),dtype=np.complex128,order='F')
            except Exception as caught:
                error=caught
        status=self.library.pq_consensus(self.context,1 if error else 0)
        if status:
            raise ValueError(str(error) if error else 'Invalid root matrix input')
        self._check(self.library.pq_solve_root(self.context,n,_pointer(h),_pointer(s),_pointer(w),_pointer(z)))
        result=None
        error=None
        if self.rank==0:
            try:
                count=n if nbands is None else nbands
                w,z=w[:count].copy(),z[:,:count].copy()
                residual=orthogonality=float("nan")
                sample=np.unique(np.r_[np.arange(min(8,count)),np.arange(max(0,count-4),count)])
                if diagnostics:
                    v=z[:,sample]
                    sv=v if s is None else s@v
                    residual=float(np.max(np.linalg.norm(h@v-sv*w[sample],axis=0)))
                    orthogonality=float(np.max(np.abs(v.conj().T@sv-np.eye(len(sample)))))
                if verify and (not np.isfinite(w).all() or residual>1e-7 or orthogonality>1e-8):
                    raise RuntimeError(f'Eigenpair check failed: residual={residual:g}, overlap={orthogonality:g}')
                result=EigenResult(w,z,residual,orthogonality)
            except Exception as caught:
                error=caught
        status=self.library.pq_consensus(self.context,1 if error else 0)
        if status:
            raise RuntimeError(str(error) if error else 'Root eigenpair verification failed')
        return result

    def fft(self, values: np.ndarray, inverse: bool = False) -> np.ndarray:
        """Threaded local 3D FFTW, NumPy-compatible inverse normalization.

        This function is local, not collective. FFTW MPI slab transforms are
        a separate optional C interface; they are not silently emulated here.
        """
        self._main()
        values=np.array(values,dtype=np.complex128,order='C',copy=True)
        if values.ndim!=3 or not np.isfinite(values).all():
            raise ValueError('FFTW requires a finite 3D complex array')
        output=np.empty_like(values)
        self._check(self.library.pq_fft3(self.context,*values.shape,_pointer(values),_pointer(output),int(inverse)))
        return output

    def profile(self, reset: bool = False) -> dict[str, float]:
        """Group-collective MAX of cumulative rank-local MPI_Wtime intervals.

        Slots are inclusive: solve_local contains pzheevd; solve_root contains
        solve_local. Reset after untimed validation/warmup and before samples.
        """
        self._main()
        values = np.empty(8, dtype=np.float64)
        self._check(self.library.pq_profile(self.context, _pointer(values), int(reset)))
        names = ('solve_root', 'solve_local', 'pzheevd', 'fft_total',
                 'fft_plan', 'fft_execute', 'assemble', 'reserved')
        return dict(zip(names, map(float, values)))

    def reduce(self, values: np.ndarray, sum: bool = False) -> np.ndarray:
        """Allreduce MAX (or SUM) across the original communicator, all k groups."""
        self._main()
        values = np.ascontiguousarray(values, dtype=np.float64)
        if not values.size or values.size > np.iinfo(np.int32).max:
            raise ValueError('Invalid reduction size')
        output = np.empty_like(values)
        self._check(self.library.pq_reduce_base(self.context, _pointer(values),
                    _pointer(output), values.size, int(sum)))
        return output

    def barrier(self) -> None:
        """Barrier across the original communicator, including all k groups."""
        self._main()
        self._check(self.library.pq_barrier_base(self.context))

    def assemble(self, local: np.ndarray, kinetic: np.ndarray,
                 beta: np.ndarray, strengths: np.ndarray) -> np.ndarray:
        """Root-local OpenMP kernel H=Vlocal+diag(T)+B D B†.

        Arbitrary Hermitian projector couplings are supported. Projector
        generation, Fourier lookup and PAW one-center physics stay in Python.
        Input conversion/output allocation are included in wrapper wall time.
        """
        self._main()
        local = np.asfortranarray(local, dtype=np.complex128)
        kinetic = np.ascontiguousarray(kinetic, dtype=np.float64)
        beta = np.ascontiguousarray(beta, dtype=np.complex128)
        strengths = np.ascontiguousarray(strengths, dtype=np.complex128)
        n = len(kinetic)
        if (kinetic.ndim != 1 or local.shape != (n,n) or beta.ndim != 2
                or beta.shape[0] != n or strengths.shape != (beta.shape[1],)*2
                or not all(np.isfinite(a).all() for a in (local,kinetic,beta,strengths))):
            raise ValueError('Invalid local/projector matrix dimensions or values')
        output = np.empty((n,n), dtype=np.complex128, order='F')
        self._check(self.library.pq_assemble(self.context,n,beta.shape[1],
                    _pointer(local),_pointer(kinetic),_pointer(beta),
                    _pointer(strengths),_pointer(output)))
        return output

    def close(self) -> None:
        """Collectively release BLACS and group communicator; never implicit in __del__."""
        if self.context:
            self._main()
            self._check(self.library.pq_context_destroy(self.context))
            self.context=ct.c_void_p()

    def __enter__(self) -> 'ParallelEigensolver':
        return self

    def __exit__(self,*args) -> None:
        self.close()
