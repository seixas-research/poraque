"""Validated kernel benchmarks for Python and MPI/OpenMP/ScaLAPACK.

The portable workload is deterministic, synthetic, complex Hermitian and uses
positive-definite projector overlaps. It exercises the Harris band sum and
fixed-density Hartree/LDA double counting, but is NOT a PAW material accuracy
benchmark or an ext2chg prediction. Use ``production_check`` for that separate
end-to-end regression with an explicit model, deck and licensed runtime file.
All energies are eV. MPI is initialized only in native workers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import signal
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any

import numpy as np
from scipy.linalg import eigh


STAGES = ('total', 'k_loop', 'fft', 'construction', 'diagonalization')


def assert_accuracy(reference: np.ndarray, actual: np.ndarray,
                    reference_energy: float, energy: float,
                    tolerance: float = 1e-8) -> tuple[float, float]:
    """Reject nonfinite values and absolute eigenvalue/total-energy errors.

    No relative tolerance: a large energy offset must not relax 1e-8 eV.
    Comparisons use total energy per cell, never energy per atom.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Tolerance must be positive and finite')
    if reference.shape != actual.shape or not reference.size:
        raise AssertionError('Eigenvalue shapes do not match')
    if not all(np.isfinite(x).all() for x in (reference, actual,
                                             np.asarray([reference_energy, energy]))):
        raise AssertionError('Nonfinite eigenspectrum or energy')
    eigen_error = float(np.max(np.abs(reference - actual)))
    energy_error = abs(reference_energy - energy)
    if eigen_error > tolerance or energy_error > tolerance:
        raise AssertionError(f'Accuracy gate failed: eigen={eigen_error:.3e} eV, '
                             f'Harris={energy_error:.3e} eV; tolerance={tolerance:.3e}')
    return eigen_error, energy_error


def symbol_providers(library: Any) -> dict[str, str | None]:
    """Record actual ELF/macOS symbol providers, detecting MKL FFTW shims.

    dladdr is optional on other platforms; library mappings are retained too.
    This avoids assuming that a requested CMake dependency won symbol lookup.
    """
    import ctypes as ct
    class DlInfo(ct.Structure):
        _fields_ = [('filename',ct.c_char_p),('base',ct.c_void_p),
                    ('symbol',ct.c_char_p),('address',ct.c_void_p)]
    result: dict[str,str | None] = {}
    try:
        dladdr = ct.CDLL(None).dladdr
    except AttributeError:
        return result
    dladdr.argtypes = [ct.c_void_p,ct.POINTER(DlInfo)]
    dladdr.restype = ct.c_int
    for name in ('fftw_execute','fftw_plan_dft_3d','fftw_init_threads','pzheevd_'):
        info = DlInfo()
        address = ct.cast(getattr(library,name),ct.c_void_p)
        result[name] = info.filename.decode() if dladdr(address,ct.byref(info)) else None
    return result


def make_problem(n: int, nk: int, projectors: int = 8,
                 grid_size: int | None = None) -> dict[str, Any]:
    """Prepare a reproducible density, plane-wave indices and dense projectors.

    Preparation/functional evaluation is untimed; the measured solve begins
    with the effective-potential FFT. G=0 Hartree uses neutral-background
    convention. Synthetic overlap/projector strengths are not VASP PAW data.
    """
    from poraque.fields import FieldGrid
    from poraque.physics.energy import hartree_potential
    from poraque.physics.harris import integrate, lda_terms

    if min(n, nk, projectors) < 1:
        raise ValueError('Basis size, k count and projector count must be positive')
    radius = math.ceil(n**(1/3))
    axes = np.arange(-radius, radius+1)
    indices = np.stack(np.meshgrid(axes, axes, axes, indexing='ij'), axis=-1).reshape(-1,3)
    order = np.argsort(np.sum(indices**2, axis=1), kind='stable')
    indices = indices[order[:n]]
    minimum = int(4*np.max(np.abs(indices))+2)
    side = max(16, minimum) if grid_size is None else grid_size
    if side < minimum:
        raise ValueError(f'FFT side must be at least {minimum} to avoid G-difference aliasing')
    grid = FieldGrid((side,)*3, np.eye(3)*10)
    xyz = np.indices(grid.shape)*2*np.pi/side
    electrons = float(max(2, 2*(n//4)))
    density = electrons/grid.volume*(1+0.12*np.cos(xyz[0])*np.cos(xyz[1])*np.cos(xyz[2]))
    vh = hartree_potential(density, grid)
    exc, vxc = lda_terms(density, grid)
    potential = vh+vxc+0.2*(np.cos(xyz[0])+np.cos(xyz[1])+np.cos(xyz[2]))
    # E_H - int(n V_H) = -E_H; all corrections use the SAME input density.
    correction = math.fsum([-0.5*integrate(density*vh, grid),
                           -integrate(density*vxc, grid), exc])
    rng = np.random.default_rng(72819+n)
    m = min(projectors,n)
    beta = (rng.normal(size=(n,m))+1j*rng.normal(size=(n,m)))/np.sqrt(n)
    d = rng.normal(size=(m,m))+1j*rng.normal(size=(m,m))
    d = (d+d.conj().T)*0.05
    overlap = np.eye(n,dtype=complex)+0.02*beta@beta.conj().T
    address = tuple(((indices[:,None,:]-indices[None,:,:]) % side).transpose(2,0,1))
    kpoints = np.zeros((nk,3))
    kpoints[:,0] = np.arange(nk)/nk-0.5
    kinetic = [3.80998211615486*np.sum(((indices+k)@grid.reciprocal_cell)**2, axis=1)
               for k in kpoints]
    return dict(n=n,nk=nk,potential=potential,correction=correction,beta=beta,
                strengths=d,overlap=overlap,address=address,kinetic=kinetic,
                electrons=electrons,grid_shape=grid.shape)


def assemble_python(local: np.ndarray, kinetic: np.ndarray,
                    beta: np.ndarray, strengths: np.ndarray) -> np.ndarray:
    """Vectorized reference H=Vlocal+diag(T)+B D B†, complex128."""
    return local+np.diag(kinetic)+beta@strengths@beta.conj().T


def harris_energy(values: np.ndarray, problem: dict[str, Any]) -> float:
    """Global zero-temperature, spin-degenerate k-weighted band sum + DC.

    Uniform k weights allow exact equal-weight filling over the entire mesh;
    individual k points are not incorrectly filled with equal electron counts.
    """
    levels = np.sort(values.ravel())
    occupancy = np.clip(problem['electrons']*problem['nk']-2*np.arange(levels.size), 0, 2)
    band = math.fsum((levels*occupancy/problem['nk']).tolist())
    return math.fsum([band, problem['correction']])


def _python_run(problem: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    start = perf_counter()
    mark = perf_counter()
    fourier = np.fft.fftn(problem['potential'])/problem['potential'].size
    fft_time = perf_counter()-mark
    timings = dict(fft=fft_time,construction=0.,diagonalization=0.)
    values = []
    loop = perf_counter()
    for ik in range(problem['nk']):
        mark = perf_counter()
        h = assemble_python(fourier[problem['address']], problem['kinetic'][ik],
                            problem['beta'], problem['strengths'])
        timings['construction'] += perf_counter()-mark
        mark = perf_counter()
        # Full eigenvectors in BOTH implementations, no eigvals-only shortcut.
        w, _ = eigh(h, problem['overlap'], driver='gvd', check_finite=True)
        timings['diagonalization'] += perf_counter()-mark
        values.append(w)
    timings['k_loop'] = perf_counter()-loop
    values = np.array(values)
    harris_energy(values, problem)
    timings['total'] = perf_counter()-start
    return values, timings


def run_worker(config: dict[str, Any]) -> dict[str, Any] | None:
    """One subprocess worker, serial SciPy or collective native k groups.

    Validation is completed before warmups and timed samples. Rank-local
    intervals are reduced with MAX to measure critical paths. Root convenience
    solves include copies, scatter/gather and wrapper validation overhead.
    """
    if config['backend']=='production':
        return production_check(Path(config['deck']),config['models'],Path(config['potcar']),
            config['library'],config['threads'],config['repeats'],config['warmups'],config['tolerance'],
            config['repair_density'],config['spin_average'],config['unpolarized_input'])
    native = config['backend']=='native'
    solver = None
    if native:
        from poraque.hpc.binding import ParallelEigensolver
        solver = ParallelEigensolver(config['library'],ranks_per_k=config['ranks_per_k'],
                                     threads=config['threads'],block=config['block'])
    problem = (make_problem(config['n'],config['nk'],config['projectors'],config['grid_size'])
               if solver is None or solver.rank==0 else
               dict(n=config['n'],nk=config['nk'],grid_shape=None))
    try:
        providers = symbol_providers(solver.library) if solver else {}
        if any('mkl' in (path or '').lower() for name,path in providers.items() if name.startswith('fftw_')):
            raise RuntimeError('MKL FFTW compatibility symbols shadow FFTW; put FFTW before MKL in the link order')
        # A serial reference is computed only on each group root, outside timings.
        reference = _python_run(problem)[0] if solver is None or solver.rank==0 else None
        errors = (0.,0.)
        fft_error = matrix_error = 0.
        if solver:
            spectra = np.zeros((problem['nk'],problem['n']))
            root = solver.rank==0
            fourier = solver.fft(problem['potential'])/problem['potential'].size if root else None
            if root:
                fft_error = float(np.max(np.abs(fourier-np.fft.fftn(problem['potential'])/problem['potential'].size)))
            for ik in range(solver.group,problem['nk'],solver.groups):
                h = None
                if root:
                    local = fourier[problem['address']]
                    h = solver.assemble(local,problem['kinetic'][ik],problem['beta'],problem['strengths'])
                    expected = assemble_python(local,problem['kinetic'][ik],problem['beta'],problem['strengths'])
                    matrix_error = max(matrix_error,float(np.max(np.abs(h-expected))))
                state = solver.solve(h,problem['overlap'] if root else None)
                if root:
                    spectra[ik] = state.eigenvalues
            # One owner per row; worker ranks contribute zero, avoiding duplicates.
            spectra = solver.reduce(spectra,sum=True)
            failed = 0.
            if root:
                try:
                    errors = assert_accuracy(reference,spectra,harris_energy(reference,problem),
                                             harris_energy(spectra,problem),config['tolerance'])
                    if max(fft_error,matrix_error)>config['tolerance']:
                        raise AssertionError('FFT/Hamiltonian construction accuracy gate failed')
                except AssertionError:
                    failed = 1.
            if solver.reduce(np.array([failed]))[0]:
                raise AssertionError('Collective FFT/matrix/eigenspectrum/Harris accuracy gate failed')
            validation = solver.reduce(np.array([*errors,fft_error,matrix_error]))
            errors = tuple(validation[:2])
            fft_error,matrix_error = validation[2:]
        samples = []
        internals = []
        for rep in range(config['warmups']+config['repeats']):
            if solver is None:
                values,timing = _python_run(problem)
            else:
                solver.profile(reset=True)
                solver.barrier()
                start = perf_counter()
                stage = dict(fft=0.,construction=0.,diagonalization=0.)
                spectra = np.zeros((problem['nk'],problem['n']))
                mark = perf_counter()
                fourier = solver.fft(problem['potential'])/problem['potential'].size if root else None
                stage['fft'] = perf_counter()-mark if root else 0.
                loop = perf_counter()
                for ik in range(solver.group,problem['nk'],solver.groups):
                    mark = perf_counter()
                    h = solver.assemble(fourier[problem['address']],problem['kinetic'][ik],
                                        problem['beta'],problem['strengths']) if root else None
                    stage['construction'] += perf_counter()-mark if root else 0.
                    mark = perf_counter()
                    state = solver.solve(h,problem['overlap'] if root else None,
                                         verify=False,diagnostics=False)
                    stage['diagonalization'] += perf_counter()-mark if root else 0.
                    if root:
                        spectra[ik] = state.eigenvalues
                stage['k_loop'] = perf_counter()-loop if root else 0.
                spectra = solver.reduce(spectra,sum=True)
                energy = harris_energy(spectra,problem) if root else 0.
                stage['total'] = perf_counter()-start
                timing = dict(zip(STAGES,solver.reduce(np.array([stage[k] for k in STAGES]))))
                profile = solver.profile()
                profile = dict(zip(profile,solver.reduce(np.array(list(profile.values())))))
                # Recheck spectra/energy after stopping timers, on every repetition.
                failed = 0.
                if root:
                    try:
                        assert_accuracy(reference,spectra,harris_energy(reference,problem),energy,config['tolerance'])
                    except AssertionError:
                        failed = 1.
                if solver.reduce(np.array([failed]))[0]:
                    raise AssertionError('Collective timed-repetition accuracy gate failed')
                if rep>=config['warmups']:
                    internals.append(profile)
            if rep>=config['warmups']:
                samples.append(timing)
        result = dict(backend=config['backend'],n=config['n'],nk=config['nk'],
                      grid_shape=problem['grid_shape'],samples=samples,internal_samples=internals,
                      eigen_error_ev=float(errors[0]),harris_error_ev=float(errors[1]),
                      fft_error=float(fft_error),matrix_error_ev=float(matrix_error),
                      environment={k:os.environ.get(k) for k in
                                   ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')})
        result['symbol_providers'] = providers
        try:
            from threadpoolctl import threadpool_info
            result['threadpools'] = threadpool_info()
        except ImportError:
            result['threadpools'] = 'threadpoolctl not installed'
        result['affinity'] = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
        result['loaded_libraries'] = sorted({line.split()[-1] for line in
            Path('/proc/self/maps').read_text().splitlines()
            if any(name in line.lower() for name in ('mkl','blas','lapack','fftw','libmpi','libgomp','libiomp'))}) if Path('/proc/self/maps').exists() else []
        return result if solver is None or (solver.group==0 and solver.rank==0) else None
    finally:
        if solver:
            solver.close()


def _launch(config: dict[str, Any], directory: Path, launcher: list[str],
            timeout: float) -> dict[str, Any]:
    config_path = directory/'config.json'
    output = directory/'worker.json'
    config_path.write_text(json.dumps(config))
    output.unlink(missing_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(config['threads']),
               MKL_NUM_THREADS=str(config['threads']),
               OPENBLAS_NUM_THREADS=str(config['threads']), BLIS_NUM_THREADS=str(config['threads']),
               VECLIB_MAXIMUM_THREADS=str(config['threads']), NUMEXPR_NUM_THREADS='1')
    env.pop('PORAQUE_DIAGONALIZER_LIBRARY',None)
    command = [sys.executable,'-m','poraque.benchmarks.performance','--worker',str(config_path),
               '--worker-output',str(output)]
    if config['backend']=='native':
        command = launcher+['-n',str(config['ranks'])]+command
    started = perf_counter()
    process = subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               text=True,start_new_session=True)
    try:
        stdout,stderr = process.communicate(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(f'Worker failed ({shlex.join(command)}):\n{stderr[-8000:]}')
    result = json.loads(output.read_text())
    result['process_seconds'] = perf_counter()-started
    result['command'] = command
    result['stderr'] = stderr[-4000:]
    return result


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    """Median/minimum/sample standard deviation, retaining raw replicates elsewhere."""
    if not samples:
        raise ValueError('At least one timed sample is required')
    result = {}
    for stage in STAGES:
        values = np.array([s[stage] for s in samples])
        if not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError(f'Invalid timing samples for {stage}')
        result.update({stage+'_s':float(np.median(values)),stage+'_min_s':float(np.min(values)),
                       stage+'_std_s':float(np.std(values,ddof=1)) if len(values)>1 else 0.})
    return result


def add_efficiencies(rows: list[dict[str, Any]]) -> None:
    """Strong: T_C(1,1)/(P*T_C(P,t)); weak: fixed work per MPI rank, fixed threads.

    Missing reference configurations produce empty CSV cells, never invented
    efficiencies. Core count is requested ranks*threads, not detected busy cores.
    """
    for row in rows:
        row['native_speedup'] = row['parallel_efficiency'] = None
        row['weak_efficiency'] = None
        candidates = [r for r in rows if r['mode']==row['mode'] and r['n']==row['n']
                      and r['python_threads']==row['python_threads'] and r['ranks']==1
                      and r['ranks_per_k']==1]
        if row['mode']=='strong':
            candidates = [r for r in candidates if r['nk']==row['nk'] and r['threads']==1]
            if candidates:
                speedup = candidates[0]['c_total_s']/row['c_total_s']
                row['native_speedup'] = speedup
                row['parallel_efficiency'] = speedup/(row['ranks']*row['threads'])
        else:
            candidates = [r for r in candidates if r['threads']==row['threads']
                          and r['nk']*row['ranks']==row['nk']]
            if candidates:
                row['weak_efficiency'] = candidates[0]['c_total_s']/row['c_total_s']


def production_check(deck: Path, models: str, potcar: Path, library: str,
                     threads: int, repeats: int, warmups: int, tolerance: float,
                     repair_density: bool = False, spin_average: bool = False,
                     unpolarized_input: bool = False) -> dict[str, Any]:
    """Separate serial full ext2chg/PAW regression; no synthetic-density fallback.

    Initialization includes inference. Warm timings repeat total_energy on the
    same predicted density; cold initialization and per-evaluation times are
    reported separately. Native path offloads diagonalization only, consistent
    with the current production runtime adapter. No licensed data is serialized.
    """
    from poraque.ml.bloch import VaspBlochHarris
    from poraque.ml.training import resolve_bundle_path
    from poraque.hpc import runtime

    previous = os.environ.pop('PORAQUE_DIAGONALIZER_LIBRARY',None)
    previous_threads = os.environ.get('PORAQUE_HPC_THREADS')
    if runtime._solver is not None:
        raise RuntimeError('Run production check in a fresh single-process interpreter')
    try:
        start = perf_counter()
        pipeline = VaspBlochHarris(deck,models,potcar,device='cpu',repair_density=repair_density,
                                  spin_average=spin_average,unpolarized_input=unpolarized_input)
        initialization = perf_counter()-start
        if pipeline.kpoints.weights is None:
            raise ValueError('Production energy regression needs a k mesh, not a line path')
        reference = pipeline.total_energy()
        os.environ['PORAQUE_DIAGONALIZER_LIBRARY'] = library
        os.environ['PORAQUE_HPC_THREADS'] = str(threads)
        actual = pipeline.total_energy()
        errors = assert_accuracy(reference.eigenvalues,actual.eigenvalues,reference.total,actual.total,tolerance)
        assert_accuracy(reference.eigenvalues,actual.eigenvalues,reference.free_energy,actual.free_energy,tolerance)
        samples = {'python':[], 'native':[]}
        stage_samples = {'python':[], 'native':[]}
        profiles = []
        from .hooks import production_profile
        for rep in range(warmups+repeats):
            # Alternate order to reduce systematic thermal/cache bias.
            for backend in (('python','native') if rep%2==0 else ('native','python')):
                if backend=='native':
                    os.environ['PORAQUE_DIAGONALIZER_LIBRARY'] = library
                    runtime._solver.profile(reset=True)
                else:
                    os.environ.pop('PORAQUE_DIAGONALIZER_LIBRARY',None)
                start = perf_counter()
                with production_profile() as stages:
                    result = pipeline.total_energy()
                elapsed = perf_counter()-start
                assert_accuracy(reference.eigenvalues,result.eigenvalues,reference.total,result.total,tolerance)
                if rep>=warmups:
                    samples[backend].append(elapsed)
                    stage_samples[backend].append(stages)
                    if backend=='native':
                        profiles.append(runtime._solver.profile())
        return dict(kind='ext2chg_full_PAW_single_process',initialization_s=initialization,
                    eigen_error_ev=errors[0],harris_error_ev=errors[1],
                    harris_python_ev=reference.total,harris_native_ev=actual.total,
                    samples=samples,stage_samples=stage_samples,native_internal_samples=profiles,
                    speedup=float(np.median(samples['python'])/np.median(samples['native'])),
                    density_diagnostics=pipeline.density_diagnostics,
                    deck=str(deck.resolve()),models=models,
                    model_sha256=hashlib.sha256(Path(resolve_bundle_path(models)).read_bytes()).hexdigest(),
                    environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                    potentials={e:d.sha256 for e,d in pipeline.datasets.items()},
                    note='Production assembly remains Python; native full-spectrum solve vs production subset solve')
    finally:
        if runtime._solver is not None:
            runtime._solver.close()
            runtime._solver = None
        os.environ.pop('PORAQUE_DIAGONALIZER_LIBRARY',None)
        if previous:
            os.environ['PORAQUE_DIAGONALIZER_LIBRARY'] = previous
        if previous_threads is None:
            os.environ.pop('PORAQUE_HPC_THREADS',None)
        else:
            os.environ['PORAQUE_HPC_THREADS'] = previous_threads


def main() -> None:
    """Orchestrate independent workers; write no CSV until every accuracy gate passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--worker-output',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--sizes',type=int,nargs='+',default=[64,128,256])
    parser.add_argument('--ranks',type=int,nargs='+',default=[1,2,4])
    parser.add_argument('--threads',type=int,nargs='+',default=[1])
    parser.add_argument('--ranks-per-k',type=int,nargs='+',default=[1],
                        help='K groups have this many ScaLAPACK ranks; must divide ranks')
    parser.add_argument('--nk',type=int,default=4)
    parser.add_argument('--mode',choices=['strong','weak','both'],default='both')
    parser.add_argument('--projectors',type=int,default=8)
    parser.add_argument('--grid-size',type=int)
    parser.add_argument('--block',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--warmups',type=int,default=1)
    parser.add_argument('--tolerance',type=float,default=1e-8)
    parser.add_argument('--python-threads',type=int,default=1,
                        help='Controlled BLAS baseline threads, independent of native threads')
    parser.add_argument('--launcher',default='mpiexec --bind-to none',help='Launcher and flags, without rank count')
    parser.add_argument('--timeout',type=float,default=600)
    parser.add_argument('--output',type=Path,default=Path('hpc_performance.csv'))
    parser.add_argument('--production-deck',type=Path)
    parser.add_argument('--models')
    parser.add_argument('--potcar',type=Path,help='Explicit user runtime file; valid VASP license required')
    parser.add_argument('--repair-density',action='store_true')
    parser.add_argument('--spin-average',action='store_true')
    parser.add_argument('--unpolarized-input',action='store_true')
    parser.add_argument('--production-only',action='store_true')
    args = parser.parse_args()
    if args.worker:
        result = run_worker(json.loads(args.worker.read_text()))
        if result is not None:
            args.worker_output.write_text(json.dumps(result,allow_nan=False))
        return
    if not args.library or not args.library.is_file():
        parser.error('--library must name a built backend')
    if (min(*args.sizes,*args.ranks,*args.threads,*args.ranks_per_k,args.nk,args.projectors,
            args.block,args.repeats,args.python_threads)<1 or args.warmups<0
            or not np.isfinite(args.tolerance) or args.tolerance<=0):
        parser.error('Counts must be positive; warmups >= 0 and tolerance > 0')
    if args.production_only and not args.production_deck:
        parser.error('--production-only requires --production-deck')
    if args.production_deck and not (args.models and args.potcar):
        parser.error('Production regression requires --models and --potcar')
    args.library = args.library.resolve()
    metadata: dict[str, Any] = dict(schema_version=1,platform=platform.platform(),
        python=sys.version,numpy=np.__version__,cpu_count=os.cpu_count(),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                        if line.startswith('model name')),None) if Path('/proc/cpuinfo').exists() else None,
        library=str(args.library),library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},runs=[])
    rows = []
    base = dict(library=str(args.library),projectors=args.projectors,grid_size=args.grid_size,
                block=args.block,repeats=args.repeats,warmups=args.warmups,tolerance=args.tolerance)
    modes = ['strong','weak'] if args.mode=='both' else [args.mode]
    with tempfile.TemporaryDirectory(prefix='poraque-perf-') as temp:
        directory = Path(temp)
        cache = {}
        if not args.production_only:
            for mode in modes:
                for n in args.sizes:
                    for ranks in args.ranks:
                        nk = args.nk if mode=='strong' else args.nk*ranks
                        key = (n,nk)
                        if key not in cache:
                            config = dict(base,backend='python',n=n,nk=nk,ranks=1,
                                          threads=args.python_threads,ranks_per_k=1)
                            cache[key] = _launch(config,directory,shlex.split(args.launcher),args.timeout)
                            metadata['runs'].append(dict(config=config,result=cache[key]))
                        py = summarize_samples(cache[key]['samples'])
                        for threads in args.threads:
                            for rpk in args.ranks_per_k:
                                if ranks%rpk:
                                    continue
                                config = dict(base,backend='native',n=n,nk=nk,ranks=ranks,
                                              threads=threads,ranks_per_k=rpk)
                                result = _launch(config,directory,shlex.split(args.launcher),args.timeout)
                                metadata['runs'].append(dict(config=config,result=result))
                                native = summarize_samples(result['samples'])
                                row = dict(mode=mode,n=n,nk=nk,ranks=ranks,threads=threads,
                                           ranks_per_k=rpk,python_threads=args.python_threads,
                                           eigen_error_ev=result['eigen_error_ev'],harris_error_ev=result['harris_error_ev'])
                                for stage in STAGES:
                                    for suffix in ('_s','_min_s','_std_s'):
                                        row['python_'+stage+suffix] = py[stage+suffix]
                                        row['c_'+stage+suffix] = native[stage+suffix]
                                    row[stage+'_speedup'] = py[stage+'_s']/native[stage+'_s']
                                fraction = py['diagonalization_s']/py['total_s']
                                row['python_diagonalization_fraction'] = fraction
                                row['amdahl_diagonalization_only_limit'] = 1/max(1-fraction,np.finfo(float).eps)
                                for name in result['internal_samples'][0]:
                                    row['c_internal_'+name+'_s'] = float(np.median([p[name] for p in result['internal_samples']]))
                                rows.append(row)
                                print(f'{mode:6s} N={n:5d} k={nk:3d} ranks={ranks} threads={threads} '
                                      f'rpk={rpk} Python={py["total_s"]:.6f}s C={native["total_s"]:.6f}s '
                                      f'S={row["total_speedup"]:.3f} eigen={row["eigen_error_ev"]:.2e} '
                                      f'energy={row["harris_error_ev"]:.2e}',flush=True)
    if args.production_deck:
        with tempfile.TemporaryDirectory(prefix='poraque-production-perf-') as temp:
            config = dict(base,backend='production',deck=str(args.production_deck.resolve()),
                models=args.models,potcar=str(args.potcar.resolve()),threads=args.threads[0],ranks=1,
                repair_density=args.repair_density,spin_average=args.spin_average,
                unpolarized_input=args.unpolarized_input)
            metadata['production'] = _launch(config,Path(temp),[],args.timeout)
        print('Full ext2chg/PAW speedup:',metadata['production']['speedup'],flush=True)
    add_efficiencies(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with args.output.open('w',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
