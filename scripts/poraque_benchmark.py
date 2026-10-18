"""Benchmark ext2chg/PAW Harris against a user-licensed local VASP 6.6.1 binary.

POTCAR is proprietary. You must hold a valid VASP license. Supply external
POTCAR files explicitly in the manifest. The runner never generates, copies,
bundles, prints or redistributes potential contents; it creates only a private
runtime symlink. Keep VASP runtime outputs private and outside source checkouts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
from time import perf_counter
import uuid
import tempfile

import numpy as np

from poraque.fields import FieldGrid, ChargeDensity, SpinDensity
from poraque.fields.vasp import Incar, Poscar
from poraque.fields.vasp.augmentation import parse_augmentation
from poraque.fields.vasp.volumetric import read_augmentation_blocks
from poraque.physics.bloch import ElectronicSettings
from poraque.physics.density_repair import inspect_density, repair_density
from poraque.physics.harris import integrate
from poraque.physics.paw_bloch import PawHarrisInput
from poraque.physics.spin_xc import SpinDensityInput
from poraque.ml.bloch import VaspBlochHarris
from poraque.benchmarks.analysis import read_vasp_reference, energy_difference, summarize, stationarity_fit

DEFAULT_VASP = '/home/leseixas/Codes/vasp/6.6.1/bin/vasp_std'


def _external_work_root(path: Path) -> Path:
    """Prevent proprietary VASP runtime output from being generated in a checkout."""
    path = path.expanduser().resolve()
    # Some execution environments put orchestration git metadata directly in
    # the system temporary root. A private child there is not a source tree.
    temporary_root=Path(tempfile.gettempdir()).resolve()
    if any((parent/'.git').exists() and parent!=temporary_root for parent in (path,*path.parents)):
        raise ValueError('VASP work-root must be outside source checkouts; use a private runtime directory')
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def execute_vasp(command: list[str], directory: Path, timeout: float, threads: int) -> float:
    """Run without a shell; terminate the process group on timeout/interruption."""
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads))
    start = perf_counter()
    with (directory/'vasp.stdout').open('w') as log:
        process = subprocess.Popen(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,
                                   env=env,start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
            raise
    if code:
        raise RuntimeError(f'VASP exited {code}; inspect private runtime logs')
    return perf_counter()-start


def scf_diagnostic(solver: VaspBlochHarris, runtime: Path, reference_energy: float,
                   repair_scf: bool = False) -> dict:
    """Separate diagnostic only: SCF inputs never replace the production ML result."""
    if solver.settings.spinor:
        return {'status':'unavailable','reason':'Noncollinear CHGCAR diagnostic reader is not implemented'}
    field = (SpinDensity if solver.settings.ispin==2 else ChargeDensity).read(runtime/'CHGCAR',dtype='float64')
    if field.grid.shape != solver.grid.shape or not np.allclose(field.grid.cell,solver.grid.cell):
        raise ValueError('SCF diagnostic grid mismatch; no resampling is permitted')
    charge = field.total if hasattr(field,'total') else field.data
    m = field.magnetization if hasattr(field,'magnetization') else np.zeros(solver.grid.shape)
    _, raw_blocks = read_augmentation_blocks(runtime/'CHGCAR')
    blocks = [parse_augmentation(block) for block in raw_blocks]
    expected = 2 if solver.settings.ispin==2 else 1
    if len(blocks)!=expected or any(len(block)!=solver.structure.natoms for block in blocks):
        raise ValueError('SCF diagnostic lacks complete PAW on-site records')
    records = tuple(np.array([block[i] for block in blocks]) for i in range(solver.structure.natoms))
    diagnostics=inspect_density(charge,solver.grid,solver.evaluator.electrons,m)
    adjustment={'enabled':repair_scf,'raw_electrons':diagnostics.raw_electrons,
                'negative_charge_e':diagnostics.negative_electrons,
                'charge_defect_e':diagnostics.charge_defect,
                'factor':1.0,'stationary_reference_preserved':True}
    if repair_scf:
        # Explicit diagnostic approximation only: raw VASP interpolation can
        # produce tiny negative grid values and a nonintegral CHGCAR charge.
        # The same module repairs the ML prediction, so the reference and the
        # prediction are corrected identically. Record it rather than calling
        # the repaired density exactly stationary.
        repair=repair_density(charge,solver.grid,solver.evaluator.electrons,m,records)
        charge,m,records=repair.charge,repair.magnetization,repair.records
        adjustment.update(factor=repair.normalization_factor,
                          clipped_electrons=repair.clipped_electrons,
                          stationary_reference_preserved=False)
    scf = PawHarrisInput(SpinDensityInput(np.array(charge,float),np.array(m,float)),records)
    energies = []
    lambdas = np.array([0.,.125,.25,.5,1.])
    for lam in lambdas:
        trial = PawHarrisInput(SpinDensityInput(scf.density.charge+lam*(solver.density.charge-scf.density.charge),
            scf.density.magnetization+lam*(solver.density.magnetization-scf.density.magnetization)),
            tuple(r+lam*(p-r) for r,p in zip(scf.records,solver.prediction.records)))
        result = solver.evaluator.evaluate(trial,solver.kpoints)
        energies.append(result.free_energy)
    return {'status':'ok','reference_density_adjustment':adjustment,
        'backend_offset_at_reference_density_ev':energies[0]-reference_energy,
        'density_and_occupancy_effect_ev':energies[-1]-energies[0],
        'density_l1_electrons':integrate(np.abs(solver.density.charge-charge),solver.grid),
        'stationarity':stationarity_fit(lambdas,np.array(energies))}


def main() -> None:
    """Read manifest cases containing deck_directory and an external potcar file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--models',required=True)
    parser.add_argument('--vasp-bin',default=DEFAULT_VASP,type=Path)
    parser.add_argument('--vasp-ncl',type=Path,help='Explicit user-provided 6.6.1 noncollinear binary for SOC')
    parser.add_argument('--launcher',default='',help='e.g. "mpiexec -n 2"; parsed as arguments, never a shell')
    parser.add_argument('--work-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--timeout',type=float,default=600)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--repair-density',action='store_true')
    parser.add_argument('--spin-average',action='store_true')
    parser.add_argument('--unpolarized-input',action='store_true')
    parser.add_argument('--diagnose-scf-density',action='store_true')
    parser.add_argument('--diagnostic-repair-scf',action='store_true',
                        help='Explicitly repair diagnostic SCF charge; invalidates exact stationarity attribution')
    args = parser.parse_args()
    if args.threads<1 or args.timeout<=0:
        parser.error('Require positive threads and timeout')
    root = _external_work_root(args.work_root)
    manifest = json.loads(args.manifest.read_text())
    cases = manifest['cases']
    if not cases or len({case['id'] for case in cases})!=len(cases):
        parser.error('Manifest requires a nonempty case list with unique ids')
    report = {'schema':1,'energy_convention':'free energy: Poraque F_Harris versus VASP TOTEN',
              'potcar_policy':'Explicit user-provided licensed inputs; no potential data in report',
              'resources':{'threads':args.threads,'launcher':shlex.split(args.launcher)},
              'model_sha256':hashlib.sha256(Path(args.models).read_bytes()).hexdigest(),
              'cases':[]}
    for case in cases:
        row = {'id':case['id'],'status':'failed'}
        runtime = root/uuid.uuid4().hex
        runtime.mkdir(mode=0o700)
        row['private_runtime_directory'] = str(runtime)
        solver = None
        try:
            deck = Path(case['deck_directory']).expanduser().resolve(strict=True)
            potcar = Path(case['potcar']).expanduser().resolve(strict=True)
            if not potcar.is_file() or potcar.suffix.lower() in ('.gz','.z'):
                raise ValueError('Benchmark VASP requires an explicit uncompressed POTCAR file')
            for name in ('POSCAR','KPOINTS'):
                (runtime/name).write_bytes((deck/name).read_bytes())
            incar = Incar.from_file(deck/'INCAR')
            settings = ElectronicSettings.from_incar(incar)
            if settings.functional!='pbe':
                raise ValueError('Automated matched benchmark currently requires PBE; LDA PZ/PW92 matching is not validated')
            if incar.get_int('NSW',0)!=0 or incar.get_int('ICHARG',2)>=10:
                raise ValueError('Benchmark requires fixed-geometry SCF inputs (NSW=0, ICHARG<10)')
            structure = Poscar.from_file(runtime/'POSCAR')
            grid = FieldGrid(incar.fine_shape,structure.cell) if incar.fine_shape else FieldGrid.from_encut(
                structure.cell,settings.encut,prec='accurate')
            # Matching explicit density grids and supported full angular PAW XC.
            incar.update(NSW='0',IBRION='-1',ISTART='0',ICHARG='2',LCHARG='.TRUE.',LWAVE='.FALSE.',
                         LASPH='.TRUE.',PREC='Accurate',ISYM='-1')
            for key,value in zip(('NGXF','NGYF','NGZF'),grid.shape):
                incar[key]=str(value)
            (runtime/'INCAR').write_text('\n'.join(f'{key} = {value}' for key,value in incar.items())+'\n')
            # Symlink only: Poraque never assembles or copies potential data.
            (runtime/'POTCAR').symlink_to(potcar)
            row['natoms']=structure.natoms
            row['input_hashes']={name:hashlib.sha256((runtime/name).read_bytes()).hexdigest()
                                 for name in ('POSCAR','INCAR','KPOINTS')}
            row['potcar_sha256']=hashlib.sha256(potcar.read_bytes()).hexdigest()
            try:
                start=perf_counter()
                solver=VaspBlochHarris(runtime,args.models,potcar,args.device,args.repair_density,
                                        args.spin_average,args.unpolarized_input)
                prepared=perf_counter()
                result=solver.total_energy()
                end=perf_counter()
                row['poraque']={'free_ev':result.free_energy,'internal_ev':result.total,
                    'prepare_seconds':prepared-start,'solve_seconds':end-prepared,'total_seconds':end-start,
                    'density':solver.density_diagnostics,'paw_terms_ev':dict(solver.evaluator.last_terms),
                    'max_eigen_residual_ev':result.max_residual}
                # Retain all magnetic multipoles in the optional diagnostic CHGCAR.
                lmax=2*max(max(d.setup.channels) for d in solver.datasets.values())
                incar['LMAXMIX']=str(lmax)
                (runtime/'INCAR').write_text('\n'.join(f'{key} = {value}' for key,value in incar.items())+'\n')
                row['input_hashes']['INCAR']=hashlib.sha256((runtime/'INCAR').read_bytes()).hexdigest()
            except Exception as error:
                row['poraque_error']=f'{type(error).__name__}: {error}'
            binary = args.vasp_ncl if settings.spinor else args.vasp_bin
            if binary is None:
                raise ValueError('SOC/noncollinear cases require explicit --vasp-ncl; vasp_std cannot run spinors')
            binary = binary.expanduser().resolve(strict=True)
            command = shlex.split(args.launcher)+[str(binary)]
            seconds = execute_vasp(command,runtime,args.timeout,args.threads)
            reference = read_vasp_reference(runtime)
            row['vasp']={**reference.as_dict(),'seconds':seconds}
            if hashlib.sha256(potcar.read_bytes()).hexdigest()!=row['potcar_sha256']:
                raise ValueError('Runtime POTCAR changed during the benchmark; comparison rejected')
            if reference.version!='6.6.1' or not reference.converged or not reference.finished:
                raise ValueError('VASP reference must be a completed, explicitly converged 6.6.1 SCF run')
            if 'poraque' not in row:
                raise ValueError('Poraque failed; SCF energy retained without an accuracy comparison')
            row['comparison']=energy_difference(row['poraque']['free_ev'],reference.free_ev,structure.natoms)
            row['speedup']=seconds/row['poraque']['total_seconds']
            row['status']='ok'
            if args.diagnose_scf_density:
                try:
                    row['scf_density_diagnostic']=scf_diagnostic(solver,runtime,reference.free_ev,args.diagnostic_repair_scf)
                except Exception as error:
                    row['scf_density_diagnostic']={'status':'failed','error':f'{type(error).__name__}: {error}'}
        except Exception as error:
            row['error']=f'{type(error).__name__}: {error}'
        report['cases'].append(row)
        report['summary']=summarize(report['cases'])
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'id':row['id'],'status':row['status'],'comparison':row.get('comparison'),
                          'error':row.get('error')},allow_nan=False),flush=True)
    if report['summary']['failed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
