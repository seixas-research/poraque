"""Standalone CHON ext2chg/Harris benchmark against the same local LDA SCF.

Run from the repository: python scripts/poraque_harris.py --help.
This is an internal local-pseudopotential DFT baseline, not a PAW baseline.
"""
from __future__ import annotations

import argparse
import json
from time import perf_counter
from pathlib import Path

import numpy as np
from ase.build import molecule

from poraque.ml.harris import Ext2ChgHarris
from poraque.physics.harris import HarrisEvaluator, HarrisResult, integrate


def scf_reference(evaluator: HarrisEvaluator, max_steps: int = 150,
                  mixing: float = 0.15, tolerance: float = 1e-6
                  ) -> tuple[HarrisResult, int, float]:
    """Independent uniform start; linear-mixed zero-temperature local LDA SCF.

    Return only a converged reference: integrated absolute density residual
    below ``tolerance`` electrons. Failure never becomes an accuracy datum.
    This simple reference solver can require smaller mixing for difficult cells.
    """
    density = np.full(evaluator.grid.shape, evaluator.electrons / evaluator.grid.volume)
    for step in range(1, max_steps + 1):
        result = evaluator.evaluate(density)
        residual = integrate(np.abs(result.output_density - density), evaluator.grid)
        if residual < tolerance:
            return result, step, residual
        density = (1 - mixing) * density + mixing * result.output_density
    raise RuntimeError(f'SCF failed in {max_steps} steps; density residual={residual:g} e')


def main() -> None:
    """Evaluate closed-shell CHON molecules; persist all successes and failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', required=True, help='Bundle containing ext2chg')
    parser.add_argument('--potcar-dir', required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=(16, 16, 16))
    parser.add_argument('--cell', type=float, default=10.0, help='Cubic cell length in Angstrom')
    parser.add_argument('--molecules', nargs='+', default=['CH4', 'NH3', 'H2O', 'H2CO'])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repair-density', action='store_true')
    parser.add_argument('--spin-average', action='store_true')
    parser.add_argument('--scf', action='store_true', help='Run same-Hamiltonian local LDA SCF')
    parser.add_argument('--max-scf-steps', type=int, default=150)
    parser.add_argument('--mixing', type=float, default=0.15)
    parser.add_argument('--output', type=Path, default=Path('harris_chon.json'))
    args = parser.parse_args()
    if not 0 < args.mixing <= 1 or args.max_scf_steps < 1:
        parser.error('Require 0 < mixing <= 1 and max-scf-steps >= 1')
    start = perf_counter()
    pipeline = Ext2ChgHarris(args.models, args.potcar_dir, tuple(args.shape),
                            args.device, args.repair_density, args.spin_average)
    report = {'backend': 'Gamma local pseudopotential LDA/PW92; not PAW',
              'settings': {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
              'model_load_seconds': perf_counter() - start, 'systems': []}
    failed = False
    for name in args.molecules:
        row = {'molecule': name}
        try:
            atoms = molecule(name)
            if not set(atoms.get_chemical_symbols()) <= set('CHON'):
                raise ValueError('Only CHON species are allowed')
            atoms.set_cell([args.cell] * 3)
            atoms.center()
            atoms.pbc = True
            start = perf_counter()
            prediction = pipeline.predict(atoms)
            seconds = perf_counter() - start
            result = prediction.result
            row.update(status='ok', harris_ev=result.total, band_ev=result.band,
                       hartree_ev=result.hartree, xc_ev=result.xc,
                       hartree_double_counting_ev=result.hartree_double_counting,
                       xc_double_counting_ev=result.xc_double_counting,
                       ion_ion_ev=result.ion_ion, pipeline_seconds=seconds,
                       raw_electrons=prediction.raw_electrons,
                       input_electrons=result.electrons,
                       clipped_electrons=prediction.clipped_electrons,
                       normalization_factor=prediction.normalization_factor,
                       eigen_residual_ev=result.residual,
                       density_residual_e=integrate(np.abs(result.output_density -
                                                           prediction.input_density),
                                                    prediction.evaluator.grid))
            if args.scf:
                start = perf_counter()
                reference, iterations, residual = scf_reference(
                    prediction.evaluator, args.max_scf_steps, args.mixing)
                scf_seconds = perf_counter() - start
                row.update(scf_ev=reference.total, scf_iterations=iterations,
                           scf_seconds=scf_seconds, scf_density_residual_e=residual,
                           energy_error_ev=result.total - reference.total,
                           error_ev_per_atom=(result.total - reference.total) / len(atoms),
                           speedup=scf_seconds / seconds)
        except Exception as error:
            row.update(status='failed', error=f'{type(error).__name__}: {error}')
            failed = True
        report['systems'].append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
