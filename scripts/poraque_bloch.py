"""Run ext2chg + official PAW Bloch Harris energy or line-mode bands from a VASP deck."""
import argparse
import json
from pathlib import Path

from poraque.ml.bloch import VaspBlochHarris


def main() -> None:
    """Write structured eigenvalues, BZ energies and density diagnostics to JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--models', required=True)
    parser.add_argument('--potcar', required=True, type=Path,
                        help='User-provided POTCAR file; a valid VASP license is required')
    parser.add_argument('--unpolarized-input', action='store_true')
    parser.add_argument('--bands', type=Path, help='Optional additional line-mode KPOINTS file')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--repair-density', action='store_true')
    parser.add_argument('--spin-average', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('bloch_harris.json'))
    args = parser.parse_args()
    pipeline = VaspBlochHarris(args.directory, args.models, args.potcar, args.device,
                               args.repair_density, args.spin_average, args.unpolarized_input)
    report = {'backend': 'official VASP PAW, generalized plane-wave spin LDA/PBE',
              'density': pipeline.density_diagnostics,
              'encut_ev': pipeline.settings.encut, 'functional':pipeline.settings.functional,
              'lsorbit':pipeline.settings.lsorbit, 'ispin':pipeline.settings.ispin,
              'potentials':{e:{'path':str(d.path),'sha256':d.sha256}
                            for e,d in pipeline.datasets.items()},
              'scf_metadata': {'EDIFF': pipeline.settings.ediff, 'NELM': pipeline.settings.nelm}}
    if pipeline.kpoints.weights is None:
        report['bands'] = pipeline.bands()
    else:
        result = pipeline.total_energy()
        report['energy'] = {'harris_internal_ev': result.total, 'harris_free_ev': result.free_energy,
            'band_ev': result.band, 'hartree_ev': result.hartree,
            'hartree_double_counting_ev': 2 * result.hartree,
            'xc_ev': result.xc, 'xc_double_counting_ev': result.xc_double_counting,
            'ion_ion_ev': result.ion_ion, 'entropy_term_ev': result.entropy_term,
            'fermi_ev': result.fermi, 'max_eigen_residual_ev': result.max_residual,
            'kpoints_fractional': pipeline.kpoints.points.tolist(),
            'weights': result.weights.tolist(), 'eigenvalues_ev': result.eigenvalues.tolist(),
            'occupations': result.occupations.tolist(), 'spin_mode':result.spin_mode,
            'paw_terms_ev':pipeline.evaluator.last_terms}
        if args.bands:
            report['bands'] = pipeline.bands(args.bands)
            report['bands']['reference_fermi_ev'] = result.fermi
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
