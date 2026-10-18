"""Official-local-PAW ext2chg smoke deck; no potential data is copied.

A very low cutoff and coarse H2 cell check software only, not DFT accuracy.
"""
import argparse
from pathlib import Path

from ase.build import molecule
from poraque.fields.structure import Structure
from poraque.fields.vasp import Poscar
from poraque.ml.bloch import VaspBlochHarris


def main() -> None:
    """Generate POSCAR/INCAR/mesh/path and run with the designated official library."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',required=True)
    parser.add_argument('--potcar',required=True,type=Path,help='Licensed user-provided H POTCAR file')
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--mode',choices=['scalar','collinear','spinor'],default='spinor')
    args=parser.parse_args()
    directory=args.output_dir
    directory.mkdir(parents=True,exist_ok=True)
    atoms=molecule('H2');atoms.set_cell([5]*3);atoms.center()
    Poscar.from_structure(Structure.from_ase(atoms)).write(directory/'POSCAR')
    spin={'scalar':'ISPIN=1','collinear':'ISPIN=2','spinor':'LSORBIT=.TRUE.; SAXIS=0 0 1'}[args.mode]
    (directory/'INCAR').write_text('ENCUT=20; NBANDS=8; ISMEAR=0; SIGMA=0\n'
        'GGA=PE; LASPH=.TRUE.; '+spin+'\nNGXF=9; NGYF=9; NGZF=9\n')
    (directory/'KPOINTS').write_text('mesh\n0\nMonkhorst-Pack\n2 1 1\n0 0 0\n')
    path=directory/'KPATH'
    path.write_text('Gamma-X\n5\nLine-mode\nReciprocal\n0 0 0 ! Gamma\n.5 0 0 ! X\n')
    pipeline=VaspBlochHarris(directory,args.models,args.potcar,repair_density=True,
                              spin_average=args.mode=='scalar')
    energy=pipeline.total_energy()
    bands=pipeline.bands(path)
    import json
    result={'interpretation':'official PAW software smoke only; unconverged cutoff/grid',
        'mode':args.mode,'density':pipeline.density_diagnostics,'harris_ev':energy.total,
        'max_residual_ev':energy.max_residual,'paw_terms_ev':pipeline.evaluator.last_terms,
        'bands':bands,'potentials':{e:{'path':str(d.path),'sha256':d.sha256}
                                   for e,d in pipeline.datasets.items()}}
    (directory/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'{args.mode}: Harris={energy.total:.9f} eV; residual={energy.max_residual:.3g} eV')


if __name__=='__main__':
    main()
