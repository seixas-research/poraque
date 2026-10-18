"""VASP input deck -> ext2chg -> norm-conserving Bloch Harris/band evaluation."""
from pathlib import Path

import numpy as np

from ..fields import FieldGrid
from ..fields.vasp import Incar, Poscar
from ..fields.vasp.kpoints import Kpoints
from ..physics.bloch import ElectronicSettings, BlochHarrisResult
# Imported as a module: the constructor's ``repair_density`` flag would
# otherwise shadow the function of the same name.
from ..physics import density_repair as sanity
from ..physics.energy import total_density
from ..physics.spin_xc import SpinDensityInput
from ..physics.paw_bloch import PawHarrisInput, PawHarrisEvaluator, official_local_potential
from ..fields.vasp.user_paw import UserPawDataset
from .training import load_bundle


class VaspBlochHarris:
    """Official-VASP-only PAW Hamiltonian from an ext2chg prediction.

    Both grid density and atom occupancy records must come from ext2chg. Every
    potential is supplied explicitly by the user at runtime, either as one
    concatenated POTCAR or a species-to-file mapping. A valid VASP license is
    required. No potential file is generated, bundled, searched for or copied.
    Scalar magnetic predictions can be embedded along SAXIS for spinor jobs;
    vector predictions explicitly use Cartesian components. MAGMOM never
    substitutes for a missing predicted magnetization field.
    """

    def __init__(self, directory: str | Path, bundle: str,
                 potcar_files: dict[str,str | Path] | str | Path | None = None, device: str = 'cpu',
                 repair_density: bool = False, spin_average: bool = False,
                 unpolarized_input: bool = False) -> None:
        self.directory=Path(directory)
        self.structure=Poscar.from_file(self.directory/'POSCAR')
        self.incar=Incar.from_file(self.directory/'INCAR')
        self.settings=ElectronicSettings.from_incar(self.incar)
        if not self.incar.get_bool('LASPH',True):
            raise ValueError('This variational PAW solver requires LASPH=True')
        mesh_tags=('NGXF','NGYF','NGZF')
        shape=None
        if any(key in self.incar for key in mesh_tags):
            if not all(key in self.incar for key in mesh_tags):
                raise ValueError('Specify all three NGXF/NGYF/NGZF dimensions')
            sizes=[self.incar.numbers(key) for key in mesh_tags]
            if any(len(v)!=1 or not np.isfinite(v[0]) or v[0]<=0 or int(v[0])!=v[0] for v in sizes):
                raise ValueError('Density mesh dimensions must be positive integers')
            shape=tuple(int(v[0]) for v in sizes)
        self.grid=(FieldGrid(shape,self.structure.cell) if shape else
                   FieldGrid.from_encut(self.structure.cell,self.settings.encut,prec='accurate'))
        if potcar_files is None:
            raise ValueError('Explicit user-provided POTCAR file(s) are required; a valid VASP license is required')
        self.datasets={element:UserPawDataset.load(
            potcar_files[element] if isinstance(potcar_files,dict) else potcar_files,
            self.settings.functional,element=element) for element in self.structure.elements}
        for element,dataset in self.datasets.items():
            if dataset.setup.element!=element:
                raise ValueError(f'Official PAW variant element mismatch for {element}')
        electrons=sum(self.datasets[e].setup.zval*c for e,c in zip(self.structure.elements,self.structure.counts))
        if 'NELECT' in self.incar:
            values=self.incar.numbers('NELECT')
            if len(values)!=1:
                raise ValueError('NELECT requires one finite value')
            electrons=values[0]
        if not np.isfinite(electrons) or electrons<=0:
            raise ValueError('NELECT must be finite and positive')
        external=official_local_potential(self.structure,self.grid,self.datasets)
        operator=load_bundle(bundle,'ext2chg',device=device)
        if hasattr(operator.model,'layouts'):
            from ..fields.structure import symbol_to_z
            for element,dataset in self.datasets.items():
                if list(operator.model.layouts.get(symbol_to_z(element),[]))!=list(dataset.setup.channels):
                    raise ValueError('ext2chg projector channels do not match the official PAW dataset')
        prediction=operator.predict(external)
        sets=operator.predict_occupancies(external)  # Mandatory predicted on-site information.
        raw=np.array(total_density(prediction),dtype=float,copy=True)
        mz=getattr(prediction,'magnetization',None)
        self.raw_density=raw.copy()
        self.raw_magnetization=None if mz is None else np.array(mz,copy=True)
        self.raw_records=tuple(np.array([values[i] for values in sets]) for i in range(self.structure.natoms))
        records=[value.copy() for value in self.raw_records]
        axis=np.asarray(self.settings.saxis)/np.linalg.norm(self.settings.saxis)
        if self.settings.magnetic:
            if spin_average:
                raise ValueError('spin_average cannot discard magnetization in a magnetic calculation')
            if mz is None and not unpolarized_input:
                raise ValueError('Magnetic Hamiltonian requires predicted magnetization; or explicit unpolarized_input')
            m=np.zeros(self.grid.shape) if mz is None else np.array(mz,dtype=float,copy=True)
            expected=4 if self.settings.spinor else 2
            if self.settings.spinor and m.ndim==3:
                m=axis[:,None,None,None]*m
            if not self.settings.spinor and m.ndim!=3:
                raise ValueError('ISPIN=2 requires collinear model magnetization')
            converted=[]
            for record in records:
                if record.shape[0]==expected:
                    converted.append(record)
                elif record.shape[0]==2 and expected==4:
                    converted.append(np.concatenate([record[:1],axis[:,None]*record[1]]))
                elif record.shape[0]==1 and unpolarized_input:
                    converted.append(np.concatenate([record,np.zeros((expected-1,record.shape[1]))]))
                else:
                    raise ValueError('ext2chg occupancy magnetization sets are missing or incompatible')
            records=converted
        else:
            if mz is not None and np.max(np.abs(mz))>1e-10 and not spin_average:
                raise ValueError('Scalar calculation requires explicit spin_average for a magnetic prediction')
            m=np.zeros(self.grid.shape)
            records=[record[:1] for record in records]
        # The floor acts on the eigen-densities (n +/- |m|)/2 and one factor
        # rescales density, magnetization and on-site records together
        # (physics/density_repair.py, shared with the Gamma evaluator and the
        # benchmark's SCF diagnostic). Without the explicit repair a defective
        # prediction fails here, naming the defect.
        if repair_density:
            repair=sanity.repair_density(raw,self.grid,electrons,m,records)
        else:
            repair=sanity.DensityRepair.unrepaired(raw.copy(),m,records,
                sanity.require_valid_density(raw,self.grid,electrons,m,what='ext2chg density'))
        self.repair=repair
        density,m,records=repair.charge,repair.magnetization,list(repair.records)
        self.density=SpinDensityInput(density,m)
        self.prediction=PawHarrisInput(self.density,tuple(records))
        self.density_diagnostics={**repair.as_dict(),
            'spin_averaged':spin_average,'unpolarized_embedding':mz is None and self.settings.magnetic,
            'collinear_embedding':self.settings.spinor and mz is not None and np.ndim(mz)==3}
        self.evaluator=PawHarrisEvaluator(self.structure,self.grid,electrons,self.settings,self.datasets)
        self.evaluator.hamiltonian(self.prediction)
        self.kpoints=self.read_kpoints(self.directory/'KPOINTS')

    def read_kpoints(self, path: str | Path) -> Kpoints:
        """Read another mesh or path, including the POSCAR Cartesian scale."""
        lines = (self.directory / 'POSCAR').read_text().splitlines()
        values = [float(value) for value in lines[1].split()]
        scale = values[0] if len(values) == 1 else None
        if scale is not None and scale < 0:
            unscaled = np.array([[float(x) for x in line.split()] for line in lines[2:5]])
            scale = (abs(scale) / abs(np.linalg.det(unscaled)))**(1/3)
        return Kpoints.from_file(path, self.structure.cell, scale=scale)

    def total_energy(self) -> BlochHarrisResult:
        """Use the input KPOINTS quadrature; line mode raises an explicit error."""
        return self.evaluator.evaluate(self.prediction, self.kpoints)

    def bands(self, path: str | Path | None = None) -> dict:
        """Eigenvalues in eV, fractional k coordinates, path distances and labels.

        Use the Fermi level from a separate converged k-mesh calculation for
        alignment; a line-mode path does not define a BZ electron count.
        """
        kpoints = self.kpoints if path is None else self.read_kpoints(path)
        return self.evaluator.bands(self.prediction, kpoints)
