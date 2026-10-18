"""Strict ext2chg -> fixed-density Harris integration for atomic geometries."""
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..fields import ExternalPotential, FieldGrid
from ..fields.structure import Structure
from ..fields.vasp.potcar import Potcar
from ..physics.density_repair import (DensityRepair, repair_density,
                                      require_valid_density)
from ..physics.energy import ewald_energy, total_density
from ..physics.harris import HarrisEvaluator, HarrisResult
from .training import load_bundle


@dataclass
class HarrisPrediction:
    """Input prediction, explicitly processed density and energy diagnostics.

    ``repair`` is the :class:`~poraque.physics.density_repair.DensityRepair`
    record of what was done to the prediction (``repair.repaired`` says
    whether anything was); the scalar fields beside it are kept for callers
    that read them directly.
    """

    result: HarrisResult
    raw_density: np.ndarray
    input_density: np.ndarray
    raw_electrons: float
    clipped_electrons: float
    normalization_factor: float
    evaluator: HarrisEvaluator
    repair: DensityRepair


class Ext2ChgHarris:
    """Load only the ext2chg task; never substitute another density model.

    Coordinates/species/cell enter as ASE Atoms. POTCAR local tables build
    the model input. PAW occupancies are not used by this local backend;
    predictions therefore cannot be compared directly to PAW total energies.
    A spin-polarized prediction requires explicit ``spin_average=True``.
    Density clipping/normalization require explicit ``repair_density=True``
    and their magnitude is reported, preserving the original prediction.
    """

    def __init__(self, bundle: str, potcar_dir: str, shape: tuple[int, int, int],
                 device: str = 'cpu', repair_density: bool = False,
                 spin_average: bool = False) -> None:
        self.operator = load_bundle(bundle, 'ext2chg', device=device)
        self.potcar_dir, self.shape = potcar_dir, shape
        self.repair_density, self.spin_average = repair_density, spin_average

    def predict(self, atoms: Any) -> HarrisPrediction:
        """Predict rho_in from atomic geometry, then diagonalize exactly once."""
        structure = Structure.from_ase(atoms)
        grid = FieldGrid(self.shape, structure.cell)
        potcar = Potcar.from_library(self.potcar_dir, structure.elements)
        charges = {entry.element: entry.zval for entry in potcar}
        electrons = sum(charges[s] * c for s, c in zip(structure.elements, structure.counts))
        external = ExternalPotential.from_potcar_tables(structure, grid, potcar)
        prediction = self.operator.predict(external)
        if hasattr(prediction, 'magnetization') and not self.spin_average:
            if np.max(np.abs(prediction.magnetization)) > 1e-10:
                raise ValueError('Spin prediction requires explicit spin_average=True for this backend')
        raw = np.array(total_density(prediction), dtype=np.float64, copy=True)
        # One module decides what "nonnegative and normalized" means and what
        # the repair does (physics/density_repair.py); the strict path raises
        # naming every defect, the explicit one floors, rescales and reports.
        if self.repair_density:
            repair = repair_density(raw, grid, electrons)
        else:
            repair = DensityRepair.unrepaired(
                raw.copy(), None, (),
                require_valid_density(raw, grid, electrons, what='ext2chg density'))
        evaluator = HarrisEvaluator(grid, external.data, electrons,
                                    ewald_energy(structure, charges))
        return HarrisPrediction(evaluator.evaluate(repair.charge), raw, repair.charge,
                                repair.raw_electrons, repair.clipped_electrons,
                                repair.normalization_factor, evaluator, repair)
