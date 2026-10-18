"""Four-channel ext2chg prediction in a fixed Cartesian spin frame."""
from dataclasses import dataclass, field
import numpy as np

from .grid import FieldGrid
from .structure import Structure


@dataclass
class VectorSpinDensity:
    """Charge plus Cartesian mx,my,mz, all in electrons/Angstrom^3.

    This inference container retains unprocessed model values. Positive
    semidefiniteness is enforced by the electronic-structure evaluator or by
    explicit density repair, never silently at decoding time.
    """

    total: np.ndarray
    magnetization: np.ndarray
    grid: FieldGrid
    structure: Structure
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.total=np.asarray(self.total)
        self.magnetization=np.asarray(self.magnetization)
        if self.total.shape!=self.grid.shape or self.magnetization.shape!=(3,*self.grid.shape):
            raise ValueError('Noncollinear density requires charge and three Cartesian components')

    @property
    def data(self) -> np.ndarray:
        """Channel-major (n,mx,my,mz) prediction."""
        return np.concatenate([self.total[None],self.magnetization])
