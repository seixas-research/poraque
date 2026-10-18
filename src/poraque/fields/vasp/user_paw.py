"""Read-only parser for explicitly user-provided, licensed PAW POTCAR inputs.

No potential is bundled, generated, downloaded, or located automatically.
Users must hold a valid VASP license. Runtime POTCAR parsing is read-only;
Poraque does not grant or verify a VASP license or certify dataset authenticity.
"""
from dataclasses import dataclass
import hashlib
import warnings
from pathlib import Path

import numpy as np

from .paw_setup import PawSetup
from .potcar import Potcar, read_potcar_text

class PotcarLicenseWarning(UserWarning):
    """User-supplied proprietary VASP data requires a valid VASP license."""



@dataclass(frozen=True)
class UserPawDataset:
    """User-supplied radial/projector tables with runtime file provenance."""

    setup: PawSetup
    potcar: Potcar
    path: Path
    sha256: str

    @classmethod
    def load(cls, path: str | Path, functional: str,
             element: str | None = None) -> 'UserPawDataset':
        """Parse an explicitly supplied POTCAR (optionally concatenated).

        A valid VASP license is required. Never copy POTCAR contents into logs,
        fixtures, results or a distribution. ``element`` selects a unique block
        of an externally supplied concatenation; without it exactly one block
        is required. Paths must identify files, never automatic search roots.
        """
        if functional not in ('lda', 'pbe'):
            raise ValueError('PAW backend supports lda or pbe')
        if path is None or not str(path):
            raise ValueError('An explicit user-provided POTCAR file is required')
        path = Path(path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError('Provide a POTCAR file, not a potential-library directory')
        warnings.warn('POTCAR is proprietary VASP data. You must hold a valid VASP license '
                      'to supply and use it. Poraque does not redistribute potential data.',
                      PotcarLicenseWarning, stacklevel=2)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        text = read_potcar_text(path)
        blocks = [block for block in text.split('End of Dataset') if block.strip()]
        if element is not None:
            blocks = [block for block in blocks
                      if Potcar.from_string(block, parse_tables=False)[0].element == element]
        if len(blocks) != 1:
            raise ValueError('Provide one unique PAW dataset per species in the runtime POTCAR input')
        setup = PawSetup.from_block(blocks[0])
        expected = {'lda': ('CA', 'PW'), 'pbe': ('PE',)}[functional]
        if str(setup.parameters.get('LEXCH', '')).upper() not in expected:
            raise ValueError(f'{path.name}: dataset LEXCH={setup.parameters.get("LEXCH")} '
                             f'does not match requested {functional}; no fallback is allowed')
        if not setup.title.startswith('PAW'):
            raise ValueError('User-provided POTCAR is not a PAW dataset')
        count = len(setup.channels)
        if (len(setup.projector_reciprocal) != count or len(setup.projector_real) != count or
                setup.projector_qmax is None or setup.projector_qmax <= 0):
            raise ValueError('Incomplete PAW nonlocal projector data')
        for name in ('grid', 'aepotential', 'core charge-density', 'kinetic energy-density',
                     'local pseudopotential core',
                     'pspotential valence only', 'core charge-density (pseudized)'):
            values = setup.tables.get(name)
            if values is None or values.shape != setup.grid.shape or not np.isfinite(values).all():
                raise ValueError(f'Incomplete official PAW radial table: {name}')
        for arrays in (setup.projector_reciprocal, setup.projector_real,
                       tuple(value for pair in setup.partial_waves for value in pair)):
            if any(not np.isfinite(value).all() or len(value) < 4 for value in arrays):
                raise ValueError('Invalid official PAW projector/partial-wave table')
        if any(value.shape!=setup.grid.shape for pair in setup.partial_waves for value in pair):
            raise ValueError('PAW partial waves do not match the radial grid')
        optional=setup.tables.get('mkinetic energy-density pseudized')
        if optional is not None and (optional.shape!=setup.grid.shape or not np.isfinite(optional).all()):
            raise ValueError('Invalid pseudized kinetic-energy table')
        if (not np.all(np.diff(setup.grid) > 0) or not np.all(setup.grid > 0) or
                setup.d0.shape != (count, count) or not np.isfinite(setup.d0).all() or
                not np.allclose(setup.d0, setup.d0.T)):
            raise ValueError('Invalid PAW grid or DION')
        setup.augmentation_charges  # Require complete Q_ij and reference occupancy tables.
        setup.atomic_occupancies
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('User-provided POTCAR changed while being parsed')
        return cls(setup, Potcar.from_string(blocks[0], parse_tables=True), path, digest)

    def validate_source(self) -> None:
        """Reject removal or modification of the user-provided source after loading."""
        if not self.path.is_file() or hashlib.sha256(self.path.read_bytes()).hexdigest() != self.sha256:
            raise ValueError('User-provided POTCAR source changed after parsing')
