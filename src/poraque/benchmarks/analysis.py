"""Energy conventions, convergence checks and Harris error decomposition."""
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?'

#: 1 kcal/mol per atom, the conventional chemical-accuracy threshold the
#: benchmark judges |E_Harris[rho_ext2chg] - E_VASP| against.
CHEMICAL_ACCURACY_MEV_PER_ATOM = 43.0


def _number(value: str) -> float:
    return float(value.replace('D', 'E').replace('d', 'e'))


@dataclass(frozen=True)
class VaspReference:
    """Different VASP energy definitions must never be interchanged implicitly."""

    free_ev: float
    internal_ev: float | None
    sigma_zero_ev: float | None
    converged: bool
    finished: bool
    version: str | None
    electronic_steps: int | None
    source: str

    def as_dict(self) -> dict:
        return asdict(self)


def read_vasp_reference(directory: str | Path) -> VaspReference:
    """Read final energies; require explicit OUTCAR evidence of SCF convergence.

    OSZICAR F is the free energy, E0 is sigma->0 (not the internal energy).
    Energy parsing alone does not establish convergence. The runner also checks
    subprocess success, fixed geometry, SCF settings, and VASP version 6.6.1.
    """
    directory = Path(directory)
    outcar = directory/'OUTCAR'
    text = outcar.read_text(errors='replace') if outcar.exists() else ''
    version = re.search(r'vasp[.\s]+(\d+\.\d+\.\d+)', text, re.I)
    free = re.findall(r'free\s+energy\s+TOTEN\s*=\s*('+NUMBER+')', text, re.I)
    pair = re.findall(r'energy\s+without\s+entropy\s*=\s*('+NUMBER+r')\s+energy\(sigma->0\)\s*=\s*('+NUMBER+')', text, re.I)
    oszicar = directory/'OSZICAR'
    osz = oszicar.read_text(errors='replace') if oszicar.exists() else ''
    fallback = re.findall(r'\bF=\s*('+NUMBER+r')\s+E0=\s*('+NUMBER+')', osz)
    if not free and not fallback:
        raise ValueError('No final VASP energy found')
    # Match the final electronic loop, rather than accepting convergence of an
    # earlier ionic step followed by a failed final loop.
    iterations = list(re.finditer(r'Iteration\s+\d+\(\s*\d+\)', text))
    last_loop = text[iterations[-1].start():] if iterations else text
    converged = 'aborting loop because EDIFF is reached' in last_loop
    finished = 'General timing and accounting informations' in text
    steps = re.findall(r'^\s*(?:DAV|RMM|CG|DMP|RMM-DIIS):\s*(\d+)', osz, re.M)
    result = VaspReference(_number(free[-1] if free else fallback[-1][0]),
        _number(pair[-1][0]) if pair else None,
        _number(pair[-1][1]) if pair else (_number(fallback[-1][1]) if fallback else None),
        converged, finished, version.group(1) if version else None,
        int(steps[-1]) if steps else None, 'OUTCAR' if free else 'OSZICAR')
    if not np.isfinite(result.free_ev):
        raise ValueError('Nonfinite VASP energy')
    return result


def energy_difference(poraque_ev: float, reference_ev: float, natoms: int) -> dict:
    """Signed error ΔE=E_Poraque-E_VASP and size-normalized absolute errors."""
    if natoms < 1 or not np.isfinite([poraque_ev, reference_ev]).all():
        raise ValueError('Require finite energies and positive atom count')
    delta = float(poraque_ev-reference_ev)
    per_atom_mev = 1000*abs(delta)/natoms
    return {'delta_ev': delta, 'delta_ev_per_atom': delta/natoms,
            'absolute_error_ev': abs(delta), 'absolute_error_mev_per_atom': per_atom_mev,
            'within_chemical_accuracy': bool(per_atom_mev < CHEMICAL_ACCURACY_MEV_PER_ATOM)}


def summarize(rows: Iterable[dict]) -> dict:
    """Aggregate only successful matched/converged comparisons; count failures.

    ``within_chemical_accuracy`` counts the valid cases under
    :data:`CHEMICAL_ACCURACY_MEV_PER_ATOM`; ``all_within_chemical_accuracy``
    is the one flag a tracker watches, and it is ``False`` while there is no
    valid case at all, so an empty run never reads as a pass.
    """
    rows = list(rows)
    successful = [row for row in rows if row.get('status') == 'ok']
    delta = np.array([row['comparison']['delta_ev_per_atom'] for row in successful])
    within = int(sum(1000*abs(d) < CHEMICAL_ACCURACY_MEV_PER_ATOM for d in delta))
    output = {'attempted': len(rows), 'valid': len(successful), 'failed': len(rows)-len(successful),
              'chemical_accuracy_mev_per_atom': CHEMICAL_ACCURACY_MEV_PER_ATOM,
              'within_chemical_accuracy': within,
              'all_within_chemical_accuracy': bool(len(delta) and within == len(delta))}
    if len(delta):
        output.update(bias_ev_per_atom=float(delta.mean()), mae_ev_per_atom=float(np.abs(delta).mean()),
                      rmse_ev_per_atom=float(np.sqrt(np.mean(delta**2))),
                      max_abs_ev_per_atom=float(np.max(np.abs(delta))),
                      max_abs_mev_per_atom=float(1000*np.max(np.abs(delta))))
    return output


def stationarity_fit(lambdas: np.ndarray, energies: np.ndarray) -> dict:
    """Fit E(lambda)-E(0)=a*lambda+b*lambda²; expose a nonstationary backend.

    Interpolation must vary both grid charge/magnetization and PAW on-site
    records. Quadratic Harris behavior is local and requires a common stationary
    functional, charge constraint, basis, magnetic state and occupations. A
    VASP SCF density need not be stationary for an incompletely matched backend.
    """
    x, e = np.asarray(lambdas, float), np.asarray(energies, float)
    if x.shape != e.shape or len(x) < 3 or not np.isfinite([x,e]).all() or x[0] != 0:
        raise ValueError('Require finite interpolation data starting at lambda=0')
    coefficients, *_ = np.linalg.lstsq(np.column_stack([x, x*x]), e-e[0], rcond=None)
    predicted = coefficients[0]*x+coefficients[1]*x*x
    keep = (x>0)&(np.abs(e-e[0])>1e-10)
    power = float(np.polyfit(np.log(x[keep]), np.log(np.abs(e[keep]-e[0])),1)[0]) if keep.sum()>1 else None
    return {'linear_coefficient_ev': float(coefficients[0]), 'quadratic_coefficient_ev': float(coefficients[1]),
            'fit_rms_ev': float(np.sqrt(np.mean((e-e[0]-predicted)**2))), 'observed_power': power,
            'lambdas': x.tolist(), 'energies_ev': e.tolist()}
