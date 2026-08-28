"""
Is the Euler-Lagrange residual physics, or is it the grid?

Every other script here reads a downsampled cache, so the first question to
settle is whether the number they report survives the truncation. It is a real
risk: the residual is built from second derivatives of a density that has been
band-limited, and Gibbs ringing near a core is exactly the kind of thing that
manufactures a residual out of nothing.

Run direct from the native calculations, resampling in-process, so the *only*
variable is the cutoff. If the answer is flat above some resolution, the cache
that other scripts read is fine and the violation is real.
"""
import glob
import os

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: Calculation directories to sample. A handful is enough: this measures a
#: convergence trend, not a dataset average.
RUNS = sorted(glob.glob(os.path.join(REPO, "data", "vasp", "structures",
                                     "structure_*")))[:4]
POTCARS = os.environ.get("PORAQUE_POTCAR_DIR",
                         os.path.expanduser("~/Simulations/vasp/POTCARs"))

from poraque.fields import ChargeDensity, ExternalPotential
from poraque.fields.resample import downsample_shape, resample_field
from poraque.fields.vasp import Potcar
from poraque.ml.physics import (
    hartree_potential,
    thomas_fermi_potential,
    von_weizsacker_potential,
    xc_potential,
)

torch.set_default_dtype(torch.float64)

LAMBDA = 1.0 / 9.0


def residual(density, potential, use_xc):
    """``std`` of the Euler-Lagrange residual, and the scale to read it against."""
    def tensor(field):
        return torch.tensor(np.asarray(field.data),
                            dtype=torch.float64)[None, None]

    rho, vext = tensor(density), tensor(potential)
    cell = torch.tensor(np.asarray(density.grid.cell),
                        dtype=torch.float64)[None]
    total = (thomas_fermi_potential(rho)
             + LAMBDA * von_weizsacker_potential(rho, cell)
             + vext + hartree_potential(rho, cell))
    if use_xc:
        total = total + xc_potential(rho, "pbe", cell=cell)
    return (total - total.mean()).std().item(), vext.std().item()


if __name__ == "__main__":
    print(f"{len(RUNS)} structures, TF + (1/9) vW, resampled in-process\n")
    header = (f"  {'grid':>10} {'std(r) xc off':>15} {'/std(vext)':>11} "
              f"{'std(r) xc on':>14} {'/std(vext)':>11}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for resolution in (24, 32, 48, 64, None):
        off, on, scale, shape = [], [], [], None
        for run in RUNS:
            density = ChargeDensity.read(os.path.join(run, "CHGCAR"))
            if resolution:
                density = resample_field(
                    density,
                    downsample_shape(density.grid.shape,
                                     target_max=resolution))
            shape = density.grid.shape
            potcar = Potcar.from_library(POTCARS, density.structure.symbols)
            potential = ExternalPotential.from_potcar_tables(
                density.structure, density.grid, potcar)
            a, s = residual(density, potential, use_xc=False)
            b, _ = residual(density, potential, use_xc=True)
            off.append(a)
            on.append(b)
            scale.append(s)
        label = f"{shape[0]}^3" + ("" if resolution else "  native")
        print(f"  {label:>10} {np.mean(off):>15.3f} "
              f"{np.mean(off) / np.mean(scale):>11.3f} "
              f"{np.mean(on):>14.3f} "
              f"{np.mean(on) / np.mean(scale):>11.3f}")
