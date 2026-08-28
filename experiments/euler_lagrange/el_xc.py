"""
Does using the RIGHT exchange-correlation functional shrink the residual?

The Pt reference data was computed with PBE (PAW_PBE potentials, LEXCH = PE).
The first pass through `el_floor.py` used an LDA v_xc, so part of the residual
it reported was not the error of the kinetic functional at all: it was the
difference between two exchange-correlation potentials, of order 1 eV in a
valence region, sitting in a total of 1.7 eV.

This re-runs the same measurement with each functional so the mislabelled part
can be separated out. If PBE lowers the floor, the earlier number was
pessimistic and part of what the third operator was being asked to learn was
our own bookkeeping error.
"""
import os

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: Prepared cache to read. Override with PORAQUE_EL_CACHE to point at another
#: resolution or another dataset without editing this file.
CACHE = os.environ.get("PORAQUE_EL_CACHE",
                       os.path.join(REPO, "data", "cache", "res32_potcar_spin"))

from poraque.fields import ChargeDensity, ExternalPotential
from poraque.fields.io import resolve_xc
from poraque.ml.data import discover_materials
from poraque.ml.physics import (
    hartree_potential,
    thomas_fermi_potential,
    von_weizsacker_potential,
    xc_potential,
)

torch.set_default_dtype(torch.float64)


def load(record):
    chg = ChargeDensity.read(record.files["CHGCAR"])
    ext = ExternalPotential.read(record.files["EXTCAR"], grid=chg.grid)
    rho = torch.tensor(np.asarray(chg.data))[None, None]
    vext = torch.tensor(np.asarray(ext.data))[None, None]
    cell = torch.tensor(np.asarray(chg.grid.cell))[None]
    return rho, vext, cell


if __name__ == "__main__":
    records = discover_materials(CACHE)

    detected = resolve_xc(os.path.join(REPO, "data", "vasp", "structures", "structure_0000"),
                          declared="auto")
    print(f"detected from the calculation: xc = {detected!r}")
    print(f"({len(records)} structures)\n")

    lambdas = [(0.0, "TF"), (1.0 / 9.0, "TF + vW/9"), (1.0 / 5.0, "TF + vW/5"),
               (1.0 / 3.0, "TF + vW/3"), (1.0, "TF + vW")]
    functionals = ["none", "lda", "pbe"]

    print("std of the Euler-Lagrange residual [eV], ground-state data")
    header = f"  {'kinetic baseline':<16}" + "".join(
        f"{f:>12}" for f in functionals)
    print(header)
    print("  " + "-" * (len(header) - 2))

    best = {}
    for lam, label in lambdas:
        row = f"  {label:<16}"
        for functional in functionals:
            spreads = []
            for record in records:
                rho, vext, cell = load(record)
                kin = thomas_fermi_potential(rho)
                if lam:
                    kin = kin + lam * von_weizsacker_potential(rho, cell)
                total = (kin + vext + hartree_potential(rho, cell)
                         + xc_potential(rho, functional, cell=cell))
                spreads.append((total - total.mean()).std().item())
            value = float(np.mean(spreads))
            best[(label, functional)] = value
            row += f"{value:>12.3f}"
        print(row)

    lda_best = min(v for (_, functional), v in best.items()
                   if functional == "lda")
    pbe_best = min(v for (_, functional), v in best.items()
                   if functional == "pbe")
    print(f"\n  best with LDA : {lda_best:.3f} eV")
    print(f"  best with PBE : {pbe_best:.3f} eV")
    change = 100 * (pbe_best - lda_best) / lda_best
    print(f"  using the functional the data was computed with: "
          f"{change:+.1f}%")
