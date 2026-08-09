"""
How badly is the orbital-free Euler-Lagrange equation violated on data that is,
by construction, the ground state?

This is the measurement that decides whether a third operator has anything
learnable to learn. The residual

    r = dTs/drho + v_ext + v_H[rho] + v_xc[rho] - mean

must vanish for the exact functional at the exact ground-state density. Every
term but the first is exact here: v_ext is the tabulated pseudopotential
construction, v_H is one FFT, v_xc is LDA. So whatever r turns out to be is
the error of the kinetic potential plus whatever physics the local-potential
picture is missing (the nonlocal part of the pseudopotential, above all).

Reported against the scale of v_ext, because a residual of 5 eV means one
thing in a cell whose potential swings by 10 eV and another in one that swings
by 300.
"""
import os
import sys

import numpy as np
import torch

REPO = ("/Users/leseixas/Library/CloudStorage/Dropbox/Repositories/"
        "seixas-research/poraque")
CACHE = os.path.join(REPO, "data", "cache", "res32_potcar")

from poraque.fields import ChargeDensity, ExternalPotential, KineticEnergyDensity
from poraque.ml.data import discover_materials
from poraque.ml.physics import (
    hartree_potential,
    integrate,
    thomas_fermi_potential,
    von_weizsacker_potential,
    thomas_fermi_tau,
    von_weizsacker_tau,
    xc_potential,
)

torch.set_default_dtype(torch.float64)


def load(record):
    """The three reference fields of one material, as (1,1,N,N,N) tensors."""
    chg = ChargeDensity.read(record.files["CHGCAR"])
    ext = ExternalPotential.read(record.files["EXTCAR"], grid=chg.grid)
    tau = KineticEnergyDensity.read(record.files["TAUCAR"], grid=chg.grid)

    def t(field):
        return torch.tensor(np.asarray(field.data), dtype=torch.float64)[None, None]

    cell = torch.tensor(np.asarray(chg.grid.cell), dtype=torch.float64)[None]
    return t(chg), t(ext), t(tau), cell


def stats(name, field):
    a = field.flatten()
    return (f"{name:<22} mean {a.mean():>10.3f}  std {a.std():>10.3f}  "
            f"min {a.min():>10.3f}  max {a.max():>10.3f}")


if __name__ == "__main__":
    records = discover_materials(CACHE)
    print(f"{len(records)} structures from {CACHE}\n")

    # ---------------------------------------------------------------- #
    # 1. The size of each term, on one structure, so the residual can be
    #    read against the things it is made of.
    # ---------------------------------------------------------------- #
    rho, vext, tau, cell = load(records[0])
    vh = hartree_potential(rho, cell)
    vxc = xc_potential(rho, "pbe", cell=cell)
    vtf = thomas_fermi_potential(rho)
    vvw = von_weizsacker_potential(rho, cell)

    print(f"--- term magnitudes, {records[0].identifier} (eV) ---")
    for name, f in [("v_ext", vext), ("v_H[rho]", vh), ("v_xc[rho]", vxc),
                    ("dT_TF/drho", vtf), ("dT_vW/drho", vvw)]:
        print("  " + stats(name, f))
    print("  " + stats("rho (e/Ang^3)", rho))
    print("  " + stats("tau (eV/Ang^3)", tau))

    # ---------------------------------------------------------------- #
    # 2. The residual, over every structure, for a range of lambda and
    #    with the xc term on and off.
    # ---------------------------------------------------------------- #
    print(f"\n--- Euler-Lagrange residual on ground-state reference data ---")
    print(f"    r = dTs/drho + v_ext + v_H + v_xc, cell average removed\n")
    header = (f"  {'kinetic functional':<26} {'xc':<5} "
              f"{'std(r) [eV]':>12} {'std(r)/std(vext)':>18} "
              f"{'max|r| [eV]':>12}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    configurations = []
    for lam, label in [(0.0, "TF"), (1.0 / 9.0, "TF + (1/9) vW"),
                       (1.0 / 5.0, "TF + (1/5) vW"), (1.0, "TF + vW"),
                       (None, "vW only")]:
        for use_xc in (False, True):
            configurations.append((lam, label, use_xc))

    table = {}
    for lam, label, use_xc in configurations:
        ratios, stds, maxes = [], [], []
        for record in records:
            rho, vext, tau, cell = load(record)
            if lam is None:
                kin = von_weizsacker_potential(rho, cell)
            else:
                kin = thomas_fermi_potential(rho)
                if lam:
                    kin = kin + lam * von_weizsacker_potential(rho, cell)
            total = kin + vext + hartree_potential(rho, cell)
            if use_xc:
                total = total + xc_potential(rho, "pbe", cell=cell)
            r = total - total.mean()
            ratios.append((r.std() / vext.std()).item())
            stds.append(r.std().item())
            maxes.append(r.abs().max().item())
        table[(label, use_xc)] = (np.mean(stds), np.mean(ratios), np.mean(maxes))
        print(f"  {label:<26} {'on' if use_xc else 'off':<5} "
              f"{np.mean(stds):>12.3f} {np.mean(ratios):>18.3f} "
              f"{np.mean(maxes):>12.1f}")

    # ---------------------------------------------------------------- #
    # 3. Is the residual structured, or is it noise? A learnable field
    #    should correlate with something.
    # ---------------------------------------------------------------- #
    print(f"\n--- is the residual structured? (TF + (1/9) vW, xc on) ---")
    print(f"  Pearson r of the residual against candidate explanatory fields,")
    print(f"  averaged over structures. |corr| near 1 means a local functional")
    print(f"  of that field could reproduce it.\n")

    def corr(a, b):
        a, b = a.flatten(), b.flatten()
        a = a - a.mean(); b = b - b.mean()
        return (a @ b / (a.norm() * b.norm())).item()

    fields = {"rho": [], "rho^(1/3)": [], "rho^(2/3)": [], "v_ext": [],
              "v_H": [], "tau": [], "tau/rho": []}
    for record in records:
        rho, vext, tau, cell = load(record)
        kin = (thomas_fermi_potential(rho)
               + (1.0 / 9.0) * von_weizsacker_potential(rho, cell))
        total = kin + vext + hartree_potential(rho, cell) + xc_potential(rho, "pbe", cell=cell)
        r = total - total.mean()
        safe = rho.clamp_min(1e-8)
        fields["rho"].append(corr(r, rho))
        fields["rho^(1/3)"].append(corr(r, safe.pow(1 / 3)))
        fields["rho^(2/3)"].append(corr(r, safe.pow(2 / 3)))
        fields["v_ext"].append(corr(r, vext))
        fields["v_H"].append(corr(r, hartree_potential(rho, cell)))
        fields["tau"].append(corr(r, tau))
        fields["tau/rho"].append(corr(r, tau / safe))
    for name, values in fields.items():
        print(f"  {name:<12} {np.mean(values):>8.4f}  "
              f"(spread {np.std(values):.4f})")
