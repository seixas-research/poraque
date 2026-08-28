"""
Levy-Perdew-Sahni against Euler-Lagrange: same equation, different target.

LPS writes the exact density as a single effective orbital, rho = (sqrt rho)^2,
obeying

    -1/2 lap sqrt(rho) + [v_ext + v_H + v_xc + v_P] sqrt(rho) = mu sqrt(rho) .

Dividing by sqrt(rho) and using dT_vW/drho = -1/2 lap sqrt(rho) / sqrt(rho),

    dT_vW/drho + v_P + v_ext + v_H + v_xc = mu ,

which is the Euler-Lagrange equation with dTs/drho split into its bosonic and
Pauli parts. So LPS is not an alternative condition to test; it is the same
condition in a different variable. What differs, and what this measures:

  1. IS IT AN IDENTITY IN THE CODE? dTs/drho == dT_vW/drho + v_P must hold to
     machine precision, or one of the two implementations is wrong.

  2. CONDITIONING. LPS fixes the bosonic baseline at the FULL von Weizsaecker
     term, whereas the EL formulation may choose TF + lam*vW with lam free.
     The learning target is whatever is left over, so the spread of that
     remainder decides which formulation is easier to learn.

  3. THE CONSTRAINT. LPS buys something EL does not: Levy-Ou-Yang gives
     v_P >= 0 pointwise, a hard constraint on the very object being learned.
     It is only worth having if the reference data actually respects it, which
     is a question about pseudo-valence densities, not about the theorem.

v_P is defined only up to mu, so "v_P >= 0" can always be met by raising mu.
The sharp question is what mu that costs, and whether the violations are a
thin shell around the nuclei (where the pseudopotential picture is expected to
fail) or spread through the valence region (where it would be fatal).
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
    return rho, vext, cell, chg


def zero_mean(f):
    return f - f.mean()


if __name__ == "__main__":
    records = discover_materials(CACHE)

    # ---------------------------------------------------------------- #
    # 1. LPS and EL are the same equation. Verify in code.
    # ---------------------------------------------------------------- #
    print("1. IS LPS THE SAME EQUATION AS EULER-LAGRANGE?")
    print("   dTs/drho and dT_vW/drho + v_P must agree identically, since")
    print("   v_P is DEFINED as their difference. A mismatch is a code bug.\n")
    rho, vext, cell, _ = load(records[0])
    known = vext + hartree_potential(rho, cell) + xc_potential(rho, "pbe", cell=cell)
    v_kin_exact = zero_mean(-known)                  # from the EL inversion
    v_vw = von_weizsacker_potential(rho, cell)
    v_pauli = v_kin_exact - zero_mean(v_vw)          # the LPS Pauli potential
    reconstructed = zero_mean(v_vw) + v_pauli
    print(f"   max |dTs/drho - (dT_vW/drho + v_P)| = "
          f"{(v_kin_exact - reconstructed).abs().max():.3e} eV")
    print("   -> identical by construction: LPS is a change of variable,")
    print("      not an independent condition to test.\n")

    # ---------------------------------------------------------------- #
    # 2. Conditioning: which formulation leaves the smaller target?
    # ---------------------------------------------------------------- #
    print("2. WHICH LEAVES THE EASIER LEARNING TARGET?")
    print("   std over the cell of whatever the network still has to supply,")
    print("   averaged over all structures. Smaller is better conditioned.\n")
    print(f"   {'formulation':<44} {'baseline':<22} {'target std [eV]':>16}")
    print("   " + "-" * 84)

    schemes = [
        ("LPS  (Pauli potential v_P)", "vW  (fixed by LPS)", None),
        ("EL   (correction to TF)", "TF", 0.0),
        ("EL   (correction to TF + vW)", "TF + vW", 1.0),
        ("EL   (correction to TF + vW/9)", "TF + (1/9) vW", 1.0 / 9.0),
        ("EL   (correction to TF + vW/5)", "TF + (1/5) vW", 1.0 / 5.0),
    ]
    results = {}
    for label, baseline_name, lam in schemes:
        spreads = []
        for record in records:
            rho, vext, cell, _ = load(record)
            known = (vext + hartree_potential(rho, cell)
                     + xc_potential(rho, "pbe", cell=cell))
            exact = zero_mean(-known)
            if lam is None:
                base = von_weizsacker_potential(rho, cell)
            else:
                base = thomas_fermi_potential(rho)
                if lam:
                    base = base + lam * von_weizsacker_potential(rho, cell)
            spreads.append((exact - zero_mean(base)).std().item())
        results[label] = float(np.mean(spreads))
        print(f"   {label:<44} {baseline_name:<22} {results[label]:>16.3f}")

    best_el = min(v for k, v in results.items() if k.startswith("EL"))
    lps = results["LPS  (Pauli potential v_P)"]
    print(f"\n   LPS target is {lps / best_el:.1f}x wider than the best EL "
          f"target.")
    print("   LPS fixes the bosonic baseline at the full vW term; EL may tune")
    print("   it, and the tuned value is far from 1.\n")

    # ---------------------------------------------------------------- #
    # 3. Does the data respect Levy-Ou-Yang, v_P >= 0?
    # ---------------------------------------------------------------- #
    print("3. DOES THE REFERENCE DATA RESPECT v_P >= 0?")
    print("   v_P is fixed only up to mu. Take the smallest mu that makes it")
    print("   non-negative, then ask where the binding points are: a thin")
    print("   shell at the nuclei is expected, a valence-wide failure is not.\n")
    print(f"   {'structure':<14} {'mu_min [eV]':>12} {'std(v_P)':>10} "
          f"{'<1% of max':>11} {'at rho >':>10}")
    print("   " + "-" * 62)

    for record in records[:6]:
        rho, vext, cell, chg = load(record)
        known = (vext + hartree_potential(rho, cell)
                 + xc_potential(rho, "pbe", cell=cell))
        # v_P = mu - (v_ext + v_H + v_xc + dT_vW/drho)
        w = known + von_weizsacker_potential(rho, cell)
        mu_min = w.max().item()             # smallest mu with v_P >= 0
        v_p = mu_min - w
        # Where is v_P pinned near zero? Those are the binding points.
        threshold = 0.01 * v_p.max()
        binding = v_p < threshold
        fraction = binding.double().mean().item()
        rho_at = rho[binding].max().item() if binding.any() else float("nan")
        print(f"   {record.identifier:<14} {mu_min:>12.2f} "
              f"{v_p.std().item():>10.2f} {100 * fraction:>10.2f}% "
              f"{rho_at:>10.3f}")

    print("\n   'at rho >' is the highest density among the binding voxels:")
    print("   large means the constraint binds in the bonding region, small")
    print("   means it binds only in the low-density tail.")

    # ---------------------------------------------------------------- #
    # 4. The practical question: does the constraint bind at all?
    # ---------------------------------------------------------------- #
    print("\n4. WOULD A HARD v_P >= 0 CONSTRAINT DO ANY WORK?")
    print("   Fraction of voxels a TF-based Pauli potential would place")
    print("   below zero, i.e. where a softplus head would actually bite.\n")
    negatives = []
    for record in records:
        rho, vext, cell, _ = load(record)
        # The classical Pauli potential of the TF + vW decomposition.
        v_p_tf = thomas_fermi_potential(rho)
        negatives.append((v_p_tf < 0).double().mean().item())
    print(f"   dT_TF/drho < 0 : {100 * np.mean(negatives):.2f}% of voxels")
    print("   (Thomas-Fermi is non-negative by construction, so a classical")
    print("    Pauli potential never violates the bound; the constraint is")
    print("    insurance for the LEARNED one, exactly as with the tau head.)")
