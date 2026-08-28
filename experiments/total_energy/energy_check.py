"""
Can the trained pair produce a total energy, and how close is it?

Two reference points, and they answer different questions.

**The same expression on the DFT fields.** Every term this expression omits --
the PAW one-centre energies, the non-local pseudopotential -- is identical
whether it is handed a predicted density or a reference one, so it cancels
exactly in the difference. What is left is purely the model error propagated
into an energy, which is the number that says whether the operators are good
enough for energetics.

**VASP's own TOTEN**, for the one structure that ships an OUTCAR. That
comparison carries a large fixed offset, and the offset is bookkeeping rather
than error -- see vasp_compare.py, which accounts for it term by term.

The chain consumes no DFT field: V_ext is built from the POTCAR tables, and
rho and tau are predicted from it.
"""
import os
import re
import sys

import numpy as np

BUNDLE, CACHE = sys.argv[1], sys.argv[2]
POTCARS = "/Users/leseixas/Simulations/vasp/POTCARs"

from poraque.fields import ChargeDensity, ExternalPotential, KineticEnergyDensity
from poraque.fields.vasp import Potcar
from poraque.ml import load_bundle
from poraque.ml.data import discover_materials
from poraque.physics import EnergyCalculator
from poraque.physics.energy import total_density

records = discover_materials(CACHE)
ext2chg = load_bundle(BUNDLE, "ext2chg", device="cpu")
chg2tau = load_bundle(BUNDLE, "chg2tau", device="cpu")
print(f"{len(records)} materials from {CACHE}\n")

potcar = None
rows = []
for record in records:
    chg = ChargeDensity.read(record.files["CHGCAR"])
    tau = KineticEnergyDensity.read(record.files["TAUCAR"], grid=chg.grid)
    if potcar is None:
        potcar = Potcar.from_library(POTCARS, chg.structure.symbols)

    # Built here rather than read from the cache: the constructed field carries
    # the valence charges, and without them there is no Ewald term and no
    # nominal electron count -- so no total energy, only a fragment of one.
    ext = ExternalPotential.from_potcar_tables(chg.structure, chg.grid, potcar)
    pscore = {e.element: e.pscore for e in potcar if e.pscore is not None}
    calc = EnergyCalculator.from_potential(ext, pscore=pscore, functional="pbe")

    # --- the chain: V_ext -> rho -> tau. No DFT field is consumed. ---
    rho_p = ext2chg.predict(ext)
    # The electron count is fixed by the pseudopotentials and is the one part
    # of the prediction that is known exactly, so it is imposed before any
    # 1e4 eV electrostatic term integrates the density. This is what the ASE
    # calculator does; not doing it here would measure a different pipeline.
    rho_p = rho_p.normalized(calc.nominal_electrons)
    tau_p = chg2tau.predict(rho_p)

    ref = calc.compute(chg, tau, ext)
    pred = calc.compute(rho_p, tau_p, ext)
    l2_rho = (np.linalg.norm(np.asarray(total_density(rho_p)) - chg.data)
              / np.linalg.norm(chg.data))
    l2_tau = (np.linalg.norm(np.asarray(total_density(tau_p)) - tau.data)
              / np.linalg.norm(tau.data))
    rows.append((record.identifier, ref, pred, chg, l2_rho, l2_tau))

natoms = sum(rows[0][3].structure.counts)
print(f"{'material':<18} {'E_ref [eV]':>14} {'E_pred [eV]':>14} {'diff [eV]':>11} "
      f"{'meV/atom':>10} {'relL2 rho':>10} {'relL2 tau':>10}")
print("-" * 92)
for name, ref, pred, _, l2r, l2t in rows:
    d = pred.total - ref.total
    print(f"{name:<18} {ref.total:>14.3f} {pred.total:>14.3f} {d:>11.3f} "
          f"{1000 * d / natoms:>10.1f} {l2r:>10.4f} {l2t:>10.4f}")

d = np.array([r[2].total - r[1].total for r in rows])
print(f"\n{len(d)} structures: mean {d.mean():+.3f} eV  std {d.std():.3f}  "
      f"|max| {np.abs(d).max():.3f}")
print(f"  per atom: mean {1000 * d.mean() / natoms:+.1f}  "
      f"std {1000 * d.std() / natoms:.1f}  "
      f"|max| {1000 * np.abs(d).max() / natoms:.1f} meV/atom")

print("\n--- energy differences between structures ---")
ref_t = np.array([r[1].total for r in rows])
pred_t = np.array([r[2].total for r in rows])
pairs = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))]
err = np.array([abs((pred_t[i] - pred_t[j]) - (ref_t[i] - ref_t[j]))
                for i, j in pairs])
spread = ref_t.max() - ref_t.min()
print(f"  reference spread over the set : {spread:.3f} eV "
      f"({1000 * spread / natoms:.1f} meV/atom)")
print(f"  |error| in dE over {len(pairs)} pairs : mean {err.mean():.3f} eV  "
      f"median {np.median(err):.3f}  max {err.max():.3f}")
print("  a constant offset would give zero here; it does not cancel because "
      "the\n  error varies from structure to structure.")

name, ref, pred, chg, _, _ = rows[0]
print(f"\n--- components, {name} (eV) ---")
print(f"  {'term':<16} {'reference':>15} {'predicted':>15} {'diff':>12}")
for term in ("kinetic", "external", "alpha_z", "hartree", "xc", "ewald"):
    a, b = getattr(ref, term), getattr(pred, term)
    if a is None:
        print(f"  {term:<16} {'--':>15} {'--':>15}")
        continue
    print(f"  {term:<16} {a:>15.3f} {b:>15.3f} {b - a:>12.3f}")
print(f"  {'TOTAL':<16} {ref.total:>15.3f} {pred.total:>15.3f} "
      f"{pred.total - ref.total:>12.3f}")
print(f"  electrons: reference {ref.n_electrons:.4f}  predicted "
      f"{pred.n_electrons:.4f}  nominal {ref.nominal_electrons}")

outcar = "data/vasp/structures/structure_0000/OUTCAR"
if os.path.exists(outcar):
    text = open(outcar, errors="replace").read()
    toten = float(re.findall(r"free  energy   TOTEN\s*=\s*(-?\d+\.\d+)",
                             text)[-1])
    for nm, ref, pred, ch, _, _ in rows:
        if nm != "structure_0000":
            continue
        n = sum(ch.structure.counts)
        print("\n--- structure_0000, against VASP ---")
        print(f"  VASP TOTEN                    {toten:>14.3f} eV  "
              f"{toten / n:>9.4f} eV/atom")
        print(f"  Poraque on DFT fields         {ref.total:>14.3f} eV  "
              f"{ref.total / n:>9.4f} eV/atom")
        print(f"  Poraque on PREDICTED fields   {pred.total:>14.3f} eV  "
              f"{pred.total / n:>9.4f} eV/atom")
        print(f"  model error                   "
              f"{pred.total - ref.total:>14.3f} eV  "
              f"{1000 * (pred.total - ref.total) / n:>9.1f} meV/atom")
        print(f"  fixed offset vs TOTEN         {ref.total - toten:>14.3f} eV  "
              f"(bookkeeping; see vasp_compare.py)")
