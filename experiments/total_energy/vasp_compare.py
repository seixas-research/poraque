"""Poraque's energy terms against VASP's own decomposition, on the DFT fields.

No model here: this asks whether the energy *expression* is right, using the
reference density and tau. Whatever it gets wrong, a predicted field can only
get more wrong.
"""
import re

RUN = "data/vasp/structures/structure_0000"
POTCARS = "/Users/leseixas/Simulations/vasp/POTCARs"

from poraque.fields import ChargeDensity, ExternalPotential, KineticEnergyDensity
from poraque.fields.vasp import Potcar
from poraque.physics import EnergyCalculator

text = open(f"{RUN}/OUTCAR", errors="replace").read()
vasp = {}
for key in ("PSCENC", "TEWEN", "DENC", "XCENC", "EBANDS", "EATOM", "EENTRO"):
    m = re.findall(rf"{key}\s*=\s*(-?\d+\.\d+)", text)
    vasp[key] = float(m[-1])
m = re.findall(r"PAW double counting\s*=\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)", text)
vasp["PAW"] = float(m[-1][0]) + float(m[-1][1])
vasp["TOTEN"] = float(re.findall(r"free  energy   TOTEN\s*=\s*(-?\d+\.\d+)", text)[-1])

chg = ChargeDensity.read(f"{RUN}/CHGCAR")
tau = KineticEnergyDensity.read(f"{RUN}/TAUCAR", grid=chg.grid)
potcar = Potcar.from_library(POTCARS, chg.structure.symbols)
ext = ExternalPotential.from_potcar_tables(chg.structure, chg.grid, potcar)
pscore = {e.element: e.pscore for e in potcar if e.pscore is not None}
calc = EnergyCalculator.from_potential(ext, pscore=pscore, functional="pbe")
c = calc.compute(chg, tau, ext)

natoms = sum(chg.structure.counts)
print(f"{RUN}   {natoms} atoms, {c.n_electrons:.4f} electrons "
      f"(nominal {c.nominal_electrons})\n")

print("terms with an exact VASP counterpart")
print(f"  {'term':<26} {'Poraque':>16} {'VASP':>16} {'diff':>12} {'rel':>10}")
for label, ours, theirs in (
        ("alpha Z    (PSCENC)", c.alpha_z, vasp["PSCENC"]),
        ("Ewald      (TEWEN)",  c.ewald,   vasp["TEWEN"]),
        ("Hartree    (-DENC)",  c.hartree, -vasp["DENC"])):
    d = ours - theirs
    print(f"  {label:<26} {ours:>16.4f} {theirs:>16.4f} {d:>12.4f} "
          f"{abs(d/theirs):>10.2e}")

print("\nterms with no direct counterpart (VASP splits them differently)")
print(f"  {'kinetic  int(tau)':<26} {c.kinetic:>16.4f}")
print(f"  {'external int(rho V)':<26} {c.external:>16.4f}")
print(f"  {'xc':<26} {c.xc:>16.4f}   VASP XCENC {vasp['XCENC']:>12.4f}")

print("\ntotals")
print(f"  {'Poraque pseudo-valence':<30} {c.total:>16.4f} eV   "
      f"{c.total/natoms:>10.4f} eV/atom")
print(f"  {'VASP TOTEN':<30} {vasp['TOTEN']:>16.4f} eV   "
      f"{vasp['TOTEN']/natoms:>10.4f} eV/atom")
print(f"  {'difference':<30} {c.total - vasp['TOTEN']:>16.4f} eV   "
      f"{(c.total - vasp['TOTEN'])/natoms:>10.4f} eV/atom")
print("\n  what VASP has and Poraque does not:")
print(f"    {'EATOM (atomic reference)':<30} {vasp['EATOM']:>16.4f}")
print(f"    {'PAW double counting':<30} {vasp['PAW']:>16.4f}")
print(f"    {'sum':<30} {vasp['EATOM'] + vasp['PAW']:>16.4f} eV   "
      f"{(vasp['EATOM'] + vasp['PAW'])/natoms:>10.4f} eV/atom")
print(f"    {'unexplained remainder':<30} "
      f"{c.total - vasp['TOTEN'] + vasp['EATOM'] + vasp['PAW']:>16.4f} eV")

# Where the remainder comes from, checked rather than asserted.
# VASP:    EBANDS = T_s + int(rho V_H) + int(rho v_xc) + int(rho V_ext^loc) + E_nl
# Poraque: kinetic + external = T_s + int(rho V_ext^loc)
# so the two bookkeepings of the same block differ by E_nl plus the PAW
# one-centre kinetic/external pieces folded into EBANDS.
int_rho_vh = 2.0 * c.hartree                      # E_H = 1/2 int(rho V_H)
int_rho_vxc = c.xc - vasp["XCENC"]                # XCENC = E_xc - int(rho v_xc)
vasp_block = vasp["EBANDS"] - int_rho_vh - int_rho_vxc
ours_block = c.kinetic + c.external
print("\n  the remainder, accounted for:")
print(f"    {'VASP  T_s + ext + nonlocal':<34} {vasp_block:>14.4f}")
print(f"    {'Poraque T_s + ext (local only)':<34} {ours_block:>14.4f}")
print(f"    {'difference':<34} {vasp_block - ours_block:>14.4f}")
print(f"    {'remainder above':<34} "
      f"{-(c.total - vasp['TOTEN'] + vasp['EATOM'] + vasp['PAW']):>14.4f}")
print(f"    {'entropy EENTRO (in TOTEN, not here)':<34} {vasp['EENTRO']:>14.4f}")
