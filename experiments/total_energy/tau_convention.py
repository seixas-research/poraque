"""
Is TAUCAR multiplied by the cell volume? Four independent tests.

The density's convention is *known*, not assumed: CHGCAR holds rho*Omega, and
that is verified absolutely by the electron count coming out at exactly 320 =
32 x ZVAL(10). Everything below leans on that anchor.
"""
import numpy as np
from poraque.fields import ChargeDensity
from poraque.fields.vasp.volumetric import read_volumetric
from poraque.fields.density import von_weizsacker_tau, thomas_fermi_tau

HA, A0 = 27.211386245988, 0.529177210903
BULK = "data/vasp/structures/structure_0000"
ATOM = "data/vasp/isolated_atoms/Pt"


def raw_blocks(path):
    _, first, extra = read_volumetric(path, read_all=True)
    return np.asarray(first, float), [np.asarray(e, float) for e in extra]


def report(label, run, n_atoms, zval=10.0):
    rho_f = ChargeDensity.read(f"{run}/CHGCAR")
    grid = rho_f.grid
    rho = np.asarray(rho_f.data, float)          # e/Ang^3, convention verified
    b1, extra = raw_blocks(f"{run}/TAUCAR")
    raw = b1 + sum(extra)                        # the numbers in the file
    Om, ne = grid.volume, grid.integrate(rho)

    print(f"\n{'='*74}\n{label}   Omega = {Om:.1f} Ang^3   "
          f"electrons = {ne:.4f} (nominal {n_atoms*zval:.0f})")
    print(f"  TAUCAR blocks: {1+len(extra)}   "
          f"block2/block1 (mean) = {sum(extra).mean()/b1.mean():.5f}"
          if extra else "  TAUCAR blocks: 1")

    for name, tau in (("A. as written  (tau = file)", raw),
                      ("B. volume-scaled (tau = file/Omega)", raw / Om)):
        integral = grid.integrate(tau)
        per_e = integral / ne
        vw = von_weizsacker_tau(rho, grid)
        peak = rho.max()
        sig = rho > 1e-3 * peak
        viol = np.count_nonzero(sig & (tau < vw * 0.95 - 1e-6)) / max(sig.sum(), 1)
        tf = grid.integrate(thomas_fermi_tau(rho))
        print(f"\n  {name}")
        print(f"    int(tau)                 = {integral:14.3f} eV")
        print(f"    per valence electron     = {per_e:14.4f} eV "
              f"= {per_e/HA:.4f} Ha")
        print(f"    int(tau) / int(tau_TF)   = {integral/tf:14.4f}")
        print(f"    tau < tau_vW at          = {100*viol:13.3f} % of "
              f"significant points   <-- a THEOREM")
    return grid.integrate(raw) / ne, grid.integrate(raw / Om) / ne, Om


print(__doc__)
a_written, a_scaled, om_a = report("ISOLATED Pt ATOM", ATOM, 1)
b_written, b_scaled, om_b = report("BULK Pt, 32 atoms", BULK, 32)

print(f"\n{'='*74}")
print("TEST 4: the same physical quantity, in two cells of different volume.")
print(f"  The atom's box is {om_a/om_b:.3f}x the bulk cell. A per-electron")
print("  kinetic energy is a property of the electrons, not of the box, so")
print("  the two must agree under the CORRECT convention and must differ by")
print("  the volume ratio under the wrong one.\n")
print(f"  {'convention':<34} {'atom':>12} {'bulk':>12} {'ratio':>10}")
print(f"  {'A. as written  [eV/electron]':<34} {a_written:>12.4f} "
      f"{b_written:>12.4f} {a_written/b_written:>10.4f}")
print(f"  {'B. volume-scaled [eV/electron]':<34} {a_scaled:>12.4f} "
      f"{b_scaled:>12.4f} {a_scaled/b_scaled:>10.4f}")
print(f"  {'volume ratio (what B tracks)':<34} {'':>12} {'':>12} "
      f"{om_a/om_b:>10.4f}")

# Uniform-gas yardstick for "is 218 eV/atom high?"
rho_f = ChargeDensity.read(f"{BULK}/CHGCAR")
mean_rho_bohr = float(np.mean(rho_f.data)) * A0**3
rs = (3.0 / (4.0 * np.pi * mean_rho_bohr))**(1/3)
ueg = 0.3 * (9.0 * np.pi / 4.0)**(2/3) / rs**2          # Ha per electron
print(f"\n{'='*74}")
print("Is the value itself plausible?")
print(f"  mean valence density        {float(np.mean(rho_f.data)):.4f} e/Ang^3"
      f"  -> r_s = {rs:.3f} bohr")
print(f"  uniform electron gas T_s/N  {ueg*HA:.3f} eV/electron "
      f"({ueg:.4f} Ha)")
print(f"  Thomas-Fermi on this rho    "
      f"{rho_f.grid.integrate(thomas_fermi_tau(np.asarray(rho_f.data,float)))/320:.3f}"
      f" eV/electron")
print(f"  measured, convention A      {b_written:.3f} eV/electron")
print(f"  measured, convention B      {b_scaled:.5f} eV/electron")
