"""
Can a third operator learn the Euler-Lagrange residual, and does it transfer?

The construction. At the ground state the EL equation holds pointwise,

    dTs/drho + v_ext + v_H[rho] + v_xc[rho] = mu ,

and every term but the first is known exactly. So on reference data the
equation can be *inverted* for the one term that has no label:

    v_kin_exact = -(v_ext + v_H + v_xc)      (up to the constant mu)

This is the exact kinetic potential of the ground-state density. It is not
obtainable from TAUCAR: a functional derivative needs a functional, and tau is
a field. The EL equation is the only route to it.

The operator therefore learns a correction to a known baseline,

    dv[rho] = v_kin_exact - v_baseline ,     v_baseline = TF + lam*vW

as a functional of rho ALONE. Not of v_ext: T_s[rho] is universal, and a
"functional" that consumed the external potential would not be one.

Why this is a clean test. Everything is zero-mean, so if dv is predicted
exactly the residual vanishes identically, and the error in the residual IS
the error in dv. That makes the baseline unambiguous:

    std(dv_target)             = the residual with no correction
    std(dv_target - dv_pred)   = the residual with it

and their ratio is the operator's skill, with no metric to argue about.

Two controls decide whether the architecture earns its place:
  * predicting the training-set mean field (does it beat "learn nothing"?)
  * a semi-local fit on (rho, |grad rho|, lap rho) at each point, which is a
    GGA-level functional. The FNO is only worth having if it beats this,
    because the exact kinetic potential is known to be non-local.
"""
import argparse
import os
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#: Prepared cache to read. Override with PORAQUE_EL_CACHE to point at another
#: resolution or another dataset without editing this file.
CACHE = os.environ.get("PORAQUE_EL_CACHE",
                       os.path.join(REPO, "data", "cache", "res32_potcar_spin"))

from poraque.fields import ChargeDensity, ExternalPotential
from poraque.ml.data import discover_materials
from poraque.ml.fno import FNO3d
from poraque.ml.physics import (
    hartree_potential,
    spectral_gradient,
    spectral_laplacian,
    thomas_fermi_potential,
    von_weizsacker_potential,
    xc_potential,
)

SEED = 42
LAM = 1.0 / 3.0          # best classical baseline; see el_xc.py
XC = "pbe"               # the functional the reference data was computed with
HOLDOUT = 4              # of 17 structures


def zero_mean(field):
    return field - field.mean(dim=(-3, -2, -1), keepdim=True)


def build(record, dtype=torch.float64):
    """rho, the correction target dv, and the cell, for one material."""
    chg = ChargeDensity.read(record.files["CHGCAR"])
    ext = ExternalPotential.read(record.files["EXTCAR"], grid=chg.grid)

    rho = torch.tensor(np.asarray(chg.data), dtype=dtype)[None, None]
    vext = torch.tensor(np.asarray(ext.data), dtype=dtype)[None, None]
    cell = torch.tensor(np.asarray(chg.grid.cell), dtype=dtype)[None]

    # The EL inversion: everything on the right is exact.
    known = vext + hartree_potential(rho, cell) + xc_potential(rho, XC, cell=cell)
    v_kin_exact = zero_mean(-known)

    baseline = zero_mean(thomas_fermi_potential(rho)
                         + LAM * von_weizsacker_potential(rho, cell))
    return rho, zero_mean(v_kin_exact - baseline), cell


def features(rho, cell):
    """Semi-local descriptors: the hypothesis space of a GGA."""
    grad = spectral_gradient(rho.squeeze(1), cell).pow(2).sum(1).sqrt()
    lap = spectral_laplacian(rho.squeeze(1), cell)
    return torch.stack([rho.squeeze(1), grad, lap], dim=1)


def split(records):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(records))
    held = {records[i].identifier for i in order[:HOLDOUT]}
    return ([r for r in records if r.identifier not in held],
            [r for r in records if r.identifier in held])


def report(name, targets, predictions):
    """Residual before and after, in eV, plus the skill ratio."""
    rows = []
    for target, prediction in zip(targets, predictions):
        before = target.std().item()
        after = (target - prediction).std().item()
        rows.append((before, after))
    before = float(np.mean([r[0] for r in rows]))
    after = float(np.mean([r[1] for r in rows]))
    print(f"  {name:<34} {before:>10.3f} {after:>10.3f} "
          f"{before / max(after, 1e-12):>9.2f}x")
    return before, after


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    torch.set_num_threads(6)

    records = discover_materials(CACHE)
    train_records, test_records = split(records)
    print(f"{len(train_records)} training structures, "
          f"{len(test_records)} held out "
          f"({', '.join(r.identifier for r in test_records)})")
    print(f"baseline kinetic functional: TF + {LAM:.3f} vW, xc = {XC.upper()}\n")

    train = [build(r) for r in train_records]
    test = [build(r) for r in test_records]

    def cast(items):
        return [(r.float(), d.float(), c.float()) for r, d, c in items]

    train_f, test_f = cast(train), cast(test)

    # Normalise: the network sees standardised fields, the metrics are in eV.
    rho_all = torch.cat([r.flatten() for r, _, _ in train])
    dv_all = torch.cat([d.flatten() for _, d, _ in train])
    rho_mean = rho_all.mean().float()
    rho_std = rho_all.std().float()
    dv_std = dv_all.std().float()
    print(f"  target dv: std {dv_std:.3f} eV over the training set")
    print("  (this is the uncorrected Euler-Lagrange residual)\n")

    header = (f"  {'model':<34} {'before':>10} {'after':>10} {'skill':>10}")

    # ------------------------------------------------------------------ #
    # Control 1: predict the mean. Any model must beat this.
    # ------------------------------------------------------------------ #
    print("HELD OUT (eV, std of the Euler-Lagrange residual)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    # The targets are zero-mean by construction and the grids differ between
    # structures, so the mean-field control degenerates to predicting zero,
    # which is exactly "apply no correction" and scores 1.00x by definition.
    report("no correction (baseline)",
           [d for _, d, _ in test], [torch.zeros_like(d) for _, d, _ in test])

    # ------------------------------------------------------------------ #
    # Control 2: semi-local (GGA-level) least squares on rho, |grad|, lap.
    # ------------------------------------------------------------------ #
    def design(rho, cell):
        f = features(rho, cell).reshape(3, -1).T
        return torch.cat([torch.ones(f.shape[0], 1, dtype=f.dtype),
                          f, f.pow(2), f[:, :1].pow(1 / 3),
                          f[:, :1].pow(2 / 3)], dim=1)

    A = torch.cat([design(r, c) for r, _, c in train])
    b = torch.cat([d.flatten() for _, d, _ in train])
    coefficients = torch.linalg.lstsq(A, b.unsqueeze(1)).solution
    report("semi-local least squares (GGA)",
           [d for _, d, _ in test],
           [(design(r, c) @ coefficients).reshape(d.shape)
            for r, d, c in test])

    # ------------------------------------------------------------------ #
    # The operator.
    # ------------------------------------------------------------------ #
    model = FNO3d(in_channels=1, out_channels=1, width=args.width,
                  modes=args.modes, n_layers=args.layers,
                  projection_channels=64)

    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs)

    start = time.perf_counter()
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_f))
        total = 0.0
        for i in order:
            rho, dv, cell = train_f[i]
            optimiser.zero_grad()
            prediction = model((rho - rho_mean) / rho_std, cell) * dv_std
            loss = (prediction - dv).pow(2).mean() / dv_std.pow(2)
            loss.backward()
            optimiser.step()
            total += loss.item()
        schedule.step()
        if (epoch + 1) % max(1, args.epochs // 8) == 0:
            with torch.no_grad():
                held = np.mean([
                    ((model((r - rho_mean) / rho_std, c) * dv_std - d)
                     .std() / d.std()).item() for r, d, c in test_f])
            print(f"    epoch {epoch + 1:>4}  train {total / len(train_f):.4f}"
                  f"   held-out rel {held:.4f}")
    elapsed = time.perf_counter() - start

    with torch.no_grad():
        predictions = [model((r - rho_mean) / rho_std, c) * dv_std
                       for r, _, c in test_f]
        train_pred = [model((r - rho_mean) / rho_std, c) * dv_std
                      for r, _, c in train_f]
    print()
    report(f"FNO (w{args.width} m{args.modes} L{args.layers})",
           [d for _, d, _ in test_f], predictions)
    print()
    report("  ... same model, TRAINING fit",
           [d for _, d, _ in train_f], train_pred)
    print(f"\n  {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{elapsed:.0f} s")
