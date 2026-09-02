# -*- coding: utf-8 -*-
# file: demo.py
"""
Read a symbolic expression straight out of a trained KAN activation.

This is a *readout*, not a fit: every learnable variant in
``poraque.ml.kan`` already is a fixed functional form (SiLU plus a small
residual built from stored coefficients), so `symbolic_expression` builds a
SymPy expression directly from a channel's own `state_dict` entries -- no
symbolic-regression search involved. That is the different, more literal
sense of "KAN interpretability" from `poraque.ml.symbolic`, which fits a
short expression to what a whole trained *operator* computes; see that
module (and its own `experiments/euler_lagrange`) for the search-based kind.

For every one of the eight checkpoints trained in this repo -- the four
base-carrying variants (`kan_cheby`, `kan_bspline`, `kan_rbf`,
`kan_rational`; `models/pt_w16_m8_l3_<variant>/`) and their four
`kan_use_base: false` ("pure") twins (`models/pt_w16_m8_l3_<variant>_purekan/`)
-- this script:

1. reads channel 0 of Fourier layer 0's activation off the real trained
   weights, as a closed-form SymPy expression;
2. runs the real operator on a real, held-out platinum structure
   (`struct_016`, never trained on by any of these eight runs -- see
   FUTURE.md's per-structure validation tables) and captures the actual
   pre-activation values a real forward pass sends into that channel;
3. evaluates the symbolic expression at those exact values and diffs it
   against what the network's own `forward()` produced there.

The base-carrying and pure printouts make concrete what
`kan_use_base: false` actually removes from the expression: a pure
channel's `phi(x)` has no SiLU base term at all -- no `exp` from it, only
whatever the residual itself is built from (which for `kan_rbf` is *also*
built from `exp`, so "no exp" is not the right thing to look for there;
what's actually absent is the linear `w_c * x/(1+exp(-x))` term).

Run from the repository root, in the `poraque` conda env::

    conda run -n poraque python experiments/kan_symbolic/demo.py
"""

import sys
from pathlib import Path

import numpy as np
import sympy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from poraque.fields import ExternalPotential  # noqa: E402
from poraque.ml import load_bundle  # noqa: E402
from poraque.ml.kan import BSplineKANActivation, symbolic_expression  # noqa: E402

BASE_CHECKPOINTS = {
    "kan_cheby": "models/pt_w16_m8_l3_kancheby/pt_w16_m8_l3_kancheby.poraque",
    "kan_bspline": "models/pt_w16_m8_l3_kanbspline/pt_w16_m8_l3_kanbspline.poraque",
    "kan_rbf": "models/pt_w16_m8_l3_rbf/pt_w16_m8_l3_rbf.poraque",
    "kan_rational": "models/pt_w16_m8_l3_rational/pt_w16_m8_l3_rational.poraque",
}
PURE_CHECKPOINTS = {
    "kan_cheby (pure)": "models/pt_w16_m8_l3_kancheby_purekan/pt_w16_m8_l3_kancheby_purekan.poraque",
    "kan_bspline (pure)": "models/pt_w16_m8_l3_kanbspline_purekan/pt_w16_m8_l3_kanbspline_purekan.poraque",
    "kan_rbf (pure)": "models/pt_w16_m8_l3_kanrbf_purekan/pt_w16_m8_l3_kanrbf_purekan.poraque",
    "kan_rational (pure)": "models/pt_w16_m8_l3_kanrational_purekan/pt_w16_m8_l3_kanrational_purekan.poraque",
}
STRUCTURE = "data/cache/res32_potcar/struct_016/EXTCAR"
LAYER, CHANNEL, N_VOXELS = 0, 0, 8


def show(name, rel_path, field, x_symbol):
    operator = load_bundle(str(ROOT / rel_path), "ext2chg", device="cpu")
    activation = operator.model.blocks[LAYER].activation

    # Capture what the real forward pass actually sends into (and gets
    # back from) this channel -- no synthetic input anywhere.
    captured = {}

    def hook(module, inputs, output, captured=captured):
        captured["in"] = inputs[0].detach().clone()
        captured["out"] = output.detach().clone()

    handle = activation.register_forward_hook(hook)
    operator.predict(field)
    handle.remove()

    x_real = captured["in"][0, CHANNEL].flatten()[:N_VOXELS]
    y_real = captured["out"][0, CHANNEL].flatten()[:N_VOXELS]

    expr = symbolic_expression(activation, CHANNEL, decimals=4)
    # sympy.lambdify evaluates the raw (unfolded) Piecewise tree just
    # fine -- a simple depth-first branch selection, same cost as the
    # tensor forward pass. What does NOT scale is sympy.piecewise_fold,
    # which tries to *merge* nested Piecewise-of-Piecewise-of-Piecewise
    # into one flat case list: for kan_bspline's 3-level Cox-de Boor
    # recursion over 9 basis functions that is a combinatorial blow-up
    # (minutes, not seconds) for no benefit here, so it is deliberately
    # never called -- see the printing branch below instead.
    fn = sympy.lambdify(x_symbol, expr, modules=["scipy", "numpy"])
    y_symbolic = np.asarray(fn(x_real.numpy().astype(float)), dtype=float)

    print(f"=== {name}  (layer {LAYER}, channel {CHANNEL}, "
          f"use_base={activation.use_base}) ===")
    if isinstance(activation, BSplineKANActivation):
        # A B-spline KAN's "closed form" is honestly a coefficient
        # vector against a fixed, known basis, not one algebraic line --
        # printing the ~23,000-character unfolded Piecewise would be
        # exact but unreadable, and folding it is what hangs. This is
        # the same information, just in the form it is actually usable.
        coeffs = activation.spline_coeff[CHANNEL].tolist()
        low, high = activation.grid_range
        base_term = (f"{activation.base_weight[CHANNEL].item():.4f}*SiLU(x) + "
                    if activation.use_base else "")
        print(f"phi(x) = {base_term}sum_i a_i * B_i(clamp(x, {low}, {high}))")
        print(f"  B_i: degree-{activation.spline_order} B-spline basis, "
              f"{activation.n_basis} fixed knot intervals")
        print(f"  a_i = {[round(c, 4) for c in coeffs]}")
    else:
        print(f"phi(x) = {expr}")
        print()
        print(f"LaTeX:  {sympy.latex(expr)}")
    print()
    print(f"  {N_VOXELS} real pre-activation values from struct_016 "
          f"(held out of this run's training split):")
    print(f"    x        = {np.round(x_real.numpy(), 4)}")
    print(f"    forward  = {np.round(y_real.numpy(), 6)}")
    print(f"    symbolic = {np.round(y_symbolic, 6)}")
    print(f"    max |forward - symbolic| = "
          f"{np.abs(y_real.numpy() - y_symbolic).max():.3e}")
    print()


def main():
    field = ExternalPotential.read(str(ROOT / STRUCTURE))
    x_symbol = sympy.Symbol("x")

    print("############################################################")
    print("# Base-carrying (kan_use_base: true, the default)")
    print("############################################################")
    print()
    for name, rel_path in BASE_CHECKPOINTS.items():
        show(name, rel_path, field, x_symbol)

    print("############################################################")
    print("# Pure (kan_use_base: false) -- no SiLU base term at all")
    print("############################################################")
    print()
    for name, rel_path in PURE_CHECKPOINTS.items():
        show(name, rel_path, field, x_symbol)


if __name__ == "__main__":
    main()
