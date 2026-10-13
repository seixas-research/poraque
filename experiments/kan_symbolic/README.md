# KAN activations and symbolic distillation (archived)

On 2026-10-13, from version 26.10.6, two features were taken out of the
`poraque` package and put here: the **Kolmogorov-Arnold (KAN) activations**
(`poraque.ml.kan`) and **symbolic distillation** (`poraque.ml.symbolic`,
`poraque.ml.gp`). They are kept whole, with their tests, their configs and
their documentation, so that nothing has to be rediscovered if either returns.
Nothing here is imported by the package.

## Why

**KAN.** `silu` beat `gelu` and every KAN variant on this project's Pt data;
dropping the base term always lost. The measurement tables are below; they
are the only record, since the checkpoints behind them were deleted. A learnable activation per
channel bought cost (up to 5.5x per step for the B-spline variant) and no
accuracy.

**Symbolic distillation.** It fitted `pseudo2tau` only: a short algebraic form
for the Pauli enhancement factor, found by genetic programming with the
physical limits as fitness. It is being set aside, not abandoned, and may come
back once the operator it distills is accurate enough to be worth distilling.

What the package does now: `model.activation` (under `ext2chg:` or `chg2tau:`
since the 2026-10-13 schema) accepts `silu`, `gelu`, `relu`
and `tanh` (`poraque.ml.fno.ACTIVATIONS`); `model.kan_setup` and the whole
`symbolic:` section raise, naming this directory; the `--symbolic*` flags of
`poraque-train`, the `symbolic distillation` profiling stage, the
`SYMBOLIC DISTILLATION <TASK>` log block, the PDF report's symbolic section
and the metrics JSON's `symbolic` key are gone, and so is the `[symbolic]`
(SymPy) install extra.

## Measurements (2026-08-15 → 17)

The checkpoints behind these tables were deleted; the tables are the only
record. Protocol throughout: `ext2pseudo`, `width 16, modes 8, n_layers 3`,
resolution 32, `seed 42`, `epochs 500, early_stopping 300`, the same 14/3
structure split of the *early* 17-structure platinum set. **One seed per row.**
Measured seed spread on this operator is 1–53 % (median ~12 %), so the ordering
is evidence, not proof.

Held-out relative L², SiLU-base KAN variants (the design in the tree today):

| activation | rel L² | R² | best epoch | wall-clock (MPS) |
|---|---|---|---|---|
| `silu` | **0.02337** | 0.99910 | 340 | 507 s |
| `kan_rbf` | 0.02398 | 0.99905 | 150 | 734 s |
| `kan_chebyshev` | 0.02464 | 0.99899 | 370 | 663 s |
| `kan_rational` | 0.02467 | 0.99899 | 260 | 697 s |
| `gelu` | 0.02504 | 0.99896 | 330 | ≈500 s |
| `kan_bspline` | 0.02834 | 0.99867 | 350 | 2580 s |

`kan_setup.use_base: false` ("pure" KAN) against the base-carrying row:

| activation | with base | pure |
|---|---|---|
| `kan_bspline` | 0.02834 | 0.03137 |
| `kan_rbf` | 0.02398 | 0.03712 |
| `kan_rational` | 0.02467 | 0.03860 |
| `kan_chebyshev` | 0.02464 | 0.03915 |

CPU forward+backward cost, single 24³ sample, relative to `gelu`: `silu` 1.00×,
`kan_chebyshev` 1.36×, `kan_rational` 1.54×, `kan_rbf` 2.00×, `kan_bspline`
5.50×.

`pseudo2tau` 5-fold CV, `silu` 0.0351 ± 0.0027 vs `gelu` 0.0364 ± 0.0029, `silu`
ahead on 4 of 5 folds — **measured on the invalid pre-2026-08-27 τ and void as
a τ result**; the `silu > gelu` ordering rests on the `ext2pseudo` rows, which
never read a `TAUCAR`.

Conclusions carried: `silu` is the default; three of four KAN variants beat
`gelu`; none beats `silu`; dropping the base term always loses; `kan_bspline`
tolerates a pure form best and is worst with one.

## Layout

| Path | Holds |
| --- | --- |
| `src/kan.py`, `symbolic.py`, `gp.py` | the three modules, as they were in `src/poraque/ml/` |
| `src/symbolic_config.py` | the `symbolic` section of `ml/config.py` |
| `src/report_symbolic.py` | the cut-out report code (`TrainingReport.pareto`, the PDF section) |
| `src/train_symbolic.py` | the post-fit stage and CLI flags of `scripts/poraque_train.py` |
| `tests/` | `test_kan.py`, `test_gp.py`, `test_symbolic.py`, `test_report_symbolic.py`, and a `conftest.py` that puts `../src` on `sys.path` |
| `configs/` | the eight `compare_kan_*_ext2pseudo.yaml` comparison runs |
| `docs/` | `symbolic_manual.md` (the Manual's old chapter) and `06_symbolic.tex` (the user guide's) |
| `demo.py` | the KAN readout demo described below |

## What runs, and what does not

- `kan.py`, `symbolic.py` and `gp.py` import as **top-level modules** from
  `src/` (`import kan`), not as `poraque.ml.*`.
- `python -m pytest experiments/kan_symbolic/tests` passes the symbolic, GP and
  KAN-module tests. The **43 tests that build an `FNO3d` or a configuration
  with a KAN activation fail**: `FNO3d` no longer accepts one.
- `report_symbolic.py` and `train_symbolic.py` are verbatim cut-outs. They lean
  on helpers that stayed behind and **do not run**; `symbolic_config.py` still
  imports `split_enable_block` from `poraque.ml.config`.
- The configs in `configs/` here are rejected by the current schema (`kan_setup`
  is a retired key, and the top-level `model:`, `task:` and `output:` sections
  they use went with the 2026-10-13 rewrite).
- `demo.py` loads four checkpoints that have been deleted, and those are in the
  retired `poraque-bundle-1` layout besides. The sample output below is the
  only record of what it printed.

## Bringing KAN back

`FNO3d` would have to take an activation **module or factory** instead of a
name looked up in `ACTIVATIONS` (a KAN activation is sized by the layer it sits
in: `C` functions in a Fourier block, `C_P` in the read-out), the checkpoint's
architecture record would have to store the factory's arguments, and the
configuration would need a key to carry them (the old one was
`model.kan_setup`, selected by `activation: kan`). Then `src/kan.py` moves back
to `src/poraque/ml/` and `tests/` back to `tests/`. For symbolic distillation,
restore the `symbolic` section from `symbolic_config.py`, remove it from
`RETIRED_SECTIONS`, and put the stage in `train_symbolic.py` back into
`run_task`. Do it against two seeds: measured seed spread is 1-53 %.

# Reading a symbolic expression out of a trained KAN

Two different things were called "symbolic". `src/symbolic.py`
(`SymbolicDistiller`) **searched** the space of short algebraic expressions for
one that reproduces what a trained *operator* computes: a regression against a
black box, over probe points. `kan.symbolic_expression`, demonstrated by
`demo.py`, is not a search. Every learnable activation in `kan.py` already
*is* a fixed functional form -- SiLU plus a residual built from a small,
explicit set of learned numbers (Chebyshev coefficients, spline coefficients,
RBF weights, or a rational function's numerator/denominator coefficients), or
(with `use_base: false`) the residual alone. Reading the function one channel
computes is therefore a **readout of parameters that were symbolic all along**,
not a fit to anything. The scope is deliberately narrow: one channel, one
Fourier layer's elementwise nonlinearity `sigma_c` in
`v[l] = v[l-1] + sigma(y[l])`. It says nothing about the spectral weights, the
pointwise mixing, or how the channels combine into a prediction of the charge
density.

## What the demo does

For each of the eight checkpoints trained in this repo -- the four
base-carrying variants (`models/pt_w16_m8_l3_kanchebyshev`, `_kanbspline`,
`_rbf`, `_rational`) and their four `kan_use_base: false` ("pure") twins
(`models/pt_w16_m8_l3_<variant>_purekan`; see the measurements above for how all
eight compare on accuracy), it:

1. reads channel 0 of Fourier layer 0's activation straight off the real
   trained weights, as a closed-form expression;
2. runs the real operator on `struct_016` -- a structure held out of every
   one of these eight training runs (see the per-structure validation
   tables in each run's log) -- and captures the actual pre-activation
   values a real forward pass sends into that channel, via a forward hook;
3. evaluates the symbolic expression at those exact values and diffs it
   against what `forward()` actually produced there.

## Sample output (2026-08-17, layer 0, channel 0, 4-decimal coefficients)

Base-carrying (`kan_use_base: true`, the default) -- every expression opens
with the SiLU base term, `x/(1+exp(-x))`, no `erf` anywhere (SiLU's exact
closed form needs none; an earlier version of this readout used GELU,
`x/2*(1+erf(x/sqrt(2)))`, which did):

```
kan_chebyshev:
  phi(x) = 0.9929*x/(1+exp(-x)) + 1.3152*tanh(x)**6 - 0.4272*tanh(x)**5
           - 2.08*tanh(x)**4 + 0.0892*tanh(x)**3 + 0.4974*tanh(x)**2
           + 0.2118*tanh(x) + 0.1765
  max |forward - symbolic| on 8 real struct_016 voxels: 1.1e-04

kan_rbf:
  phi(x) = 0.9895*x/(1+exp(-x))
           + 0.0447*exp(-2.0*(x+2.0)**2) + 0.0298*exp(-2.0*(x+1.5)**2)
           - 0.0041*exp(-2.0*(x+1.0)**2) - 0.0067*exp(-2.0*(x+0.5)**2)
           + 0.001*exp(-2.0*(x-0.5)**2)  + 0.0042*exp(-2.0*(x-1.0)**2)
           + 0.0415*exp(-2.0*(x-1.5)**2) + 0.0235*exp(-2.0*(x-2.0)**2)
           + 0.0408*exp(-2.0*x**2)
  max |forward - symbolic| on 8 real struct_016 voxels: 8.0e-05

kan_rational:
  phi(x) = 0.9815*x/(1+exp(-x))
           + (0.0316*x**4 - 0.0456*x**3 - 0.0555*x**2 + 0.0009*x + 0.0544)
             / (0.1088*x**8 + 0.225*x**6 + 0.148*x**4 + 0.1988*x**2 + 1)
  max |forward - symbolic| on 8 real struct_016 voxels: 7.3e-05

kan_bspline: (see "Why kan_bspline prints differently" below)
  phi(x) = 0.9599*SiLU(x) + sum_i a_i * B_i(clamp(x, -2.0, 2.0))
  max |forward - symbolic| on 8 real struct_016 voxels: 2.0e-05
```

Pure (`kan_use_base: false`) -- the same four channels, from the separate
checkpoints trained with the base term switched off. The SiLU term is gone
**entirely**, not zeroed out -- `kan_rbf (pure)`'s expression still contains
`exp`, but only from its own Gaussian residual, never from a base function:

```
kan_chebyshev (pure):
  phi(x) = -0.1408*tanh(x)**6 - 0.1664*tanh(x)**5 + 0.0648*tanh(x)**4
           + 0.0084*tanh(x)**3 - 0.1864*tanh(x)**2 + 0.0632*tanh(x) + 0.1672
  max |forward - symbolic| on 8 real struct_016 voxels: 1.3e-04

kan_rbf (pure):
  phi(x) = -0.0898*exp(-2.0*(x+2.0)**2) - 0.0971*exp(-2.0*(x+1.5)**2)
           - 0.1194*exp(-2.0*(x+1.0)**2) - 0.0903*exp(-2.0*(x+0.5)**2)
           - 0.0191*exp(-2.0*(x-0.5)**2) + 0.0063*exp(-2.0*(x-1.0)**2)
           + 0.0674*exp(-2.0*(x-1.5)**2) + 0.0904*exp(-2.0*(x-2.0)**2)
           - 0.0066*exp(-2.0*x**2)
  max |forward - symbolic| on 8 real struct_016 voxels: 5.0e-05

kan_rational (pure):
  phi(x) = (0.043*x**4 - 0.023*x**3 - 0.0755*x**2 + 0.0254*x + 0.0386)
           / (0.0252*x**8 + 0.1426*x**6 + 0.0894*x**4 + 0.1708*x**2 + 1)
  max |forward - symbolic| on 8 real struct_016 voxels: 5.2e-05

kan_bspline (pure):
  phi(x) = sum_i a_i * B_i(clamp(x, -2.0, 2.0))    -- no base_weight at all
  max |forward - symbolic| on 8 real struct_016 voxels: 2.3e-05
```

Every error is consistent with rounding coefficients to 4 decimals for
readability, not with any mismatch in the readout itself --
`symbolic_expression(..., decimals=None)` (full float32 precision) agrees
with `forward()` to ~1e-6, the unit tests in `tests/test_kan.py::
TestSymbolicExpression` check this at points from -6 to +6, well outside
every variant's clamp/decay range, not only the well-behaved interior.

## Why `kan_bspline` prints differently

A degree-3 B-spline residual with `grid_size=8` is, honestly, **11
coefficients against a fixed, known basis of piecewise cubics** -- not one
algebraic line. `symbolic_expression` still returns the exact SymPy
`Piecewise` (built by literally running the Cox-de Boor recursion
symbolically, three levels deep, over 9 knot intervals), and it evaluates
correctly and quickly (`sympy.lambdify` + a numeric point is a simple
branch-selection). But the *unfolded* tree prints as a ~23,000-character
expression, and `sympy.piecewise_fold` -- which merges nested
`Piecewise`-of-`Piecewise`-of-`Piecewise` into one flat case list -- blows
up combinatorially trying to collapse it (minutes, not seconds, on this
grid). The demo deliberately never calls `piecewise_fold`; for `kan_bspline`
it prints the coefficient vector against the known basis instead, which is
the same information in the form that is actually readable. This is a
genuine, structural difference between the four variants, not a limitation
of the readout: a Chebyshev/RBF/rational residual really is one formula, a
B-spline residual really is a coefficient table.

## Caveat

Read for one (layer, channel) pair on one trained checkpoint. It says
nothing about which channels' learned functions departed meaningfully from
their base function (most channels, most layers, stay close to it at these
accuracy levels -- see `src/kan.py`'s "close to silu at init" property,
which training only partially erodes) or about whether any of this readout
*explains* the accuracy differences between activations or between
base-carrying and pure mode. That would need reading every channel of every
layer across all eight checkpoints and is future work, not attempted here.
