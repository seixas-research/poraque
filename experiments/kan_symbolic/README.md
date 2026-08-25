# Reading a symbolic expression out of a trained KAN

```bash
conda run -n poraque python experiments/kan_symbolic/demo.py
```

## Two different things both called "symbolic"

`poraque.ml.symbolic` (`SymbolicDistiller`, used by `experiments/euler_lagrange`)
**searches** the space of short algebraic expressions for one that
reproduces what a trained *operator* computes -- a regression against a
black box, over probe points.

`poraque.ml.kan.symbolic_expression`, demonstrated here, is not a search.
Every learnable activation in `poraque.ml.kan` already *is* a fixed
functional form -- SiLU plus a residual built from a small, explicit set of
learned numbers (Chebyshev coefficients, spline coefficients, RBF weights,
or a rational function's numerator/denominator coefficients), or (with
`kan_use_base: false`) the residual alone, no base function at all. Reading
the symbolic function one channel computes is therefore a **readout of
parameters that were symbolic all along**, not a fit to anything. This is
the literal content of the claim that a KAN is interpretable: the
interpretation was never hidden inside the weights, it *is* the weights.

The scope is deliberately narrow: one channel, one Fourier layer's
elementwise nonlinearity `sigma_c` in `v[l] = v[l-1] + sigma(y[l])`. It says
nothing about the spectral weights, the pointwise mixing, or how the 16 (or
so) channels combine across three layers into a prediction of the charge
density -- that whole-operator question is what `poraque.ml.symbolic`
answers, by search, and only for the semi-local GGA/reduced-gradient
features it is given.

## What the demo does

For each of the eight checkpoints trained in this repo -- the four
base-carrying variants (`models/au_w16_m8_l3_kancheby`, `_kanbspline`,
`_rbf`, `_rational`) and their four `kan_use_base: false` ("pure") twins
(`models/au_w16_m8_l3_<variant>_purekan`; see `FUTURE.md` for how all eight
compare on accuracy), it:

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
`x/2*(1+erf(x/sqrt(2)))`, which did -- see FUTURE.md's "Base-function
correction"):

```
kan_cheby:
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
kan_cheby (pure):
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
accuracy levels -- see `poraque.ml.kan`'s "close to silu at init" property,
which training only partially erodes) or about whether any of this readout
*explains* the accuracy differences between activations or between
base-carrying and pure mode. That would need reading every channel of every
layer across all eight checkpoints and is future work, not attempted here.
