# Symbolic distillation

A Fourier Neural Operator reproduces $\rho \mapsto \tau$ to a few percent and
explains nothing. Symbolic distillation searches short algebraic expressions
for one that reproduces the same mapping — trading accuracy for a formula you
can read, publish, and check against known physics.

```{note}
Off by default, and an optional install. Switch it on with
`enable_symbolic_distillation: true` in the `symbolic` block of the config, or
with `--symbolic` on the command line.
```

```bash
pip install -e ".[symbolic]"     # PySR + sympy; not installed by default
poraque-train --config configs/train_config.yaml --task chg2tau --symbolic
```

The extra is separate because PySR carries a **Julia toolchain**, which it
fetches the first time a search runs. Without it everything else in the package
works unchanged, and a run with distillation enabled reports the missing
dependency and keeps its trained model rather than failing.

## What it can and cannot find

The features are evaluated **at a point**, so the search space is exactly the
semi-local functionals. Whatever the operator learned that is *non-local*
cannot be expressed in that space at all, and appears as irreducible residual.

That makes a poor fit informative rather than disappointing: it puts a number
on how much of the learned map is not semi-local. A near-perfect fit would say
the operator found nothing a GGA-level functional could not have.

Only `chg2tau` is attempted. `ext2chg` is the Hohenberg–Kohn map, whose whole
content is non-local — a semi-local expression for it would be meaningless
rather than merely inaccurate.

## The physical variables

Raw $\rho$ is not enough to build a robust functional. The engine receives the
density together with its **dimensionless reduced derivatives**, which is the
form a semi-local kinetic functional is actually written in:

$$
k_F = (3\pi^2\rho)^{1/3},
\qquad
p = \frac{|\nabla\rho|}{2 k_F \rho},
\qquad
q = \frac{\nabla^2\rho}{4 k_F^2 \rho}.
$$

| Variable | Name in the equation | Meaning |
| --- | --- | --- |
| $\rho$ | `rho` | electron density, atomic units ($e/a_0^3$) |
| $p$ | `p` | reduced gradient — the GGA variable |
| $q$ | `q` | reduced Laplacian — the meta-GGA variable |

Those names are passed to the engine verbatim, so they are the names that
appear in the result.

Why the reduced forms rather than $|\nabla\rho|$ and $\nabla^2\rho$ directly:
they are dimensionless and invariant under the coordinate scaling
$\rho_\lambda(\mathbf r) = \lambda^3\rho(\lambda \mathbf r)$ that fixes $T_s$,
so a functional expressed in them holds at every density scale instead of
having to be rediscovered at each. Raw derivatives make every coefficient carry
units, which a genetic search spends its budget approximating.

### Derivatives are spectral

$\nabla \to i\mathbf{G}$ and $\nabla^2 \to -|\mathbf{G}|^2$, evaluated by FFT.
This is *exact* for the band-limited periodic fields a plane-wave grid carries;
a finite-difference stencil would introduce an error that the search would then
model as physics.

### Feature schemes

| `features` | Variables | Target |
| --- | --- | --- |
| `gga` (default) | `rho`, `p`, `q` | $\tau$ |
| `enhancement` | `p`, `q` | $F = \tau/\tau_\mathrm{TF}$ |
| `raw` | `rho`, `grad_rho`, `lap_rho` | $\tau$ |

`enhancement` is worth knowing: it is the form the literature writes kinetic
functionals in, so the answer is directly comparable — Thomas–Fermi is $F = 1$
and von Weizsäcker is $F = 5p^2/3$ — and every constant to be found is order
unity.

## Regularization in vacuum

$p$ and $q$ both divide by $\rho$, which decays exponentially into vacuum. Two
things happen, and both are needed:

1. **Every denominator is clamped** at `epsilon`, so $k_F$, $p$ and $q$ stay
   finite even where the density underflows.
2. **Voxels with $\rho \le$ `epsilon` are dropped.** In vacuum $p$ and $q$ are
   ratios of two vanishing numbers — noise with a plausible magnitude, which
   corrupts a fit far more quietly than a `NaN` would. Vacuum carries no
   information about the functional, so nothing is lost by removing it.

`epsilon` defaults to `1e-8`, in atomic units ($e/a_0^3$). The mask also
removes the slightly negative voxels that band-limiting leaves around the core
peaks (Gibbs ringing), where a fractional power of $\rho$ would be complex.

```{tip}
For a bulk crystal nothing is dropped — the minimum density is far above the
threshold. It matters for slabs, molecules and anything with real vacuum.
```

## Configuration

```yaml
symbolic:
  enable_symbolic_distillation: false
  target: model          # model | reference
  features: gga          # gga | enhancement | raw
  epsilon: 1.0e-08       # vacuum threshold, e/a0^3
  unary_operations: [exp, log, sqrt, abs]
  binary_operations: ['+', '-', '*', /, ^]
  iterations: 40
  population_size: 33
  populations: 15
  max_depth: 10
  max_size: 30
  parsimony: 0.0032
  n_samples: 4000
  seed: 0
```

`target` selects what is fitted. `model` distils the trained operator's own
predictions — what the network learned, faithful to it including its errors.
`reference` fits the DFT data directly, which is plain symbolic regression
against ground truth and answers a different question.

`unary_operations` and `binary_operations` are passed to the engine untouched.
Keep the alphabet small: the search space grows combinatorially, and an
operator that cannot appear in the answer only dilutes the population. `/` and
`log` are the usual sources of singularities on a density approaching zero.

`n_samples` caps the rows handed to the search. A single 32³ structure is
already 32 768 voxels and the cost is linear in them.

## Output

The expression, its accuracy/complexity front, and the fit quality are printed
to the terminal, written to the run's JSON summary, and typeset into the PDF
report as display mathematics with $\rho$, $p$ and $q$ rendered properly.

```text
  expression   : F = (q * 2.454276) + 0.9163395
  variables    : p, q  [dimensionless (F = tau / tau_TF)]
  complexity   : 5 nodes
  fit          : R2 0.9487   relative L2 0.0736
```

**Read the front, not just the winner.** A single expression hides the trade
that produced it; the front shows what each extra node bought.

## Engine

[PySR](https://github.com/MilesCranmer/PySR) is the backend, installed with the
`symbolic` extra. It fetches a Julia toolchain the first time a search runs, so
the first distillation of a session is slower than the rest.

The engine is injected rather than imported at module scope, so everything
except the search itself works and is testable without it:

```python
from poraque.ml.symbolic import SymbolicDistiller, build_features

table = build_features(rho, tau, grid, scheme="gga")
result = SymbolicDistiller(config, engine=my_engine).fit(table)
```

```{warning}
The search is stochastic. Two runs with the same `seed` can still differ unless
PySR is also put in deterministic, serial mode — it warns about this itself.
Treat a single expression as one sample, not as *the* answer.
```
