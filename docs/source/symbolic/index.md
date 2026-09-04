# Symbolic distillation

A Fourier Neural Operator reproduces $\rho \mapsto \tau$ to a few percent and
explains nothing. Symbolic distillation searches short algebraic expressions
for one that reproduces the same mapping — trading accuracy for a formula you
can read, publish, and check against known physics.

```{note}
Off by default, and an optional install. Switch it on with
`enable: true` in the `symbolic` block of the config, or with `--symbolic` on
the command line. It was `enable_symbolic_distillation` until 26.9.8, which
restated its own section in its own name; the old spelling now raises and says
so.
```

```bash
pip install -e ".[symbolic]"     # sympy, for LaTeX and the limit checks
poraque-train --config configs/train.yaml --task chg2tau --symbolic
```

The extra is one package, and only for *reporting*: SymPy renders the result
as LaTeX and checks its asymptotic limits. The search itself needs nothing that
training does not. Without the extra everything else in the package works
unchanged, and a run with distillation enabled reports the missing dependency
and keeps its trained model rather than failing.

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

### Feature schemes and templates

Two independent knobs: `features` picks the **input variables**, `template`
picks how the **target is factorised**.

| `features` | Variables given to the engine |
| --- | --- |
| `gga` (default) | `rho`, `p`, `q` |
| `reduced` | `p`, `q` |
| `raw` | `rho`, `grad_rho`, `lap_rho` (dimensional) |

| `template` | Fitted target | Reported formula |
| --- | --- | --- |
| `none` (default) | $\tau$ | $\tau = f(\dots)$ |
| **`pauli`** | $F = \dfrac{\tau - \tau_\mathrm{vW}}{\tau_\mathrm{TF}}$ | $\tau = \tau_\mathrm{vW} + \tau_\mathrm{TF}\,f(\dots)$ |
| `thomas_fermi` | $F = \tau/\tau_\mathrm{TF}$ | $\tau = C_\mathrm{TF}\rho^{5/3}\,f(\dots)$ |

`pauli` is the physically right choice. $\tau_\mathrm{vW} =
|\nabla\rho|^2/8\rho$ is known in closed form and $\tau - \tau_\mathrm{vW}
\ge 0$ by Hoffmann–Ostenhof, so subtracting it leaves exactly the quantity a
kinetic functional actually has to model. Leaving it in means fitting something
mostly already known — and near the von Weizsäcker limit, fitting a
near-cancellation between two large numbers.

A template **gives away the part of the physics that is already known**.
Thomas–Fermi supplies the density scaling exactly, so the search stops spending
its budget rediscovering $\rho^{5/3}$ and works on what is actually unknown —
and every constant it must find becomes order unity. It is also the form the
literature writes kinetic functionals in, so the answer is directly comparable:
Thomas–Fermi is $F = 1$, von Weizsäcker is $F = 5p^2/3$.

The discovered expression is multiplied back into the template before it is
reported, so the console, the JSON and the PDF all show the complete physical
formula rather than the factor that was fitted.

The pairing the shipped config uses is `features: reduced` with
`template: pauli` — the Pauli enhancement factor $F_\theta(p, q)$ on the two
reduced variables and nothing else:

```yaml
symbolic:
  features: reduced        # inputs: p, q
  template: pauli          # target: F = (tau - tau_vW) / tau_TF
```

$\rho$ is dropped deliberately rather than merely omitted. $p$ and $q$ are
invariant under the coordinate scaling that fixes $T_s$, so a dimensionless
$F$ *cannot* depend on the density; offering $\rho$ as a third variable gives
the search a way to fit the particular densities in the dataset and nothing
else. What comes back is the kinetic energy density in full,

$$
\tau = \tau_\mathrm{vW}[\rho] + C_\mathrm{TF}\,\rho^{5/3}\,F_\theta(p, q),
$$

built from two terms already known in closed form and one that was searched
for.

```{note}
`features: enhancement` is kept as an alias for `features: reduced` plus
`template: pauli`, which is exactly what it used to mean when one name
selected both. It therefore **overrides** any `template` set beside it — the
explicit pair says the same thing and cannot silently ignore half of itself.
```

### Operator constraints

```yaml
constraints:
  "^": [-1, 1]
```

Per-operator limits on argument complexity, passed straight to the engine. The
default holds the **exponent** of `^` to a single node while leaving the base
unconstrained (`-1`).

Unconstrained exponents are the main source of nonsense from a power operator:
a fractional power of a negative quantity leaves the reals, and an exponent
that is itself a subtree is unreadable and almost never physical. Real
functionals have simple exponents — $5/3$, $4/3$, $2$.

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
  enable: false
  target: model          # model | reference
  features: gga          # gga | reduced | raw
  template: none         # none | pauli | thomas_fermi
  epsilon: 1.0e-08       # vacuum threshold, e/a0^3
  constraints:           # per-operator argument-complexity limits
    "^": [-1, 1]
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

## The Pareto knee

The front states a trade and refuses to resolve it. The **knee** is where
resolving it costs least: the candidate nearest the ideal corner $(0, 0)$ once
both axes are rescaled to $[0, 1]$ across the front,

$$
d_i = \sqrt{\hat c_i^{\,2} + \hat{\mathcal{L}}_i^{\,2}}, \qquad
\hat x_i = \frac{x_i - \min x}{\max x - \min x} .
$$

Both rescalings are necessary — a node count and a loss have no common unit,
so a distance between them is meaningless until they are made comparable.

The loss is compared on a **logarithmic** scale. A front spans orders of
magnitude, and on a linear scale every candidate but the most accurate
collapses onto $\hat{\mathcal{L}} \approx 1$; the knee would then be the
shortest expression regardless of what it cost.

```python
from poraque.ml.symbolic import pareto_knee

knee = pareto_knee(result.pareto)
knee["complexity"], knee["loss"], knee["distance"]
```

`pareto_knee` also writes `distance` onto every entry of the front in place, so
the report's table can show the column it ranked on and the figure and the
table cannot disagree.

The run prints both candidates:

```text
  pareto knee  : complexity 7 nodes, loss 0.021  (lowest loss: 18 nodes, 0.0188)
```

A knee is a heuristic, not a theorem. With one candidate it is that candidate;
where the front is a straight line every point is equidistant. It answers
"which expression is worth its length", not "which expression is right" — that
is what the asymptotic limits are for.

## Rounding

Constants are rounded to **three decimal places** wherever an expression is
displayed. A search returns them at full float precision —
`0.33333334326744079589843750` is a real example — and three of those in one
formula overrun the width of a PDF page.

Three places is also what the numbers are worth: the search is stochastic and
its constants move in the third place between seeds, so printing twenty digits
states a precision the method does not have. `expression_to_latex(...,
decimals=None)` keeps them verbatim.

## Physical asymptotic compliance

A symbolic fit is a numerical statement until it is checked against physics it
was never shown. Every candidate on the front is tested against the two limits
that pin a kinetic functional, both written for the enhancement factor
$F = \tau/\tau_\mathrm{TF}$:

Both are statements about the **Pauli** enhancement factor
$F = (\tau - \tau_\mathrm{vW})/\tau_\mathrm{TF}$:

| Limit | Condition | Physical regime |
| --- | --- | --- |
| Thomas–Fermi | $F(0,0) = 1$ | uniform density: $\tau_\mathrm{vW}\to0$, $\tau\to\tau_\mathrm{TF}$ |
| von Weizsäcker | $F \to 0$ as $p \to \infty$ | single orbital: $\tau\to\tau_\mathrm{vW}$, nothing left over |

Every template is converted to this one convention before checking, using
$\tau_\mathrm{vW}/\tau_\mathrm{TF} = 5p^2/3$ exactly — so a `thomas_fermi`
or `none` fit is held to the same physical standard.

The check is analytic — `sympy.limit`, with the symbols bound at parse time —
falling back to a converged numerical probe when SymPy cannot resolve a deeply
nested expression. Which route was used is recorded, because an analytic limit
is a proof and a numerical one is evidence.

```{important}
**Neither textbook functional passes both.** As a Pauli factor Thomas–Fermi is
$F = 1 - 5p^2/3$ — correct at the origin, divergent at infinity; von Weizsäcker
is $F = 0$ — correct at infinity, wrong at the origin. Failing one limit is
therefore not damning on its own. Failing *both* means the expression
reproduces the training data without the physics that constrains it outside
that range, and it should not be extrapolated.

A form that passes both, such as $F = e^{-p^2}$ or $F = 1/(1+p^2)$,
interpolates between the two regimes — which is what a usable semi-local
kinetic functional has to do.
```

Whether $F$ tends to *any* finite limit is reported apart from whether that
limit is zero: a functional settling on a finite non-zero constant stays
bounded but never reduces to von Weizsäcker, which is a repairable failure —
unlike one that diverges.

Each front entry carries a `TF`/`vW` badge in the console, the JSON summary and
the PDF:

```text
  accuracy/complexity front (TF/vW = asymptotic limits satisfied):
      nodes          loss   limits  expression
          1          0.45    TF/--  1.0
          5          0.09    TF/--  1 + 5*p**2/27 + 20*q/9
          9         0.012    TF/vW  1 + 5*p**2/3
  2 of 4 candidates satisfy BOTH limits; the simplest is:
      1 + 5*p**2/3
```

The most accurate expression is frequently the least physical, so the compliant
candidates are listed separately — a slightly worse expression that obeys both
limits is usually the better functional. The PDF report gets a **Physical
asymptotic compliance** section with the per-limit findings and a score.

## Figures

Three are written after a search, all embedded in the report's Symbolic
Distillation section:

| File | Shows |
| --- | --- |
| `<task>_symbolic_pareto.png` | the front, with the lowest-loss candidate and the knee marked |
| `<task>_symbolic_parity.png` | the **lowest-loss** expression against the DFT reference |
| `<task>_symbolic_parity_knee.png` | the **knee** expression, on identical voxels |

The two parity plots are laid side by side in the report, because the question
they answer is a comparison: does the shorter formula give anything away? Where
the clouds are indistinguishable, the extra nodes bought nothing.

Both are evaluated on the **held-out** structures and compared against the DFT
reference — not against whatever was fitted, since with `target: model` the two
are different things. With no validation split they fall back to the fitted
voxels and the caption says so.

Read them beside the operator's own parity plot: the gap between them is what
the closed form gives up. A formula that tracks the identity line through the
bulk and bends away at the high-density end is the usual picture — semi-local
features cannot resolve the core peaks.

```python
from poraque.ml.symbolic import check_asymptotic_limits

result = check_asymptotic_limits("exp(-p**2)", ["rho", "p", "q"],
                                 scheme="reduced", template="pauli")
result.passes        # True
result.badge()       # 'TF/vW'
```

## Engine

[`poraque.ml.gp`](../api/index.rst) is the backend: genetic programming over
expression trees in NumPy, with SciPy fitting the constants of the final front.
Populations evolve under tournament selection with a small elite, reseeding
around it when one stops improving, and the accuracy/complexity front is kept
as the best expression at every size.

It replaced [PySR](https://github.com/MilesCranmer/PySR) on 2026-09-03. PySR is
excellent and searches harder per second than this does; it also installs a
**Julia toolchain** on first use, which on a supercomputer is a network fetch
from a compute node, a writable depot on a filesystem that may be read-only or
purged between jobs, a precompilation pass per architecture, and a second
language runtime inside an MPI job — for minutes of work at the end of a run
that already took hours.

**The objective is unchanged, term for term.** The physics is enforced as
fitness over the same probe points, with the same weights and the same penalty
ceiling, so a constrained `loss` from this engine is comparable with one from
the old.

The engine is still injected rather than imported at module scope, so a
different backend — PySR included, for anyone who wants it and has the
toolchain — stays a parameter rather than a rewrite:

```python
from poraque.ml.symbolic import SymbolicDistiller, build_features

table = build_features(rho, tau, grid, scheme="gga")
result = SymbolicDistiller(config, engine=my_engine).fit(table)
```

```{note}
The search is stochastic, but a `seed` is now an unconditional promise: it runs
in one process, so the same seed gives the same front and `deterministic` is
accepted and ignored rather than traded against parallelism. Two *different*
seeds will still differ, so treat a single expression as one sample rather than
as *the* answer.
```
