# Query by committee: measuring disagreement on 3D fields

**Scope:** `poraque.ml.committee`, `FieldOperator(init_seed=...)` ·
**Companion:** `docs/notes/roadmap.md`

## Why we want this

Reference data costs a plane-wave DFT run. With 12 structures of one element,
the binding question is *which structure to compute next* — and the honest
answer today is a guess. A committee turns that guess into a measurement.

The mechanism: train N operators that share their data, architecture and batch
order, differing only in `init_seed`. Where they agree, the data determined the
answer. Where they diverge, the initialisation did — which is to say the data
did not pin it down.

```bash
for s in 0 1 2 3 4; do
  poraque-train --config configs/train.yaml \
      --init-seed $s --json logs/committee_$s.json \
      --checkpoint-dir models/committee_$s
done

poraque-committee --models "models/committee_*" --task ext2chg
```

`init_seed` is deliberately **separate from `seed`**. `FieldOperator` saves and
restores the global RNG state around the weight draw, so members differ in
their weights and in nothing else. Had a single seed governed both, two members
would also see different batch orders, and their disagreement would mix
optimisation variance with a reshuffled dataset — measuring neither.

## What makes our setup unusual

Standard query-by-committee assumes a scalar output and takes the variance
across members. Here the output is a **3D field**, which changes what
disagreement even is: it is a field too. That is an advantage, and it means
there are four distinct things to measure, not one.

### 1. Pointwise spread — where the doubt lives

$$\sigma(\mathbf r) = \mathrm{std}_k\, f_k(\mathbf r)$$

A field, on the same grid as the prediction, written out in `CHGCAR` format
like anything else. It answers *where* the committee is unsure.

This is worth more than it sounds. We already know the failure modes are
spatially structured — the density peaks at the ionic cores, the Gibbs ringing
from band-limiting lives there too, and the Gaussian-blur study showed 50 % of
its total change inside 1 Å of an ion. If $\sigma(\mathbf r)$ concentrates on
the cores, the remedy is resolution or a better core treatment; if it spreads
through the interstitial region, the remedy is more chemistry. A scalar cannot
distinguish those.

### 2. Relative spread — comparable to the error we already quote

$$\frac{\|\sigma\|_2}{\|\bar f\|_2}$$

Deliberately the same shape as the relative $L^2$ the models are scored with,
so the two sit on one axis: a committee spread of 0.02 against a measured error
of 0.05 says immediately that the committee is under-reporting by 2.5×.

### 3. Integrated quantities — the ones the energy is built from

$\int\rho$ is the electron count, $\int\tau$ is the kinetic energy $T_s$. A
committee can be tight pointwise and still disagree on the integral, because a
small constant offset integrates. Both are reported and neither implies the
other.

This matters here specifically: the predicted electron count is already off by
0.4 %, and every electrostatic term in the energy is linear or quadratic in
$\rho$.

### 4. Energy spread — the operationally relevant number

Run each member's $(\rho, \tau)$ through `EnergyCalculator` and take the spread
of the totals. Because the energy is a near-cancellation of terms of order
$10^4$ eV, this is where a small field disagreement becomes a large one — and
it is the quantity a user of the ASE calculator actually consumes.

Expect it to be *much* larger in relative terms than the field spread. That is
not a defect of the measure; it is the cancellation problem restated.

### The chain compounds

`ext2chg → chg2tau` means an uncertain $\rho$ feeds an operator that is itself
uncertain. Two options, and they answer different questions:

- **Per-stage committees** — feed each member the *same reference* $\rho$.
  Isolates which of the two operators is the weak link.
- **End-to-end committees** — chain member $k$'s $\rho$ into member $k$'s
  $\tau$. Gives the uncertainty a user actually experiences.

Do the end-to-end one for active learning; the per-stage one when deciding
where to spend modelling effort.

## Validating it: the only question that matters

A disagreement measure is worthless until it is shown to **rank** correctly.
The check is a correlation between committee spread and true error across
held-out structures — `disagreement_error_correlation` returns both Pearson and
Spearman.

**Read Spearman.** Active learning consumes an ordering, and a measure can rank
perfectly while being badly scaled. A strong Pearson with a weak Spearman is
the opposite, and is not useful.

### The test we can run today, against known ground truth

We already measured something the committee should be able to rediscover: in
the 12-structure cross-validation, the two **32-atom** cells are 2.2–2.5×
harder than the ten 27-atom ones.

| Subset | `ext2chg` | `chg2tau` |
| --- | --- | --- |
| 27-atom (10 structures) | 0.0205 ± 0.0064 | 0.0355 ± 0.0069 |
| 32-atom (2 structures) | 0.0445 ± 0.0182 | 0.0894 ± 0.0420 |

So: **does a committee rank `struct_010` and `struct_011` as its two most
uncertain?** If yes, the measure has recovered a real generalisation gap
without being told about it, and is worth trusting to pick the next
calculation. If no, it is measuring optimisation noise and nothing else.

### Result: it failed (2026-08-04)

Four members, `--init-seed 0..3`, `--valid-fraction 0`, `ext2chg` at
resolution 32, scored against the cross-validated errors in
`logs/kfold12b.json`:

| Check | Result |
| --- | --- |
| Spearman, JSD vs cross-validated error | **+0.133** |
| Spearman, L² spread vs cross-validated error | **+0.091** |
| Rank of the two 32-atom cells (of 12) | #1 and **#5** |
| JSD / ln K, max over all structures | **5×10⁻⁶** (bound is 1) |

Both correlations are indistinguishable from zero. `struct_010` came first but
`struct_011` — equally a 32-atom cell, equally hard — landed mid-pack, which is
what chance looks like at n = 12.

The diagnostic is the last row. **JSD/ln K ≈ 5×10⁻⁶** means the committee is
degenerate: four members trained on the same data converge to nearly the same
function, so the spread is numerical noise, not uncertainty. The
over-confidence is quantified too — the L² spread is 0.72× the *training* error
but only 0.19× the cross-validated one, so it under-reports held-out error
five-fold.

**The test design was also flawed.** With `valid_fraction: 0` every member saw
every structure it was then scored on; asking a committee about data it
memorised guarantees agreement. That confounds "the measure is weak" with "the
test was unfair".

The honest form is a **per-fold committee**: for each CV fold, train K members
that all exclude that fold, and measure disagreement on structures none of them
saw. Cost is K× a K-fold run — about 2.5 hours for `ext2chg`.

Two readings survive, and both are testable:

1. **Init-seed variation is too weak a perturbation.** Likely. Bootstrapping
   the training structures, or varying `width`/`modes`, samples far more of the
   uncertainty and is the standard remedy.
2. **The test was unfair** and a per-fold committee would show real signal.

Faint counter-evidence worth keeping: the 32-atom mean JSD sits 36 % above the
27-atom mean (6.25×10⁻⁶ against 4.61×10⁻⁶), which is the right direction. Not
enough to act on.

Until one of those is resolved, **do not use committee disagreement to choose
the next calculation.**

## What this cannot tell us

> **Members differ only in initialisation, so they explore optimisation
> variance — not epistemic uncertainty over the data distribution.**

Three consequences, all of which bound the claim:

1. **Systematically over-confident.** Deep ensembles under-estimate error; the
   `ratio` field in `committee_spread` reports by how much, and it is expected
   to be below 1.
2. **Blind to missing chemistry.** Every member trained on the same twelve
   gold structures. Shown a silicon cell they may agree confidently and all be
   wrong together. No amount of committee size fixes this — it is a property of
   the training set, not the ensemble.
3. **Not an error bar.** Use it to order candidates, not to put $\pm$ on a
   number.

Widening the ensemble to also vary the architecture or bootstrap the training
structures would sample more of the uncertainty, at proportionally more
compute. Worth doing only after the ranking test above passes.

## Cost

N× training. With early stopping the folds converged in 90–120 epochs rather
than running to the epoch cap, so a 5-member committee at `resolution: 32` is
roughly 20 minutes per task on one GPU — comparable to a single K-fold run, and
cheaper than one DFT calculation on a 32-atom cell.
