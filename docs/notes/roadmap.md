# Poraquê roadmap

**Updated:** after the autograd functional derivative landed
**Companions:** `docs/notes/model2_architecture.md` (why Model 2 exists), `docs/notes/pi_fno.md` (physics-informed detail), `docs/notes/fno_physics.md` (DFT ↔ FNO mapping)

---

## Where the project actually stands

| Component | State | Evidence |
|---|---|---|
| Shared-grid field model | done | electron count exact: 297.0000 |
| Local pseudopotential reconstruction | done | rel. $L^2$ $2$–$6\times10^{-5}$ vs VASP, $r=1.000000$ |
| Code-agnostic ingestion | VASP done; QE/GPAW scaffolded | contract is four methods |
| Spectral resampling | done | integral preserved to $5.6\times10^{-17}$ |
| FNO, ragged grids | done | one model, 3 grid shapes, CPU/MPS agree to $10^{-6}$ |
| Universal training | done | `models/{ext2chg,chg2tau}.poraque` |
| K-fold cross-validation | done | 0.0295 ± 0.0025 / 0.0525 ± 0.0031 |
| Pauli-residual head | done | −18.3 % error, trains 17.8 % faster |
| Exact spectral $\nabla$, $\nabla^2$, $v_H$ | done | Poisson residual $10^{-12}$ |
| **$\delta T_s/\delta\rho$ by autograd** | **done** | TF exact to $5\times10^{-16}$; FD-verified |
| Joint two-model training | **not started** | ← the next real step |
| OF-DFT minimiser | not started | the test that matters |
| More than one element | **not started** | the binding constraint |

Two things are true at once: the infrastructure is essentially complete and
well-verified, and **nothing yet demonstrates the scientific claim**. The
roadmap below is ordered by which of those it moves.

---

## The three things that actually block progress

Everything else is refinement.

### B1. The dataset is five structures of one element

Every number in the project measures interpolation between nearby geometries of
platinum. No amount of architectural work changes what that supports. This is the
**single largest limitation** and it is a data problem, not a code problem —
the ingestion layer, the shape-bucketed sampler and the configuration system
were all built for a heterogeneous set.

### B2. The kinetic potential is now computable but unvalidated *in use*

`kinetic_potential` is verified against Thomas-Fermi (machine precision) and
finite differences. What is **not** established is whether the *learned*
$\delta T_s/\delta\rho$ is any good — nothing in the current results bears on
it, because training optimises $\tau$ and only ever reports $\tau$.

### B3. There is no downstream test

The definitive question — does the learned functional produce the right density
when placed inside a variational loop? — has never been asked. Until it is, the
project has a well-measured proxy and no result.

**Update (2026-08-03): the first downstream test now exists, and it fails.**
`poraque.physics.energy` integrates the predicted fields into the Kohn–Sham
total energy, and `poraque.calculator.Poraque` exposes it through ASE. On the
seven reference structures:

| Quantity | Value |
| --- | --- |
| True spread of $E$ across the seven | 7.9 eV |
| MAE of predicted energy *differences* | 22.3 eV (0.83 eV/atom) |
| Correlation of differences | $r = 0.61$ |

The error is **three times the signal**. This is not a bug in the energy module
— that is validated against exact Madelung constants ($10^{-11}$), an analytic
Hartree integral ($10^{-16}$) and uniform-electron-gas limits. It is
cancellation: the total is a sum of terms of order $10^4$ eV whose relevant
variation is a relative $2.5\times10^{-4}$, and a field-level relative $L^2$ of
$3\times10^{-2}$ cannot survive that.

Two candidate explanations were tested and rejected:

- **Grid resolution.** Reference-field energies shift by 0.08 eV out of 31215
  (0.003 eV/atom) between resolution 32 and native 128³. Not the limit.
- **Electron-count drift.** Rescaling $\rho$ to the exact valence count makes
  it *worse* (0.83 → 3.20 eV/atom), because $E_\mathrm{ext}$ is linear in
  $\rho$ and a 0.3 % rescale moves a $-12690$ eV term by 38 eV. The drift is a
  symptom, not the cause.

**This changes the target.** Chasing a 1000× better density is almost certainly
unreachable. The alternative is to make the energy *insensitive* to the field
error, which is exactly what variationality buys: at the true density the
Kohn–Sham energy is stationary, so a first-order density error costs only
second order in energy. That property requires evaluating the energy with a
self-consistent functional — i.e. Phase 5 — rather than a fitted $\tau$. It
raises Phase 5 from "the test that settles things" to "the only route to a
usable energy".

---

## Phase 1 — Validate the derivative (days)

*Unblocks B2. Cheap, and everything after depends on it.*

1. **Measure the learned kinetic potential against a reference.** For each
   validation structure, compare $\delta T_s/\delta\rho$ from the trained
   `chg2tau` model against the analytic TF+vW potential and against the
   derivative implied by the reference $\tau$. Report the error in the
   *derivative*, not in $\tau$.
2. **Add $\delta T_s/\delta\rho$ error to the reported metrics.** If it is much
   worse than the $\tau$ error — which §3 of `model2_architecture.md` predicts,
   since differentiation amplifies high-frequency error — that is a finding,
   and it changes the objective.
3. **Enable the $H^1$ (Sobolev) loss and re-measure.** It exists precisely to
   penalise gradient error; this is the first test of whether it helps the
   quantity that matters.

**Falsifiable:** if the derivative error tracks the $\tau$ error, the current
objective is fine and Phase 3 is lower priority. If it does not, training must
change.

## Phase 2 — Establish the residual floor (days)

*Makes the Euler–Lagrange constraint interpretable.*

Evaluate $r(\mathbf r)$ with the **reference** density and the analytic KEDF.
It will not vanish — the functional is approximate. Record that floor. A
learned functional must beat it to be adding anything, and without the number
the residual loss is uninterpretable.

Then repeat with the learned functional via `euler_lagrange_residual(...,
kinetic=tau_fn)`, which is now wired.

## Phase 3 — Couple the two models (weeks)

*The research contribution. Depends on Phases 1–2.*

Train `ext2chg` with the Euler–Lagrange residual built from the learned
`chg2tau` functional. Two configurations:

- **Frozen Model 2** — a fixed physics oracle. Simple, stable; do this first.
- **Joint** — mutual regularisation, and Model 2 finally sees the densities
  Model 1 actually produces, which attacks the distribution-shift problem.

> **The hazard, restated:** the residual alone has trivial solutions. With both
> models free, the pair can drive $r\to0$ by co-adapting without either being
> correct. It is a *consistency* condition, not a *correctness* one. Keep both
> data terms dominant; the collapse signature is a residual that falls while
> held-out error rises.

**The payoff to measure:** label efficiency. Train Model 1 on $k$ labelled
structures plus $m$ unlabelled ones under the residual, and show the curve beats
$k$ labelled alone. If the residual cannot buy label efficiency, its main
practical argument fails — and that is worth knowing early.

## Phase 4 — Grow the dataset (weeks, mostly compute)

*Unblocks B1. Can run in parallel with 1–3.*

Priority order, by what each buys:

1. **More platinum geometries** (strains, vacancies, surfaces) — tests transfer
   within one chemistry; cheapest, and the current numbers become meaningful
   rather than degenerate.
2. **A second element**, ideally a simple metal (Al, Na) — the first real test
   of chemical transfer, and where Thomas-Fermi is *not* hopeless, so the
   baseline comparison becomes informative rather than a walkover.
3. **A binary compound** — exercises the multi-species path in the external
   potential, which is implemented but **untested**.
4. **A spin-polarised system** — `ISPIN=2` writes a second block in `CHGCAR`
   and `TAUCAR` that the reader currently ignores. Silent truncation today.

**Also do at this point:** re-run K-fold with $k < N$ so the folds are not
leave-one-out, giving an honest variance estimate.

## Phase 5 — The downstream test (weeks)

*Unblocks B3. The only test that settles anything.*

Insert the learned KEDF into an orbital-free minimiser: minimise $E[\rho]$
using $\delta T_s/\delta\rho$ from Model 2, and measure the error in the
**converged density and energy** against Kohn–Sham.

This is where the distribution-shift risk becomes concrete — the minimiser
visits non-converged densities that are, strictly, off the $v$-representable
manifold where $\tau[\rho]$ is defined. Expect this to be the hard part.

## Phase 6 — Stretch

- **Charge normalisation head** — makes $\int\rho = N$ exact by construction
  rather than the current 1–3 % drift. Cheap, and strictly better than the
  penalty.
- **Pauli potential constraint** — Levy–Ou-Yang gives $v_P \ge 0$, a constraint
  on the *derivative*, which §3 argues is the object that should be
  constrained.
- **Coordinate-scaling augmentation** — $T_s[\rho_\lambda]=\lambda^2 T_s[\rho]$
  is exact and label-free: free data.
- **Deep-equilibrium solver** — replace Model 1's forward pass with a
  fixed-point solve, differentiated by the implicit function theorem. The
  network then *is* an OF-DFT solver.
- **Density-to-potential inversion** — a cycle-consistency loss that is
  stronger than the EL residual because it does not depend on a KEDF
  approximation.

---

## Engineering debt worth clearing

Small, and none of it blocks the science.

| Item | Why it matters |
|---|---|
| `ISPIN=2` second block ignored | silent truncation on spin-polarised data |
| Multi-species external potential untested | implemented, never exercised |
| `from_encut` grid heuristic is approximate | off by 2× on the reference data; `from_file` is the reliable path |
| Spline boundary condition | $6\times10^{-5}$ residual vs VASP; would need porting `SPLCOF` |
| QE / GPAW readers are stubs | the contract is fixed; each is a day's work |
| Two report kinds are easy to confuse | universal (training fit) vs k-fold (generalisation) differ only by filename |

---

## What to do next week

If only one thing: **Phase 1**. It is days of work, it uses code that already
exists, and its outcome determines whether Phase 3 is worth attempting as
designed. Measuring the error in $\delta T_s/\delta\rho$ is the first honest
test of the thing this project is actually building.

Run Phase 4.1 (more platinum geometries) in parallel, since it is compute-bound
rather than thought-bound.
