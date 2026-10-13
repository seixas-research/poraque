# Can the two operators produce a total energy, and is it close?

```bash
python experiments/total_energy/vasp_compare.py     # the expression vs VASP
python experiments/total_energy/energy_check.py <bundle.poraque> <cache>
```

`vasp_compare.py` uses no model: it hands the **DFT** fields to Poraquê's energy
expression and compares term by term against the `OUTCAR` decomposition. That
isolates the expression from the operators — whatever it gets wrong, a predicted
field can only get more wrong. `energy_check.py` then runs the real chain,
`V_ext → rho → tau → E`, consuming no DFT field.

## Answer 1: the absolute total is not a VASP total energy

On `structure_0000`:

| | eV | eV/atom |
|---|---|---|
| VASP `TOTEN` | −195.343 | −6.1045 |
| Poraquê, on the DFT fields | −29172.077 | −911.6274 |
| offset | −28976.734 | −905.5229 |

The offset is bookkeeping, not error, and it closes to 0.3 eV:

| what VASP counts and Poraquê does not | eV |
|---|---|
| `EATOM`, the atomic reference | +23331.582 |
| PAW double counting | +810.066 |
| non-local + one-center block, inside `EBANDS` | −4835.860 |
| entropy `EENTRO` | −0.582 |

Poraquê integrates **pseudo-valence** fields. There is no arrangement of them
that produces `TOTEN`, and none is attempted. What is comparable is the
*variation* of the total with geometry, and the terms that have exact
counterparts.

## Answer 2: those terms are right

| term | Poraquê | VASP | rel. diff |
|---|---|---|---|
| `alpha_z` vs `PSCENC` | 1947.9522 | 1947.9525 | 1.7e-7 |
| `Ewald` vs `TEWEN` | −26623.033 | −26623.300 | 1.0e-5 |
| `Hartree` vs `−DENC` | 2060.717 | 2060.737 | 9.9e-6 |

So the electrostatics are sound and the offset is not hiding a bug.

## Answer 3: the models are ~100x too coarse for energetics

`w16 m8 l3`, resolution 32, 25 training / 6 held-out structures, 400 epochs.
Predicted fields against reference fields, same expression both sides, so every
omitted term cancels exactly and what remains is the model error alone.

| | rel L² rho | rel L² tau | mean abs dE | per atom |
|---|---|---|---|---|
| 25 training | 0.0014 | 0.0037 | 9.6 eV | 301 meV |
| **6 held out** | **0.0113** | **0.0216** | **119 eV** | **3722 meV** |

Energy differences, which is what the total is actually for:

    reference spread over the set : 16.584 eV  (518 meV/atom)
    |error| in dE over 465 pairs  : mean 51.768 eV, median 17.206, max 283.971

The error is larger than the signal. A fixed offset would cancel in a
difference; this one does not, because it varies from structure to structure.

### Why: the kinetic term, multiplied by seven thousand

`|dE|` correlates at **0.97** with `relL2(tau) * int(tau)`, and `int(tau)` is
about 6987 eV. So tau's *relative* error is multiplied by that, and the realized
error runs from near zero — where the tau error oscillates and cancels in the
integral — up to the full bound. On `structure_0000` the kinetic term is 96 % of
the total error (+32.40 eV of +33.77 eV).

What that demands:

| target | needs rel L²(tau) |
|---|---|
| the set's own spread, 16.6 eV | 2.4e-3 |
| chemical accuracy, 1.6 eV (50 meV/atom x 32) | 2.3e-4 |
| 10 meV/atom | 4.6e-5 |

Held-out tau reaches 2.2e-2. That is roughly **100x short** of chemical
accuracy, and short of even resolving the set's own spread.

## What this is not

It is not a measurement of the method's ceiling. Twenty-five training structures
of one element, with held-out fields 4-10x worse than fitted ones, is
overfitting rather than a converged result. What it does establish is the
*shape* of the requirement: `ext2pseudo` is already at rel L² 0.0014 on training
data and the energy still misses by hundreds of meV/atom, so the binding
constraint is `pseudo2tau`, and the target to quote for it is an absolute one —
`relL2(tau) <= 2.3e-4` — not a comparison against the previous run.

## One bug this found

The chain could not produce an energy at all for a spin-polarized model.
`EnergyCalculator.compute` does `np.asarray(density)`, which a `SpinDensity` is
not; `SpinDensity.as_charge_density` existed for exactly this and nothing called
it. Since `data.spin: auto` resolves against the data, that was every model
trained on these `ISPIN = 2` runs. Fixed in `physics/energy.py:total_density`,
with the ASE calculator's electron-count check and Hellmann-Feynman call made
channel-correct alongside. `E_xc` is still evaluated on the total density, which
is the unpolarized form and therefore drops LSDA — small and second order in
`m/rho` for a cell this weakly polarized, and stated rather than hidden.

---

# The PAW route: which of the three inputs spends the error budget?

```bash
python experiments/total_energy/paw_budget.py --potcar <POTCAR> \
    --json experiments/total_energy/paw_budget.json
```

Everything above concerns the grid-only chain, `ext2pseudo → pseudo2tau`, whose
two outputs are both grids. The PAW route has **three** model outputs:
`ext2chg` predicts the pseudo-density grid *and* every atom's augmentation
occupancies, and `chg2tau` predicts τ. A target accuracy for the whole says
nothing about how the three divide it, and the occupancies are a new kind of
output — 8832 numbers per cell against 1.26 M voxels — with no reason to obey
the grid's arithmetic.

So this measures the derivative directly. Start from the energy on VASP's own
fields, perturb **one** input to a known relative error, and read `|ΔE|`. Every
omitted term cancels exactly between the two evaluations.

## The baseline is intact

| | eV |
|---|---|
| `paw_total_energy` on VASP's fields | −195.342013 |
| VASP `TOTEN` | −195.343223 |
| **difference** | **1.210 meV** (0.038 meV/atom) |

`structure_0000`, 32 Pt, 108³ grid, `LASPH = .FALSE.`, `EENTRO` from the
`OUTCAR`. ∫τ = 6986.8 eV and the one-center block is 7120.9 eV, so the two
large numbers the total is a small difference of are both of order 7000 eV.

## A relative L² does not determine ΔE

The energy is a *functional*, so what matters is not the size of the error but
which part of it survives an integral. Three perturbation shapes bracket that,
at a relative error of 1e-2 (12 seeds, drawn reproducibly from the case index;
`coherent` is deterministic, so one evaluation is the whole answer):

| input | shape | mean \|ΔE\| | sd | dominant term |
|---|---|---|---|---|
| density | smooth (≤16 Å⁻¹) | 0.54 eV | 0.37 | `xc` |
| density | white | 4.31 eV | 0.13 | `xc` |
| density | **coherent** | **109.89 eV** | — | `local` |
| τ | smooth | 0.35 eV | 0.18 | `kinetic` |
| τ | white | 0.09 eV | 0.06 | `kinetic` |
| τ | **coherent** | **69.87 eV** | — | `kinetic` |
| occupancies | independent | 31.87 eV | 21.11 | `one_center_linear` |
| occupancies | **coherent** | **63.10 eV** | — | `one_center_linear` |
| magnetization | smooth, vs ‖ρ‖ | 1.35 eV | 0.01 | `xc` |

`smooth` band-limits the perturbation to the band the platinum runs retain
(g_max 16 Å⁻¹, 0.37 of this grid's Nyquist); `coherent` scales the input by
1 + ε, so nothing cancels. Every row is first order in ε (the ratio of
`|ΔE|` at 1e-2 to 1e-3 is 9–13) **except** two, and both exceptions are
physics rather than noise:

- **`density white` reads 120**, not 10. A white perturbation scaled to 1 % of
  the *global* norm is enormous where ρ is small, and `grid_energy` clamps the
  spin densities at 1e-10 before `pbe_xc`. That row is the clamp, not a
  sensitivity; read `density smooth` instead.
- **`magnetization` reads 100**, i.e. quadratic — which is correct and worth
  stating. E_xc is even in m, so at m ≈ 0 its derivative vanishes and the
  leading cost of an error in m is `‖δm‖²`. On this cell ‖m‖/‖ρ‖ = 3.2e-6, so
  the coherent row is numerically zero. **A magnetically live cell would
  restore the linear term**, and nothing here measures that case.

## τ's coherent bound reproduces the earlier number exactly

`tau coherent` gives a tolerance of **2.29e-4** for 1.6 eV, which is the
`relL²(τ) ≤ 2.3e-4` quoted above. The two are not independent derivations —
both are 1.6 eV / ∫τ, since a coherent perturbation of τ moves the energy by
exactly ε∫τ and the `kinetic` term is the only one it touches. The agreement is
a consistency check on the machinery, not new evidence.

What *is* independent, and what decides which column to design against, is the
measurement reported above: a trained operator's `|ΔE|` correlates at **0.97**
with `relL2(τ)·∫τ`, i.e. with the coherent bound rather than the incoherent
floor two orders of magnitude below it. **Real model error behaves coherently,
so the random rows are a floor the model does not enjoy.**

## The budget, and the one input that cannot hope for cancellation

Relative error each input may carry alone for 1.6 eV (50 meV/atom × 32):

| input | incoherent floor | coherent ceiling | ratio |
|---|---|---|---|
| density | 2.9e-2 | **1.46e-4** | 201× |
| τ | 4.6e-2 | **2.29e-4** | 202× |
| occupancies | 4.9e-4 | **2.54e-4** | **1.9×** |

The three ceilings are within a factor of 2 of each other, so at the coherent
end no single output dominates — all three need ~2e-4. What separates them is
the **ratio**. A grid has 1.26 M voxels, so an incoherent error averages down
by ~√N in the integral and the realized cost depends entirely on how systematic
the model's error happens to be. The occupancies enter `one_center_linear`
(8620 eV) through a linear form over ~8832 numbers; there is almost nothing to
average, so their tolerance is **2.5–4.9e-4 whichever shape the error takes**.

That makes the occupancies the one input whose target can be quoted without
knowing the error's spectrum — and the one where no amount of error structure
will rescue a loose fit.

### Against what the occupancy head achieves

The CHON `ext2chg` study (`experiments/chon_ext2chg/`) reaches a validation
occupancy relative RMS of **6.4e-2** at best (0.101 for the baseline arm). Set
against a 2.5–4.9e-4 requirement that is **130–250× short**, on a different
chemistry from the platinum cell measured here but the same order of magnitude
of the quantity.

So the PAW route does not relax the accuracy problem; it adds a third output
that is subject to it. The earlier conclusion — that `pseudo2tau` is the
binding constraint of the grid-only chain — becomes, for the PAW chain, that
**all three outputs are binding at ~2e-4 and the occupancies are the one with
no cancellation to hope for.**

## One asymmetry worth exploiting

The density's coherent bound (1.46e-4) is the tightest of the three, and it is
also the one the pipeline already defends against: `poraque-inference`
renormalizes the predicted grid to the valence electron count
(`ChargeDensity.normalized`), and a uniform scaling error is *exactly* what
renormalization projects out. The density's worst error mode is therefore
largely unrealized.

There is no analogous projection for the other two. In PAW the total electron
count is ∫ñ plus an on-site contribution that is linear in the occupancies, so
a sum rule of the same kind exists for them — and nothing in the pipeline
enforces it: the grid is renormalized and the records are attached as
predicted. **[next]** Enforcing the occupancy sum rule at inference, the way
the density's is enforced, would remove their coherent component and leave only
the floor — which for the occupancies is a factor of 1.8, but for a systematic
bias is the whole of it. Worth checking against the identity before building
it; this measurement does not verify the numerical sum rule, only that nothing
applies one.

## Scope

One structure, one element, one grid. The sensitivities are properties of
*this* cell's energy terms — ∫τ ≈ 7000 eV and a 7121 eV one-center block — and
a cell with different term magnitudes will scale them. No model prediction
enters: the perturbations are synthetic, which is what makes the attribution
exact and also what leaves the realized error of a *trained* operator
unmeasured. Doing that needs a loadable `ext2chg` and `chg2tau` pair on this
platinum set, and neither exists yet — the CHON bundles in `models/` predate
the British→American parameter rename and no `chg2tau` operator has been
trained at all.
