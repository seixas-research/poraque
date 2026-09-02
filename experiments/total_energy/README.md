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
| non-local + one-centre block, inside `EBANDS` | −4835.860 |
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
about 6987 eV. So tau's *relative* error is multiplied by that, and the realised
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
*shape* of the requirement: `ext2chg` is already at rel L² 0.0014 on training
data and the energy still misses by hundreds of meV/atom, so the binding
constraint is `chg2tau`, and the target to quote for it is an absolute one —
`relL2(tau) <= 2.3e-4` — not a comparison against the previous run.

## One bug this found

The chain could not produce an energy at all for a spin-polarised model.
`EnergyCalculator.compute` does `np.asarray(density)`, which a `SpinDensity` is
not; `SpinDensity.as_charge_density` existed for exactly this and nothing called
it. Since `data.spin: auto` resolves against the data, that was every model
trained on these `ISPIN = 2` runs. Fixed in `physics/energy.py:total_density`,
with the ASE calculator's electron-count check and Hellmann-Feynman call made
channel-correct alongside. `E_xc` is still evaluated on the total density, which
is the unpolarised form and therefore drops LSDA — small and second order in
`m/rho` for a cell this weakly polarised, and stated rather than hidden.
