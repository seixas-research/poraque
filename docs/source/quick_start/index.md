# Quick start

The pipeline turns a set of DFT calculations into two trained neural operators
and, from those, predicts the electronic structure of a new crystal from its
geometry alone.

## 1. Arrange the data

One directory per material, each holding a standard VASP calculation:

```
data/vasp/
    struct_000/   POSCAR  INCAR  POTCAR  CHGCAR  TAUCAR
    struct_001/   POSCAR  INCAR  POTCAR  CHGCAR  TAUCAR
    ...
```

Only `CHGCAR` and `TAUCAR` are required as *targets*. The external potential is
computed by Poraquê from `POSCAR`/`INCAR`/`POTCAR`, so **no special build of
VASP is needed**.

```{note}
Grids may differ between materials — they usually do, since `ENCUT` and the
cell shape fix `NGXF, NGYF, NGZF`. The pipeline handles that; see
[Grids that differ between materials](../ml/index.md).
```

## 2. Compute an external potential

```python
from poraque.fields import ExternalPotential

potential = ExternalPotential.from_calculation("data/vasp/struct_000")
potential.write("EXTCAR")

print(potential)                 # range, units, grid
print(potential.mean())          # ~1e-16 eV: the G=0 convention
```

The result is written in `CHGCAR` format.

## 3. Train

```bash
poraque-train --config configs/train.yaml
```

This trains **one** `ext2chg` model and **one** `chg2tau` model on the combined
data of every structure, then writes:

| Output | Contents |
| --- | --- |
| `models/poraque_models.poraque` | both trained operators, in one file |
| `logs/*.log`, `logs/*.json` | metrics, and the resolved configuration |
| `results/plots/` | loss curves, field cross-sections, parity plots |
| `reports/*.pdf` | a typeset report per model |

## 4. Predict a new structure

```bash
poraque-inference new_structure/ --output predictions/new_structure
```

Both operators come from the single `models/poraque_models.poraque` written by
step 3. The grid is sized from `--encut` (default **200 eV**) at `PREC=Normal`,
unless `--grid`, `--like` or `--resolution` is given.

```{tip}
200 eV at `PREC=Normal` lands on **32³** for the 27-atom reference cells and
28³ for the 32-atom ones — the training `resolution` is 32, so the operator is
interpolating rather than extrapolating. Raising it to 400 eV would give a
substantially finer mesh than the models have seen.
```

### Matching a VASP grid

VASP's `PREC` sets the multiplier between the wavefunction cutoff and the
density FFT grid: `Normal` uses the cheaper 3/2 rule, `Accurate` and `High` a
factor of 2 so the density grid is wrap-around free. Two flags expose it:

```bash
# the Accurate rule at the default cutoff: 42³ instead of 32³
poraque-inference struct/ --prec-accurate

# take ENCUT and PREC from a real INCAR: 450 eV, Accurate -> 64³
poraque-inference struct/ --from-incar struct/INCAR
```

```{important}
**`--from-incar` takes precedence.** When it is given, `--encut` and
`--prec-accurate` are ignored entirely and anything overridden is named in the
log. A run described by an input file should reproduce that file's grid; a flag
quietly modifying it would make the two disagree while appearing to agree.
```

### `--to-vasp`: the grid VASP itself would build

```bash
poraque-inference struct/ --to-vasp --from-incar struct/INCAR --add-paw
```

The sizing above is how a plane-wave code *should* choose a grid. It is not
what VASP does, and for a restart that difference matters — the `CHGCAR`
declares its own `NGXF NGYF NGZF`, and `ICHARG=1` wants those to be the grid
the run itself would build.

VASP derives the density grid in **two stages** (`main.F`):

```fortran
GRID%NGPTAR(1) = XCUTOF*WFACT + 0.5   ! WFACT = 4 for high/accurate/single, else 3
CALL FFTCHK(GRID%NGPTAR)              ! rounded here
GRIDC%NGPTAR(1) = GRID%NGPTAR(1)*2    ! then doubled
CALL FFTCHK(GRIDC%NGPTAR)
```

**The order is the algorithm.** For a 27-atom platinum cell at 450 eV the coarse
grid is `4 × 15.25 = 61 → 64`, so the density grid is **128**; computing the
density size directly and rounding once gives 64 — a factor of two, and a file
VASP would reject.

`--to-vasp` reproduces this exactly. Validated against every reference
calculation in the project: **17 of 17**, including the anisotropic
`(120,128,128)` and `(108,108,112)` cases.

```{tip}
Sizes are rounded by VASP's `FFTCH1` rule — 7-smooth **and** even, so 61 → 64
and 109 → 112. The helper is
{py:func}`~poraque.fields.vasp.fftgrid.get_valid_fft_grid_size`.
```

Grid selection overall, first match winning: `--grid`, then `--like`, then
`--to-vasp`, then `--resolution`, then the cutoff path (`--from-incar`, else
`--encut` with `--prec-accurate`).

Geometry → external potential (analytic) → charge density (Model 1) → kinetic
energy density (Model 2). Every output is a `CHGCAR`-format file.

### Restarting VASP from the prediction

```bash
poraque-inference new_structure/ --like new_structure/CHGCAR --add-paw
```

A `CHGCAR` that VASP accepts for `ICHARG=1` carries, after the grid block, one
*augmentation occupancies* record per atom — the one-centre PAW terms, inside
the augmentation spheres.

```{warning}
Those terms are **not representable on the plane-wave grid**, so no grid-based
model predicts them. `--add-paw` copies them verbatim from a reference
calculation in the same directory; the result is interstitial density from the
model and core-region occupancies from the reference.

It follows that `--add-paw` needs a converged `CHGCAR` for the geometry — and
if you have one, you do not need a prediction to restart from. The flag is for
the case where the reference is a *near-neighbour* geometry whose on-site
occupancies are still a good approximation.
```

**Without a reference calculation**, the model falls back to a per-element
table it carries. Training reads the augmentation records off the training
`CHGCAR`s, averages them per element, and stores them in the `.poraque` bundle:

```text
  PAW reference: Pt  138 values, averaged over 494 atoms in 17 structure(s)
  PAW reference   -> stored in the bundle (Pt)
```

At inference the table is used when the directory has no reference of its own:

```text
  using the bundle's PAW reference: Pt (averaged over 494 atoms in 17 structures)
  !! these are AVERAGED on-site terms, not this structure's.
```

```{warning}
The averaged table reproduces the true occupancies to about **9 % RMS** on the
reference dataset, and the dominant component varies by a factor of two across
sites. It is a defensible starting guess for `ICHARG=1`, not a converged
on-site density — and it covers only the elements the model was trained on.
```

A real calculation beside the structure always wins over the table: its records
are *that* system's, where the table is an average over others.

Three checks run first, each refusing rather than writing a file VASP would
reject: a record count that disagrees with the atom count, a grid that differs
from the reference (a warning — the records are grid-independent, but
`ICHARG=1` wants the CHGCAR grid to be the run's `NGXF`; use `--like`), and an
element the table does not cover, which yields no records at all rather than a
partial block.

## 5. Evaluate generalisation

Step 3 already holds back a fifth of the structures, so its score is genuinely
held out. For a tighter estimate — every structure scored by a model that never
saw it — run the cross-validation protocol instead:

```bash
poraque-train --config configs/train.yaml --kfold
```

That is the only variation on the training protocol. It fits *K* models and
reports a spread, so it produces a number to quote rather than a model to
deploy; set `k_folds` to the number of structures for leave-one-out.

For the artefact you actually ship, set `valid_fraction: 0` so training uses
every structure — accepting that its own metrics are then a training fit.

## 6. Drive it from ASE

```python
from ase.build import bulk
from poraque.calculator import Poraque

atoms = bulk("Pt", "fcc", a=3.92, cubic=True)
atoms.calc = Poraque("models/poraque_models.poraque", potcar_dir="POTCARs")

rho = atoms.calc.get_charge_density(atoms)   # ChargeDensity, e/Ang^3
print(rho.electron_count())                  # 44.0, the valence count
rho.write("CHGCAR_pred")                     # readable by any DFT tool

charges = atoms.get_charges()                # net charge per atom, +e
print(atoms.calc.charge_analysis)            # populations, valence, method
```

`potcar_dir` is a POTCAR **library** — one subdirectory per element,
`<potcar_dir>/Pt/POTCAR` — not a single POTCAR file. The entries for the
elements the `Atoms` contains are assembled on demand and cached per
composition, which is what lets one calculator serve arbitrary structures.

The fields are the prediction; a total energy is one thing integrated from
them, and `atoms.get_potential_energy()` returns it. See
{doc}`../energy/index` for what that number does and does not mean, and
{doc}`../analysis/index` for the partitioning behind `get_charges`. Forces and
stress are not implemented, so this is for single points rather than
relaxations.
