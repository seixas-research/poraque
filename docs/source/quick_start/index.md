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

The result is written in `CHGCAR` format and opens directly in VESTA.

## 3. Train

```bash
python scripts/run_train.py --write-config configs/train_config.yaml
python scripts/run_train.py --config configs/train_config.yaml
```

This trains **one** `ext2chg` model and **one** `chg2tau` model on the combined
data of every structure, then writes:

| Output | Contents |
| --- | --- |
| `models/poraque_models.pth` | both trained operators, in one file |
| `logs/*.log`, `logs/*.json` | metrics, and the resolved configuration |
| `results/plots/` | loss curves, field cross-sections, parity plots |
| `reports/*.pdf` | a typeset report per model |

## 4. Predict a new structure

```bash
python scripts/run_eval.py new_structure/ --output predictions/new_structure
```

Both operators come from the single `models/poraque_models.pth` written by
step 2. The grid is sized from `--encut` (default **200 eV**) unless `--grid`,
`--like` or `--resolution` is given.

```{tip}
200 eV is chosen to land near the grid the models were trained on — 42³ for the
27-atom reference cells, against a training `resolution` of 32. Raising it to
400 eV would give 60³, a substantially finer mesh than the models have seen.
Pass `--resolution 32` to evaluate exactly where they were fitted.
```

Geometry → external potential (analytic) → charge density (Model 1) → kinetic
energy density (Model 2). Every output is a `CHGCAR`-format file.

## 5. Evaluate generalisation

Training on everything produces the best model but no held-out data, so its
metrics are a training fit. For a generalisation estimate run the
cross-validation protocol instead:

```bash
python scripts/run_train.py --config configs/train_config.yaml \
    --mode leave_one_out
```

Each structure is held out in turn and scored against a model that never saw
it. Use `universal` for the model you deploy and `leave_one_out` for the number
you quote.

## 6. Drive it from ASE

```python
from ase.build import bulk
from poraque.calculator import Poraque

atoms = bulk("Au", "fcc", a=4.08, cubic=True)
atoms.calc = Poraque("models/poraque_models.pth", potcar="POTCAR")

energy = atoms.get_potential_energy()
print(atoms.calc.components)          # the full energy decomposition
```

Forces and stress are not implemented, so this is for single points rather than
relaxations. See {doc}`../energy/index` — including what the number does and
does not mean.
