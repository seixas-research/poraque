# Charges and population analysis

Once a density has been predicted, the questions that follow are *how much
charge is there* and *where does it sit*. This page covers both: the
conservation check that must pass first, and the three partitionings that turn
a grid into per-atom numbers.

Nothing here is learned. Every quantity is a deterministic functional of a
density that has already been predicted, so **the error in a partial charge is
inherited from the density and is never smaller than it**.

## Charge conservation

A Kohn–Sham density integrates to the valence count the pseudopotentials fix.
That number is an *input* to a DFT calculation, not an output of one — so it is
exactly known, and it is the one property of a predicted density that can be
checked without a reference calculation.

```python
from poraque.analysis import verify_total_charge

check = verify_total_charge(rho.data, rho.grid.cell, expected_electrons=297.0)
print(check)
# charge check OK: 297.000001 electrons against an expected 297.000000
#                  (+1.918e-09 relative, tolerance 0.001)
```

The integral is the voxel sum times the voxel volume,

$$\int\rho\,d^3r \;=\; \sum_{ijk}\rho_{ijk}\,\mathrm{d}V,
\qquad \mathrm{d}V = \frac{\Omega}{N_x N_y N_z},$$

which is *exact* for a band-limited field on a uniform periodic mesh — the
rectangle and trapezoid rules coincide under periodic boundary conditions, so
there is no quadrature error to argue about, only the field itself.

The default tolerance of $10^{-3}$ is not arbitrary. The electrostatic energy
terms are of order $10^4$ eV, so a relative drift of $10^{-3}$ already moves a
total energy by roughly 10 eV — far more than any energy difference worth
computing.

:::{warning}
The shipped `ext2chg` operator predicts densities whose electron count drifts
by up to **1.7 %** — nearly five electrons out of 297. Left uncorrected this
moves totals by tens of eV, differently for each structure, so it does not
cancel in a difference.
:::

### Automatic normalization

The calculator rescales the density to the exact count by default:

```python
atoms.calc = Poraque("models/poraque_models.pfno", potcar_dir="POTCARs",
                     normalize_density=True)   # the default

atoms.calc.verify_charge()          # passes by construction
atoms.calc.raw_electron_drift       # what the operator actually did
```

`normalize_density=True` guarantees the check passes, so the *interesting*
number afterwards is `raw_electron_drift` — the drift of the raw prediction,
recorded before the repair. Set `normalize_density=False` to measure the
operator rather than the repair.

Rescaling is a repair, not a cure: a density whose integral is wrong by 1.7 %
has the wrong shape too, and a single global factor does not fix a shape.

## Partial charges

All three schemes have the same form,

$$q_A = Z^{\rm val}_A - \int w_A(\mathbf r)\,\rho(\mathbf r)\,d^3r,
\qquad \sum_A w_A(\mathbf r) = 1,$$

and differ only in the weight $w_A$. Because the partitions are exhaustive,
**all three conserve charge exactly** — the populations sum to $\int\rho$
whatever the weights are.

| Method | $w_A(\mathbf r)$ | Cost | Character |
| --- | --- | --- | --- |
| `voronoi` | 1 for the nearest atom | fast | purely geometric |
| `hirshfeld` | $\rho^{\rm at}_A / \sum_B \rho^{\rm at}_B$ | fast | needs a promolecule |
| `bader` | 1 inside $A$'s zero-flux basin | moderate | intrinsic to $\rho$ |

### From the ASE calculator

```python
atoms.calc = Poraque("models/poraque_models.pfno", potcar_dir="POTCARs",
                     references="data/vasp/ref")

atoms.get_charges()                              # standard ASE, Bader default
atoms.calc.get_charges(method="hirshfeld")
atoms.calc.get_charges(method="voronoi")

print(atoms.calc.charge_analysis)                # full decomposition
```

`get_charges` returns net charges in units of $+e$, positive for
electron-deficient. The full analysis — populations, the valence subtracted,
which promolecule or Bader backend was used — is left on `charge_analysis`.

### Directly on a field

```python
from poraque.analysis import partial_charges

result = partial_charges(rho, method="bader", valence={"Pt": 11.0})
result.charges          # net charge per atom
result.populations      # electrons per atom
result.total_charge     # ~0 for a neutral cell
```

### Voronoi

Each voxel goes wholly to its nearest atom, under the minimum-image
convention. In a non-orthogonal cell the surrounding images are searched rather
than trusting a wrap of the fractional coordinates — in a skewed lattice the
nearest image can be a diagonal neighbour.

Voxels exactly equidistant from several atoms are **shared equally** rather
than awarded to the lowest atom index. This is rare on a real density and
common on a symmetric cell sampled by a commensurate grid, which is exactly
where an index-order tie-break would bias a result that should come out
symmetric. The count is reported as `details["shared_voxels"]`.

Purely geometric: the density enters only through the integral, never through
where the boundary is drawn. For atoms of unequal size that is a poor chemical
partition, which is why it is offered as a baseline rather than a default.

### Hirshfeld

Weights by the free atoms, so the partition is smooth and chemically sensible
— but it is *defined by its reference*, and a poor promolecule gives poor
charges with no other symptom.

Supply real isolated-atom densities wherever possible:

```python
partial_charges(rho, method="hirshfeld", valence={"Pt": 11.0},
                references="data/vasp/ref")
```

Each `<references>/<Element>/CHGCAR` is spherically averaged into
$\rho^{\rm at}(r)$. Where no reference exists, the fallback is an exponential
free atom, $\rho(r) = \frac{N}{8\pi a^3}e^{-r/a}$, normalized to the valence
count with the decay length set from the covalent radius through
$\langle r\rangle = 3a$. Which was used for each element is reported in
`details["promolecule"]` — check it, because the fallback is crude.

Hirshfeld charges are known to be small in magnitude compared with other
schemes. For an elemental solid they are near zero by construction, since the
promolecule is built from identical free atoms; that makes a useful sanity
check.

### Bader (QTAIM)

The only scheme whose definition is intrinsic to the density: basins are
bounded by surfaces of zero flux in $\nabla\rho$. Two backends:

```python
partial_charges(rho, method="bader", backend="native")    # pure NumPy
partial_charges(rho, method="bader", backend="external")  # Henkelman `bader`
partial_charges(rho, method="bader", backend="auto")      # external if present
```

**Native** is a vectorized on-grid steepest ascent: every voxel points at
whichever of its 26 neighbours gives the largest density increase *per unit
distance*, and those pointers are followed to a local maximum by repeated
pointer doubling. The distance weighting matters — without it the diagonal
neighbours, $\sqrt3$ further away, win too often and the basins acquire a
staircase bias along the cell diagonals.

**External** writes a temporary `CHGCAR`, runs the Henkelman group's `bader`
program, and parses `ACF.dat`. `backend="external"` raises if the program is
absent rather than falling back silently — the two backends are not identical,
and a result reported as "Bader (Henkelman)" should not quietly have been
something else.

A density may have more maxima than atoms: spurious ones from grid noise, or
genuine non-nuclear attractors in a metal. Each maximum is assigned to its
nearest atom, so the mapping is many-to-one and `details["maxima"]` reports how
many were found.

## What these charges are not

:::{warning}
All three partition the **pseudo** valence density. The PAW core is absent, so
these are not all-electron populations: a Bader volume here is not the
all-electron Bader volume, and the charges are systematically compressed toward
zero. This matters most for Bader, whose basin boundaries are set by the
density's topology near the nuclei — exactly where the pseudisation is largest.

For a proper Bader analysis VASP's own guidance is to sum `AECCAR0` and
`AECCAR2`; Poraquê does not predict those.

Treat the numbers as comparative across a series, not as absolute.
:::

## In the reports

Pass a `PartialCharges` to the report builder and the per-atom table, the
method, and the caveat above are typeset into the PDF:

```python
from poraque.vis import ModelReport

ModelReport("reports").build(task="ext2chg", per_material=metrics,
                             charges=analysis)
```
