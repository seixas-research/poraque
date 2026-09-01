# Energies and the ASE calculator

Once $\rho$ and $\tau$ have been predicted, {mod}`poraque.physics` turns them
into energies, and {class}`poraque.calculator.Poraque` wraps the whole chain
behind the ASE calculator protocol.

```{warning}
Read [What the number is not](#what-the-number-is-not) before comparing any of
these energies to a DFT result. They are **pseudo-valence** energies, and on
the current models the error on energy *differences* is larger than the
differences themselves.
```

## The energy expression

$$
E = \underbrace{\int\tau\,d^3r}_{T_\mathrm{s}}
  + \underbrace{\int\rho V_\mathrm{ext}\,d^3r + E_{\alpha Z}}_{E_\mathrm{ext}}
  + \underbrace{\tfrac12\int\rho v_\mathrm{H}\,d^3r}_{E_\mathrm{H}}
  + E_\mathrm{xc}[\rho]
  + E_\mathrm{Ewald}
$$

| Term | How it is obtained |
| --- | --- |
| $T_\mathrm{s}$ | integral of the predicted `TAUCAR` |
| $E_\mathrm{ext}$ | $\int\rho V_\mathrm{ext}$, with $\mathbf G = 0$ excluded |
| $E_{\alpha Z}$ | the finite $\mathbf G = 0$ remainder, from the `POTCAR` `PSCORE` |
| $E_\mathrm{H}$ | Poisson solve in reciprocal space, $v_\mathrm{H}(\mathbf G{=}0) = 0$ |
| $E_\mathrm{xc}$ | PBE by default; LDA available — see [below](#exchange-correlation) |
| $E_\mathrm{Ewald}$ | Ewald sum over the pseudo-ions with a neutralizing background |

(the-g-0-bookkeeping)=
### The $\mathbf G = 0$ bookkeeping

Each of the three electrostatic terms diverges at $\mathbf G = 0$ and the
divergences cancel in a neutral cell. Poraquê follows the standard plane-wave
treatment: drop $\mathbf G = 0$ from all three, then add back the finite
remainder of the electron-ion term,

$$E_{\alpha Z} = \frac{N_\mathrm{elec}}{\Omega}\sum_s N_s\,\mathrm{PSCORE}_s ,$$

which is VASP's `alpha Z`. It is of order eV *per atom*, so it is read from the
`POTCAR` tables when they are available and reported as `None` when they are
not — never silently assumed to be zero.

```{note}
{attr}`~poraque.physics.energy.EnergyComponents.missing` lists any term that
was skipped, and `str(components)` prints `incomplete:` when the list is
non-empty. A total that is missing a term says so.
```

(exchange-correlation)=
### Choosing the exchange-correlation functional

| `functional` | Exchange | Correlation |
| --- | --- | --- |
| `"pbe"` (**default**) | PBE | PBE |
| `"lda"` | Dirac | PW92 |
| `"pbe-x"` | PBE | — |
| `"lda-x"` / `"x-only"` | Dirac | — |
| `"none"` | — | — |

PBE is the default because it matches the reference data: the calculations
Poraquê ingests use `PAW_PBE` pseudopotentials with `LEXCH = PE`, so $\rho$ and
$\tau$ are PBE quantities. Evaluating an LDA $E_\mathrm{xc}$ on a PBE density
is neither a PBE energy nor an LDA one. On the reference Pt supercells the two
differ by **−0.92 eV/atom** (0.65 % of $E_\mathrm{xc}$), so the choice is not
cosmetic.

Both are validated against the limit that defines them: on a uniform density
PBE reduces to LDA *bit-exactly*, because $F_\mathrm{x}(0) = 1$ and
$H \to 0$. Dirac exchange reproduces $-0.458165293/r_s$ Ha per electron
exactly, and PW92 matches the published table to $10^{-6}$.

```{warning}
PBE is semilocal and needs $\nabla\rho$. On a *predicted* field that gradient
carries the network's noise and the enhancement factor amplifies it; on a
band-limited grid it also aliases wherever the density has sharp core peaks.
The extra physics is only worth having once the density is good enough for its
gradient to mean something — which, on the current models, it is not (see
[below](#what-the-number-is-not)).
```

Set it wherever the energy is computed:

```python
EnergyCalculator.from_potential(vext, functional="lda")
Poraque("ext2chg.pfno", "chg2tau.pfno", potcar_dir=..., functional="lda")
```
```bash
poraque-inference run/ --functional lda
```

## Computing energies from files

```python
from poraque.fields import ChargeDensity, ExternalPotential, FieldGrid, KineticEnergyDensity
from poraque.fields.vasp.potcar import Potcar
from poraque.physics import EnergyCalculator

grid = FieldGrid.from_file("run/CHGCAR")
rho  = ChargeDensity.read("run/CHGCAR", grid=grid)
tau  = KineticEnergyDensity.read("run/TAUCAR", grid=grid)
vext = ExternalPotential.from_calculation("run", grid=grid)

potcar = Potcar.from_file("run/POTCAR", parse_tables=True)
calc = EnergyCalculator.from_potential(
    vext, pscore={e.element: e.pscore for e in potcar})

print(calc.compute(rho, tau, vext))
```

```text
  kinetic      T_s            6207.068300 eV
  external     E_ext        -12634.744762 eV
  alpha Z                     1752.512988 eV
  Hartree      E_H            3230.363304 eV
  xc (pbe)                   -3845.824650 eV
  Ewald                     -25949.345827 eV
  ----------------------------------------
  potential                 -37447.038948 eV
  TOTAL                     -31239.970647 eV
  electrons                    297.000001
```

```{tip}
`electrons` is the cheapest available diagnostic. It should equal the sum of
the `ZVAL`s — 297 for 27 Pt atoms above — to a few parts in $10^4$. A predicted
density that has drifted off it invalidates every electrostatic term, since
they are all linear or quadratic in $\rho$.
```

`EnergyCalculator` accepts plain arrays as readily as field objects, so a
prediction can be scored without writing it to disk first.

## The ASE calculator

{class}`poraque.calculator.Poraque` behaves like MACE or NequIP at the
interface:

```python
from ase.build import bulk
from poraque.calculator import Poraque

atoms = bulk("Pt", "fcc", a=3.92, cubic=True)
atoms.calc = Poraque("models/poraque_models.pfno", potcar_dir="POTCARs")

energy = atoms.get_potential_energy()
print(atoms.calc.components)          # the full decomposition
rho = atoms.calc.fields["density"]    # the predicted CHGCAR
```

Each call runs `Atoms → Structure → FieldGrid → V_ext → ρ → τ → E`. The three
fields stay on the calculator afterwards, so a prediction can be written out
with `rho.write("CHGCAR")`.

### Options

| Argument | Meaning |
| --- | --- |
| `ext2chg`, `chg2tau` | checkpoint paths, or already-loaded `FieldOperator`s |
| `potcar` | a single `POTCAR` covering the species present |
| `potcar_dir` | a `POTCAR` *library*, searched per composition |
| `charges` | `{element: Z_val}`, used only when no `POTCAR` is available |
| `resolution` | longest grid axis; defaults to the checkpoint's training resolution |
| `functional` | exchange-correlation approximation (default `"pbe"`) |
| `device` | `auto`, `cuda`, `mps` or `cpu` |

### Supplying pseudopotentials

`potcar=` is right when the composition is fixed. For a calculator that must
serve **arbitrary** structures, point `potcar_dir=` at a POTCAR library and the
entries for whatever elements an `Atoms` contains are assembled on demand and
cached per composition:

```python
atoms.calc = Poraque("models/poraque_models.pfno",
                     potcar_dir="/opt/vasp/potpaw_PBE")
```

Recognised layouts, in preference order — each accepting a `.gz` or `.Z`
suffix:

1. `<dir>/Pt/POTCAR` — what VASP ships;
2. `<dir>/Pt_pv/POTCAR` — a variant, used only if it is the *only* candidate,
   with a warning;
3. `<dir>/POTCAR.Pt` or `<dir>/Pt.POTCAR` — flat.

```{note}
If several variants match — `Fe_pv` and `Fe_sv`, say — the lookup **raises**
rather than picking one. They differ in `ZVAL` and therefore in every energy,
so the choice is the user's; name the one you want with an explicit `potcar=`.
```

```{warning}
With no POTCAR at all the external potential falls back to the **Gaussian
pseudo-ion model**, which differs from the tabulated potential by a relative
$L^2$ of about `0.13` against `2×10⁻⁵` for the tabulated one. The operators
were trained on tabulated potentials, so the Gaussian model feeds them an
input outside their training distribution. The calculator warns once and
proceeds; treat the result as a smoke test, not a prediction.
```

### Forces and stress

`get_forces()` returns the analytic **Hellmann-Feynman** force: the
electron-ion term $-\int\rho\,\partial V_\mathrm{ext}/\partial\mathbf R$
evaluated in reciprocal space from the same `POTCAR` form factor the potential
was built from, plus the analytic Ewald force. Both are differentiated in
closed form, not by finite differences — on a fixed grid those would be
dominated by the grid's own discontinuity as atoms cross voxel boundaries.

The implementation is exact: each term agrees with a central finite difference
of the energy it differentiates to $10^{-7}$ eV/Å.

:::{warning}
The **physics** is incomplete for PAW datasets. Hellmann-Feynman is the whole
force only for a *local* pseudopotential; a PAW calculation adds a projector
force and a one-centre force, and neither is recoverable from $\rho$, $\tau$
and $V_\mathrm{loc}$ on a grid. Measured against VASP on platinum, using VASP's own
density: MAE 0.83 eV/Å against forces of 1.66 eV/Å. The magnitude is right, the
direction is not. **Geometry optimisation and molecular dynamics remain
unavailable.**

The cancellation is why it is delicate: the electron-ion and ion-ion terms are
each $\approx 100$ eV/Å and cancel to $\approx 0.5$ eV/Å, so a relative error in
$\rho$ arrives in the force amplified roughly 200-fold.
:::

`get_stress()` still raises. The stress needs the energy's response to a
strain, which deforms the cell *and* the grid the fields live on.

`implemented_properties` is `["energy", "free_energy", "forces"]`. The first
two are the same number: no electronic smearing enters this pipeline, and ASE
optimizers request `free_energy` by name.

(cohesive-energies)=
### Cohesive energies

A total energy means nothing on its own — it is referenced to whatever the
pseudopotential generator chose, and Poraquê's own totals additionally carry a
$\approx 10^3$ eV per atom offset from the absent PAW one-centre terms. The
**cohesive energy** removes the arbitrary part:

$$\Delta E = E_\mathrm{total} - E_\mathrm{ref},
\qquad E_\mathrm{ref} = \sum_i E_\mathrm{iso}(Z_i)$$

with $E_\mathrm{iso}$ the energy of one isolated atom of that species. What
survives is the energy released on assembling the solid from free atoms — the
bonding, and nothing else.

Point the calculator at a directory holding one subdirectory per element, each
an ordinary single-point calculation of one atom in a large box:

```text
data/vasp/ref/
    Pt/     POSCAR POTCAR OSZICAR OUTCAR CHGCAR TAUCAR
    N/      ...
```

```python
atoms.calc = Poraque("models/poraque_models.pfno",
                     potcar_dir="POTCARs",
                     references="data/vasp/ref")

atoms.get_potential_energy()                  # total, as always
atoms.calc.get_cohesive_energy()              # ΔE, in eV
atoms.calc.get_cohesive_energy(per_atom=True) # eV/atom
```

#### Which reference to subtract

`ReferenceEnergies.from_directory` takes a `method`, and the choice matters
more than anything else in this section.

| `method` | $E_\mathrm{iso}$ from | Pt cohesive energy |
| --- | --- | --- |
| `"poraque"` (default) | Poraquê's own energy expression on the reference fields | **−1.9 eV/atom** |
| `"code"` | VASP's `OUTCAR`/`OSZICAR` | −1157 eV/atom |

A cohesive energy is only meaningful when the two energies being subtracted
carry the *same* systematic error. Subtracting VASP's atomic energy from
Poraquê's total leaves the PAW offset entirely intact; subtracting Poraquê's
own atomic energy cancels it, because the same terms are missing from both
sides. That is why `"poraque"` is the default. Use `"code"` when the reference
calculations are what you want to compare *against*.

#### What referencing does not change

Two things are worth stating because they are commonly assumed otherwise, and
both are exact identities rather than approximations:

- **Forces are untouched.** $E_\mathrm{ref}$ depends on composition, not on
  coordinates, so $\nabla_\mathbf{R} E_\mathrm{ref} = 0$ and
  $\nabla_\mathbf{R}\Delta E = \nabla_\mathbf{R} E_\mathrm{total}$ identically.
- **Differences at fixed composition are untouched.** Two structures with the
  same formula share $E_\mathrm{ref}$, which cancels exactly in
  $\Delta E_1 - \Delta E_2$. An energy-volume curve or a polymorph ranking is
  numerically identical, bit for bit.

Referencing earns its place across *different* compositions — binding
energies, cohesive energies per atom, formation energies — where the offset
does not cancel and without a reference state the comparison is undefined.

(hartree-potential)=
### The Hartree potential

$v_\mathrm{H}$ is **solved, not predicted**. Poisson's equation relates it to
$\rho$ exactly, and on a periodic plane-wave grid the reciprocal-space form is
the solution rather than an approximation to it:

$$\nabla^2 v_\mathrm{H} = -4\pi e^2\rho
\quad\Longleftrightarrow\quad
v_\mathrm{H}(\mathbf G) = \frac{4\pi e^2\rho(\mathbf G)}{G^2},
\qquad v_\mathrm{H}(\mathbf G = 0) = 0 .$$

Two FFTs, exact for any band-limited density. A third learned operator would be
strictly worse: it would introduce error into a quantity that has none, and
would not be guaranteed to satisfy the equation that defines it.

```python
potential = atoms.calc.get_hartree_potential()
potential.write("LOCPOT")

# v_H + V_ext, the total local potential a plain VASP LOCPOT holds
total = atoms.calc.get_hartree_potential(with_external=True)
```

The $\mathbf G = 0$ term is set to zero, the same neutralizing-background
convention {ref}`used for $V_\mathrm{ext}$ <the-g-0-bookkeeping>` — which is
what makes the two potentials addable and what makes $E_\mathrm{H}$ of a
uniform density come out at exactly zero rather than infinite.

Written to `LOCPOT`, the field is stored **unscaled**: unlike `CHGCAR`, which
holds $\rho\Omega$, a `LOCPOT` holds the potential itself in eV. From the
command line:

```bash
poraque-inference structure/ --output predictions/ --write-locpot
poraque-inference structure/ --output predictions/ --write-locpot --locpot-total
```

The field accessors are symmetric — `get_external_potential()`,
`get_charge_density()`, `get_kinetic_energy_density()`,
`get_hartree_potential()` — and all four return fields on one shared grid.

(what-the-number-is-not)=
## What the number is not

**It is not a DFT total energy.** `CHGCAR` holds the valence *pseudo* density
and `TAUCAR` the valence pseudo kinetic energy density, so the PAW one-centre
terms are absent entirely. For the 27-atom Pt cell above the result is
$\approx -31215$ eV against a VASP `TOTEN` of order $-100$ eV. Nothing in this
module can close that gap.

**Energy differences are not yet usable either.** Measured on the current
seventeen-structure Pt dataset, comparing within each composition group:

| Source of the fields | MAE on $\Delta E$ | vs. signal |
| --- | --- | --- |
| VASP's own $\rho$ and $\tau$ — the method ceiling | 0.18–0.24 eV/atom | 1.5–2× |
| The shipped operators | 0.87–1.93 eV/atom | 7–15× |
| *True spread being resolved* | *0.125 eV/atom* | — |

The error **exceeds the signal** in both rows, and the correlation with the
true ordering is consistent with zero, so the predicted ranking of structures
carries no information.

The first row is the important one: it is measured with the model removed from
the problem entirely, so it is not a training failure. It is the neglected PAW
one-centre and non-local energy, which is *not* a per-atom constant. The
electrostatic terms themselves are correct — $E_{\alpha Z}$, Ewald and
$E_\mathrm{H}$ reproduce VASP's `PSCENC`, `TEWEN` and `DENC` to better than
0.01 eV in a difference.

The underlying difficulty is one of scale. The total is a cancellation of terms
of order $10^4$ eV down to a result of order $10^0$ eV, so resolving
$\Delta E$ to 10 meV/atom needs a relative accuracy of about $10^{-6}$ in the
fields. The operators currently deliver $10^{-2}$. See
`docs/plan/future_roadmap.pdf` for what would close this.

The cause is cancellation, not a bug: the total is a sum of terms of order
$10^4$ eV whose physically relevant variation is a fraction of an eV per atom,
i.e. a relative $\sim10^{-4}$. A field-level relative $L^2$ of
$2\times10^{-2}$ cannot survive that. Reaching 0.01 eV/atom would need
densities accurate to roughly $10^{-5}$ relative — three orders of magnitude
beyond the current models.

```{note}
The grid is *not* the limiting factor. On reference fields the total energy
changes by 0.08 eV out of 31215 (0.003 eV/atom) between `resolution: 32` and
the native 128³ mesh, and $T_\mathrm{s}$ and the electron count are preserved
exactly, because spectral resampling is an exact band-limited projection.
```

Rescaling the predicted density to the known electron count does **not** help:
$E_\mathrm{ext}$ is linear in $\rho$, so correcting a 0.3% electron-count error
shifts a $-12690$ eV term by about 38 eV. It was measured and rejected, not
assumed.
