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
| $E_\mathrm{xc}$ | Dirac exchange + PW92 correlation (LDA) |
| $E_\mathrm{Ewald}$ | Ewald sum over the pseudo-ions with a neutralizing background |

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
  xc (lda)                   -3820.927158 eV
  Ewald                     -25949.345827 eV
  ----------------------------------------
  potential                 -37422.141455 eV
  TOTAL                     -31215.073155 eV
  electrons                    297.000001
```

```{tip}
`electrons` is the cheapest available diagnostic. It should equal the sum of
the `ZVAL`s — 297 for 27 Au atoms above — to a few parts in $10^4$. A predicted
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

atoms = bulk("Au", "fcc", a=4.08, cubic=True)
atoms.calc = Poraque(ext2chg="models/ext2chg.pt",
                     chg2tau="models/chg2tau.pt",
                     potcar="POTCAR")

energy = atoms.get_potential_energy()
print(atoms.calc.components)          # the full decomposition
rho = atoms.calc.fields["density"]    # the predicted CHGCAR
```

Each call runs `Atoms → Structure → FieldGrid → V_ext → ρ → τ → E`. The three
fields stay on the calculator afterwards, so a prediction can be written out
with `rho.write("CHGCAR")` and opened in VESTA.

### Options

| Argument | Meaning |
| --- | --- |
| `ext2chg`, `chg2tau` | checkpoint paths, or already-loaded `FieldOperator`s |
| `potcar` | `POTCAR` for the species present — **strongly preferred** |
| `charges` | `{element: Z_val}`, used only when `potcar` is absent |
| `resolution` | longest grid axis; defaults to the checkpoint's training resolution |
| `functional` | exchange-correlation approximation |
| `device` | `auto`, `cuda`, `mps` or `cpu` |

```{warning}
Without `potcar` the external potential falls back to the **Gaussian
pseudo-ion model**, which differs from the tabulated potential by a relative
$L^2$ of about `0.13` against `2×10⁻⁵` for the tabulated one. The operators
were trained on tabulated potentials, so the Gaussian model feeds them an
input outside their training distribution. The calculator warns once and
proceeds; treat the result as a smoke test, not a prediction.
```

### Forces and stress

`get_forces()` and `get_stress()` raise `NotImplementedError`, so **geometry
optimisation and molecular dynamics are unavailable**. Single points,
energy-volume scans and ranking of fixed geometries work.

Two independent pieces are missing: the derivative of $V_\mathrm{ext}$ with
respect to the ionic positions (analytic, a Hellmann-Feynman term), and
back-propagation of $\partial E/\partial\rho$ through both operators. A
finite-difference stand-in is not a shortcut here — on a fixed grid it would be
dominated by the grid's own discontinuity as atoms cross voxel boundaries.

`implemented_properties` is `["energy", "free_energy"]`. The two are the same
number: no electronic smearing enters this pipeline, and ASE optimizers request
`free_energy` by name.

(what-the-number-is-not)=
## What the number is not

**It is not a DFT total energy.** `CHGCAR` holds the valence *pseudo* density
and `TAUCAR` the valence pseudo kinetic energy density, so the PAW one-centre
terms are absent entirely. For the 27-atom Au cell above the result is
$\approx -31215$ eV against a VASP `TOTEN` of order $-100$ eV. Nothing in this
module can close that gap.

**Energy differences are not yet usable either.** Measured on the seven
reference Au supercells, with the models evaluated on their own training data:

| Quantity | Value |
| --- | --- |
| True spread of $E$ across the seven structures | 7.9 eV |
| MAE of the predicted differences | 22.3 eV (0.83 eV/atom) |
| Correlation of predicted vs. true differences | $r = 0.61$ |

The error is roughly **three times the signal**. The cause is cancellation, not
a bug: the total is a sum of terms of order $10^4$ eV whose physically relevant
variation is a few eV, i.e. a relative $2.5\times10^{-4}$. A field-level
relative $L^2$ of $3\times10^{-2}$ cannot survive that. Reaching 0.01 eV/atom
would need densities accurate to roughly $10^{-5}$ relative — three orders of
magnitude beyond the current models.

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
