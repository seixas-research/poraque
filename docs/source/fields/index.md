# Scalar fields

## The shared-grid invariant

For one material, *every* field lives on a single
{py:class}`~poraque.fields.FieldGrid` and is serialised in a single format.
That invariant is what makes the fields comparable, composable, and usable as
aligned input/target pairs.

| Class | File | Quantity | Unit |
| --- | --- | --- | --- |
| {py:class}`~poraque.fields.ExternalPotential` | `EXTCAR` | $V_\mathrm{ext}$ | eV |
| {py:class}`~poraque.fields.ChargeDensity` | `CHGCAR` | $\rho$ | e/Å³ |
| {py:class}`~poraque.fields.KineticEnergyDensity` | `TAUCAR` | $\tau$ | eV/Å³ |

All three derive from {py:class}`~poraque.fields.ScalarField`, which owns the
`CHGCAR`-format reader and writer, so one code path handles every field.
Reading with a supplied grid *imposes* that grid; a mismatch raises rather than
passing silently.

```{note}
VASP stores $\rho\Omega$ and $\tau\Omega$ — the field multiplied by the cell
volume — but stores potentials unscaled. The classes declare their own
convention (`volume_scaled`), so `integrate()` returns the electron count
directly.
```

## The external potential

The valence electrons see a local ionic potential built from pseudo-ions of
charge $Z^\mathrm{val}_s$. It is assembled in reciprocal space, where the
Coulomb kernel is diagonal:

$$
V_\mathrm{ext}(\mathbf G) = \frac{1}{\Omega}\sum_s S_s(\mathbf G)
  \left[ v^s_\mathrm{short}(G) - \frac{4\pi Z^\mathrm{val}_s e^2}{G^2} \right],
\qquad V_\mathrm{ext}(\mathbf 0) \equiv 0 .
$$

Dropping the $\mathbf G = 0$ term is the neutralising-background convention of
periodic plane-wave codes; it fixes the cell average of the potential to zero.
The real-space field follows from one inverse FFT, so the construction is
$\mathcal O(N\log N)$, exactly periodic, and free of any Ewald parameter.

### Pseudo-ion models

| `model` | Form factor | Notes |
| --- | --- | --- |
| `"potcar"` | tabulated $v_\mathrm{short}(q)$ | **exact**; reproduces a reference `EXTCAR` to $2\times10^{-5}$ |
| `"gaussian"` | $e^{-G^2\sigma^2/2}$ | smooth, one width per species |
| `"coulomb"` | $1$ | bare point ions; visibly aliased near nuclei |
| `"auto"` | *(default)* | tabulated when the POTCAR tables are readable, else Gaussian |

The tabulated route is the default because no analytic form factor can
reproduce a real pseudopotential: inverting a reference potential shows the
true $f(G)$ **oscillating about zero** at large $G$, the signature of a
repulsive core.

### Optional Gaussian smoothing

```python
potential = ExternalPotential.from_calculation(
    directory, gaussian_blur=0.15, blur_method="spectral")
```

Smoothing is applied *on top of* the tabulated potential; it never replaces it.

```{warning}
`blur_method="ndimage"` uses {py:func}`scipy.ndimage.gaussian_filter` with
`mode="wrap"`, which blurs along **lattice** axes. On a non-orthogonal cell
that is anisotropic in Cartesian space — on an fcc cell the two methods differ
by 2–6 %. `"spectral"` multiplies by $e^{-G^2\sigma^2/2}$ using the true
reciprocal metric and is the default for that reason.
```

## Grids

A grid may be obtained in three ways, in decreasing order of reliability:

1. `FieldGrid.from_file(path)` — adopt an existing file's grid. The only way to
   guarantee point-by-point comparability with a reference calculation.
2. explicit `NGXF/NGYF/NGZF` tags from the input file;
3. `FieldGrid.from_encut(...)` — derived from the plane-wave cutoff.

```{warning}
The cutoff-derived shape is an *estimate*. Exact rounding is
version-dependent, and on the reference data it differs from what VASP chose.
Read the grid from the files when a match matters.
```

## Spectral resampling

Changing resolution is a **basis-truncation** problem, not an interpolation
problem. The field is a finite Fourier series, so restricting it to a coarser
grid means keeping the coefficients that grid can represent. The result is the
exact band-limited projection: still exactly periodic, and with the cell
average — hence the electron count — preserved to machine precision.

Interpolation would alias high frequencies onto low ones, break periodicity at
the cell boundary, and shift the integral.

```{note}
Band-limiting a strictly positive field with sharp core peaks rings slightly
(Gibbs), so a small number of voxels can go marginally negative. That is an
artefact of the truncation, and it is why the dataset uses a sign-tolerant
`asinh` normalisation rather than a logarithm.
```

## Ingesting other codes

Support for a new plane-wave code must not ripple through grids, fields,
datasets and models. A reader implements four methods:

| Method | Returns |
| --- | --- |
| `read_structure` | {py:class}`~poraque.fields.Structure` (Å, fractional, species-grouped) |
| `read_parameters` | cutoff in **eV**, precision, explicit grid shape |
| `read_pseudopotentials` | `{element: PseudopotentialInfo}` |
| `read_field` / `write_field` | {py:class}`~poraque.fields.ScalarField` |

Units are normalised at the boundary, so a Quantum ESPRESSO reader converts its
Rydberg `ecutwfc` there and nothing downstream needs to know. Field kinds are
referred to by neutral names (`external`, `density`, `kinetic`) rather than by
filename.

```python
from poraque.fields.io import register_reader, CalculationReader

class MyCodeReader(CalculationReader):
    code = "mycode"
    structure_files = ("mycode.in",)
    field_files = {"external": "...", "density": "...", "kinetic": "..."}
    ...

register_reader(MyCodeReader)
```

After that, `ExternalPotential.from_calculation(directory, code="mycode")` and
the dataset loader work unchanged.
