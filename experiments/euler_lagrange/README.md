# Is the Euler-Lagrange equation satisfied, and what does the violation look like?

Measured on the 31-structure platinum set (`data/cache/res32_potcar_spin`, built
from `data/vasp/structures` at `resolution: 32`). Reproduce with:

```bash
python experiments/euler_lagrange/el_resolution.py  # is the number the grid?
python experiments/euler_lagrange/el_floor.py       # how badly is EL violated?
python experiments/euler_lagrange/el_xc.py          # does the right xc help?
python experiments/euler_lagrange/lps_check.py      # is LPS a better route?
```

`PORAQUE_EL_CACHE` overrides the cache path; `PORAQUE_POTCAR_DIR` overrides the
pseudopotential library `el_resolution.py` reads.

All of them use **PBE**, which is what the reference data was computed with
(`PAW_PBE` potentials). `poraque.fields.io.resolve_xc` reads that off the
calculation where it is recorded, and warns and assumes PBE where it is not —
these runs set no `GGA` tag and keep no `POTCAR`, so the assumption is the one
in force here and it is the right one.

## The construction

At the ground state the Euler-Lagrange equation holds pointwise,

    dTs/drho + v_ext + v_H[rho] + v_xc[rho] = mu

and every term but the first is known exactly. So on reference data it can be
**inverted** for the one quantity that has no label:

    dTs/drho = mu - v_ext - v_H - v_xc

This is the point of the whole exercise. A functional derivative needs a
functional, and `TAUCAR` is a field, so no amount of it yields `dTs/drho`. The
Euler-Lagrange equation is the only route from a ground-state density to a
pointwise target for the quantity orbital-free theory actually consumes.
Implemented as `poraque.ml.exact_kinetic_potential`.

"Is EL satisfied" is therefore not a question about the data — inverting it
makes it hold by construction. It is a question about the **approximate**
kinetic functionals: how far from constant is `dTs/drho + v_ext + v_H + v_xc`
when `dTs/drho` is one of the standard orbital-free forms? That is what the
residual measures, and its cell average is removed because `mu` is the constant
the equation is allowed to have.

## 0. Is the residual physics or the grid? (`el_resolution.py`)

Settled first, because the residual is built from second derivatives of a
band-limited density and Gibbs ringing near a core would manufacture one out of
nothing. Resampled in-process from the native calculations, so the only
variable is the cutoff:

| grid | std(r) [eV], xc off | std(r) [eV], xc on |
|---|---|---|
| 24³ | 3.556 | 6.006 |
| **32³** | **2.740** | **5.707** |
| 48³ | 2.746 | 5.708 |
| 64³ | 2.751 | 5.709 |
| 108³ (native) | 2.752 | 5.710 |

**Converged from 32³ up** — 0.4 % between the cache resolution and the native
grid. Only 24³ is under-resolved. Everything below is measured on a grid that
is not limiting it.

## 1. The residual floor (`el_floor.py`)

Term magnitudes on one structure, so the residual can be read against the
things it is made of (eV):

| field | mean | std | min | max |
|---|---|---|---|---|
| `v_ext` | 0.000 | 31.630 | -116.159 | 33.043 |
| `v_H[rho]` | 0.000 | 13.627 | -19.151 | 54.632 |
| `v_xc[rho]` | -12.801 | 3.417 | -24.002 | -8.592 |
| `dT_TF/drho` | 25.132 | 14.930 | 9.271 | 88.713 |
| `dT_vW/drho` | -4.195 | 10.626 | -18.052 | 32.096 |

The residual, over all 31 structures:

| kinetic functional | xc | std(r) [eV] | std(r)/std(v_ext) | max abs r [eV] |
|---|---|---|---|---|
| TF | off | 3.662 | 0.114 | 11.5 |
| TF | on | 6.774 | 0.210 | 19.3 |
| TF + (1/9) vW | off | 2.766 | 0.086 | 10.0 |
| TF + (1/9) vW | on | 5.697 | 0.177 | 17.9 |
| **TF + (1/5) vW** | off | **2.245** | **0.070** | 9.9 |
| TF + (1/5) vW | on | 4.869 | 0.151 | 16.7 |
| TF + vW | off | 7.832 | 0.243 | 38.3 |
| TF + vW | on | 4.979 | 0.154 | 27.1 |
| vW only | off | 9.271 | 0.287 | 38.9 |
| vW only | on | 12.298 | 0.381 | 47.9 |

**Euler-Lagrange is not satisfied by any of them.** The best case leaves a
residual of 2.2 eV standard deviation against a `v_ext` that swings by 31.6 eV
— 7 %, with excursions to 9.9 eV. So the violation is a correction rather than
the leading term, but it is far too large to ignore: 7 % of the potential is
several times the energy differences the whole pipeline exists to resolve.

The tuned coefficient sits near 1/5, neither the 1/9 of the second-order
gradient expansion nor the 1 of von Weizsaecker.

### The residual is structured, and almost entirely pointwise

Pearson correlation against candidate explanatory fields, averaged over
structures:

| field | corr | field | corr |
|---|---|---|---|
| `rho` | -0.9354 | `v_ext` | +0.9416 |
| `rho^(1/3)` | -0.9257 | `v_H` | -0.8726 |
| `rho^(2/3)` | -0.9413 | **`tau`** | **-0.9610** |
| | | `tau/rho` | -0.3116 |

Correlation assumes a straight line, so the sharper measurement drops that
assumption: bin by a field, take the mean residual per bin, and ask how much
variance that one-dimensional curve accounts for.

| best pointwise function of | R² | left over |
|---|---|---|
| `rho` | 0.9794 | 2.1 % |
| **`tau`** | **0.9867** | **1.3 %** |
| `v_ext` | 0.9695 | 3.1 % |

**98.7 % of the violation is a pointwise function of tau alone.** That is the
central result: on this data the Euler-Lagrange violation is *local*, and at
most 1.3 % of its variance is anything else.

The caveat is not decoration. These are 31 fcc-like platinum cells at similar
densities, and non-locality is a real property of the exact functional that
such a set never exercises. The number bounds what is true *here*, not what is
true in general.

## 1b. Does the right functional help? (`el_xc.py`)

std of the residual [eV], by kinetic baseline and xc:

| baseline | none | lda | **pbe** |
|---|---|---|---|
| TF | 3.662 | 6.558 | 6.774 |
| TF + vW/9 | 2.766 | 5.489 | 5.697 |
| TF + vW/5 | 2.245 | 4.671 | 4.869 |
| TF + vW/3 | **2.109** | 3.560 | 3.730 |
| TF + vW | 7.832 | 5.179 | 4.979 |

Best with LDA 3.560 eV, best with PBE 3.730 eV: using the functional the data
was actually computed with is **4.8 % worse**, and omitting exchange and
correlation altogether is better than either.

**That is not evidence that `v_xc` does not belong.** It belongs — the equation
is not the equation without it. What the table shows is that the terms are
trading against one another: `v_xc` is a local function of `rho` with 3.4 eV of
spatial variation, the error in the approximate kinetic potential is also very
nearly a local function of `rho` (R² = 0.979 above), and the two partly cancel.
Dropping `v_xc` moves along that same one-dimensional curve. Read together with
the pointwise result, the honest conclusion is that no individual term here is
separately right, and the apparent improvement from omitting one of the terms
that *is* known exactly is a coincidence of sign, not a finding.

What the residual most likely contains, beyond the kinetic-functional error, is
the **non-local part of the PAW pseudopotential** — large for a 5d metal,
atom-centred, and absent from a local `v_ext` by construction. This measurement
cannot separate the two contributions, and nothing here should be read as
attributing the whole residual to `dTs/drho`.

## 2. Levy-Perdew-Sahni (`lps_check.py`)

LPS writes the density as one effective orbital,

    -1/2 lap sqrt(rho) + [v_ext + v_H + v_xc + v_P] sqrt(rho) = mu sqrt(rho)

Dividing by `sqrt(rho)` and using `dT_vW/drho = -1/2 lap sqrt(rho)/sqrt(rho)`
gives back Euler-Lagrange with `dTs/drho` split into bosonic and Pauli parts.
Verified in code to **7.1e-15 eV**: LPS is a change of variable, not an
independent condition. There is no separate equation to test.

What differs is conditioning, and it is worse:

| formulation | bosonic baseline | target std [eV] |
|---|---|---|
| LPS (Pauli potential) | vW, **fixed by the equation** | 12.298 |
| EL, best tuned | TF + (1/5) vW | **4.869** |

LPS pins the bosonic baseline at the full von Weizsaecker term, and experiment 1
showed the best coefficient is near 1/5. The LPS target is **2.5x wider** than
the EL target, purely because the equation removes the freedom to tune it.

LPS does buy one thing EL does not: Levy-Ou-Yang gives `v_P >= 0` pointwise, a
hard constraint on the derivative, which is the object that should be
constrained. On this data it barely binds. Taking the smallest `mu` that makes
`v_P` non-negative, the binding voxels are 0.03-0.78 % of the cell and sit at
`rho ~ 0.16 e/Ang^3` against a cell mean of 0.64 — the low-density tail, never
the bonding region. A Thomas-Fermi Pauli potential is non-negative by
construction and never violates the bound at all, so the constraint is
insurance for a *learned* potential, exactly as with the tau head.

**Verdict:** LPS is not an alternative to the current formulation, it is the
same formulation with the baseline frozen at a worse value. The part worth
importing is the constraint, not the change of variable, and it can be imposed
inside the EL formulation by parameterising the learned correction so that
`v_kin - dT_vW/drho >= 0`.

## 3. What is left, and what it would take to model it

The pointwise result above settles the shape of the answer: on this data 98.7 %
of the violation is a function of `tau` at the same point, and a nine-parameter
local fit would already reach R² ≈ 0.98. So the useful next measurement is not
"can the residual be learned" — it plainly can — but **what remains after a
pointwise function of `tau` is subtracted**, and whether that remainder has any
length scale at all or is just the noise floor of the construction.

That is a measurement on the existing fields: fit the pointwise function by
bin-and-average, subtract it, and look at the autocorrelation of what is left.
If the remainder is structureless, the Euler-Lagrange violation on this data is
a local correction to `TF + lam*vW` and belongs in a closed form —
`poraque.ml.symbolic` already searches exactly that space, and the target here
is one variable rather than two.

The caveat stated throughout applies with full force: 31 fcc-like cells of one
element at similar densities is precisely the setting in which a non-local
effect would not appear even if it dominated elsewhere. The number bounds this
dataset, not the physics.

## What `v_xc` required

`euler_lagrange_residual` takes a `v_xc` argument, and for a long time nothing
in the package could produce one. Now `poraque.ml.xc_potential` covers LDA
(Dirac plus PW92, closed form) and **PBE** (the divergence term
`-div(de/d grad rho)` taken by autograd rather than derived by hand). Validated
against `poraque.physics.energy` by finite differences to 1e-8, against the
exchange virial relation exactly, and against the requirement that PBE collapse
onto LDA on a uniform density (3.6e-15 eV). It is differentiable with respect to
the density, so an Euler-Lagrange residual built from a *predicted* density
still trains. See `tests/test_xc_potential.py`.

`poraque.fields.io.resolve_xc` resolves the functional in the order VASP does:
the `INCAR` `GGA` tag, else the pseudopotential `LEXCH` tag. It warns rather
than substituting silently when a functional is recognised but not implemented
(PW91, PBEsol, RPBE), because a mislabelled `v_xc` in the residual would read as
an error of the kinetic functional.
