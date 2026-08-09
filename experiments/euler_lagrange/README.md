# Does a third operator for the Euler-Lagrange residual earn its place?

Three experiments, run on the 17-structure Au cache (`data/cache/res32_potcar`)
because it was what existed. **Re-run all three on the new data set before
drawing any conclusion from them**: 13 training structures of one element is
not a basis for a claim about architecture, and the overfitting numbers below
say so explicitly.

```bash
python experiments/euler_lagrange/el_floor.py       # how badly is EL violated?
python experiments/euler_lagrange/el_xc.py          # does the right xc help?
python experiments/euler_lagrange/el_operator.py    # can an operator fix it?
python experiments/euler_lagrange/lps_check.py      # is LPS a better route?
```

All four use **PBE**, which is what the reference data was computed with
(`PAW_PBE` potentials, `LEXCH = PE`). `poraque.fields.io.resolve_xc` reads that
off the calculation, so `data.xc: auto` is enough; declare `data.xc: pbe`
explicitly for data whose settings are not recorded beside it.

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

## 1. The residual floor (`el_floor.py`)

How far from satisfied is EL on data that is by construction the ground state?

| kinetic functional | xc | std(r) [eV] | std(r)/std(v_ext) |
|---|---|---|---|
| TF | off | 1.99 | 0.055 |
| TF | LDA | 3.45 | 0.095 |
| TF + (1/9) vW | LDA | 2.37 | 0.065 |
| **TF + (1/5) vW** | **LDA** | **1.73** | **0.048** |
| TF + vW | LDA | 8.57 | 0.236 |
| vW only | LDA | 11.56 | 0.319 |

So the violation is a **correction**, not the leading term, and it is strongly
structured: Pearson correlation -0.85 with tau, -0.76 with rho^(1/3), +0.75
with v_ext. Structured means learnable.

Note the tuned coefficient is near 1/5, not the 1/9 of the second-order
gradient expansion and not the 1 of von Weizsaecker. That number matters for
experiment 3.

## 1b. Does the right functional help? (`el_xc.py`)

The first pass used an LDA `v_xc` against PBE data. That is wrong, and worth
measuring rather than assuming, because the error would land in the residual
and read as the error of the kinetic functional.

std of the residual [eV], by kinetic baseline and xc:

| baseline | none | lda | **pbe** |
|---|---|---|---|
| TF | 1.99 | 3.45 | 3.74 |
| TF + vW/9 | 2.56 | 2.37 | 2.66 |
| TF + vW/5 | 3.32 | **1.73** | 1.97 |
| TF + vW/3 | 4.65 | 1.71 | **1.73** |
| TF + vW | 12.05 | 8.57 | 8.33 |

**It does not.** Best with LDA 1.714 eV, best with PBE 1.731 eV: +1.0 %. The
xc error is largely reabsorbed by re-tuning lambda, whose optimum moves from
1/5 to 1/3. So use PBE because it is what the data is, not because it shrinks
anything. The residual is dominated by the kinetic functional and by whatever
the local-potential picture is missing, not by the choice of xc.

## 2. The operator (`el_operator.py`)

Learns `dv[rho] = v_kin_exact - (TF + lam*vW)` as a functional of **rho alone**.
Not of v_ext: `T_s[rho]` is universal, and a functional that consumed the
external potential would not be one.

Everything is zero-mean, so a perfect prediction annihilates the residual and
the error in the residual *is* the error in `dv`. That makes the baseline
unarguable: predicting zero is "apply no correction".

Held out (4 of 17 structures, split by material, seed 42):

| model | parameters | held-out (LDA, lam=1/5) | held-out (**PBE, lam=1/3**) | training fit |
|---|---|---|---|---|
| no correction | 0 | 1.00x | 1.00x | 1.00x |
| **semi-local least squares** | **9** | **3.06x** | **1.90x** | - |
| FNO w8 m6 L2 | 114k | 2.44x | - | 4.1x |
| FNO w16 m8 L3 | 1.58M | 2.42x | - | 24x |
| FNO w24 m10 L4 | 9.23M | 3.24x | 2.13x | 920-1030x |

Read the two columns together, not separately. The PBE/lam=1/3 baseline is
slightly better on its own (1.675 vs 1.712 eV) and leaves a *less structured*
remainder, so both the fit and the operator score lower skill on it. In
absolute terms the LDA/lam=1/5 chain still ends lower: 1.712 -> 0.529 eV
against 1.675 -> 0.788 eV. A baseline that leaves a larger but more structured
residual can beat one that leaves a smaller but noisier residual. With 13
structures that difference is not significant, and it is recorded here as a
thing to re-measure, not a finding.

**The idea works and the architecture does not earn its keep.** The residual is
learnable, and a 2-3x reduction is real. But a nine-parameter linear
least-squares fit on `(rho, |grad rho|, lap rho)` reaches 3.06x where 9.2
million FNO parameters reach 3.24x. Six percent, for a million-fold increase in
parameters. Under PBE the gap is wider in relative terms (1.90x vs 2.13x) and
still small in absolute ones.

The training-fit column is the diagnosis: 1030x on the data it saw against
3.24x on data it did not. It is memorising 13 structures, not learning a
functional.

Two readings, and this data cannot separate them:

* The residual left after a *tuned* semi-local baseline is itself mostly
  semi-local, so there is little non-local content for the FNO to find.
* Thirteen fcc-like gold structures at similar densities never exercise the
  non-locality, which is a real property of the exact functional.

The second is why this must be re-run on chemically diverse data before the
operator is abandoned.

## 3. Levy-Perdew-Sahni (`lps_check.py`)

LPS writes the density as one effective orbital,

    -1/2 lap sqrt(rho) + [v_ext + v_H + v_xc + v_P] sqrt(rho) = mu sqrt(rho)

Dividing by `sqrt(rho)` and using `dT_vW/drho = -1/2 lap sqrt(rho)/sqrt(rho)`
gives back Euler-Lagrange with `dTs/drho` split into bosonic and Pauli parts.
Verified in code to **1.8e-15 eV**: LPS is a change of variable, not an
independent condition. There is no separate equation to test.

What differs is conditioning, and it is worse:

| formulation | bosonic baseline | target std [eV] |
|---|---|---|
| LPS (Pauli potential) | vW, **fixed by the equation** | 11.56 |
| EL, tuned | TF + (1/5) vW | **1.73** |

LPS pins the bosonic baseline at the full von Weizsaecker term, and experiment
1 showed the best coefficient is near 1/5. So the LPS target is **6.7x wider**
than the EL target, purely because the equation removes the freedom to tune it.

LPS does buy one thing EL does not: Levy-Ou-Yang gives `v_P >= 0` pointwise, a
hard constraint on the derivative, which is the object that should be
constrained. On this data it does not bind. Taking the smallest mu that makes
`v_P` non-negative, the binding voxels are 0.00-0.04 % of the cell and sit in
the low-density tail (rho ~ 0.2 e/Ang^3 against a cell mean of 0.61), never in
the bonding region.

**Verdict:** LPS is not an alternative to the current formulation, it is the
same formulation with the baseline frozen at a worse value. The part worth
importing is the constraint, not the change of variable, and it can be imposed
inside the EL formulation by parameterising the learned correction so that
`v_kin - dT_vW/drho >= 0`.

## What was fixed along the way

`euler_lagrange_residual` has always taken a `v_xc` argument and nothing in the
package could produce one, so **every residual ever computed here silently
omitted exchange and correlation**. In a valence region `v_xc` is about -16 eV,
the same order as the kinetic potential it is weighed against.

Now `poraque.ml.xc_potential`, covering LDA (Dirac plus PW92, closed form) and
**PBE** (the divergence term `-div(de/d grad rho)` taken by autograd rather
than derived by hand). Validated against `poraque.physics.energy` by finite
differences to 1e-8, against the exchange virial relation exactly, and against
the requirement that PBE collapse onto LDA on a uniform density (3.6e-15 eV).
It is differentiable with respect to the density, so an Euler-Lagrange residual
built from a *predicted* density still trains. See `tests/test_xc_potential.py`.

`poraque.fields.io.resolve_xc` resolves the functional in the order VASP does:
the `INCAR` `GGA` tag, else the pseudopotential `LEXCH` tag. It warns rather
than substituting silently when a functional is recognised but not implemented
(PW91, PBEsol, RPBE), because a mislabelled `v_xc` in the residual would read
as an error of the kinetic functional.
