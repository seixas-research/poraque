# What the two FNOs actually learn, in DFT terms

**Scope:** `poraque.ml` · **Companion:** `docs/notes/pi_fno.md` (the physics-informed roadmap), `scripts/poraque_train.py`

---

## 1. The two models are the two halves of orbital-free DFT

It is tempting to read `EXTCAR → CHGCAR` and `CHGCAR → TAUCAR` as two unrelated
regressions that happen to share a data format. They are not. They are the two
maps that, composed, constitute a complete orbital-free DFT calculation.

The total energy of an electronic system is

$$
E[\rho] \;=\; \underbrace{T_s[\rho]}_{\textbf{Model 2}} \;+\; \int V_{\rm ext}\,\rho\,d^3r \;+\; E_H[\rho] \;+\; E_{xc}[\rho],
$$

and the ground state is the density that minimises it subject to
$\int\rho\,d^3r = N$. So:

| | Model | Learns | Status in DFT |
| --- | --- | --- | --- |
| **1** | $V_{\rm ext}\mapsto\rho$ | the *solution* of the variational problem | a **theorem** (Hohenberg–Kohn); exists and is unique |
| **2** | $\rho\mapsto\tau$ | the *functional* being minimised | the **open problem** of OF-DFT |

That asymmetry is the single most important thing to understand about these
two models, and it governs everything below.

---

## 2. Model 1 — `EXTCAR → CHGCAR` is the Hohenberg–Kohn map

### 2.1 Why the map is well-posed

The first Hohenberg–Kohn theorem states that the ground-state density
determines the external potential up to a constant, and conversely that for a
given electron number the ground-state density is a *functional of
$V_{\rm ext}$ alone*:

$$
V_{\rm ext}(\mathbf r) \;\longleftrightarrow\; \rho_0(\mathbf r).
$$

This is a strong statement about what the network is being asked to do. The
target is not a fitted correlation — it is a mathematically guaranteed,
single-valued function of the input. A supervised model of it is
learning an object that provably exists.

### 2.2 What makes it hard

The map is *non-local and self-consistent*. In Kohn–Sham language the density
is reached only after solving

$$
\left(-\tfrac12\nabla^2 + V_{\rm ext} + v_H[\rho] + v_{xc}[\rho]\right)\psi_i = \varepsilon_i\psi_i,
\qquad \rho = \sum_i^{\rm occ} f_i|\psi_i|^2,
$$

to self-consistency. The density at a point depends on the potential
*everywhere* — through the Hartree term, which is a $1/|\mathbf r - \mathbf r'|$
convolution, and through the orbitals, which are delocalised Bloch states.

This is precisely why an FNO is the right architecture rather than a CNN. A
convolutional network has a finite receptive field and would have to be made
very deep to propagate information across the cell. The FNO's spectral
convolution is **global in one layer**: multiplying in reciprocal space is
convolving over the whole cell in real space. The Hartree kernel it must
partly represent, $4\pi e^2/G^2$, is *diagonal* in exactly the basis the FNO
operates in. The architecture matches the physics of the operator.

The FFT also imposes Born–von Kármán boundary conditions automatically, which
is not a convenience but a correctness requirement: a crystal unit cell *is*
periodic, and a padded CNN would spend capacity learning to imitate that.

### 2.3 Measured result

Same protocol, in e/Å³ on the held-out material:

| Predictor | relative $L^2$ | $R^2$ | MAE |
| --- | --- | --- | --- |
| **FNO (learned)** | **0.067** | **0.992** | **0.039** |
| mean density | 0.776 | 0.000 | 0.522 |

The electron count $\int\rho\,d^3r$ is recovered to within 2.3–2.8 % without
any constraint enforcing it — which is precisely the degree of freedom the
charge-normalisation head of `docs/notes/pi_fno.md` §3.2 would pin exactly, for free.

### 2.4 Two caveats worth stating plainly

**The map is conditional on $N$.** HK fixes $\rho$ given $V_{\rm ext}$ *at
fixed electron number*. In the current dataset $N = \sum_a Z_a^{\rm val}$ is
implied by the geometry, so the network can infer it — but for charged systems
or a mixed-chemistry dataset, $N$ must become an explicit input. The
architecture already supports this through FiLM conditioning
(`fno.CellEncoder`).

**Our $V_{\rm ext}$ is a model potential.** `ExternalPotential` builds the
long-range local ionic potential from `ZVAL` with a Gaussian-smeared
pseudo-ion; VASP's `EXTCAR` is the true PAW local pseudopotential. Validation
against the reference (see `scripts/validate_vasp_data.py`) gives Pearson
$r = 0.992$ at the best-fit width but a residual relative $L^2$ of $\simeq 0.13$,
concentrated near the cores. The learned map is therefore the HK map *for our
model potential*, which is well-defined and self-consistent across the dataset,
but is not literally VASP's. Training on the provided `EXTCAR` removes the
ambiguity and is the recommended default.

---

## 3. Model 2 — `CHGCAR → TAUCAR` is the kinetic energy density functional

### 3.1 Why this is the valuable one

$T_s[\rho]$ is the *only* term of $E[\rho]$ with no accurate explicit density
functional. Kohn–Sham DFT sidesteps the problem by reintroducing orbitals and
computing $T_s$ exactly from them — which is also the reason KS-DFT costs
$\mathcal{O}(N^3)$. Orbital-free DFT eliminates the orbitals and scales
near-linearly, but it is only as good as its KEDF, and the classical
approximations are not good enough for chemistry.

A learned $\rho\mapsto\tau$ is a direct attack on that gap. **This model is the
scientific contribution; Model 1 is the infrastructure around it.**

### 3.2 The exact constraints that make it tractable

Unlike the HK map, $\tau$ obeys several relations that are *theorems*, and they
are what make learning it well-posed rather than an unconstrained fit:

$$
\tau(\mathbf r) \;\ge\; \tau_{\rm vW}(\mathbf r) = \frac{|\nabla\rho|^2}{8\rho}
\qquad\text{(Hoffmann-Ostenhof; exact for one nodeless orbital)}
$$

$$
\tau \;\longrightarrow\; \tau_{\rm TF} = C_{\rm TF}\,\rho^{5/3}
\qquad\text{(uniform-gas limit)}
$$

$$
T_s[\rho_\lambda] = \lambda^2\,T_s[\rho],\quad \rho_\lambda(\mathbf r) = \lambda^3\rho(\lambda\mathbf r)
\qquad\text{(exact coordinate scaling)}
$$

The decomposition $\tau = \tau_{\rm vW} + \tau_P$ with the Pauli term
$\tau_P \ge 0$ is the natural target: $\tau_{\rm vW}$ is known analytically
from $\rho$, so a network that predicts only $\tau_P$ starts from a correct
baseline instead of re-learning a closed-form expression. See §3 of
`docs/notes/pi_fno.md` for the recommended $\tau = \tau_{\rm vW} + \mathrm{softplus}(f_\theta)$
head, which makes the bound structural rather than penalised.

### 3.3 Measured result

On the Au₂₇ dataset (2 structures, 32³ spectral downsample, 200 epochs),
leave-one-out, evaluated in physical units (eV/Å³) on the **held-out**
material:

| Predictor | relative $L^2$ | $R^2$ | MAE |
| --- | --- | --- | --- |
| **FNO (learned)** | **0.125** | **0.967** | **1.43** |
| von Weizsäcker | 0.739 | 0.015 | 8.90 |
| Thomas-Fermi | 1.348 | −2.289 | 9.41 |
| TF + vW/9 | 1.358 | −2.338 | 9.38 |

The integrated kinetic energy $T_s=\int\tau\,d^3r$ — the quantity that actually
enters the total energy — is reproduced to **2.0 %** on both folds.

The classical functionals are not merely worse — Thomas-Fermi has *negative*
$R^2$, i.e. it is a worse pointwise predictor of $\tau$ than the constant mean.
That is expected and is exactly the known failure of TF for a $d$-band metal
with strong density inhomogeneity: TF is the uniform-gas limit, and gold's
core and $d$-shell regions are as far from uniform as matter gets. The learned
operator beating it by an order of magnitude is the result that motivates the
whole programme.

**Caveat, stated once and meant:** with two closely related structures these
numbers measure interpolation between two nearby geometries of one element.
They demonstrate that the pipeline is correct and that the architecture has
enough capacity. They say nothing about transfer to new chemistry, and should
not be quoted as if they did.

---

## 4. How the two models close into one physical loop

Trained separately, the two models are useful. Coupled, they become an OF-DFT
solver. The link is the Euler–Lagrange equation obtained by minimising
$E[\rho]$ at fixed $N$:

$$
\boxed{\;\frac{\delta T_s}{\delta\rho}(\mathbf r) \;+\; V_{\rm ext}(\mathbf r) \;+\; v_H[\rho](\mathbf r) \;+\; v_{xc}[\rho](\mathbf r) \;=\; \mu\;}
$$

with $\mu$ **constant** in space. Every symbol here is something we have:

* $V_{\rm ext}$ — the input to Model 1 (`EXTCAR`);
* $\rho$ — the output of Model 1 (`CHGCAR`);
* $v_H[\rho]$ — computed *exactly* by FFT, `physics.hartree_potential`;
* $\delta T_s/\delta\rho$ — obtained by autograd through Model 2, since
  $T_s = \int\tau_\theta[\rho]\,d^3r$;
* $\mu$ — eliminated by subtracting the cell average, since it is a constant.

So the residual

$$
r(\mathbf r) = \frac{\delta T_s}{\delta\rho} + V_{\rm ext} + v_H[\rho] + v_{xc}[\rho] - \overline{(\cdots)}
$$

is computable from the two networks and their own inputs, with **no additional
labels**, and must vanish for a correct pair $(\rho, \tau)$.

This is the concrete connection to the physics-informed programme:

```
        ┌──────────── Euler–Lagrange residual (label-free) ────────────┐
        │                                                              │
 EXTCAR ──▶ [ FNO 1 : V_ext ↦ ρ ] ──▶ CHGCAR ──▶ [ FNO 2 : ρ ↦ τ ] ──▶ TAUCAR
        │                     │                                 │
        │                     └──▶ v_H[ρ]  (exact, FFT)         │
        └───────── δT_s/δρ  via autograd through FNO 2 ◀─────────┘
```

Three consequences, in order of practical value:

1. **Model 2 gets regularised on Model 1's output distribution.** A KEDF
   trained only on VASP-converged densities fails when placed inside an SCF
   loop, because it then sees *non-converged* densities it was never shown.
   Joint training on the residual exposes it to exactly that distribution.
2. **The optimised quantity becomes the used quantity.** OF-DFT needs
   $\delta T_s/\delta\rho$, not $\tau$. A model with small error in $\tau$ can
   have a poor functional derivative. Training through the derivative fixes the
   objective mismatch.
3. **Inference becomes physically consistent.** A predicted $(\rho,\tau)$ pair
   satisfies the variational condition it is supposed to satisfy, rather than
   merely resembling the training data.

---

## 5. Why physics-informed terms are cheap *here* specifically

Physics-informed training normally costs a great deal: derivatives must be
obtained by autograd through the network or by finite differences, and the
latter introduces truncation error the loss then misattributes to the model.

For an FNO on a plane-wave grid, neither problem arises:

| Physical operator | On this grid | Cost |
| --- | --- | --- |
| $\nabla$ | $i\mathbf{G}$ | one FFT, **exact** |
| $\nabla^2$ | $-G^2$ | one FFT, **exact** |
| $v_H[\rho]$ | $4\pi e^2\rho(\mathbf G)/G^2$ | one FFT, **exact** |
| $\tau_{\rm vW}$ | $\lvert\nabla\rho\rvert^2/8\rho$ | one FFT, **exact** |

These are *exact* for band-limited fields, which is precisely what a plane-wave
DFT grid carries — and the FNO is already computing FFTs in every layer. The
physics and the architecture share the same mathematical substrate. This is the
strongest technical argument for choosing an FNO over a CNN for this problem,
independent of accuracy.

All of the above is implemented and unit-tested in `poraque/ml/physics.py`;
`docs/notes/pi_fno.md` gives the staged roadmap for switching the terms on.

---

## 6. Summary

* **Model 1** learns a map guaranteed to exist by Hohenberg–Kohn. Its
  difficulty is non-locality and self-consistency, which the FNO's global
  spectral convolution addresses directly.
* **Model 2** learns the object orbital-free DFT is missing. It is constrained
  by exact theorems ($\tau \ge \tau_{\rm vW}$, the uniform-gas limit, coordinate
  scaling) that should be built into the architecture rather than penalised.
* Measured on Au₂₇, the learned KEDF beats Thomas-Fermi and von Weizsäcker by
  roughly an order of magnitude in relative $L^2$ — on two related structures,
  which bounds what may be concluded.
* The **Euler–Lagrange equation** couples the two models through quantities
  that are all either network outputs or exactly computable by FFT, giving a
  label-free training signal and turning the pair into a differentiable OF-DFT
  solver.
