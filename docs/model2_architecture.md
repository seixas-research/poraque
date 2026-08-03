# Why Model 2 exists: `chg2tau` as a learned kinetic energy functional

**Scope:** architectural rationale for `poraque.ml` · **Companions:** `plan/pi_fno.md` (roadmap), `plan/fno_physics.md` (physics mapping), `docs/ked_formulation_analysis.md` (τ representation)

---

## 1. The question, stated without charity

A skeptic's version of the objection is worth writing down, because the weak
answers to it are the ones most often given:

> You already have Model 1, which predicts ρ from the geometry. If you have a
> converged DFT calculation you have τ for free, and if you don't have one, you
> have no ρ to feed Model 2 either. So what is `chg2tau` *for*?

Three answers are available. Two are weak. The third is the reason the model
exists.

### 1.1 "τ is a useful observable" — weak

τ enters meta-GGA functionals, the electron localisation function, and bonding
analysis. True, but it does not survive the objection: anyone holding a
converged `CHGCAR` also holds a `TAUCAR`. Predicting a quantity you already
have is a benchmark, not an application.

### 1.2 "It completes the pipeline" — weak

Chaining `V_ext → ρ → τ` produces a τ for a structure that was never computed.
Also true, and genuinely useful for screening. But it makes Model 2 a
convenience: nothing in it is more than interpolation, and its errors do not
feed back into anything.

### 1.3 "It *is* the orbital-free kinetic energy functional" — the actual reason

$T_s[\rho]$ is the **only** term of the DFT total energy with no accurate
explicit density functional. Kohn–Sham theory evades the problem by
reintroducing orbitals, and pays $\mathcal{O}(N^3)$ for the privilege.
Orbital-free DFT removes the orbitals and scales near-linearly, but it is only
as good as its KEDF — and the classical approximations are not good enough for
chemistry.

A model of $\rho \mapsto \tau$ **is** a KEDF, because

$$
T_s[\rho] \;=\; \int \tau[\rho](\mathbf r)\, \mathrm{d}^3r .
$$

That is the whole claim. Model 2 is not a predictor of a convenient
observable; it is a candidate solution to the central open problem of
orbital-free DFT, expressed in the one form that a neural network can
represent and that autograd can differentiate.

**Model 2 is the scientific contribution. Model 1 is the infrastructure around
it.**

---

## 2. Is τ even a functional of ρ?

Worth establishing before building on it, because the answer is subtler than
"Hohenberg–Kohn says so".

For the **integrated** quantity the argument is clean. HK gives
$\rho \to v_s$ uniquely (for non-interacting *v*-representable densities), and
$v_s$ determines the Kohn–Sham orbitals, hence $T_s$. So $T_s[\rho]$ exists as
a functional.

For the **density** $\tau(\mathbf r)$ the same chain applies —
$\rho \to v_s \to \{\psi_i\} \to \tau$ — so $\tau[\rho]$ also exists. But note
what that chain contains: solving the Kohn–Sham equations. The functional is
therefore

- **exact but non-local**: τ at one point depends on ρ *everywhere*, because
  the orbitals are delocalised Bloch states;
- **not semi-local**: no finite gradient expansion converges to it, which is
  precisely why Thomas–Fermi and the gradient corrections fail;
- **expensive to evaluate exactly**: the definition requires the very
  diagonalisation orbital-free DFT is trying to avoid.

Two architectural consequences follow directly:

1. **The architecture must be non-local.** A CNN with a finite receptive field
   is structurally wrong for this map. The FNO's spectral convolution is global
   in a single layer, which is why it was chosen (Chapter 3 of the technical
   guide).
2. **The target is well-posed.** Unlike most ML targets, $\tau[\rho]$ is a
   genuine single-valued functional, not a fitted correlation. Failure to learn
   it is a capacity or data problem, not an ill-posedness problem.

### 2.1 A caveat that is often skipped

τ is a functional of ρ *for ground-state, non-interacting v-representable
densities*. Inside an SCF loop the intermediate densities are **not** ground
states of any $v_s$. Strictly, $\tau[\rho]$ is undefined there. In practice one
uses the functional anyway — that is what every OF-DFT calculation does — but
it means a model trained only on converged densities is being *extrapolated*
every time it is used variationally. Section 6 returns to this; it is the
single largest risk in the programme.

---

## 3. What the model must get right — and it is not τ

This is the most important design point in this document, and it is easy to get
wrong.

Orbital-free DFT does not consume τ. It consumes the **functional
derivative**:

$$
v_s^{\rm kin}(\mathbf r) \;\equiv\; \frac{\delta T_s}{\delta \rho(\mathbf r)} .
$$

That is the term that appears in the Euler–Lagrange equation and therefore the
term that determines the density. A model can have small
$\lVert \tau_\theta - \tau \rVert$ and a **useless** derivative, because
differentiation amplifies high-frequency error: a small ripple of amplitude
$\epsilon$ and wavevector $G$ contributes $\epsilon$ to the value and
$\epsilon G$ to the gradient.

Consequences:

- Reporting only pointwise τ error is reporting a **proxy**. The relative $L^2$
  numbers in the technical guide are necessary, not sufficient.
- The $H^1$ (Sobolev) data loss exists for exactly this reason: it penalises
  the gradient error the plain $L^2$ ignores.
- The definitive test of Model 2 is **downstream**: insert it into an
  orbital-free minimiser and measure the error in the converged density and
  energy. Everything before that is a proxy.

---

## 4. The mechanism: δT_s/δρ by autograd

This is what makes a *neural* KEDF more useful than a tabulated one, and it is
mechanically simple.

$T_s$ is a scalar computed from ρ by a differentiable program. PyTorch will
therefore produce $\partial T_s / \partial \rho_i$ for every grid point at the
cost of one backward pass — no finite differences, no analytic derivative to
derive by hand, and it works for *any* architecture.

```python
rho = rho_physical.detach().requires_grad_(True)         # (B,1,Nx,Ny,Nz), e/Ang^3

tau  = target_transform.inverse(model2(input_transform(rho), cell))   # eV/Ang^3
T_s  = integrate(tau, cell)                                          # (B,), eV

grad, = torch.autograd.grad(T_s.sum(), rho, create_graph=True)
dTs_drho = grad / volume_element                          # eV per (e/Ang^3), see below
```

### 4.1 The discretisation factor (easy to get silently wrong)

With $T_s = \sum_i \tau_i \,\Delta v$ and the functional derivative defined by
$\delta T_s = \int (\delta T_s/\delta\rho)\,\delta\rho \,\mathrm{d}^3r
\approx \sum_i (\delta T_s/\delta\rho)_i \,\delta\rho_i \,\Delta v$,
while autograd returns $\delta T_s = \sum_i (\partial T_s/\partial\rho_i)\,\delta\rho_i$,
we get

$$
\boxed{\;\frac{\delta T_s}{\delta \rho}(\mathbf r_i)
   \;=\; \frac{1}{\Delta v}\,\frac{\partial T_s}{\partial \rho_i}\;}
$$

where $\Delta v = \Omega / (N_1N_2N_3)$. **Omitting the $1/\Delta v$ rescales
the whole kinetic potential by the number of grid points** — a factor of
$3\times10^4$ here — which would silently destroy any Euler–Lagrange residual
built from it. `create_graph=True` is likewise required if the derivative is to
appear in a loss that is itself backpropagated.

### 4.2 Sanity checks that must pass before trusting it

The derivative is only meaningful if it is verified independently:

| Test | Expectation |
|---|---|
| Substitute the analytic Thomas–Fermi functional for Model 2 | autograd reproduces $\tfrac{5}{3}C_{\rm TF}\rho^{2/3}$ |
| Substitute von Weizsäcker | autograd reproduces $-\tfrac12\nabla^2\sqrt\rho/\sqrt\rho$ |
| Finite-difference a few random voxels | agrees to the FD accuracy |
| Uniform density | $\delta T_s/\delta\rho$ constant in space |

These are cheap and they catch the $\Delta v$ error, the transform-chain error,
and any accidental `detach()`.

---

## 5. Coupling: how Model 2 trains Model 1

### 5.1 The equation

Minimising $E[\rho]$ at fixed particle number gives, pointwise,

$$
\frac{\delta T_s}{\delta\rho}(\mathbf r)
 + V_{\rm ext}(\mathbf r)
 + v_H[\rho](\mathbf r)
 + v_{xc}[\rho](\mathbf r) \;=\; \mu ,
$$

with $\mu$ **constant in space**. Every term is available:

| Term | Source | Exact? |
|---|---|---|
| $V_{\rm ext}$ | Model 1's *input* | exact (tabulated pseudopotential) |
| $\rho$ | Model 1's *output* | learned |
| $v_H[\rho]$ | $4\pi e^2\rho(\mathbf G)/G^2$, one FFT | **exact** |
| $\delta T_s/\delta\rho$ | autograd through Model 2 | learned |
| $v_{xc}[\rho]$ | LDA/GGA, or omitted | approximate |
| $\mu$ | eliminated | — |

$\mu$ is unknown and material-dependent, but it is a *constant*: subtracting the
cell average of the left-hand side removes it exactly. What remains,

$$
r(\mathbf r) \;=\;
  \frac{\delta T_s}{\delta\rho} + V_{\rm ext} + v_H[\rho] + v_{xc}[\rho]
  \;-\; \overline{\Big(\cdots\Big)} ,
$$

must vanish for the true ground-state density.

### 5.2 Why this is worth the trouble

$r(\mathbf r)$ is computable from **Model 1's own input and output**, plus
Model 2. It needs **no additional labels** — no extra DFT calculations, no
targets. It is therefore usable on:

- unlabelled structures (geometry only, no reference `CHGCAR`);
- perturbed or interpolated geometries generated on the fly;
- the model's own predictions during training.

That converts Model 2 from a passive predictor into an **active physical
constraint on Model 1**. This is the coupling the architecture is built around.

### 5.3 The loss

$$
\mathcal{L}_1 =
  \underbrace{\big\lVert \rho_\theta - \rho^{\rm DFT} \big\rVert_{\rm rel}}_{\text{data}}
  \;+\; \lambda_{\rm EL}\,
  \underbrace{\Big\langle \big(r(\mathbf r)/s\big)^2 \Big\rangle}_{\text{Euler–Lagrange}}
  \;+\; \lambda_{N}\,
  \underbrace{\Big(\tfrac{\int\rho_\theta - N}{N}\Big)^2}_{\text{particle number}}
$$

with $s = \operatorname{std}(V_{\rm ext})$ making the residual term
dimensionless and comparable across materials whose potentials differ by an
order of magnitude.

Three design rules, each learned the hard way elsewhere in this project:

1. **Physics terms act on the prediction decoded to physical units**, never on
   the normalised representation. A constraint is a statement about physics,
   not about preprocessing.
2. **Every term is dimensionless**, or one loss weight cannot serve a
   heterogeneous dataset.
3. **Warm up.** An Euler–Lagrange residual evaluated at random initialisation
   is noise; ramp $\lambda_{\rm EL}$ only after the data term has converged.

### 5.4 Freeze Model 2, or train jointly?

Both are defensible, and the failure modes differ.

**Frozen Model 2** — a fixed physics oracle. Simple, stable, and the constraint
is exactly as good as the KEDF. Recommended first.

**Joint training** — mutual regularisation. Model 2 is then exposed to the
densities Model 1 actually produces, which directly attacks the
distribution-shift problem of §2.1. But there is a real hazard:

> **The Euler–Lagrange residual alone has trivial solutions.** With both models
> free, the pair can drive $r \to 0$ by co-adapting: Model 2 can learn a
> functional whose derivative cancels $V_{\rm ext} + v_H + v_{xc}$ for whatever
> ρ Model 1 emits, without either being correct.

The residual is a *consistency* condition, not a *correctness* condition. Joint
training is therefore only safe with **both data terms retained and dominant**.
Any schedule that lets $\lambda_{\rm EL}$ overwhelm them invites collapse, and
the diagnostic is a residual that falls while the held-out data error rises.

---

## 6. Critical assessment

Stated plainly, because the case for Model 2 is strong enough not to need
overselling.

### 6.1 Genuine strengths

- The target is a well-posed functional, not a correlation (§2).
- It attacks the actual bottleneck of OF-DFT, not a peripheral quantity.
- The derivative is free and exact via autograd — no hand-derived functional
  derivative, which is the traditional obstacle to new KEDFs.
- Exact constraints exist and can be imposed structurally: $\tau \ge \tau_{\rm vW}$
  is already enforced by construction, supplying ~31 % of τ analytically.
- On this data it beats every classical KEDF by roughly an order of magnitude,
  with Thomas–Fermi achieving *negative* $R^2$.

### 6.2 Real weaknesses

- **Distribution shift is the central risk.** Trained on converged densities,
  used on non-converged ones. §2.1 shows this is not merely a practical
  inconvenience but a domain question: the functional is strictly undefined
  off the v-representable manifold.
- **We optimise τ and use $\delta T_s/\delta\rho$** (§3). Until training runs
  through the derivative, the objective is a proxy.
- **The derivative is unvalidated.** Nothing in the current results says
  anything about $\delta T_s/\delta\rho$. The §4.2 checks are not yet
  implemented.
- **Data scale.** Three structures of one element. The measured numbers
  characterise the pipeline, not the physics.
- **The Pauli potential constraint is unused.** Levy–Ou-Yang gives
  $v_P = \delta T_P/\delta\rho \ge 0$ — a constraint on the *derivative*, which
  is exactly the object §3 argues we should be constraining. It is not yet
  enforced.

### 6.3 The τ representation is a gauge choice

Relevant here because it bounds what the choice can matter for. The two common
forms differ by a total derivative,

$$
\tau_L = \tau_{\rm PD} - \tfrac{\hbar^2}{4m}\nabla^2\rho ,
$$

and $\int \nabla^2\rho \,\mathrm{d}^3r = 0$ on a torus. Therefore

$$
T_s^{L} = T_s^{\rm PD}
\qquad\text{and}\qquad
\frac{\delta T_s^{L}}{\delta\rho} = \frac{\delta T_s^{\rm PD}}{\delta\rho} .
$$

**Both representations give the same OF-DFT physics.** Verified numerically:
$|\int\tau_L - \int\tau_{\rm PD}| = 3.6\times10^{-12}$ eV.

What *does* differ is the learning problem — sign-definiteness, spectral
content, and which structural constraints apply. That is the subject of
`docs/ked_formulation_analysis.md`.

---

## 7. Verification plan

Falsifiable, in dependency order:

1. **Derivative correctness** — the §4.2 substitution tests. Blocking: nothing
   downstream is meaningful until these pass.
2. **Residual vanishes on truth** — evaluate $r(\mathbf r)$ with the *reference*
   ρ and a classical KEDF. It will not be zero (the KEDF is approximate); record
   the floor, because a learned functional must beat it to be adding anything.
3. **Constraint helps** — matched ablation, $\lambda_{\rm EL} = 0$ vs $> 0$, on
   held-out material. Same protocol as the Pauli-head comparison, which showed
   an 18.3 % gain.
4. **Label efficiency** — the real promise. Train Model 1 on $k$ labelled
   structures plus $m$ unlabelled ones under the residual, and show the curve
   beats $k$ labelled alone. If the residual cannot buy label efficiency, its
   main practical argument fails.
5. **Downstream** — insert the KEDF into an OF-DFT minimiser; measure converged
   density and energy error. The only test that matters.

---

## 8. Status

| Component | State |
|---|---|
| $\rho \mapsto \tau$ supervised | done — rel. $L^2$ 0.0620, $R^2$ 0.9924 |
| $\tau \ge \tau_{\rm vW}$ structural | done — `PauliResidualOperator` |
| Exact spectral $\nabla$, $\nabla^2$, $v_H$ | done — `poraque.ml.physics` |
| Euler–Lagrange residual, analytic KEDF | done — `euler_lagrange_loss` |
| $\delta T_s/\delta\rho$ via autograd | **not implemented** |
| Joint training | not implemented |
| Pauli potential $v_P \ge 0$ | not implemented |
| OF-DFT minimiser integration | not implemented |

The gap between rows 4 and 5 is the whole programme: everything up to the
analytic residual is infrastructure; the autograd derivative is what turns two
regressions into a differentiable orbital-free solver.
