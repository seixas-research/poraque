# Physics-Informed Fourier Neural Operator (PI-FNO)

**Status:** technical plan · **Scope:** `poraque.ml` · **Companion code:** `poraque/ml/physics.py`, `poraque/ml/losses.py`

---

## 1. Summary

We have two supervised operator-learning tasks defined on a shared 3D grid:

| Task | Map | Physical meaning |
| --- | --- | --- |
| `ext2chg` | `EXTCAR` → `CHGCAR`, i.e. $V_{\rm ext}\mapsto\rho$ | the Hohenberg–Kohn map |
| `chg2tau` | `CHGCAR` → `TAUCAR`, i.e. $\rho\mapsto\tau$ | the kinetic energy density functional (KEDF) |

Trained purely on data, both are curve-fits that happen to have a physical
label. This document specifies how to turn them into a **physics-informed**
system in which the governing equations of KS-DFT and OF-DFT enter the
optimisation directly.

The central claim is that these are not two independent regressions. They are
the two legs of one variational statement:

$$
E[\rho] \;=\; \underbrace{\int \tau\,d^3r}_{\text{task 2}} \;+\; \int V_{\rm ext}\rho\,d^3r \;+\; E_H[\rho] \;+\; E_{xc}[\rho],
\qquad
\frac{\delta E}{\delta\rho}\bigg|_{\rho_0} = \mu .
$$

The stationarity condition couples $V_{\rm ext}$, $\rho$ and $\tau$ in a single
pointwise equation. Enforcing it makes the output of `chg2tau` a constraint on
the output of `ext2chg`, and vice versa — the two networks stop being
independent and start being a *discretised DFT solver with learned components*.

**Why an FNO is unusually well suited to this.** A physics-informed loss needs
derivatives: $\nabla\rho$, $\nabla^2\sqrt\rho$, and the Hartree convolution
$\rho\mapsto v_H$. An FNO already transforms every layer to reciprocal space.
On a plane-wave grid, $\nabla \to i\mathbf{G}$ and $\nabla^2 \to -G^2$ are
*exact*, not approximations, and the Hartree kernel $4\pi e^2/G^2$ is
diagonal — an $\mathcal{O}(N\log N)$ multiply. A physics-informed CNN would
need finite-difference stencils, whose truncation error the loss would then
wrongly attribute to the network. Here the physics operators and the
architecture share the same mathematical substrate.

---

## 2. Governing equations

### 2.1 Kohn–Sham DFT

$$
\left(-\tfrac{1}{2}\nabla^2 + v_s[\rho](\mathbf r)\right)\psi_i = \varepsilon_i\psi_i,
\qquad
v_s = V_{\rm ext} + v_H[\rho] + v_{xc}[\rho],
$$

$$
\rho(\mathbf r) = \sum_i^{\rm occ} f_i |\psi_i(\mathbf r)|^2,
\qquad
\tau(\mathbf r) = \tfrac{1}{2}\sum_i^{\rm occ} f_i |\nabla\psi_i(\mathbf r)|^2 .
$$

Two exact relations follow that need **no orbitals** and are therefore directly
usable as losses:

* **Poisson / Hartree.** $\nabla^2 v_H = -4\pi e^2\rho$, solved exactly in
  reciprocal space with $v_H(\mathbf G{=}0)=0$ — the same neutralising
  background convention `ExternalPotential` uses, so $v_H$ and $V_{\rm ext}$ are
  directly addable. Implemented: `physics.hartree_potential`.
* **Particle number.** $\int\rho\,d^3r = N_{\rm val}$, exactly, where
  $N_{\rm val}=\sum_a Z^{\rm val}_a$ is already known from the `POTCAR`.

### 2.2 Orbital-free DFT

Minimising $E[\rho]$ under $\int\rho = N$ gives the Euler–Lagrange equation

$$
\boxed{\;\frac{\delta T_s}{\delta\rho}(\mathbf r) + V_{\rm ext}(\mathbf r) + v_H[\rho](\mathbf r) + v_{xc}[\rho](\mathbf r) = \mu\;}
$$

with $\mu$ a **constant**. This is the single most valuable constraint we
have, for three reasons:

1. It contains all three fields of the dataset.
2. It requires **no additional labels** — $V_{\rm ext}$ is the network input and
   $\rho$ is its output.
3. $\mu$ is unknown and material-dependent, but subtracting the cell average of
   the left-hand side eliminates it, leaving a residual that must vanish
   pointwise for the exact density. Implemented:
   `physics.euler_lagrange_residual`.

### 2.3 Exact properties of $\tau$

These constrain `chg2tau` and are theorems, not heuristics:

| Property | Statement | Use |
| --- | --- | --- |
| Non-negativity | $\tau \ge 0$ | hard, via output parameterisation |
| **Hoffmann-Ostenhof bound** | $\tau \ge \tau_{\rm vW}=\dfrac{\lvert\nabla\rho\rvert^2}{8\rho}$ | hard, via output parameterisation |
| Uniform-gas limit | $\tau\to\tau_{\rm TF}=C_{\rm TF}\rho^{5/3}$ as $\nabla\rho\to0$ | soft penalty / augmentation |
| **Coordinate scaling** | $\rho_\lambda(\mathbf r)=\lambda^3\rho(\lambda\mathbf r)\Rightarrow T_s[\rho_\lambda]=\lambda^2T_s[\rho]$ | exact, label-free augmentation |
| Pauli positivity | $\tau_P=\tau-\tau_{\rm vW}\ge0$ and $v_P=\delta T_P/\delta\rho\ge0$ | hard / soft |
| One-orbital exactness | $\tau=\tau_{\rm vW}$ for a nodeless single orbital | unit test |

The scaling relation deserves emphasis: it is an **exact, label-free data
augmentation**. Any training sample can be rescaled and its target $T_s$ known
in closed form, which multiplies the effective dataset at zero DFT cost.

---

## 3. Design principle: constrain by construction, not by penalty

A soft penalty $\lambda\lVert\text{violation}\rVert^2$ trades accuracy against
constraint satisfaction and never fully achieves either. Where a constraint can
be built into the architecture it should be, leaving penalties for constraints
that cannot.

**Three constraints move into the architecture.**

### 3.1 Exact positivity — output parameterisation

Predict $\log\rho$ and exponentiate. `transforms.Log` already does this: its
`inverse` returns `exp(y) - epsilon`, so $\rho > -\epsilon$ *identically*, for
every weight configuration, at initialisation and after any optimiser step.
No penalty, no weight to tune.

### 3.2 Exact electron count — a normalisation head

$$
\rho_{\rm out}(\mathbf r) \;=\; N_{\rm val}\,\frac{\tilde\rho(\mathbf r)}{\int\tilde\rho\,d^3r}
$$

One reduction and one division makes $\int\rho = N_{\rm val}$ hold to machine
precision, and it is differentiable. This is strictly better than
`electron_count_loss`, which should be kept only as a diagnostic.

### 3.3 Exact von Weizsäcker bound — residual parameterisation ✅ IMPLEMENTED

$$
\tau_{\rm out} \;=\; \tau_{\rm vW}[\rho] \;+\; s\,\mathrm{softplus}\big(f_\theta(\rho)\big)
$$

The network learns only the **Pauli term** $\tau_P\ge0$. The bound holds by
construction, and — more importantly — the network no longer has to spend
capacity re-learning $\lvert\nabla\rho\rvert^2/8\rho$, which is analytically
known and often dominates.

Implemented as `poraque.ml.heads.PauliResidualOperator`; enable with
`FieldOperator(..., pauli_residual=True)` or `train_fno.py --pauli-head`.
Notes from the implementation:

* The head consumes and returns *normalised* fields, so it is a drop-in
  replacement for a bare backbone. Internally it decodes $\rho$, evaluates
  $\tau_{\rm vW}$ spectrally (exact on a plane-wave grid), adds the positive
  residual, and re-encodes — keeping the loss well-conditioned while the
  constraint lives in physical units.
* The scale $s$ is fitted on the **training split only** (`fit_pauli_scale`)
  so the ``softplus`` operates near unit argument; it is then optimised as
  $\log s$, which cannot change sign.
* $\tau_{\rm vW}$ is a fixed function of the *input*, so it contributes no
  gradient to $\theta$ — it is a per-sample offset, not a second branch.
* The bound is enforced **non-strictly**, exactly as Hoffmann-Ostenhof states
  it: for strongly negative backbone output ``softplus`` underflows and the
  head returns $\tau = \tau_{\rm vW}$ *exactly*. That single-orbital limit is
  reachable, not merely approachable — something a soft penalty can never do.
* Verified holding at random initialisation, for weights scaled by 50×, and
  after training at a deliberately destructive learning rate
  (`tests/test_heads.py`). A control test confirms the *unconstrained*
  backbone does violate the bound, so the head is not solving a non-problem.

**Check the data first.** The inequality is a theorem for all-electron
densities; VASP's `CHGCAR`/`TAUCAR` are pseudo quantities. Use
`pauli_bound_violation` before enabling the head. On the Au dataset it holds at
every point of `struct_001`/`struct_002` and at all but one point of
`struct_000` — and that point is where spectral downsampling rang $\tau$
slightly negative, i.e. an artefact of resampling rather than of physics.
$\tau_{\rm vW}$ supplies ~31 % of $\tau$ there, which is the capacity the head
hands back to the network for free.

> **Recommendation.** Implement §3.1–3.3 *before* adding any soft physics loss.
> They cost nothing at inference, cannot destabilise training, and remove three
> hyper-parameters. `physics.positivity_loss`, `physics.electron_count_loss`
> and `physics.von_weizsacker_bound_loss` then serve as *verification* that the
> parameterisation works, not as training signal.

---

## 4. Soft constraints: the composite objective

$$
\mathcal{L} = \mathcal{L}_{\rm data} + \lambda_{\rm EL}\mathcal{L}_{\rm EL} + \lambda_{\rm TF}\mathcal{L}_{\rm TF} + \lambda_{T}\mathcal{L}_{T} + \lambda_{\rm scale}\mathcal{L}_{\rm scale}
$$

Implemented in `losses.PhysicsInformedLoss`. **Every physics weight defaults to
zero**, so the class is exactly the supervised baseline until a term is turned
on deliberately.

| Term | Definition | Task | Notes |
| --- | --- | --- | --- |
| $\mathcal{L}_{\rm data}$ | relative $L^2$, optionally $H^1$ | both | see §4.1 |
| $\mathcal{L}_{\rm EL}$ | $\lVert \text{LHS} - \overline{\text{LHS}}\rVert^2$ of the Euler–Lagrange equation | `ext2chg` | label-free |
| $\mathcal{L}_{\rm TF}$ | penalty on $\tau-\tau_{\rm TF}$ where $\lvert\nabla\rho\rvert/\rho^{4/3}$ is small | `chg2tau` | enforces the uniform limit |
| $\mathcal{L}_{T}$ | $(\int\tau - T_s^{\rm ref})^2$ relative | `chg2tau` | needs reference $T_s$ |
| $\mathcal{L}_{\rm scale}$ | violation of $T_s[\rho_\lambda]=\lambda^2T_s[\rho]$ | `chg2tau` | label-free |

**All physics terms are evaluated on the prediction decoded to physical
units**, never in the normalised representation — a constraint is a statement
about physics, not about whatever preprocessing training happens to use. This
is why `PhysicsInformedLoss.forward` takes `physical_prediction` separately.

Every term is **normalised to be dimensionless**. A raw $(\mathrm{eV/Å^3})^2$
penalty varies by orders of magnitude between a light semiconductor and a
transition-metal oxide, and no single weight could serve both.

### 4.1 Why the data term is $H^1$, not $L^2$

$\tau_{\rm vW}$ depends on $\nabla\rho$, so a model with small pointwise error
but noisy derivatives is useless downstream: the gradient noise is amplified by
$\lvert\nabla\rho\rvert^2/8\rho$ in the low-density regions where the
denominator is small. `losses.SobolevLoss` adds the relative gradient error,
computed spectrally. Expect it to matter most for `chg2tau`.

### 4.2 The functional derivative comes from autograd

$\mathcal{L}_{\rm EL}$ needs $\delta T_s/\delta\rho$. Three options, in
increasing order of ambition:

1. **Analytic surrogate** — `TF + λ·vW` with $\lambda=1/9$ (gradient expansion)
   or $\lambda=1$ (strongly inhomogeneous). This is what
   `physics.euler_lagrange_residual` does today. It is an *approximation*, so
   the term supplies a correct inductive bias, not ground truth, and must carry
   a modest weight.
2. **Learned** — take $T_s=\int\tau_\theta(\rho)\,d^3r$ from the `chg2tau`
   network and obtain $\delta T_s/\delta\rho$ by `torch.autograd.grad` through
   it. This is the standard construction for ML-KEDFs and makes the constraint
   exact in the limit of a perfect KEDF.
3. **Self-consistent** — see §5.

Option 2 is the pivot point of the whole programme: it is what couples the two
tasks.

### 4.3 Weighting and scheduling

Physics losses commonly *degrade* accuracy when introduced badly. Mitigations,
in order of preference:

* **Warm-up.** Train supervised to convergence, then ramp $\lambda$ linearly
  over ~10% of the remaining budget. A physics residual evaluated on a random
  initialisation is meaningless and its gradient is noise.
* **Gradient-norm balancing.** Set $\lambda_i$ each step so that
  $\lambda_i\lVert\nabla_\theta\mathcal{L}_i\rVert \approx \alpha\lVert\nabla_\theta\mathcal{L}_{\rm data}\rVert$
  with $\alpha\sim0.1$. Removes manual tuning; costs one extra backward pass.
* **Uncertainty weighting.** Learn $\lambda_i = 1/2\sigma_i^2$ with a
  $\log\sigma_i$ regulariser. Cheaper, less controllable.
* **Ablate one term at a time** against a frozen supervised baseline. A physics
  term that does not improve held-out error is not earning its place.

---

## 5. The closed loop: coupling the two operators

The end state is not two models but one differentiable OF-DFT cycle:

```
                    ┌──────────────── Euler–Lagrange residual ───────────────┐
                    │                                                        │
   EXTCAR ──▶ [ FNO_A : V_ext ↦ ρ ] ──▶ CHGCAR ──▶ [ FNO_B : ρ ↦ τ ] ──▶ TAUCAR
                    ▲                        │                          │
                    │                        └──▶ v_H[ρ] (exact, FFT)   │
                    │                                                   │
                    └──────── δT_s/δρ via autograd through FNO_B ◀───────┘
```

The joint residual for a sample is

$$
r(\mathbf r) \;=\; \frac{\delta}{\delta\rho}\!\int\!\tau_{\theta_B}[\rho_{\theta_A}]\,d^3r \;+\; V_{\rm ext} \;+\; v_H[\rho_{\theta_A}] \;+\; v_{xc}[\rho_{\theta_A}] \;-\; \mu,
$$

with $\mu$ removed as the cell average. Training $\theta_A$ and $\theta_B$
jointly on $\mathcal{L}_{\rm data}^A+\mathcal{L}_{\rm data}^B+\lambda\lVert r\rVert^2$
gives three things a separate training cannot:

* **Mutual regularisation.** $\theta_B$ is constrained on densities produced by
  $\theta_A$, i.e. exactly the distribution it will see at deployment, not only
  on VASP-converged densities. This directly attacks the distribution shift
  that breaks naive ML-KEDFs when they are put inside an SCF loop.
* **A usable KEDF.** The object OF-DFT actually needs is
  $\delta T_s/\delta\rho$, not $\tau$. Training through the functional
  derivative optimises the quantity that will be used.
* **Physical consistency at inference.** A predicted $(\rho,\tau)$ pair
  satisfies the variational condition it is supposed to satisfy.

**Beyond that:** replace the forward pass of `FNO_A` with a fixed-point solve
of the Euler–Lagrange equation using the learned KEDF (a deep-equilibrium
model, differentiated with the implicit function theorem). At that point the
network *is* an OF-DFT solver, `ext2chg` becomes a solver output rather than a
regression, and transferability is bounded by the KEDF alone.

---

## 6. Kohn–Sham constraints without orbitals

`TAUCAR` from VASP is the KS kinetic energy density, so `chg2tau` learns the
*non-interacting* $T_s$ — the correct target for OF-DFT. Additional KS-derived
constraints that avoid orbitals:

* **Density-to-potential inversion.** For the exact KS system, given $\rho$ one
  can recover $v_s$. Since $v_s = V_{\rm ext}+v_H+v_{xc}$ and $v_H$ is exact,
  an inversion applied to the predicted $\rho$ must return the network's own
  input $V_{\rm ext}$ up to a constant. This is a **cycle-consistency loss** and
  it is strictly stronger than the OF-DFT residual because it does not depend
  on a KEDF approximation. Cost: one inversion per sample; feasible via the
  van Leeuwen–Baerends iteration on the plane-wave grid.
* **Total-energy consistency.** If reference energies are harvested from
  `OUTCAR`, the assembled $E[\rho]$ must match. One scalar per material, but it
  constrains the integral quantities that pointwise losses leave free.
* **Virial relation.** $2T_s + \int\rho\,\mathbf r\cdot\nabla v_s\,d^3r = 0$ for
  the KS system, giving a scalar check per material with no extra labels.

---

## 7. Implementation roadmap

| Phase | Work | Deliverable | Depends on |
| --- | --- | --- | --- |
| **0** | Supervised baseline, both tasks, honest held-out split by material | baseline relative $L^2$ per task | done (`poraque.ml`) |
| **1** | Hard constraints §3.1–3.3: `Log` head, charge-normalisation head, $\tau=\tau_{\rm vW}+\mathrm{softplus}$ | constraints satisfied to machine precision; accuracy ≥ baseline | 0 |
| **2** | $H^1$ data loss; scaling-relation augmentation | improved gradient fidelity for `chg2tau` | 1 |
| **3** | $\mathcal{L}_{\rm EL}$ with the **analytic** `TF+λvW` surrogate, warm-up schedule, gradient-norm balancing | label-free physics signal on `ext2chg` | 1 |
| **4** | $\delta T_s/\delta\rho$ by autograd through `FNO_B`; joint training of both operators | coupled PI-FNO (§5) | 3 |
| **5** | Cycle-consistency via density-to-potential inversion | KEDF-independent constraint | 4 |
| **6** | Deep-equilibrium OF-DFT solver | learned solver, not regressor | 4 |

Phases 1–3 are low-risk and independently valuable. Phase 4 is the research
contribution. Phases 5–6 are stretch.

---

## 8. Risks and failure modes

| Risk | Symptom | Mitigation |
| --- | --- | --- |
| Physics term degrades accuracy | held-out error rises when $\lambda>0$ | warm-up; gradient-norm balancing; ablate individually |
| Ill-conditioned $\tau_{\rm vW}$ in vacuum | $\lvert\nabla\rho\rvert^2/8\rho$ blows up as $\rho\to0$ | density floor (already in `physics`); weight the residual by $\rho$ |
| $\lambda$ in `TF+λvW` is a guess | EL residual biases towards a wrong functional | keep $\lambda_{\rm EL}$ small until Phase 4 replaces it |
| Missing $v_{xc}$ in the residual | systematic offset | the constraint enforces *constancy* of the sum, so an omitted smooth term weakens but does not bias it; add libxc when available |
| Pseudopotential inconsistency | `EXTCAR` model ≠ VASP's PAW local potential | see §9 |
| Grid-shape correlation with chemistry | model keys on grid size instead of physics | `ShapeBucketSampler` shuffles buckets; verify with a shape-permutation control |
| Distribution shift inside an SCF loop | KEDF fails on non-converged densities | Phase 4 joint training is the designed fix |

### 8.1 A known approximation to track

`ExternalPotential` builds the **long-range local** ionic potential from
`ZVAL` with a Gaussian-smeared pseudo-ion whose width comes from the `POTCAR`
`RCORE`. It is not VASP's PAW local pseudopotential: the short-range
pseudisation inside $R_{\rm core}$ is modelled rather than read from the
tabulated `local part`, and non-local projectors are out of scope for a local
field by construction.

For learning $V_{\rm ext}\mapsto\rho$ this is defensible — the map is
well-defined for *our* $V_{\rm ext}$, and consistent across the dataset — but it
means the learned operator is the HK map for the model potential, not for VASP's.
Two checks belong in the validation protocol: (i) confirm accuracy does not
degrade for species with large $R_{\rm core}$; (ii) if it does, the tabulated
`local part` is already parsed into `PotcarSingle.local_part` and only needs
its $q$-mesh convention supplied to be usable.

---

## 9. Validation protocol

1. **Split by material**, never by grid point or by crop. Two crops of one
   material leak across the split and inflate the score.
2. **Report in physical units.** A small error under an `asinh` compression can
   hide a large error in $\rho$. `training.evaluate` decodes before scoring.
3. **Constraint report per epoch**: $\lvert\int\rho-N\rvert$, $\min\rho$,
   fraction of points violating $\tau\ge\tau_{\rm vW}$, and
   $\lVert r_{\rm EL}\rVert$ — logged even when the corresponding weight is
   zero, since they are the diagnostics that tell whether §3 is working.
4. **Resolution-transfer test.** Train at one grid density, evaluate at another
   for the same material. An FNO should degrade gracefully; a failure here
   indicates the mode-truncation convention is wrong (use
   `mode_selection="physical"`).
5. **Chemistry-transfer test.** Hold out an entire chemical family.
6. **Downstream test**, the one that matters: insert the learned KEDF into the
   OF-DFT minimiser and measure the error in the converged density and energy —
   not the error in $\tau$.

---

## 10. Mapping to the code

| Concept | Implementation |
| --- | --- |
| $\nabla$, $\nabla^2$ (spectral, exact) | `physics.spectral_gradient`, `physics.spectral_laplacian` |
| Hartree potential / Poisson | `physics.hartree_potential`, `physics.poisson_residual` |
| $\tau_{\rm TF}$, $\tau_{\rm vW}$ and their potentials | `physics.thomas_fermi_tau`, `physics.von_weizsacker_tau`, `*_potential` |
| Euler–Lagrange residual | `physics.euler_lagrange_residual`, `physics.euler_lagrange_loss` |
| Constraint penalties | `physics.electron_count_loss`, `positivity_loss`, `von_weizsacker_bound_loss`, `kinetic_energy_loss` |
| Composite objective | `losses.PhysicsInformedLoss` |
| $H^1$ data loss | `losses.SobolevLoss` |
| Operator | `fno.FNO3d`, `fno.SpectralConv3d` |
| Training | `training.train`, `training.FieldOperator` |

Not yet implemented, in roadmap order: the charge-normalisation head, the
$\tau_{\rm vW}+\mathrm{softplus}$ head, the scaling-relation augmentation, the
autograd functional derivative, joint two-operator training, density-to-potential
inversion.

---

## 11. References

* Hohenberg & Kohn, *Phys. Rev.* **136**, B864 (1964) — the $V_{\rm ext}\mapsto\rho$ map.
* Kohn & Sham, *Phys. Rev.* **140**, A1133 (1965).
* Hoffmann-Ostenhof & Hoffmann-Ostenhof, *Phys. Rev. A* **16**, 1782 (1977) — the $\tau\ge\tau_{\rm vW}$ bound.
* Levy, Perdew & Sahni, *Phys. Rev. A* **30**, 2745 (1984) — Pauli potential.
* Levy & Perdew, *Phys. Rev. A* **32**, 2010 (1985) — coordinate scaling.
* Li *et al.*, *Fourier Neural Operator for Parametric PDEs*, ICLR 2021.
* Snyder *et al.*, *Phys. Rev. Lett.* **108**, 253002 (2012) — ML kinetic functionals.
* Raissi, Perdikaris & Karniadakis, *J. Comput. Phys.* **378**, 686 (2019) — PINNs.
* Bai, Kolter & Koltun, *Deep Equilibrium Models*, NeurIPS 2019 — §5 fixed-point solver.
