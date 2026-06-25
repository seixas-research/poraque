# Claude Code response to the AGY audit

This documents how the six items in `plan/agy_report.md` were addressed, where I
diverged from the proposed code, and the issues the audit missed. All 123
pre-existing tests still pass after the changes, plus the new behaviour was
validated directly (see "Verification" below).

## Summary of what was applied

| # | Item | Status |
|---|------|--------|
| 1 | Non-orthogonal reciprocal `G`-vectors / `|G|²` | Applied as proposed |
| 2 | Unify Poisson solver via `grid.get_g2()` | Applied (dropped the redundant `.copy()`) |
| 3 | Ion-ion (Ewald / pairwise) repulsion energy | Applied, with two changes — see below |
| 4 | Respect grid PBC in MIC default | Applied, **narrowed** to avoid a new bug |
| 5 | PBE XC fail-safe | Applied, **extended** to the OF-DFT path |
| 6 | Spectral Laplacian in von Weizsäcker | **Partially declined** — made opt-in, not default |

---

## Where I diverged from the report

### #2 — dropped the redundant `.copy()`
`Grid.get_g2()` already returns a freshly allocated array
(`gx**2 + gy**2 + gz**2`), so mutating `G2[0,0,0]` cannot alias grid state. The
proposed `.copy()` is harmless but unnecessary; I omitted it.

### #4 — MIC default applied only to the branches that accept `mic`
The report's snippet places `kwargs.setdefault("mic", any(grid.pbc))` **before**
the `kind` dispatch in `build_external_potential`. That would forward a `mic`
keyword into `point_charge_potential(...)`, whose signature is
`(grid, positions, charges, rc=1e-6)` — **no `mic` parameter** — raising
`TypeError` for `kind="point"`. I moved the `setdefault` inside the `"soft"`
branch only. I deliberately did **not** change the `"ewald"` default
(`mic=False`), because there `mic=True` selects the *fast* minimum-image method
that is only valid when `r_cut ≤ min_height/2`; flipping its default would
silently trade robustness for speed. The pseudopotential change (`mic=None →
any(grid.pbc)`) was applied as proposed.

### #5 — also fixed the OF-DFT path, made it case-insensitive
The report only patched `KSDFTEngine`'s `_xc_functional`. The exact same latent
crash existed in `_ofdft_functionals`, which did
`functionals.append(LDA() if self.xc == "lda" else self.xc)` and would have
appended the bare string `"pbe"`. I factored a single `_resolve_xc()` helper
used by both paths, so `xc="pbe"` now raises the same clear
`NotImplementedError` in OF-DFT too. The helper also accepts any capitalization
(`"LDA"`, `"Lda"`), which the report's `self.xc.lower() == "lda"` allowed for KS
but the unchanged OF path (`== "lda"`) did not.

### #6 — spectral Laplacian made **opt-in**, not the default (main disagreement)
I disagree with switching von Weizsäcker to the spectral FFT Laplacian
*unconditionally*. The audit is right that 2nd-order finite differences are less
accurate than a spectral operator for a well-resolved, band-limited periodic
density — but it overlooks that the **spectral Laplacian is globally nonlocal**,
and the same `VonWeizsaecker` object is reused by the Frozen-Density-Embedding
engine (`fde.py`) to build the **nonadditive kinetic potential**, whose whole
physical content is that it is *short-ranged*.

Switching to FFT breaks that: with the proposed change,
`tests/test_fde.py::test_nonadditive_parts_vanish_when_separated` fails — for two
Gaussians ~10 Bohr apart, the nonadditive vW potential in subsystem A's region
jumps from ≈0 to 3.7e-3 (above the 1e-3 tolerance) purely from FFT ringing,
because `√n` of a coarsely sampled Gaussian is not band-limited. That is a real
regression in embedding quality, not a brittle test.

Resolution: I added a `laplacian={"fd","fft"}` option to `VonWeizsaecker` and
`TFvW`, **defaulting to `"fd"`** (preserving FDE correctness and all existing
behaviour) and exposing `"fft"` for users doing solid-state OF-DFT energetics on
well-resolved plane-wave grids, where the audit's accuracy argument is valid.
This delivers the report's benefit without the regression.

---

## Issues the audit missed

### A. Ion-ion charges are wrong under pseudopotentials (correctness gap in #3)
`compute_ion_ion_energy` uses `system.atomic_numbers` as the ionic charge `Z`.
That is correct for **all-electron** runs, but inconsistent for
**pseudopotential** runs: there the electrons and the electron-ion potential see
only the *valence* charge `Z_val` (the calculator already sets
`system.electrons = Σ Z_val`), while the ion-ion term would repel with the full
`Z·Z`. For Si (Z=14, Z_val=4) the ion-ion energy would be ~12× too large and the
total energy/forces would be unphysical.

I did **not** silently "fix" this because the per-atom valence charge is not
currently plumbed from `build_pseudopotential_potential` through to the engines.
Instead I made it explicit and safe:
`compute_ion_ion_energy(system, grid, charges=None, ...)` accepts an optional
`charges` array, and the docstring states the valence charges must be supplied
for pseudopotential runs. **Follow-up needed:** have
`build_pseudopotential_potential` return per-atom `z_valence`, store it on the
engine, and pass it through. Until then, ion-ion is only physically correct in
all-electron mode (which covers the report's H₂ verification case, since
Z_val = Z = 1 for H).

### B. Ion-ion energy was being recomputed every SCF iteration
The report wires `compute_ion_ion_energy` straight into `_total_energy` /
`compute_total_energy`, both of which run **once per SCF iteration**. The
nuclei do not move during an SCF, and the real-space Ewald term is an `O(N²·n_shifts)`
triple Python loop, so this is pure waste (and it runs inside every
finite-difference force displacement). I cached it on the engine (`self._e_ion`,
computed lazily once per engine instance), which is where the cost actually
matters.

### C. Ewald assumes full 3-D periodicity for any periodic direction
Both the new `compute_ion_ion_energy` and the existing `ewald_summation` branch
on `any(grid.pbc)` but then apply a 3-D Ewald sum. For genuinely 2-D (slab) or
1-D (wire) PBC this is not the correct electrostatic limit. This is pre-existing
behaviour and out of scope here, but worth a tracking note.

### D. `laplacian_fft` is not on the `Backend` ABC
von Weizsäcker's `"fft"` mode (and the existing Poisson/`laplacian_fft` usage)
relies on `NumpyBackend.laplacian_fft`, which is not declared in
`backends/base.py`. Harmless today (NumPy is the only backend) but it should be
added to the abstract interface so future backends are forced to implement it.

---

## Verification

- Full suite: **123 passed** (same as baseline).
- Non-periodic ion-ion: two `+1` charges 2 Bohr apart → `0.5 Ha` exactly.
- Anti-collapse: ion-ion energy diverges as `r → 0` (r=3→0.33, r=0.74→1.35,
  r=0.1→10.0 Ha). End-to-end H₂ KS-DFT now penalises compression
  (E(0.74 Å)=−27.5 eV vs E(0.50 Å)=−20.6 eV), so the molecule no longer collapses.
- Ewald correctness: total energy is independent of the splitting parameter `α`
  for a neutral `+1/−1` cell (−0.4071 Ha across α∈[0.8,1.2]) — the defining
  property of a correct Ewald implementation.
- Non-orthogonal grid: for a hexagonal cell the reciprocal lattice satisfies
  `aᵢ·bⱼ = 2π δᵢⱼ` exactly and `|G|²` matches the analytic `|b₁|`.
- PBE fail-safe raises `NotImplementedError` in both `ks` and `of` modes;
  `xc="lda"`/`None` unaffected; `kind="point"` external potential still builds.
