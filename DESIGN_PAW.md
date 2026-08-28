# DESIGN_PAW.md — making a predicted density readable by VASP

**Goal.** Turn a Poraquê prediction into a `CHGCAR` that VASP will accept for a
non-self-consistent Kohn–Sham run (`ICHARG = 11`), so a band structure can be
obtained from a density that was never converged.

**Status.** Design + implementation, 2026-08-25. Claims verified against the
VASP wiki are cited; claims verified against this repository's own data are
marked **measured** with the number; everything else is marked **TO-VERIFY**
rather than asserted.

---

## 1. What a VASP `CHGCAR` actually contains

### 1.1 The grid block

Layout, in order:

```
<POSCAR block: comment, scale, cell, species, counts, coordinates>
<blank line>
NGXF NGYF NGZF
<NGXF*NGYF*NGZF values, 5 per line, Fortran (1X,E17.11)>
```

The wiki states the ordering explicitly:

> "the density is written using the following command in Fortran:
> `WRITE(IU,FORM) (((C(NX,NY,NZ),NX=1,NGXF),NY=1,NGYF),NZ=1,NGZF)`"

so `NX` runs fastest and `NZ` slowest — Fortran order. `volumetric.py` already
reads and writes exactly this (`data.ravel(order="F")`).

**Scaling.** The values are the density multiplied by the cell volume,
`ρ(r)·Ω`. The wiki's own phrasing ("charge times FFT-grid volume", with a
formula that also divides by `V_grid`) describes a *sum* convention rather than
an integral one and is easy to misread, so this was settled **empirically
instead**: reading the three shipped platinum `CHGCAR`s with Poraquê's
`volume_scaled = True` (divide by Ω on read) and integrating gives

| material | ∫ρ d³r |
|---|---|
| `ref/Pt` (1 atom, ZVAL 11) | 11.000001 |
| `struct_000` (27 atoms) | 297.0000006 |
| `struct_015` (32 atoms) | 352.0000000 |

which are exactly `N_atoms × ZVAL`. **Measured**: Poraquê's convention is
correct and the writer needs no change.

**What the grid block physically is.** The *pseudo* valence density: the soft
plane-wave density plus the compensation charge that restores the correct
multipole moments outside the augmentation spheres. It is smooth inside the
core radius by construction. It is **not** the all-electron density, and no
amount of grid resolution recovers one from it.

### 1.2 The augmentation occupancy block

After the grid block, one record per atom, in the order the atoms appear in the
structure:

```
augmentation occupancies   1 138
  0.6333702E+01  0.5836143E-01  0.9523181E-01  0.2293921E-02 -0.3054571E-01
 -0.1086637E-01  0.0000000E+00 ...
```

Header is `("augmentation occupancies",2I4)` — record index and value count —
then `(5E15.7)` for the values. **Measured** on `struct_015/CHGCAR`: 32 records
of 138 values each, matching its 32 Pt atoms. `augmentation.py` already parses
and re-emits this byte-compatibly.

These are the PAW **one-centre** terms,

```
rho^a_ij = sum_{nk} f_nk <Psi~_nk|p~^a_i><p~^a_j|Psi~_nk>
```

transformed to `rho(ll',L,M)` by Clebsch–Gordan coefficients before writing
(`TRANS_RHOLM` in VASP's `paw_base.F`). They are contractions over converged
wavefunctions, and they live *inside* the core radius where the pseudo-density
carries no information. **A grid-based operator can never predict them.** That
is the whole reason this document exists.

They are on-site quantities and do not depend on the FFT grid, so a record read
from a 108³ calculation is valid on a 32³ one.

**How many values, and what each means.** `LMAXMIX` (default 2) sets the
highest `l` written:

> "The `CHGCAR` file will contain the one-center PAW occupancy matrices up to
> `LMAXMIX`."

**TO-VERIFY:** the exact index ordering of the 138 values for `PAW_PBE Pt` —
i.e. the map from position in the record to `(l, l', L, M)`. It is set by
`TRANS_RHOLM` and is not documented on the wiki. Poraquê never interprets the
values individually (it copies, averages and re-emits them as an opaque
vector), so nothing currently depends on knowing this. It *would* be needed for
learned corrections to the occupancies, which is why that is a `FUTURE.md` item
and not this session's work.

### 1.3 Spin

`ISPIN = 2` writes total density + its augmentation, then the magnetisation
density + *its* augmentation. Poraquê's `read_augmentation` deliberately stops
at the second grid header, so it extracts the total-density records only.

### 1.4 `TAUCAR`

> "The format is the same as the `CHGCAR` file except for the lack of
> augmentation occupancies."

The quoted sentence settles only the *layout*: a `TAUCAR` carries the grid and
no augmentation records. It does **not** settle the scaling, and Poraquê's
assumption that τ was written as τ·Ω to match `CHGCAR` turned out to be wrong —
measured against real output on 2026-08-27, `TAUCAR` holds τ itself, with no
volume factor, and under `ISPIN = 2` its two blocks are (τ_up, τ_down) whose
sum is the total. See `KineticEnergyDensity` and
`tests/test_platinum_dataset.py`, which pin both.

A `TAUCAR` exists at all only because the `INCAR` set `LTAU = .TRUE.`; that tag
is what calculates τ, and there is no other route to one.

---

## 2. What `ICHARG = 11` reads and needs

> `ICHARG = 11`: "To obtain the eigenvalues (for band-structure plots) or the
> density of states (DOS) of a given charge density read from `CHGCAR`."

The density "will be kept constant during the entire electronic
minimization" — one non-self-consistent diagonalization at fixed `ρ`. That is
precisely the operation a predicted density is wanted for.

On the augmentation block the wiki is clear that it matters and less clear
about whether it is mandatory:

> "Restarting calculations without one-center PAW occupancy matrices up to the
> appropriate l-quantum number leads to loss of information."

and

> "When the `CHGCAR` file is read and kept fixed in the course of the
> calculations (`ICHARG=11`), the results will not necessarily be identical to
> a self-consistent run."

**TO-VERIFY:** whether VASP *errors*, *warns*, or *silently proceeds with zero
one-centre occupancies* when the block is absent under `ICHARG = 11`. This
determines whether the augmentation block is a hard requirement or a quality
knob, and it is settled by one five-minute experiment on the user's machine
(run `ICHARG = 11` twice on a reference density, once with the block stripped).
Until then Poraquê writes the block whenever it can, which is correct either
way.

**Also required, and easy to get wrong:** `ENCUT` and `PREC` in the
`ICHARG = 11` run must reproduce the FFT grid the `CHGCAR` was written on, and
`LMAXMIX` must match the one that wrote the occupancies. The generated deck
(`fields/vasp/templates.py`) carries both as commented tags for that reason.

---

## 3. Strategy

### 3.1 (a) Superposition baseline and the δρ target

**The construction.** For each element (and POTCAR variant) the database stores
an isolated-atom entry. Placing them on an arbitrary target grid is done in
reciprocal space, exactly as `ExternalPotential` already places pseudo-ions:

```
rho_sup(G) = (1/Omega) * sum_s f_s(|G|) * S_s(G),
S_s(G) = sum_{a in s} exp(-i G . tau_a)
rho_sup(r) = sum_G rho_sup(G) exp(i G . r)
```

where `f_s(|G|)` is the **atomic form factor**, a radial table.

**Why reciprocal space and a radial table.** Three reasons, and the third is the
one that decides it:

1. The structure factor factorises into an outer product of three 1D phase
   vectors, so the cost is `O(N_atoms · (N1+N2+N3))` exponentials rather than
   `O(N_atoms · N1N2N3)` — `_structure_factor` in `external.py` already does
   this and is reused verbatim.
2. `f(|G|)` is grid-independent: one table serves every cell and every FFT
   shape, which is the same invariance the FNO itself is built around.
3. The electron count comes out **exact by construction**. `f_s(0) = Z_val,s`,
   so `∫ρ_sup = Σ_a Z_val,a` with no normalization step. That is what makes the
   interaction with the electron-count constraint tractable (§3.3).

**Is the atomic density radial enough for this?** **Measured** on
`data/vasp/ref/Pt` (one Pt atom, 10 Å cube, 108³): after recentring on the
atom, `f(G)` is real to `6.8e-16` relative, and the worst within-bin scatter is
**0.48 % of `f(0)`** (512 bins to the Nyquist radius, which is what the builder
records as `radial_scatter`; a coarser probe over `|G| < 8 Å⁻¹` in 80 bins gives
0.74 %). The residual is the cubic anisotropy a box calculation of an open-shell
atom necessarily has.

**The end-to-end number is the one that matters.** Superposing the stored table
back onto the reference atom's own grid reproduces its density to a relative
`L²` of **3.0e-4** — the total cost of the radial reduction *and* the binning.
The per-bin anisotropy is scattered across directions and largely averages out
of the sum, which is why the end-to-end figure is far below the 0.48 %.

Against that, the residual the baseline leaves behind on a real supercell is
**0.036** (`struct_000`) and **0.037** (`struct_015`). So the baseline removes
~96 % of the field while introducing an error two orders of magnitude smaller
than what remains to be learned, and the electron count comes back exact:
297.000037 superposed against 297.000001 in the reference, the drift being the
reference atom's own integration error carried through `f(0)`.

**Two details of the table cost an order of magnitude each, and both were
found by testing against a Gaussian atom whose `f` is known in closed form.**

- **Bin by the mean `|G|` of the points in a shell, not by the bin's
  midpoint.** Reciprocal-lattice points are not spread uniformly across a
  shell — there are few of them at small `|G|` and they cluster — so pairing
  the *mean* of `f` with the *midpoint* of the bin mismatches the two by the
  local curvature. Round-trip: 9e-3 → 6e-4.
- **Interpolate linearly in `G²`, not in `|G|`.** A spherically symmetric
  density has an `f` that is even in **G**, so it is analytic in `G²` and has
  *zero slope at the origin*: `f(G) = Z − ⟨r²⟩G²/6 + O(G⁴)`. The reference
  cell has nothing between `G = 0` and its first shell at `2π/L` (0.63 Å⁻¹ for
  the 10 Å platinum atom), and any target grid with points inside that gap is
  served by interpolation alone. A chord drawn in `|G|` cuts across the
  curvature and gets the origin slope wrong by construction: 3.3 % of `f(0)` in
  that first gap, against 0.2 % in `G²`.

**Pros of learning δρ = ρ − ρ_sup:**

- Most of the dynamic range disappears. The core peaks — which span four orders
  of magnitude and are what the `asinh` transform currently exists to absorb —
  are almost entirely in ρ_sup. What is left is the *bonding* charge, which is
  smooth, small and sign-changing.
- The residual is spatially localised in the bonding regions, so a fixed mode
  truncation covers a larger fraction of it.
- It is the standard trick in this literature. The KS-FNO paper predicts
  `n_SAD + F_θ[·]` for exactly this reason, and `report.md` §5.5 lists it as one
  of two "cheap architectural imports".
- Extrapolation to new geometries improves for free in the part of ρ that is
  pure superposition, which is most of it.

**Cons, stated honestly:**

- δρ is **signed**, so positivity is no longer a property of the target. The
  positivity constraint has to move to the reconstructed ρ (§3.3), and the
  `Asinh` transform — which is sign-tolerant — is doing a different job on a
  field whose sign genuinely alternates.
- The baseline must be *identical* at train and inference time. A change to the
  atomic database silently invalidates every model trained against it. The
  database therefore carries a hash, recorded in the checkpoint.
- A relative L² on δρ is not comparable with a relative L² on ρ. The denominator
  is far smaller, so the same physical error reports as a much larger number.
  **Every δ-mode metric must be quoted against the absolute density** or the
  comparison with existing runs is meaningless. This is the most likely way to
  fool oneself with this change.
- An element with no isolated-atom entry cannot be trained in δ-mode at all.

**Absolute mode stays available** behind `data.delta_density: false`, which is
the default. The ablation is a `FUTURE.md` item with the comparison protocol
already written down.

### 3.2 (b) Augmentation occupancies for a predicted `CHGCAR`

The plan was: take the occupancies from the isolated-atom reference per element
(nearest-atom assignment), as a zeroth-order approximation.

**This was implemented, and it is measurably the worse of the two options.**
Comparing the isolated Pt atom's record against the 32 per-site records of
`struct_015` (same POTCAR, same `LMAXMIX`, same 138 values):

| reference for a bulk site | RMS error, relative to the bulk mean |
|---|---|
| **training-set average per element** (already in the codebase) | **9.9 %** |
| isolated-atom record | **86.6 %** |

The leading component alone is `2.841` for the isolated atom against a bulk mean
of `6.795` — **58 % off**. The reason is not subtle: a free Pt atom and an Pt
atom in a metal have genuinely different on-site occupations, and the
one-centre terms are exactly where that difference lives.

**Decision, departing from the stated plan.** The isolated-atom database stores
the augmentation record — it is real provenance and it is the only option for an
element absent from the training set — but the **default source stays the
training-set average**, with the isolated-atom block as an explicit fallback
(`--paw-source atomic`). The numbers above are printed by the writer so the
choice is never silent.

The isolated-atom entries' real job is §3.1: the *density* superposition, where
they are the right object and the measurement supports them.

**Limitations, stated plainly.** Both sources ignore environment-induced changes
in the on-site density matrix. The 9.9 % figure is the site-to-site scatter
within one rattled 32-atom platinum cell; it is **not** a transferability estimate
across coordination numbers, phases or elements, and nothing here measures that.
`struct_000`, a pristine supercell, has **0.00 %** scatter — every site is
symmetry-equivalent — which shows how easily this number can look better than it
is. Quantifying it properly needs the `ICHARG = 11` validation runs, which
happen outside this session.

Learned corrections to the occupancies go to `FUTURE.md`, not here.

### 3.3 (c) Order of operations at inference — the part that must not be got wrong

δ-mode, positivity and electron-count normalization all touch the same tensor,
and the order decides whether the result is a density. The chosen order:

```
1.  network  ->  delta_rho_normalized
2.  target_transform.inverse       ->  delta_rho          (e/Ang^3, SIGNED)
3.  + baseline (rho_sup on this grid)  ->  rho            (e/Ang^3, absolute)
4.  clip negatives to zero                                (positivity)
5.  rescale to the exact electron count                   (normalization)
```

**Why the baseline is restored before steps 4 and 5, and not after:**

- **Positivity is false for δρ.** Bonding charge is negative wherever charge has
  moved away from the free-atom superposition. Clipping δρ at zero would delete
  exactly the physics the residual was introduced to represent.
- **Normalizing δρ is undefined.** `ChargeDensity.normalized` rescales by a
  global factor `N_target/∫field`. δρ integrates to approximately *zero*, so the
  factor is a small number divided by a smaller one — numerically explosive and
  physically meaningless.
- **The baseline already carries the whole electron count**, exactly
  (`f_s(0) = Z_val,s`). So after step 3 the field integrates to `N ± ε`, and
  step 5 is the small correction it is meant to be rather than a rescue.

During **training** the same reconstruction happens before the physics terms:
`physical_prediction = inverse(prediction) + baseline` and
`physical_target = target_physical + baseline`. No loss term changes — every
one of them is a statement about the absolute density, and this is what makes
them true again. The *data* term still acts on normalized δρ, which is the
point of the mode.

**The batch carries the baseline** (`FieldPairDataset` computes it once per
material and caches it), because reconstructing it inside the loss would mean
rebuilding a structure factor per step.

### 3.4 (d) The writer, and how it is validated

`write_volumetric(..., augmentation=...)` already exists and already appends the
records verbatim. The gap was that nothing proved the round-trip.

**The validation chosen: read an unmodified reference `CHGCAR`, write it back,
and compare.** Not byte-level — that would be testing `fortran_exponential`'s
rounding against VASP's, which is a different claim — but:

- the parsed structures agree,
- the grid shapes agree,
- the density arrays agree to the file's own written precision (`E17.11`, so
  ~1e-11 relative),
- the augmentation block is **byte-identical**, because it is copied as text and
  never reformatted,
- and the re-read of the written file reproduces the first read exactly.

That is the strongest statement the format supports without owning VASP's
Fortran runtime.

### 3.5 (e) Band-structure decks

`poraque-vasp bands` writes `INCAR` (`ICHARG = 11`) and a line-mode `KPOINTS`
beside a predicted `CHGCAR`, plus the `POSCAR` taken from the density's own
header. It does not write a `POTCAR` (cannot be redistributed) and does not run
VASP. Since 2026-08-27 the same command also writes the two other decks that
read a prediction back — `poraque-vasp dos` (the same `ICHARG = 11`
diagonalization on an automatic Γ-centred mesh with `ISMEAR = -5`) and
`poraque-vasp energy` (that mesh's total energy, which for `ICHARG = 11` is the
Harris–Foulkes functional at the predicted density rather than a variational SCF
energy, and the deck says so).

---

## 4. Schema

```
atomic_reference.json
{
  "version": 1,
  "entries": {
    "Pt|PAW_PBE Pt 06Sep2000|<potcar_sha16>": {
        "element": "Pt",
        "potcar_title": "PAW_PBE Pt 06Sep2000",
        "potcar_sha256": "...",
        "valence_charge": 11.0,
        "g_grid":       [...],      # |G| in 1/Ang, ascending from 0
        "form_factor":  [...],      # f(|G|), f(0) = valence_charge
        "g_max": 8.0,               # table range; f = 0 beyond it
        "radial_scatter": 0.0074,   # measured anisotropy, see 3.1
        "augmentation": [...],      # the atom's own one-centre record
        "source": "data/vasp/ref/Pt",
        "vasp_version": "6.2.0",
        "incar_sha256": "...",
        "cell_volume": 1000.0,
        "grid": [108, 108, 108]
    }
  }
}
```

Keyed by `element|potcar_title|potcar_hash` so two POTCAR variants of the same
element (`Pt` vs `Pt_pv`) never collide; lookup falls back to element alone when
only one variant is present, which is the normal case.

---

## 5. TO-VERIFY, collected

1. **Does `ICHARG = 11` require the augmentation block, or merely prefer it?**
   The wiki says omitting one-centre occupancies "leads to loss of information"
   but does not say whether VASP refuses the file. Settled by one experiment.
2. **The `(l, l', L, M)` index ordering inside an augmentation record.** Set by
   `TRANS_RHOLM`; not on the wiki. Not needed until occupancies are *predicted*
   rather than copied.
3. **Whether the compensation charge is included in the `CHGCAR` grid block as
   Poraquê assumes.** The electron-count check (§1.1) is consistent with it but
   does not isolate it; a `LAECHG = .TRUE.` run writing `AECCAR0`/`AECCAR2`
   would separate the terms and settle it.
4. **Transferability of averaged occupancies across environments.** The 9.9 %
   figure is within-cell scatter for one rattled platinum supercell. Nothing here
   measures a different coordination, phase or element.
5. **Whether `ICHARG = 11` on a predicted density gives a usable band
   structure at all** — the actual question. Runs happen outside this session;
   `poraque-vasp bands` produces the inputs, and `poraque-vasp dos` / `energy`
   ask the same question of the two other observables.
6. **Whether δ-mode helps.** Argued from the literature and from dynamic-range
   reasoning, not measured here. The ablation is specified in `FUTURE.md`.

---

## 6. Sources

- [ICHARG — VASP Wiki](https://www.vasp.at/wiki/index.php/ICHARG)
- [CHGCAR — VASP Wiki](https://www.vasp.at/wiki/index.php/CHGCAR)
- [LMAXMIX — VASP Wiki](https://www.vasp.at/wiki/index.php/LMAXMIX)
- [TAUCAR — VASP Wiki](https://vasp.at/wiki/TAUCAR)
- [LTAU — VASP Wiki](https://www.vasp.at/wiki/index.php/LTAU)
- Measurements: this repository's `data/vasp/ref/Pt`, `struct_000`,
  `struct_015`.
