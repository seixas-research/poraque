# What predicts a Pt atom's PAW augmentation occupancies, and in what form?

```bash
python experiments/paw_occupancies/occupancies.py            # extract once, then evaluate
python experiments/paw_occupancies/occupancies.py --refresh  # re-extract
```

The per-atom half of the `ext2paw` target: the 138 numbers `ρ(ll′LM)` VASP
writes after a `CHGCAR`'s grid, one set for the total density and one for the
magnetisation. Measured on the 97 platinum cells of `data/vasp/structures`
(31 bulk, 33 slabs, 33 nanoparticles; 3720 atoms), with the records read from
the native `CHGCAR`s and every field feature taken from `data/cache/res64_potcar`.
Every error below is on **whole structures held out**: five folds stratified by
family, and separately each family left out entirely. `results_res64_potcar.json`
holds every number.

## 1. The target, pinned before anything is fitted

Two conventions decide whether a model can be equivariant at all, and the file
states neither.

**The channel order is the POTCAR's projector order, d d s s p p.** The `l = 3`
line of `PAW_PBE Pt`'s Description is the local potential (`ICORE = 3`), not a
projector. With that order, the non-zero components of a bulk record sit exactly
at L = 0 and L = 4, which cubic site symmetry requires; with s s p p d d the same
fifteen numbers land at every L from 0 to 4.

**M follows the standard real spherical harmonics, m = −L…L.** The bulk L = 4
block of every d–d pair is `Y₄₀ + √(5/7) Y₄₄`: measured ratio 0.8451543, √(5/7) =
0.8451543.

So a record is a set of spherical tensors, and nothing is lost by regrouping it:

| L | pairs | components | share of Σ‖ρ‖² (total set) | rank for 99 / 99.9 / 99.99 % |
|---|---|---|---|---|
| 0 | 9 | 9 | 99.00 % | 1 / 2 / 2 |
| 1 | 8 | 24 | 0.23 % | 1 / 2 / 3 |
| 2 | 10 | 50 | 0.34 % | 2 / 4 / 5 |
| 3 | 4 | 28 | 0.02 % | 2 / 3 / 3 |
| 4 | 3 | 27 | 0.41 % | 2 / 2 / 3 |

Two things follow. **The records are nearly all monopole**, and the nine L = 0
numbers of a Pt atom are effectively two. And **the magnetisation set is not
negligible on this data**: its record norm per atom is 1.0e-3 in bulk and
1.5e-4 in slabs, but 1.41 in the nanoparticles, against 8.39 for their total set
— 17 %.

## 2. An invertible training form, and where it belongs

| transform | round trip |
|---|---|
| regroup into the L blocks above | **bit-exact** (a permutation) |
| … and scale each (L, pair) by its RMS | `CHGCAR` text **identical** |
| … and centre L = 0, rotate each L's pair space by its principal axes (shared across M) | 9.8e-15 absolute; text identical on every component above 1e-12 |

The 3.9 % of components below 1e-12 are the symmetry zeros VASP prints as
`0.35E-17` noise, and the centring moves them by 1e-15; nothing reads them.

Every one of these is equivariant: scales and rotations act on the pair index
and are shared across M, and only L = 0 is centred. The rotation is also a
compressor — keeping the ranks in the table keeps 99.9 % of the signal in 17
numbers per atom instead of 138.

**Recommendation: the form belongs in the dataset layer, not in the cache.** The
cache keeps the records verbatim, as VASP wrote them, since they are what a
`CHGCAR` needs back and they are on-site, so downsampling never touches them.
The scales, the L = 0 mean and the rotation are *statistics of a training split*,
exactly as `FieldTransform`'s are. They should be fitted per run and stored in the
checkpoint, where inference inverts them. Baked into the cache, they would leak
the held-out structures into the normalisation and pin every future split to the
one the cache was built for.

A loss on the raw 138 numbers would also be 99 % an L = 0 loss. Per-block scaling
is what lets the L > 0 blocks be learned at all.

## 3. What predicts them

Every model is linear in its features and **exactly rotation-equivariant**: one
ridge map per L, shared across M, with no intercept for L > 0.

| model | all | L = 0 | L > 0 | bulk | slab | nanoparticle | unseen bulk | unseen slab | unseen nanoparticle |
|---|---|---|---|---|---|---|---|---|---|
| element mean (the current PAW table) | 28.3 % | 26.7 % | 97 % | 21.2 % | 22.7 % | 40.3 % | 28.1 % | 23.1 % | 49.5 % |
| positions: Σⱼ Rₙ(r) Y_LM(r̂), 6 Å | 5.6 % | 0.74 % | 56 % | 2.2 % | 5.6 % | 8.2 % | 2.5 % | 11.7 % | 10.4 % |
| V_ext, grid projection | 9.6 % | 5.6 % | 77 % | 4.3 % | 9.3 % | 13.7 % | 10.7 % | 114 % | diverges |
| ∇²V_ext, spectral (the pseudo-ion charge) | 10.7 % | 6.7 % | 65 % | 7.4 % | 6.8 % | 15.3 % | 17.2 % | 8.3 % | 10.8 % |
| ρ̃, grid projection | 3.3 % | 0.44 % | 33 % | 2.4 % | 3.3 % | 4.1 % | 2.5 % | 3.8 % | 184 % |
| ρ̃, grid projection, RBF kernel on L = 0 | 3.4 % | 0.70 % | 33 % | 2.4 % | 3.3 % | 4.3 % | 2.5 % | 3.7 % | 139 % |
| **ρ̃, spectral, \|G\| ≤ 9 Å⁻¹** | 2.3 % | 0.40 % | 23 % | 1.3 % | 2.5 % | 3.0 % | 1.8 % | 2.9 % | **4.9 %** |
| **ρ̃ spectral + positions** | **2.1 %** | **0.27 %** | **21 %** | **1.0 %** | **2.2 %** | **2.9 %** | **1.5 %** | **2.9 %** | 5.4 % |

"Grid projection" is `Σ ρ̃(r) gₙ(|r − R|) Y_LM` over the cached grid points in a
2 Å sphere. "Spectral" is the same integral evaluated from the Fourier
coefficients inside one sphere |G| ≤ 9 Å⁻¹ common to every cell, which is below the
smallest Nyquist frequency in the cache.

Read in order, it says four things.

**The pseudo-density carries the occupancies, linearly.** L = 0 is predicted to
0.27–0.40 %, and an RBF kernel does *worse* than the linear map (0.70 %). That is
consistent with the PAW construction: the density VASP writes on its fine grid
includes the compensation charges, whose multipoles are linear in `ρ(ll′LM)`.
The linearity is the measurement; the mechanism is the reading of it.

**A grid-dependent feature is a trap.** The grid projection's 184 % on held-out
nanoparticles is not extrapolation. The cache's grid spacing is 0.12 Å in a bulk
cell and 0.31 Å in a nanoparticle box, so the same physical feature was being
integrated by two different quadratures. Computed over a fixed band instead, the
failure is 4.9 %. This is the lesson `mode_selection: physical` taught the
operator, now appearing in its readout.

**V_ext has a gauge, and a readout of it inherits the gauge.** The stored potential
has its G = 0 term zeroed, so its constant shifts with a cell's vacuum fraction,
and the raw projection diverges on held-out slabs and nanoparticles.
∇²V_ext has no constant and does not diverge. It carries no more than the geometry
already does, though: 10.7 %, against 5.6 % from positions.

**What is left is almost all L > 0.** In the best model 98.4 % of the remaining
squared error is in the L > 0 blocks, which hold 1 % of the signal and are
predicted to 21 %. They are orientation-dependent tensors, and a map linear in
first-order features is the least capable equivariant model there is. This is
where a real model has to earn its keep.

**The magnetisation set is not predictable from these features.** The best
nanoparticle error is 26 % (ρ̃ spectral + positions), and a held-out nanoparticle
family is 100 %, because bulk and slab carry no moment to learn from. Nothing
above contains m(r): the res64 cache stores one density channel. A spin cache
(`data.spin: auto`) supplies it. On this data, the nanoparticles' occupancies
cannot be modelled without it.

## 4. Operator families

**Graph neural operator (GNO): yes, as the grid-to-atom readout.** A GNO layer
evaluates a kernel integral from one point set to another. The ρ̃ projection
above is exactly such an integral, with the kernel fixed to `gₙ(r) Y_LM(r̂)`
rather than learned, and it already reaches 2.1 %.

The architecture this points to is the geometry-informed one:
- an FNO on the grid, as `ext2chg` already is;
- a GNO decoder from its latent field to the atomic positions, with radial-times-`Y_LM` kernels so each L block stays equivariant.

Two measured constraints on it:
- evaluate the integral **spectrally, over a common band** — the grid quadrature failed at 184 %;
- make the output blocks per L and shared across M.

A GNO over the *atoms* alone is the positions row, which gives 5.6 % from a
single linear equivariant layer. Equivariant message passing (the NequIP/MACE
family) is its deeper form, and the natural candidate for the L > 0 residual.

**Laplace neural operator (LNO): no.** Its advantage is a pole–residue kernel for
transient and non-periodic responses. These fields are periodic, static and
plane-wave by construction, and the Fourier basis the FNO already uses is exact
for them.

**Koopman neural operator (KNO): no.** It learns linear dynamics in a latent space,
for time evolution. `ext2paw` is a static map, and the data has no trajectory: the
runs ship no SCF history.

**One consequence for the project's own reasoning.** cuequivariance was left out
because a scalar-field operator carries only ℓ = 0. That argument does not extend
to this head: its outputs are spherical tensors up to L = 4, and an equivariant
network predicting them needs L = 4 tensor products.

## 5. Recommendation

*Built as `poraque.ml.paw`; `docs/source/configuration.md` (`paw`) describes the
operator and its settings.*

- **Build `ext2paw` in two stages.**
  - The field half is `ext2chg`.
  - The per-atom half reads the grid spectrally at each atom inside |G| ≤ G_max, adds the positions expansion, and maps each L block equivariantly.
  - Start with the linear maps measured here as the baseline. Give the L > 0 blocks a nonlinear equivariant head, since that is where 98 % of the remaining error is.
- **Train in the invertible form of §2**, fitted on the training split and stored in the checkpoint. Weight the loss per L block.
- **Use a spin-polarised cache** for anything containing the nanoparticles.

**The open measurement: how far these errors grow when ρ̃ is a prediction rather
than DFT's.** Every ρ̃ row above used the reference density. The features are
linear in ρ̃, so the error propagates linearly, but its size depends on the
`ext2chg` model, and every existing `.poraque` must be retrained before it can be
measured.
