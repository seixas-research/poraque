# PAW data in the checkpoint, the `ext2paw` operator, and run diagnostics

**Scope:** work of 16–17 September 2026, released as `v26.9.12`, `v26.9.13`
and `v26.9.14` · **Companions:** `experiments/paw_occupancies/README.md`,
`FUTURE.md` §4.8, `docs/source/configuration.md`

Three requests, in order, and one thread through the first two. A `CHGCAR`
that VASP will read back is a pseudo-density *and* a block of per-atom PAW
augmentation occupancies, and until now Poraquê predicted only the first. The
first release made the checkpoint carry the PAW data an inference run needs.
The second turned the occupancies into a training target, measured what
predicts them, and built an operator for them. The third is unrelated to PAW:
it makes a training run report its own cost, draw its parity over every
structure, and split a spin-polarised loss into its two channels.

| release | what it holds | tests after |
| --- | --- | --- |
| `v26.9.12` | `--from-incar` implies `--to-vasp --add-paw`; checkpoint = `{model_state_dict, config, paw_profiles}` with POTCAR core densities | 2144 passed, 81 skipped |
| `v26.9.13` | augmentation records through cache and batch; `ext2paw` task and `model.paw` block; the occupancy experiment; the `ext2paw` operator | 2215 passed |
| `v26.9.14` | resource profile; parity over every structure; charge and magnetisation parts of the loss | 2234 passed, 83 skipped |

Every count is with `PORAQUE_TEST_POTCAR_DIR` set and `-m "not gpu"`.

---

## 1. Inference and the checkpoint (`v26.9.12`)

### `--from-incar` switches on what it needs

A run that names an `INCAR` is a run whose output is going back into VASP, so
`poraque-inference --from-incar` now turns on `--to-vasp` and `--add-paw`
itself, at the top of `run()` (`apply_incar_implications`, driven by the
`INCAR_IMPLIES` table), and logs the flags it enabled. Three things had to move
with it, because each was written for a world in which the user typed those
flags:

- **The error when no PAW records exist** said to drop `--add-paw`, a flag the
  user may never have typed. It now says `--from-incar` switched it on.
- **`--resolution` beside `--from-incar`** was silently discarded, since
  `--to-vasp` uses VASP's own FFT-grid rule. It is still discarded, but the log
  now says so.
- **A documented example** promised a 64³ grid from `--from-incar`, which was
  never true. The docs now reach that grid with `--encut 450 --prec-accurate`.

### A checkpoint is exactly three keys

```python
checkpoint = {"model_state_dict": ..., "config": ..., "paw_profiles": ...}
```

- `model_state_dict` maps each task to `FieldOperator.state()`: the weights and
  everything that rebuilds the model around them (architecture record,
  normalising transforms, δ-density baseline). Without the transforms a
  prediction comes out in the wrong units, so they are not optional.
- `config` is the resolved `TrainingConfig.to_dict()`; resolution, epochs and
  fine-tuning provenance live there rather than in ad hoc keys.
- `paw_profiles` maps atomic number to that element's **radial PAW core charge
  density** and its averaged augmentation record (the one `--add-paw` writes
  when no reference calculation is available), and since `v26.9.13` the
  element's projector channels.

`save_bundle`, `read_bundle`, `bundle_tasks`, `load_bundle` and
`CHECKPOINT_KEYS` in `ml/training.py` are the whole interface. The previous
`poraque-bundle-1` layout raises by name and says to retrain; by decision there
is no compatibility layer. **Every model under `models/` is in that layout and
no longer loads.**

### Reading a core density whose units the file does not state

`PotcarSingle.core_profile` reads the `PAW radial sets` block of the POTCAR
that built V_ext (the run's own, else `data.potcar_dir`). The file gives
neither the quantity nor the mesh unit, and a parser that guessed either wrong
would store a smooth, plausible, wrong curve. Both were therefore measured:

- the table is **r²ρ₀₀(r)**: √(4π)∫ of it is exactly Z − ZVAL core electrons on
  nine POTCARs (68.0000 for Pt);
- the mesh is in **Å**: the pseudized core rejoins the all-electron core at
  `RPACOR` converted to Å, checked on five elements.

Titles are matched whole and only after the block marker, because the
reciprocal-space `core charge-density (partial)` earlier in the file is a
different table. `data/cache.py:build_paw_profiles` writes `paw_profiles.json`
beside the cache's `paw_reference.json` and reuses it when present, so a GPU
node without the POTCARs mounted cannot overwrite the profiles with nothing. On
the 97 Pt cells the table builds in 0.3 s.

**Two consequences to keep in view.** Nothing downstream reads the core density
yet; inference still takes V_ext, the electron count and the energy report from
a POTCAR. And the POTCAR header forbids redistribution without a VASP licence,
so a checkpoint trained on POTCAR-bearing data now carries licensed data
(CLAUDE.md pitfall 13).

**Files:** `scripts/poraque_inference.py`, `scripts/poraque_train.py`,
`src/poraque/ml/training.py`, `src/poraque/data/cache.py`,
`src/poraque/fields/vasp/potcar.py`, `src/poraque/fields/vasp/augmentation.py`;
tests in `tests/test_paw_profiles.py` (19), `tests/test_inference.py`,
`tests/test_ml.py`.

---

## 2. Augmentation occupancies as training data (`v26.9.13`, part 1)

### The parser bug found on the way in

A spin-polarised `CHGCAR` writes one line of per-ion `MAGMOM` values after its
first set of augmentation records. The parser read each record up to the next
header, so those values were folded into the last atom's record: 170 values
instead of 138 for atom 32 of a Pt cell, and 139 for the free atom. The damage
was already in the stored tables:

- with `data.paw_source: material` the per-element table came back **empty** on
  all 97 cells, because the oversized record failed a length check and the file
  was skipped without a message;
- with `atomic`, the default, all three caches held a **139-value** Pt record,
  its extra value the free atom's `MAGMOM` of 1.0, and `--add-paw` wrote it
  straight into a `CHGCAR`.

`parse_augmentation` now reads each record to the length its header declares
(`augmentation occupancies   i  n`). The cached tables carry
`"schema": RECORD_SCHEMA` (2) and the isolated-atom memo `SCHEMA_VERSION` 2,
so a stale table is rebuilt on the next run rather than trusted. The δ-density
fingerprint covers only the form factors, so no baseline was invalidated.

### Records through the cache and the batch

The downsampled cache used to drop the records (zero in
`res32_potcar/structure_0000/CHGCAR`). It now keeps them verbatim, one set per
density channel, in text and HDF5 alike, and a real Pt cell round-trips them
bit for bit with spin on and off. A cache built before 16 September has none,
and `FieldPairDataset.site_targets` raises and says to rebuild.

For a task whose `site_target` is `"augmentation"` a sample gains:

| key | shape | meaning |
| --- | --- | --- |
| `paw` | `(A, S, L)` | records per atom, per set (total, magnetisation) |
| `paw_lengths` | `(A,)` | record length, which depends on the element |
| `species` | `(A,)` | atomic numbers |
| `positions` | `(A, 3)` | fractional coordinates |

`collate_fields` pads atoms and values to `paw (B, A, S, L)` and adds
`paw_mask (B, A, L)` and `atom_mask (B, A)`. A member of a spin set with no
magnetisation records gets zeros, as its grid gets m ≡ 0. Both sets are kept
because VASP misreads a spin-polarised `CHGCAR` written with only the first.

### The task and its switch

`ext2paw` is a `task.type` (`TASKS` in `ml/tasks.py`, with
`site_target = "augmentation"`), but not a link of the chain: `CHAIN` is still
`("ext2chg", "chg2tau")` and `all` still means those two. The configuration
block is `model.paw: {enable, width, modes, n_layers, readout_g_max,
occupancy_weight}`, defaults 64 / 16 / 4 / `auto` / 1.0, using the schema's
existing key names rather than the request's `hidden_channels`, list-valued
`modes` and `layers`. The two switches are cross-checked in
`TrainingConfig.task_names()`: `ext2paw` with the block off raises, the block
on beside `ext2chg` or `chg2tau` raises, and `all` with the block on adds
`ext2paw`. Every other committed template states `paw: {enable: false}` (a deliberate
exception to the no-restated-defaults rule in `tests/test_precision.py`), and
the new `configs/train_ext2paw.yaml` turns it on.

**Files:** `src/poraque/fields/vasp/augmentation.py`, `src/poraque/data/cache.py`,
`src/poraque/fields/atomic.py`, `src/poraque/ml/data.py`, `src/poraque/ml/tasks.py`,
`src/poraque/ml/config.py`, the configs; tests in `tests/test_ext2paw.py` (31).

---

## 3. What predicts the occupancies (`experiments/paw_occupancies/`)

Before building a model, the request asked which operator family fits — GNO,
LNO, KNO or something else — and whether the target could be stored in a form
easier to learn yet invertible back to the `CHGCAR`. This was answered by
measurement, on the 97 Pt cells (3720 atoms), records from the native files and
field features from `res64_potcar`, with whole structures held out: five folds
stratified by family, and separately each family left out entirely.

### The target is a set of spherical tensors

VASP states neither convention, and both were pinned from bulk symmetry. The
channel order is the POTCAR's projector order, **d d s s p p**: only with it do
a bulk record's non-zero components sit at L = 0 and L = 4, as cubic site
symmetry requires. The components are **standard real spherical harmonics,
M = −L…L**: every bulk d–d L = 4 block is Y₄₀ + √(5/7) Y₄₄ to seven digits. The
138 values per atom therefore regroup into 9×L0 + 8×L1 + 10×L2 + 4×L3 + 3×L4
blocks, and a model can be exactly rotation-equivariant. 99 % of Σ‖ρ‖² is
L = 0, effectively two numbers, yet the magnetisation set is 17 % of the record
norm in the nanoparticles and cannot be dropped.

### An invertible training form, fitted per run

Regrouping into L blocks is a permutation (bit-exact); scaling each (L, pair)
block by its RMS leaves the `CHGCAR` text identical; centring L = 0 and
rotating each pair space to its principal axes loses only the ~1e-17 noise VASP
prints for symmetry zeros, and keeps 99.9 % of the signal in 17 numbers per
atom. The recommendation was to keep the cache verbatim and fit the form on the
training split, stored in the checkpoint like the field normalisations; baked
into the cache, one split's statistics would leak into every other.

### Linear, equivariant models, held out

| features | all | L = 0 | L > 0 | unseen nanoparticles |
| --- | --- | --- | --- | --- |
| element mean (today's table) | 28.3 % | 26.7 % | 97 % | 49 % |
| atom positions, 6 Å | 5.6 % | 0.74 % | 56 % | 10.4 % |
| density on the grid | 3.3 % | 0.44 % | 33 % | **184 %** |
| density, fixed \|G\| ≤ 9 Å⁻¹ band | 2.3 % | 0.40 % | 23 % | **4.9 %** |
| density band + positions | **2.1 %** | **0.27 %** | 21 % | 5.4 % |

The 184 % is the instructive row. Grid spacing is 0.12 Å in a bulk cell and
0.31 Å in a nanoparticle box, so a grid-quadrature projection computes "the same"
feature two different ways; over a fixed band in |G| it drops to 4.9 %. It is
the lesson `mode_selection: physical` taught the field operator, reappearing in
the readout. V_ext is a weak input (its arbitrary constant breaks held-out
families; ∇²V_ext is stable but no better than positions), a nonlinear kernel
on L = 0 does worse than the linear map, and 98 % of the remaining error is in
L > 0. The magnetisation set could not be predicted from a one-channel cache
(26 % on nanoparticles at best).

### The operator families

- **GNO: yes, as the grid-to-atom step.** The fixed-band density projection
  above is a fixed-kernel instance of it and already reaches 2.1 %.
- **LNO: no.** It is built for transient, non-periodic responses; these fields
  are periodic and static, and the Fourier basis is already exact for them.
- **KNO: no.** It learns time evolution, and this map has no time axis (the
  runs carry no SCF history).
- **Equivariant message passing (NequIP/MACE family)** is the natural nonlinear
  model for the L > 0 blocks. That reopens an earlier decision: cuequivariance
  was set aside because the field operator carries only ℓ = 0, and this target
  goes to L = 4.

The script, README and every number (`results_res64_potcar.json`) are in
`experiments/paw_occupancies/`; the extracted features (40 MB) are in the
gitignored `data/cache/paw_occupancies/`.

---

## 4. The `ext2paw` operator (`v26.9.13`, part 2)

Built as the experiment recommended, in `src/poraque/ml/paw.py`:

```
V_ext grid ──► FNO3d.encode ──► latent ──► 1×1 conv ──► 4 read channels ──┐
                     │                                                    ├─► SiteReadout ──┐
                     └──► FNO3d.project ──► pseudo-density ρ̃ ─────────────┘   (per atom)    │
                                               │                                            ├─► EquivariantHead ──► ρ(ll′LM)
atom positions + cell ─────────────────────────┼──────────────────► NeighbourExpansion ─────┘   (one per element)
                                               └──► density loss, as in ext2chg
```

1. **Backbone.** The ordinary `FNO3d`, sized by `model.paw`, with every other
   `model` key (equivariance, activation, precision, …) applying. Its `forward`
   was split into `encode` and `project`, so the density head and the readout
   share one pass through the Fourier blocks.
2. **`SiteReadout`, the grid-to-atom kernel integral.** For each atom at R and
   each read field f (four latent channels and ρ̃):
   c_nLM(R) = ∫ f(R + d) N_nL r^L Y_LM(d̂) exp(−r²/2s_n²) w(r) d³d, with eight
   widths s_n from 0.2 to 0.8 Å, a cosine window w at 2.5 Å and L ≤ 4. Before
   integrating, each field is cut to |G| ≤ `readout_g_max` and resampled to a
   spacing of π/2G_max, so every cell uses the same quadrature. Two grids
   holding different bands of one field give identical features (1e-15); the
   features match direct quadrature to 4e-7 at L = 0 and 6e-4 at L = 4; a grid
   too coarse for the band raises. `readout_g_max: auto` is 95 % of the coarsest
   training grid's Nyquist frequency.
3. **`NeighbourExpansion`.** A fixed sum over neighbours within 6 Å,
   Σ_j R_n(r_aj) Y_LM(r̂_aj), ten Gaussian radial functions from 2.2 to 5.8 Å.
4. **`EquivariantHead`, one per element.** An invariant path (L = 0 features
   and the norms of the L > 0 channels, through LayerNorm and a two-layer MLP)
   predicts the L = 0 blocks with a linear skip. Each L > 0 block gets a linear
   map across channels, shared over M and without bias, multiplied by sigmoid
   gates from the MLP's hidden state. Outputs are scattered back into VASP's
   record order using the layout parsed from the POTCAR's `Non local Part`
   headers (`PotcarSingle.projector_channels`): d d s s p p for Pt, p p d d s s
   for `Fe_pv`, read per element rather than hard-coded.
5. **Target and loss.** `OccupancyTransform` centres L = 0 and scales each
   (set, L, pair) block by its RMS, fitted on the training split and stored in
   the checkpoint's `site` record with the layouts and the band. The loss is a
   masked squared error in which every L block weighs the same, added to the
   field loss with `occupancy_weight`. The principal-axes rotation was dropped
   from the recommended form: the head's last linear layer absorbs it, and
   whitening would inflate the ~1e-4 tail of components.

### How equivariance holds

Each stage has one property, and together they make the map commute with
rotations. With `model.equivariant.enable` (and `use_coordinates: false`) the
backbone's multipliers depend only on |G| with real coefficients, so the latent
field and ρ̃ rotate as scalars; without it the backbone, and so the whole model,
is not equivariant. The readout band is a sphere and the resampling grid lives
in the cell's fractional coordinates, so each L block of projections transforms
by the real Wigner matrix D^L (1e-15), as does the neighbour expansion. The
head never mixes M or different L: its linear maps act on the channel index
only, L > 0 maps carry no bias, and gates and the MLP see only invariants. The
target transform centres only L = 0 and scales per block, shared over M. End
to end, a quarter turn of a cubic cell rotates the predicted occupancies by D^L
to 7e-15 in float64.

In the vocabulary of the request, this is the decoder half of a
geometry-informed GNO (the GINO pattern), with atoms as the query points. It is
not a textbook GNO: the kernel is a fixed equivariant basis whose combination is
learned, rather than a free MLP κ_θ(x, y), and that restriction is what makes
equivariance exact and the quadrature grid-independent; it is applied once;
and nothing learned passes messages between atoms.

### Wiring

- **Training.** The same `train()` loop. The log gains a `val occ` column
  (relative RMS of the held-out occupancies), and the best epoch is chosen on
  `val rel L2 + occupancy_weight × val occ`. `task.type: all` with
  `paw.enable` trains all three operators into one checkpoint.
- **Inference.** `FieldOperator.predict_occupancies(potential)` returns
  `[set][atom]` records. `poraque-inference --paw-source` gained `model`, and
  `auto` now tries reference → model → bundle. The JSON summary's record count,
  which was counting lines, was fixed on the way.
- **Refused:** `--kfold` and fine-tuning, before the cache is built
  (`validate_paw_settings`: the occupancy transform is fitted per split, and a
  pretrained bundle has no occupancy head to adapt); and f-channel elements,
  which need solid harmonics to L = 6, when the operator is built rather than
  truncated silently.

### First result on the Pt cells

`data/cache/res48_potcar_spin` was built for it: 97 cells, both record sets,
627 MiB. One run, **one seed**: equivariant backbone (width 32, modes 8, four
layers, `n_radial` 16), band `auto` = 6.65 Å⁻¹, 60 epochs on the CPU in 41 min,
19 structures held out.

| | held out |
| --- | --- |
| occupancy relative RMS | **5.2 %** (training fit 4.7 %) |
| element-mean record, same split | 28.2 % |
| pseudo-density relative L² | 0.029 |
| bulk / slab / nanoparticle, median (worst) | 2.2 % (4.3 %) / 4.1 % (5.9 %) / 8.7 % (13.5 %) |

What limits trust in that number: occupancy error was still falling at epoch 60
(12.3 → 9.8 → 7.0 → 5.9 → 5.2 %); the field error spiked twice, at epochs 10
and 35 (0.05 → 1.62 at the first), and recovered, so the learning rate is not
yet right for long runs; early stopping was off, so the model is epoch 60 (epoch
55 scored 5.19 %); and the experiment's 2.1 % used resolution 64, a 9 Å⁻¹ band
and the DFT density, so the gap between the two numbers mixes three changes.
The checkpoint stayed in the session scratchpad, not under `models/`.

**Files:** `src/poraque/ml/paw.py` (new), `src/poraque/ml/fno.py`,
`src/poraque/ml/training.py`, `src/poraque/fields/vasp/potcar.py`,
`scripts/poraque_train.py`, `scripts/poraque_inference.py`,
`configs/train_ext2paw.yaml`; tests in `tests/test_paw_operator.py` (26).

---

## 5. Run diagnostics (`v26.9.14`)

### The resource profile

`poraque-train` now ends with a table of what the run cost:

```text
RESOURCE PROFILING SUMMARY
==============================================================================
  stage                               wall time     peak RSS
  ----------------------------------------------------------
  cache                                   0.0 s    222.6 MiB
  ext2chg: setup                          0.0 s    228.4 MiB
  ext2chg: training                       0.4 s    319.7 MiB
  ext2chg: evaluation and figures         1.6 s    500.5 MiB
  ...
  checkpoint                              0.0 s    552.8 MiB
  ----------------------------------------------------------
  total (wall clock)                      3.1 s    552.8 MiB
```

`ResourceProfile` (`src/poraque/ml/profiling.py`) is created by `run()` and
passed to `run_task` and `run_task_kfold`, which mark stages imperatively:
`begin` closes whatever stage is open, synchronises the device and resets the
CUDA peak. The stages are `cache`, then per task `setup`, `training`,
`evaluation and figures`, `symbolic distillation` (when enabled) and
`PDF report` (or one `k-fold cross-validation` stage under `--kfold`), then
`checkpoint`.

The memory columns were chosen for what an allocation is sized from. **Peak
RSS** comes from `resource.getrusage`, not psutil: it is the operating system's
own high-water mark, so a spike between two samples cannot be missed, it costs
no dependency (psutil is not installed), and it is a running maximum that never
falls. Its unit differs by platform (bytes on macOS, KiB on Linux), which a
test pins with an allocation of known size. DataLoader workers are reported
from `RUSAGE_CHILDREN` on their own line. **Peak VRAM** on CUDA is
`max_memory_allocated` within each stage, with the most the caching allocator
reserved, the number `nvidia-smi` shows, in the notes. On **MPS** only
`driver_allocated_memory` at the stage's end exists, so that column is a floor.

The table prints from `run()`'s `finally`, after everything else: `--cache-only`
gets one, and a run that raises reports every finished stage and marks the one
it died in `(interrupted)`. A Slurm time-limit kill does not: it is `SIGTERM`,
which Python does not turn into an exception, so no summary is printed. The
same record is `resources` in the metrics JSON.

### Parity over every structure

The parity figure used to draw the first training structure beside the first
validation one. It now covers every structure of each split, training and
validation side by side, each panel titled with its structure and voxel counts
(one panel, every structure as its fold held it out, under `--kfold`). Holding
the voxels was never an option — 97 cells at 48³ is eleven million pairs, and a
Materials Project set is orders of magnitude more — so the figure is streamed.

`ParityAccumulator` (`src/poraque/vis/parity.py`) takes 10 000 voxels from each
structure without replacement (all of a smaller grid), seeded by
`(training.seed, structure index)` so a rerun draws the same voxels, and folds
them at once into sparse fine histograms: int64-keyed bins 0.005 decades wide
on the log table, and the first structure's span over 2000 on the linear one.
Memory follows occupied bins, not structures: in a test, ten times as many
structures from one distribution occupy fewer than three times the bins, and no
voxel is kept. The 200 plotted
bins are summed from the fine ones with `searchsorted` at the end.
`TrainingReport.global_parity` draws them through the same `_draw_parity` code
as the single-structure figure, so the two cannot drift apart.

Relative L², R², MAE and RMSE come from running sums, so they are exact for the
sampled voxels (equal to NumPy on the concatenated sample to 12 digits). They
pool the split, weighting each structure by its share of the sample, and are
therefore not the report table's mean of per-structure errors. Log axes fall
back to linear when under 1 % of sampled voxels are positive in both reference
and prediction; on log axes the CSV `count` excludes the non-positive voxels
the figure cannot draw, which for an undertrained model can be many.

### Charge and magnetisation in the training log

The request asked for a parser of the second `CHGCAR` block, a
`model.spin_polarized.enable` switch and a two-channel `ext2chg`. The first and
third already existed: `SpinDensity` reads the magnetisation block, and
`data.spin: auto|true|false` settles the channel count dataset-wide, from which
the operator's `out_channels` follows. A second switch under `model` could only
disagree with the data, so none was added; `data.spin` was instead documented in
the Manual's `data` table, where it had been missing. What was new is the log:

```text
          epoch     train loss     charge        mag     val rel L2
            3/6        1.23195    0.78913    0.94600        1.17954  *
            6/6        1.16123    0.73227    0.90121        1.11899  *
```

`charge` is `data_error` with the prediction substituted into the density
channel alone, the magnetisation left at its reference, and `mag` the reverse.
Both keep the objective's norm **and its denominator**, the norm of the whole
(ρ, m) target. That is forced: a relative error of m alone divides by m, which
is identically zero on every non-magnetic cell, and so on the whole Pt set. For
the L² objectives the parts add in quadrature, data = √(charge² + mag²) per
sample, exactly (on the synthetic spin run above, √(0.73227² + 0.90121²) =
1.16119 against a train loss of 1.16123). An H¹ objective adds its gradient
term instead, so there the columns rank the channels without summing, and the
legend says which case applies. The parts are computed under `no_grad`, never
enter the gradient, are all-reduced under DDP like the loss, and are recorded
as `history["train_loss_charge"]` and `["train_loss_magnetisation"]` only for a
two-channel `CHGCAR` target. A spin run also draws
`<task>_parity_magnetisation`, on linear axes because m changes sign.

**Files:** `src/poraque/ml/profiling.py` (new), `src/poraque/vis/parity.py`
(new), `src/poraque/vis/report.py`, `src/poraque/vis/pdf_report.py`,
`src/poraque/ml/training.py`, `scripts/poraque_train.py`; tests in
`tests/test_run_diagnostics.py` (21), which drive `run()` end to end on
synthetic cells for the summary (normal, `--cache-only`, failing) and for a
parity figure fed every structure of both splits.

---

## Documentation touched

The Manual (`configuration.md`: `model.paw`, `data.spin`, the resource profile,
the parity sidecar; `ml/index.md`: reporting; `data/index.md`,
`quick_start/index.md`, `fine_tuning/index.md`, `api/index.rst`), the user guide
(`03_training`, `04_inference`, `05b_fine_tuning`, `06_configuration`, rebuilt),
`configs/train_complete_and_commented.yaml`, CLAUDE.md and FUTURE.md §4.8.

## Open

- **Every existing model must be retrained** to load in the three-key checkpoint.
- **The core density is carried but unused.** All-electron reconstruction also
  needs the POTCAR partial waves (`ae wavefunction`, `pseudo wavefunction`),
  which are not parsed.
- **`ext2paw` at inference** predicts the records, but the written `CHGCAR`
  still takes its grid from `ext2chg`; the `ext2paw` density head trains and is
  not used.
- **The 5.2 % is one seed of an unconverged run.** Needed: a second seed, a
  learning rate that does not spike the field loss, early stopping on, and a
  resolution-64 run to compare with the experiment's 2.1 %.
- **Predicted versus reference density.** How much the occupancy error grows
  when ρ̃ comes from a trained `ext2chg` rather than DFT is unmeasured.
- **The head is limited for L > 0.** It has no products between different-L
  features, so an L > 0 output is linear in L > 0 inputs scaled by invariant
  gates; body-order > 2 equivariant layers are the next step if L > 0 stays
  short. Inversion equivariance holds by construction but is untested.
- **`ext2paw` refusals** to lift: `--kfold`, fine-tuning, f-channel elements.
- **A job killed by `SIGTERM`** leaves no profiling summary.
