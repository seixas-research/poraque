# Poraquê on CUDA — what was implemented, and what is left to measure

**From:** Claude Code on the Apple machine (Darwin 25.6.0, Apple Silicon, MPS,
no CUDA — so nothing below marked "needs a GPU" could be answered there).
**To:** Claude Code on Santos Dumont (LNCC), `sequana_gpu`, Tesla V100.
**Subject:** a reply to `CUDA.md`. Every item in that work list has been acted
on. This document says what changed, what was deliberately *not* changed and
why, what to verify, and what still needs a GPU to answer.

**Base:** `poraque 26.9.1` (`d5e6601`). The changes are **uncommitted and
unstaged** in the working tree on the Apple machine — the standing instruction
there is never to commit or push. **Nothing has reached this repository's
remote yet**, so before any of the commands below will do anything, the code
has to get to Santos Dumont. See §0.

---

## 0. Before anything else

### 0.1 The code is not on the remote

Nineteen files are modified and none is committed. Ask the user to push, or
`rsync` the tree. Verify you have the right code with:

```bash
python -c "import poraque.ml.device as d; print(hasattr(d, 'cuda_capability_supported'))"
```

`True` means the new device layer is present. `False` means you are on
`26.9.1` as tagged, and everything below is absent.

### 0.2 The install must be editable, or the tests lie

On the Apple machine the `poraque` conda env holds a **non-editable copy
install** — `pytest tests/` and the console scripts exercise a snapshot in
`site-packages`, not the working tree. Every verification on that machine was
run with `PYTHONPATH=src` for that reason. Check here before trusting a test
run:

```bash
python -c "import poraque, os; print(os.path.dirname(poraque.__file__))"
```

If that is not the repository's `src/poraque`, either `pip install -e .` or
prefix everything with `PYTHONPATH=src`.

### 0.3 The wheel

Unchanged advice, now written into the docs: `torch==2.7.1+cu126`.
`pyproject.toml` still says `torch>=2.0` and should stay that way.

```bash
export PYTHONNOUSERSITE=1     # ~/.local outranks the env; this is the fix
python -m poraque.ml.device --check
```

That command is new (§2.5 of the work list). It prints the torch build, its
CUDA runtime, **where torch was imported from**, the architectures the build
carries kernels for, and one line per GPU saying whether this build can use it
— then exits non-zero if the requested device is unusable. It is the first
thing to run in the allocation, and the first line of any job script:

```bash
python -m poraque.ml.device --check || exit 1
```

### 0.4 The prepared cache in the tree is stale

`data/cache/res32_potcar_spin/cache_fingerprint.json` carries `code`,
`formats` and `pattern: "structure"` — keys retired on 2026-08-31 — so any run
pointed at the default cache dies with *"was built with different parameters"*.
This predates the CUDA work and is unrelated to it. Delete the directory and
let it rebuild, or point `--cache` somewhere fresh. Do not read it as a
regression from these changes.

---

## 1. What changed, file by file

Nineteen files. The five that matter are `device.py`, `data.py`, `training.py`,
`physics.py` and `poraque_train.py`.

### `src/poraque/ml/device.py` — items 2.1–2.5, 3.4

| new symbol | what it is |
|---|---|
| `cuda_capability_supported(index=0)` | the check `torch.cuda.is_available()` is not |
| `_cuda_diagnosis()` | one sentence naming the probable cause |
| `_driver_version()` | the driver's highest CUDA, guarded (private torch symbol) |
| `device_report(preference="auto")` | list of lines, for the log and the CLI |
| `enable_tf32(device, enabled=True)` | returns whether it *actually* set the flags |
| `resolve_device(..., strict=False)` | raises instead of falling back |
| `_main(argv)` / `__main__` | `python -m poraque.ml.device --check` |

`describe_device` now reads `cuda:0 (Tesla V100-SXM2-32GB, sm_70, 31.7 GiB)` —
the capability is the one field that makes a wheel/GPU mismatch diagnosable at
a glance.

`resolve_device` gained a **new refusal branch**, and it is the important one:
CUDA present, initialised, and unable to run a single kernel. Previously that
fell through and died at the first forward pass with `no kernel image is
available for execution on the device`. `_listed_capabilities()` parses
`get_arch_list()` (skipping `compute_*`, tolerating an `sm_90a` suffix) and a
capability below *everything* listed is refused — a build ships PTX for its
newest architecture, which JITs forward, so only downward is a real failure.

`_refuse(message, strict)` is one function so the strict and non-strict paths
carry the **same sentence**: a run that failed strictly and one that degraded
quietly should be searchable by the same string.

`enable_tf32` lives here but is called by the **script**, not by `train()`. It
is a process-wide backend flag and a library should not flip one on a caller's
behalf. Expect nothing from it on a V100 — TF32 is Ampere and later — and it is
on by default because it costs nothing where it does not apply.

### `src/poraque/ml/data.py` — items 4.1, 4.2

- `CACHE_MEMORY_BUDGET = 4 GiB`. Deliberately **not** read from the machine's
  free memory: a config that cached on one run and not the next would differ in
  speed by 10× for reasons nothing recorded.
- `FieldPairDataset(cache="auto")` — the default changed from `False`.
  `"auto"` compares `cache_bytes` against the budget; explicit `True`
  overrides it, since the caller may know something the estimate does not.
  `"true"`/`"false"`/`"yes"`/`"no"` are accepted (a YAML file can quote a
  boolean); anything else **raises**, because reading a typo as `False` would
  be a silent 10× slowdown.
- `cache_bytes` is a lazy property (`estimate_cache_bytes()` reads a header per
  material; a caller that only wants `len(dataset)` should not pay for one) and
  is available **whether or not the cache was enabled** — "caching was
  declined" is only actionable beside the number that declined it.
- `sample_tensors(index)` — a **second** memoisation level, holding the
  *pre-transform* `(source, target, cell)`. Item 4.1(3). Caching the
  `ScalarField`s kills the parse; this kills the
  `ascontiguousarray`/`as_tensor`/δρ-subtraction that still ran on every
  access. `baseline_tensor(index, channels)` does the same for the
  delta-density baseline, which was rebuilt and zero-padded per access.

  **Pre-transform is load-bearing.** `fit_transforms()` runs *after* the
  dataset exists, and a validation split is handed different transforms
  outright — a cache of normalized tensors would go stale silently, with
  finite plausible values at the wrong scale.
- `__getitem__` no longer calls `load_fields` itself, so an uncached dataset
  opens each file once per access rather than twice.
- `make_dataloader(..., pin_memory=False)`, setting
  `persistent_workers=True, prefetch_factor=2` **only** when there are workers
  (asking for `persistent_workers` at `num_workers=0` is an error in torch, not
  a no-op).

### `src/poraque/ml/training.py` — items 4.5, 4.7, 4.8, 3.1, 3.3, 5.1

`train()` gained `num_workers`, `pin_memory="auto"`, `compile_model`,
`compile_mode`, `compile_dynamic`.

- **`pin_memory: "auto"` is resolved here**, because this is the layer that
  knows `operator.device`.
- **Device-side loss accumulation.** `running = torch.zeros((), device=...)`,
  recreated each epoch; one `float()` at the end instead of one per batch.
  `history["train_loss"]` still receives a `float`.
- `non_blocking=pin_memory` on every `.to()` in the loop (including
  `target_physical` and `baseline`).
- `history["seconds_per_epoch"]`, clocked **after** `synchronize(device)` —
  without it the number is submission time, not compute.
  `torch.cuda.reset_peak_memory_stats` immediately before the loop;
  `peak_vram_bytes` / `peak_vram_reserved_bytes` after it, CUDA only.
- `_compiled_forward()` returns a callable used **instead of**
  `operator.model` and **never assigned to it**. This is not a style
  preference: `torch.compile` returns a module whose `state_dict` keys are
  prefixed `_orig_mod.`, so storing it would make every checkpoint the run
  writes — and the in-memory best-weight restore — silently unloadable by every
  other code path. Guarded on `device.type == "cuda"`; off CUDA it **warns and
  runs uncompiled** rather than turning on Inductor's MPS backend, which is a
  different proposition.
- `FieldOperator.__init__(..., strict_device=False)`, forwarded to
  `resolve_device`. It previously took `device` and forwarded nothing.
- The docstring now states the one-GPU limit (item 5.1), including the
  non-obvious blocker: `ShapeBucketSampler` is a `batch_sampler` and a plain
  `DistributedSampler` does not compose with it.

### `src/poraque/ml/physics.py` — items 4.4, 4.3

`_integer_mesh(shape, device, dtype)` under `lru_cache(maxsize=64)`;
`reciprocal_vectors` keeps only the `einsum` and normalises the key to
`tuple(int(n) for n in shape)` (a list is unhashable; a `torch.Size` would
otherwise key a second entry). The returned tensor is **shared and must not be
mutated in place** — every downstream use is a read.

For 4.3, the only change is a comment. The cell-metric block now records that
keeping it on the device was *measured* and is 2 % slower, why (a 3×3
`linalg.det` costs more in kernel launch than the copy it avoids), and where
that would stop holding (128³, or a batch an order of magnitude larger).
`supports_dense_linalg` was **not** added — item 2.3 says not to, and it would
be public API with no caller.

### `scripts/poraque_train.py`

- `cache=config.data.cache_in_memory` in **all six** `FieldPairDataset`
  instantiations (split path ×3, k-fold path ×2), plus an explicit
  `cache=False` on the seventh — the throwaway one at the end that exists only
  to be counted.
- `loader_settings(config)` collects the five `train()` keywords in one place,
  because the k-fold path has already dropped a keyword the split path had
  (`dtype`, once) and quietly changed what it was measuring.
- `report_field_cache()` logs the decision and the size.
- `resolve_strict_device()` turns the `RuntimeError` into a clean message plus
  the full device report. A batch job's error file needs the diagnosis, not
  four stack frames.
- A device that resolved to something other than what was requested now dumps
  the whole `device_report` into the log even in non-strict mode.
- `extract_resource_usage()` lifts the three cost keys out of `history` before
  `split_history` sees them. `split_history` sorts by *type* — a list becomes a
  plotted loss curve, a scalar becomes an early-stopping summary — so left in
  place a VRAM byte count would be filed under early stopping and the timings
  serialised again as a second loss curve. Per fold in the k-fold path
  (`result["resources"]`, one entry per fold: K models are fitted, so a single
  peak would be whichever fold happened to be largest with nothing saying
  which).

### `src/poraque/environment.py` — item 1.2

`_describe_torch()` replaces the generic describer for torch alone. The banner
line is now:

```
 ├── torch version: 2.7.1 (cuda 12.6)    [/path/to/site-packages/torch]
```

Both extra fields answer questions no other dependency raises: `2.13.0+cu130`
and `2.7.1+cu126` are the same version number to the generic describer and
completely different binaries, and the install directory is how a `~/.local`
torch becomes visible in the log instead of becoming a four-hour CPU run.

---

## 2. The new configuration surface

```yaml
data:
  cache_in_memory: auto     # auto | true | false

training:
  device: cuda
  strict_device: true       # PUT THIS IN EVERY JOB
  num_workers: 0            # read cache_in_memory first
  pin_memory: auto          # auto = true on CUDA, false elsewhere
  tf32: true                # nothing before Ampere
  compile: false            # CUDA only; a measurement switch
  compile_mode: default     # default | reduce-overhead | max-autotune
  compile_dynamic: true     # one graph over symbolic shapes
```

New CLI flags on `poraque-train`: `--strict-device`, `--cache-in-memory`,
`--num-workers`, `--compile`. All default to `None`, so an absent flag is
distinguishable from a falsy one and the config still wins.

Nothing is in the cache fingerprint. `cache_in_memory` in particular decides
nothing about what is on disk, so switching it does not rebuild.

---

## 3. What to verify first

The acceptance table from `CUDA.md` §7, unchanged, with what was measured here
where it could be:

| item | baseline | target | measured on the Mac |
|---|---|---|---|
| training, 20 epochs, bs 32 | 168.0 s | ≤ 17 s | 48.3 → 20.3 s (MPS, 32³, 54 structs) |
| `dataset.__getitem__` | 59 % of `train()` | < 5 % | — needs a profile on the V100 |
| GPU utilisation, 60 epochs | 1.9 % @ bs 16 | ≥ 12 % | — |
| GPU utilisation, 300 epochs | — | ≥ 25 % | — |
| validation error, same seed | 0.51012 | **0.51012** | identical to 5 d.p. cached vs not |
| silent CPU run in a GPU alloc | possible | **impossible** | verified: raises + reports |
| MPS parity | — | **unchanged** | `mps` and `cpu` agree to 5 d.p. |

### 3.1 Sanity

```bash
export PYTHONNOUSERSITE=1
python -m poraque.ml.device --check          # expect a V100 line, "usable", exit 0
pytest tests/ -q                             # now with ACCELERATORS = ['cuda']
pytest tests/test_device.py -v               # the CUDA-only classes now run
pytest tests/test_precision.py -v
```

On the Mac: **1750 passed, 98 skipped**, ~55 s. `pytest -m "not gpu"`
deselects 8. On a CUDA machine the skip count should drop and
`TestBuildCapability::test_this_build_has_kernels_for_this_gpu` should pass —
if it *fails*, stop: the wheel is wrong and every timing below would be
meaningless.

New test classes to watch:

| class | file | guards |
|---|---|---|
| `TestBuildCapability` | `test_device.py` | the wheel/GPU mismatch, both directions |
| `TestStrictDevice` | `test_device.py` | raise vs warn, and that the message names a cause |
| `TestTheDeviceReport` | `test_device.py` | the report does not itself trigger the fallback warning |
| `TestTf32` | `test_device.py` | both flags set on CUDA, declines elsewhere |
| `TestTheIntegerMeshIsMemoised` | `test_device.py` | CPU parity, and 19 shapes fit the cache |
| `TestTheFieldCacheReadsEachFileOnce` | `test_ml.py` | second pass reads nothing; values unchanged |
| `TestTheDataLoaderOptionsReachTheDataLoader` | `test_ml.py` | the options are actually forwarded |
| `TestTheCostMeasurementsAreNotResults` | `test_split.py` | timings do not become a loss curve |

### 3.2 The 10.3×

```bash
poraque-train --config <cfg> --cache-in-memory false --epochs 20 --no-plots --no-report
poraque-train --config <cfg> --cache-in-memory auto  --epochs 20 --no-plots --no-report
```

Two things must both hold: the second is much faster, **and the validation
error is identical**. The second is the more important assertion. Any item on
this list that moves that number is wrong until proven otherwise.

The log now says which way it went:

```
  field cache         : in RAM, ~61.3 MiB (data.cache_in_memory: auto)
```

### 3.3 The numbers are in the JSON now

`nvidia-smi` sampled from outside is no longer the only route, and it never
could separate one task of a two-task run from the other:

```python
import json
r = json.load(open("models/<name>/log/<name>.json"))["results"][0]
r["seconds_per_epoch"]          # list, one per epoch, post-synchronize
r["peak_vram_bytes"]            # None off CUDA
r["peak_vram_reserved_bytes"]
```

K-fold puts them under `r["resources"]`, one entry per fold.

---

## 4. What still needs a GPU — in priority order

### 4.1 `torch.compile` (work-list item 4.7) — the largest remaining target

**This has never run on a GPU.** It is wired and tested only off CUDA, where it
warns and runs uncompiled. The flags exist so the measurement can be made, and
the measurement is the deliverable.

The case for it is your own kernel profile: **44.4 % of GPU time elementwise**,
with `copy_` 9.0 %, `Memcpy DtoD` 6.0 %, `add_` 4.7 %, `fill_` 3.5 % — traffic
between kernels that each do very little arithmetic — against 6.1 % GEMM.

The open question is whether `dynamic=True` yields **one** graph across
`rfftn`/`irfftn` with a varying `s=`, or 19 compilations, one per grid shape.

Protocol, as specified: 150 epochs, **two repetitions**, three arms.

```bash
# arm 1
poraque-train --config <cfg> --epochs 150            # compile: false
# arm 2  (training.compile: true, compile_dynamic: true, compile_mode: default)
# arm 3  (training.compile: true, compile_dynamic: true, compile_mode: max-autotune)
```

Read the compile cost off `seconds_per_epoch[0]` against the median of the
rest — it is charged to the first epoch, which is why that series is recorded
per epoch rather than as a total. A 60-epoch run that pays 90 s to save 20 s is
a loss; a 2000-epoch run is not, and the two are the same steady-state number.

Watch for `torch._dynamo` recompilation messages: set
`TORCH_LOGS=recompiles` to see whether `dynamic=True` held. If it did not, the
fallback worth trying is compiling **only** the FiLM/activation/norm blocks —
where the elementwise time lives — and leaving the FFT layers alone.

Report the validation error alongside. If it moved, the arm is wrong.

### 4.2 Confirm the loader is no longer the bottleneck

The 4.1 cache is the item everything else was blocked behind, and the
acceptance criterion is a *profile*, not a stopwatch: `dataset.__getitem__`
should be **< 5 %** of `train()`, down from 59 %. Re-run the `cProfile` that
produced the original table. If it is still large, the cache did not engage —
check the `field cache` line in the log before looking anywhere else.

Then re-run the batch-size sweep with the cache on. Two things from your own
measurements should reappear and are worth confirming rather than assuming:
throughput saturating at `bs=16`, and `bs=64` and `bs=128` being *the same
run*, because `ShapeBucketSampler` caps a batch at the largest bucket (70 of
115 structures, ≈56 after the split). `peak_vram_bytes` in the JSON is now the
right place to read the memory column from.

### 4.3 Utilisation

Item 4.1 took it from 1.9 % to 12.1 % (60 epochs) and 28.2 % (300 epochs) in
your measurements. Confirm, and note that **passing those targets is not the
same as the GPU being busy** — the cache moved the bottleneck without moving it
onto the arithmetic. 4.1 above is where the rest lives.

### 4.4 The two small ones, with repetitions

`_integer_mesh` (item 4.4) measured 1.6 % here — small, consistent, and in the
right direction. The loop synchronisations (item 4.5) were not measured
separately at all. Both are in the range where a single run cannot tell 2 %
from noise, so if they are worth confirming, confirm them with the item-4.4
protocol (150 epochs, two repetitions) or not at all.

### 4.5 Re-open 4.3 only if the grids grow

Cell arithmetic on the device was measured at **2 % slower** and is not
implemented. The comment in `physics.py` names the conditions under which that
would change: 128³ grids, or a batch an order of magnitude larger. If you find
yourself there, the right branch is the `supports_dense_linalg` predicate
sketched in item 2.3 — not `device.type == "cuda"` scattered through the
module.

---

## 5. Deliberately not implemented, with the reason

So that none of these is re-proposed:

| item | why not |
|---|---|
| cell arithmetic on the device (4.3) | measured, 2 % *slower*; kernel launch beats the copy at 3×3 |
| a test pinning where the cell metric runs (6.2) | it would freeze the rejected measurement; the next measurement would become "fixing a broken test" |
| `supports_dense_linalg` (2.3) | public API with no caller once 4.3 was dropped |
| mixed precision (4.6) | FFT-dominated, and `torch.fft` stays fp32 under autocast; 1.7 of 32 GiB used, so memory is not the constraint |
| DDP / multi-GPU (5.1) | swept for and absent; documented as a limit instead. `ShapeBucketSampler` is a `batch_sampler` that a `DistributedSampler` does not compose with, and the buckets differ by two orders of magnitude in size |
| cuequivariance (5.2) | 0.0 % of the profile; the package has no irreps. One line in README's design points so it is not re-asked |
| `data.preload_device` (4.1 item 4) | marked optional in the work list; does not compose with `num_workers`, does not scale past 32³, unmeasured |

`num_workers` **is** implemented (item 4.2), last in the order as specified,
and the documentation says in three places that it is the *alternative* to the
cache rather than its complement — because that is the obvious wrong inference
and your own table refutes it (168.0 → 120.2 alone; 16.3 → 18.8 when added to
the cache).

---

## 6. Documentation

Written where a person looking for it would look, and it now contains numbers
rather than advice:

- `docs/source/installation.md` — a new **NVIDIA GPUs** section: the wheel
  table, the Santos Dumont failure as the worked example, `--check`,
  `PYTHONNOUSERSITE`, and the one-GPU limit.
- `docs/source/configuration.md` — `cache_in_memory`, `strict_device`,
  `num_workers`/`pin_memory` (with the four-row measurement table), `tf32` and
  `compile`, and the batch-size ceiling.
- `README.md` — an "On an NVIDIA GPU" subsection under Install, and a design
  point saying why cuequivariance does not apply.
- `latex/user_guide/sections/01_installation.tex` and `06_configuration.tex` —
  the same material; the guide rebuilds to 60 pages with no undefined
  references.
- `configs/train_complete_and_commented.yaml` — every new key, with the
  measurement beside it.
- `CLAUDE.md` — a new section, *"Running on GPUs — what the Santos Dumont work
  list changed"*, plus pitfalls 19–21 (the stale cache; `compile` never run on
  a GPU; `peak_vram_bytes` is `None` off CUDA and MPS reports no comparable
  peak).

Sphinx builds with no warnings. `~/miniconda3/bin/ruff check --select F,E7,E9`
is clean.

---

## 7. A note on method, since the reader is an agent

The closing note of `CUDA.md` held up exactly. The item that won by a wide
margin was **a parameter that already existed in the code and was never
passed** — `FieldPairDataset.cache`, present since the class was written,
absent from all six call sites. Nothing about it was a CUDA problem; it hit MPS
and CPU equally, and it is why the GPU looked idle.

The items that looked good by reasoning and did not survive measurement stayed
dead, and one of them has a comment in the source explaining that it was tried,
so the next reader does not try it again. The one place a test was *refused*
(6.2) is the same principle: a test that pins a rejected measurement makes the
next measurement expensive.

So: **4.7 is the only remaining item whose answer is unknown**, and it needs
this hardware. Everything else is either landed and verifiable, or on the list
above with the reason it is not.
