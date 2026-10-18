# Parallelizing the cache build: OpenMP, MPI, or neither?

```bash
python experiments/cache_parallelism/profile_cache.py --profile --materials 4
python experiments/cache_parallelism/profile_cache.py --fixes  --materials 97
python experiments/cache_parallelism/profile_cache.py --scaling --materials 16
```

`--fixes` carries the pre-fix implementations in the script itself
(`legacy_read_block`, `legacy_grid_from_file`) and measures the shipped code
against them. Both fixes have landed, so without that it would benchmark the
new reader against itself and print 1.0x.

The answer is **processes over materials**, and it is implemented:
`runtime.workers` (`--cache-workers`; `data.cache_workers` before the schema was rewritten on 2026-10-13), default `auto`. On the 97-cell
platinum set the `--cache-only` cache stage went from **10m 57s to 4m 06s**
(2.67x from the pool, 1.54x more from two serial fixes that came with it,
4.0x against the original).

The question arrived naming OpenMP and MPI, so the first measurement was which
of them had anything to act on. Neither does, and the profile is why.

## Where the clock actually is

`scripts/slurm/poraque_cache.sbatch` used to request eight cores, export
`OMP_NUM_THREADS=8`, and explain itself: *"The spectral downsampling is FFT
work; let NumPy's BLAS/FFT use the cores this task was given."* Measured per
material on the platinum set at `resolution: 48`, spin on, **on the code
as it was before any of this landed** (`--profile` reports the build as it
is now, whose `grid header` and text-read rows are smaller):

| stage | s/material | share |
|---|---|---|
| write CHGCAR (text) | 0.614 | 22.5 % |
| read CHGCAR (text) | 0.469 | 17.2 % |
| read TAUCAR (text) | 0.430 | 15.7 % |
| write EXTCAR (text) | 0.307 | 11.2 % |
| write TAUCAR (text) | 0.306 | 11.2 % |
| compute EXTCAR (FFT + POTCAR tables) | 0.226 | 8.3 % |
| `FieldGrid.from_file` (a header read) | 0.217 | 8.0 % |
| augmentation rescan | 0.101 | 3.7 % |
| **spectral downsampling (the FFT)** | **0.060** | **2.2 %** |
| total | 2.730 | |

86 % is encoding and decoding Fortran `E17.11` text in Python, one value or
one line at a time: `fortran_exponential` once per voxel written (110 592 at
48³, 442 368 per material over four blocks), `np.array(line.split())` once per
line read (252 k lines for a 108³ density, 4.8 M for a 288³ one).

**`OMP_NUM_THREADS` had nothing to act on.** There is no BLAS call in the cache
path, and every transform in the package goes through `np.fft`, which is
pocketfft, which is single-threaded. Its whole addressable share was the 2.2 %
line, and it could not even reach that. Seven of the eight requested cores
idled for the length of the job. The sbatch now sets `OMP_NUM_THREADS=1`
deliberately, so the worker processes do not each think they own the node.

**MPI was the right model for the wrong axis.** Nothing inside one material
needs a communicator, and on a single node a pool gets the same cores without
a launcher. Across nodes it earns its keep; see the last section.

I/O was never the constraint: a 120 MB `CHGCAR` reads in 0.036 s (3.3 GB/s)
and parses in 1.7 s, a factor of 50. The build consumes **61.2 GiB** of text to
write **466 MiB** of cache. What it is doing is decoding an archive, and that
is per-material independent work.

## What was built

`runtime.workers: auto | <n> | 0`, in `build_field_cache`.

- **The pool is over materials.** `_build_one` already wrote every field to
  `.partial-<pid>` and moved it with `os.replace` — discipline bought by a
  regression at LNCC, where two jobs sharing a cold cache interleaved inside
  one `write` and left a zero-length density that *parsed*. The pool inherits
  it, so no lock is needed. A cache built by three workers is **byte-identical**
  to the serial one; `tests/test_cache_parallel.py` pinned that (until it was
  removed on 2026-10-13) for a text cache and value-for-value for HDF5 (whose store stamps a `created` timestamp, so
  comparing its bytes would pin the clock, not the cache).
- **Longest-processing-time-first.** Materials are handed out largest first
  with `chunksize=1`. The native grids of one dataset span 1.26 M to 23.89 M
  voxels and the cost is linear in them — least squares over all 97 rows of the
  build log gives `t = 0.469 + 1.033 s` per million voxels, R² = 0.9963,
  largest residual 1.55 s — a factor of 19 between cheapest and dearest. In
  discovery order the eleven 288³ nanoparticles sit at the end of the list and
  become a 25-second tail with idle cores behind it.
- **Rows print by completion** (`imap_unordered`), `i/N` counting materials
  finished. The table exists to show progress, and ordered delivery would hold
  a finished 88 MiB row behind an unfinished 1.6 GiB one.
- **The bookkeeping stayed in the parent.** `cache_summary.json` and
  `cache_fingerprint.json` are plain `open(..., "w")` writes; the parent also
  owns the progress table and the `finally` that saves the summary when a build
  aborts. No on-disk invariant moved into the pool.
- **The spin probe takes the same pool.** `_resolve_spin` asks every density in
  the set whether it carries a magnetization block, before the first field is
  written: **29.2 s on 97 materials**, 11 % of the serial build, and under
  Amdahl the term that would have capped the whole exercise at 3.2x however
  many cores were added.
- **A failing material no longer loses the archive.** A worker returns its
  exception instead of killing the pool; the parent writes the summary for
  everything that succeeded and then raises naming what did not, so a rerun
  resumes.

### Two blockers that had to be fixed first

**A `MaterialRecord` could not be pickled.** It carries its source, the source
carries its options dict, and `build_field_cache` put the build's `log`
callable in those options — so a record held a live closure over an open log
file and `pickle.dumps(record)` raised `PicklingError`. `fork` hides this;
macOS defaults to `spawn`, and `srun` has no fork either. Fixed by
`shippable_options` (strip `log`) plus sending the three strings a record is
made of and rebuilding the source in a pool `initializer` — one directory probe
per *process* instead of per material. `spawn` is used deliberately rather than
`fork`: NumPy's BLAS starts threads, and forking a threaded process is
undefined where those threads hold locks.

**`CalculationSource.charges` was lazy and fanned out.** Its fallback,
`infer_valence_charges`, parses *every density in the set* to solve
∫ρ = Σ nZ. Serially that happens once; in a pool each worker would have done it
independently — 61 GiB per worker on this set before a single field was
written. `_resolve_charges_before_forking` settles it in the parent, and only
for datasets that actually take the Gaussian route (an `os.path.exists` per
material plus the per-element answer `potcar_coverage` has already memoized).

### Two serial redundancies removed, because they compound with every worker

**`FieldGrid.from_file` parsed the whole density to read a header.** It went
through `read_volumetric`, converted 1.26 M values to float (23.89 M on a
nanoparticle), reshaped them, and returned `data.shape` — for a number the
`NGXF NGYF NGZF` line carries. New `read_grid_header` stops at that line:
**0.214 s → 0.00006 s**, identical grid, 8 % of the build.

**`_read_block` called `np.array` once per line.** A `CHGCAR` line holds five
values, so NumPy's array-construction overhead was paid once per five numbers.
One `split()` per ~1 M values instead: **1.8–2.0x**, values bit-identical. The
whole block at once is faster still and is deliberately not done — a 288³
density is 23.9 M values and the list of Python strings `split()` returns for
it costs ~1.4 GB, which is the figure that decides how many workers fit in a
node.

Together these two took one 288³ material from **25.5 s to 15.4 s** before any
parallelism (15.9 s measured standalone in a fresh process; 15.4 s is the row
in the full serial build, where the interpreter is already warm).

## Measured

Three runs of the same config over the same 97 materials, text storage,
`spin: false`, `resolution: 48`, from `poraque-train --cache-only`:

| | sum of per-material times | cache stage, wall | peak RSS |
|---|---|---|---|
| original, serial | 961.8 s¹ | — ¹ | — |
| serial + the two fixes | 623.7 s | **656.9 s** | 2.3 GiB |
| 4 workers + the two fixes | 928.4 s² | **246.1 s** | 568 MiB (parent)³ |

¹ 91 rows measured; the run was stopped at 91/97 and the remaining six 280³
cells are carried at their measured 22.2 s. No stage wall for it.
² Each material's *own* elapsed grows under contention — that is what the
third column divides by four. ³ The parent no longer touches a field; the
workers do, up to ~1.6 GiB each, so the aggregate is higher than the serial
run's 2.3 GiB. That is the trade `auto`'s memory budget exists to bound.

So, decomposed:

- **the two serial fixes: 1.54x** (961.8 → 623.7 s of per-material work). On a
  single 288³ material, 25.5 → 15.4 s.
- **four workers on top: 2.67x** (656.9 → 246.1 s wall). Not 4x: per-material
  time rises from 623.7 s to 928.4 s of aggregate work under contention, which
  is 1.49x per material — memory bandwidth on the 288³ grids, not scheduling
  (worker occupancy is 928.4/(246.1 × 4) = 0.94).
- **together, 4.0x**, and the parallel run was competing with the full test
  suite for part of its length, so it is a floor.

Pool scaling in isolation, 16 materials of 108³, `spawn`, `chunksize=1`, on
this 8-core machine (four performance + four efficiency cores):

| workers | s/material (text) | speedup | efficiency | s/material (hdf5) | speedup |
|---|---|---|---|---|---|
| 1 | 2.764 | 1.00x | 1.00 | 1.515 | 1.00x |
| 2 | 1.462 | 1.89x | 0.97 | 0.832 | 1.82x |
| 4 | 0.868 | 3.18x | 0.94 | 0.518 | 2.92x |
| 8 | 0.684 | 4.04x | 0.91 | 0.440 | 3.44x |

4.04x at 8 workers is the asymmetric cores, not contention: aggregate busy time
rises from 44.2 s to 79.5 s as the efficiency cores take work while per-worker
efficiency holds at 0.91. On a homogeneous node expect close to linear.

`data.cache.storage: hdf5` composes with all of it and removes the text writer
outright: 2.730 → 1.627 s/material serially, 0.440 with eight workers.

### How `auto` sizes itself, and why memory binds first

Three numbers, smallest wins: the cores this process **may** use
(`sched_getaffinity`, which is what Slurm's `--cpus-per-task` restricts a task
to — the node's core count is the wrong number on a shared node), a memory
budget, and the material count.

The memory model is measured, not assumed. Building one material in a process
of its own and reading `ru_maxrss`: a 108³ density (1.26 M voxels) peaked at
**498 MiB**, a 288³ one (23.89 M voxels) at **1558 MiB** — a slope of 47 B per
voxel over a ~200 MiB fixed cost. Carried at 400 MiB + 70 B/voxel, which
reproduces both from the top side. On this set:

```
workers: 4 (auto: 8 usable core(s); 4 within the 8 GiB budget at ~1995 MiB
each, sized on the largest grid at 23.89 M voxels; 97 material(s))
```

Sized on the **largest** grid, not the average: every worker may be handed the
nanoparticle, and sizing on the mean is how a build goes into swap on the last
eleven materials. The voxel count comes from the headers via `read_grid_header`
(18 ms for all 97), not from file size over eighteen columns — a
spin-polarized `CHGCAR` holds two grid blocks, so that estimate reads as twice
the grid and would halve the worker count on exactly the datasets that need
them most.

## What is left, and where MPI would come in

**OpenMP** acquires a legitimate home only if the text codec is rewritten in
C — a `_spectral.c`-shaped extension for the `E17.11` reader and writer, worth
at most the 45 % of the build that text writing and reading still is. Before
writing one, note that `data.cache.storage: hdf5` already removes the writer and is a
config key.

**MPI**, for an archive that outgrows one node. `ml/distributed.py` cannot be
reused: `discover()` deliberately returns a disabled context without CUDA —
*"NCCL is a CUDA library. Without CUDA there is nothing to distribute over, and
falling back to Gloo would let a misconfigured job train distributed across CPU
cores at a fraction of one GPU's speed"* — and a cache job is CPU-only by
construction, since `--cache-only` forces `device: cpu` and `distributed: off`.
So a multi-rank cache build needs either a two-line read of `SLURM_PROCID` and
`SLURM_NTASKS`, or `mpi4py` (present in the environment at 4.1.2, absent from
`pyproject.toml`).

No communicator is required. Each rank takes a stride of the size-sorted record
list, the atomic writes keep them from tearing, and `_all_present` makes the
work idempotent. Three things need handling, and all three are the bookkeeping
the pool deliberately left in the parent:

- `cache_summary.json` — per-rank shards merged at the end, or written by rank
  0 from `_describe_cached`; the current plain write loses updates.
- `cache_fingerprint.json` — every rank must *check* it, one must write it, and
  that write needs the `os.replace` discipline the field writes already have.
- the progress table — `Logger(silent=True)` on non-zero ranks, as under DDP.

The honest sizing: a cache is built once per (dataset, resolution), and four
minutes on a CPU partition costs no GPU-hours. The case for MPI is not this
dataset; it is a Materials Project archive of tens of thousands of densities,
where four minutes becomes hours and the 6-hour `#SBATCH --time` is what
breaks. The hybrid — `srun --ntasks=<nodes>` with a pool inside each — is the
shape that fits, and it is what exists now plus a stride.

**The `poraque-mp` download is a separate question with a separate answer:**
threads, not processes, because it is network-bound — `--workers`,
default 1 for downloads because the bound is the Materials Project's per-key rate limit and
that is unmeasured. See the Manual's data page.

**Still three scans of the same CHGCAR per material.** Fixing
`FieldGrid.from_file` removed one; `source.read` and `_augmentation_for` →
`read_augmentation_blocks` (0.101 s, 3.7 %) remain. `read_volumetric`'s third
return value is `extra` — further *grid* blocks, not augmentation records — so
merging them needs a reader that returns both, not a different call.
