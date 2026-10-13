# Reading the output

Every Poraquê command writes its output the same way. `poraque-train`,
`poraque-inference`, `poraque-atoms`, `poraque-vasp`, `poraque-mp`,
`poraque-committee`, `poraque-active-learning`, `poraque-compile` and the C
backend's report (`python -m poraque.ml.backend`) all report through one writer,
{class}`poraque.utils.logging.Logger`, so a person who has read one log can
find their way around any other, and a script that reads one can read them
all.

## The layout

The format has three parts, and that is the whole of it.

**The banner.** The logo and the version, then three blocks recording what
produced the output: `[SYSTEM]` (architecture, platform, user, host, working
directory, process ID), `[PYTHON]` (version and executable) and
`[DEPENDENCIES]` (each library's version **and the directory it was imported
from** — the line that exposes a `torch` in `~/.local` outranking the
environment a job activated; see [Installation](installation.md)).

**A title**, between two rules of `=`, naming what the command is doing.

**Blocks.** Each opens with a header `[NAME]`, upper case, at column 0, and
its body follows indented by two spaces: `key: value` lines with the colons
aligned, fixed-width tables under a dashed rule, or free text. A block ends
where the next header begins, and a blank line separates them.

An abridged `poraque-train` run:

```text
[SYSTEM]
  architecture: x86_64
  platform:     Linux
  ...

[DEPENDENCIES]
  numpy:      2.4.6    [.../site-packages/numpy]
  torch:      2.13.0+cu130 (cuda 13.0)    [.../site-packages/torch]
  ...

==============================================================================
  Poraque - Fourier Neural Operator training
==============================================================================

[RUN]
  run    : smoke
  device : cpu (12 threads)  (requested 'cpu')
  ranks  : single device (no distributed group)
  ...

[CACHE]
  path: data/cache/res8
  ...

==============================================================================
  TASK ext2pseudo: EXTCAR -> CHGCAR
==============================================================================
  Local external potential -> valence charge density.

[EXT2PSEUDO DATASET]
  field cache         : in RAM, ~36.0 KiB (data.cache.in_memory: auto)
  training structures : 2
  ...

[EXT2PSEUDO TRAINING]
          epoch     train loss     val rel L2
    -----------------------------------------
            1/2        1.34859        1.87486  *
            2/2        1.30728        1.79770  *
```

A block is the unit of meaning — the cache, the model, the training progress,
the results of one split, the resource profile — so finding something in a
long log is a search for its header, and a parser has only to recognize lines
that begin with `[`.

## Reading it from a script

{func}`~poraque.utils.logging.split_blocks` is the reference reader. It returns
`(name, lines)` per block, in order, each body with its indentation removed;
whatever precedes the first header — the logo, a title — is the block named
`""`.

```python
from poraque.utils.logging import split_blocks

blocks = dict(split_blocks(open("models/smoke/log/smoke.log").read()))
blocks["OVERALL"][0]      # 'task          rel L2 (mean)    R2 (mean) ...'
blocks["EXT2PSEUDO TRAINING"][-1]
                          # 'trained 2/2 epochs in 2.4 s   loss 1.3486 -> 1.3073'
```

A title is for the person reading, not for the parser: one that falls between
two blocks, such as a task's, lands in the body of the block before it. Key on
the headers. The machine-readable record of a training run remains its
metrics JSON; the log is what to grep while the run is still going.

## The blocks each command writes

| Command | Blocks, in order of appearance |
| --- | --- |
| `poraque-train` | `RUN`, `CONFIGURATION`, `CACHE`, then per task `<TASK> DATASET`, `<TASK> MODEL`, `<TASK> TRAINING`, `<TASK> RESULTS`, `<TASK> AGGREGATE OVER TRAINING` / `VALIDATION`, `OCCUPANCIES` (`ext2chg`), `<TASK> FILES`; then `CHECKPOINT`, `OVERALL`, `OUTPUT`, `RESOURCES` |
| `poraque-train --kfold K` | per task `<TASK> FOLDS`, `<TASK> FOLD i/K`, `<TASK> K-FOLD SUMMARY` in place of the single fit's |
| `poraque-train --cache-only` | `RUN`, `CONFIGURATION`, `CACHE`, `CACHE SUMMARY`, `RESOURCES` |
| `poraque-inference` | `RUNTIME`, `INPUT`, `GRID`, `POTENTIAL`, `DENSITY`, `PAW RECORDS`, `KINETIC ENERGY DENSITY`, `HARTREE POTENTIAL`, `COMPARISON`, `ENERGY`, `FIGURES`, `OUTPUT FILES` — those the run produces |
| `poraque-atoms` | `INPUT`, `ATOMS`, `OUTPUT FILES`, `USAGE`; with `--show`, `DATABASE`, `ENTRIES`, `NOTES` |
| `poraque-vasp` | `DENSITY`, `SETTINGS`, `OUTPUT FILES`, `NEXT STEPS`; `NOTE` and `ERROR` when they apply |
| `poraque-mp` | `SEARCH`, `TASKS`, `SIZES`, `SAMPLE`, `DOWNLOAD`, `DECOMPRESS`, `RECOMPRESS`, `DRY RUN`, `SUMMARY FILES`, `STRUCTURES` — those the mode reaches; `--estimate` adds `ESTIMATE`, `PROJECTION` and `STORAGE` |
| `poraque-committee` | `PURPOSE`, `COMMITTEE`, `RANKING`, `CALIBRATION`, `MEASURED ERRORS`, `CAVEATS`, `OUTPUT FILES` |
| `poraque-active-learning` | `PURPOSE`, `COMMITTEE`, `SCORING`, `STATISTICS`, `RANKING`, `SELECTION`, `PROMOTION`, `CAVEATS`, `OUTPUT FILES` |
| `poraque-compile` | `CONFIGURATION`, then `SPECTRAL KERNEL` and `CODEC KERNEL` (status, library, check; one only with `--only`), `CLEAN` (with `--clean`), `NEXT STEPS` |
| `python -m poraque.ml.backend` | `CONFIGURATION`, `BUILD`, `STATUS`, `BENCHMARK`, `NEXT STEPS` |

`<TASK>` is the internal task's name in upper case, so a run with both
operators and the occupancies off has an `EXT2PSEUDO TRAINING` and a
`PSEUDO2TAU TRAINING` (`EXT2CHG` and `CHG2TAU` with them on).

## Two blocks worth reading closely

**`[CACHE]`** prints one row per material as it is built — not collected and
printed at the end, since a full archive takes minutes to hours and a table
that appears afterwards is not progress:

```text
  file  material           native grid    cached grid  EXTCAR [eV]             CHGCAR [e/Ang^3]        TAUCAR [eV/Ang^3]       time
  ---------------------------------------------------------------------------------------------------------------------------------
   1/3  structure_0000           8x8x8          8x8x8  -23.4 .. 5.64           0.0184 .. 0.21          0.0282 .. 1.7           0.0 s
   2/3  structure_0001           8x8x8          8x8x8  -22.4 .. 5.14           0.0199 .. 0.21          0.032 .. 1.74           0.0 s
   3/3  structure_0002           8x8x8          8x8x8  -27.5 .. 6.98           0.0147 .. 0.21          0.0192 .. 1.77          0.0 s
```

The first column counts the materials, `i/N`, so a long build says how far it
has got. Then the two grid shapes, which say what the downsampling did, and
the value range of every cached field, which is the cheapest check that it did
it correctly: a potential that does not straddle zero, or a range orders of
magnitude from its neighbors', is visible at a glance and always means
something. A band-limited density can dip slightly below zero — Gibbs ringing
around the core peaks, an artifact of the truncation and the reason the
density is normalized with an asinh — and the range column shows it; it is not
reported a second time. The last column is the time a material took, or
`cached` for one found already built — what every row of a GPU job reads after
a [`--cache-only`](cache-only) job. (The rows above are a synthetic 8³ smoke
run; the shape is what matters.)

**`[CACHE SUMMARY]`** and **`[OVERALL]`** name the dataset's elements, read
from the cached headers — the `CHGCAR`'s, or the root group of each `fields.h5`
in an HDF5 cache — so the closing caveat about chemical breadth is computed
from the data rather than asserted.

The training progress table, the per-structure results and the resource
profile are described where their settings are: [`validation.every`](configuration.md),
[`spin`](spin-key) and [the resource profile](resource-profile).

## Where it goes

To the terminal, to a file, or to both, a line at a time and flushed at once,
so a long run can be followed while it runs. `poraque-train` writes its log to
`models/<name>/log/<name>.log` beside the metrics JSON; `poraque-inference`
writes `inference.log` into its `--output` directory, banner included. A reader that goes away — a log piped
into `head` — stops the echo and leaves the file alone. Under distributed
training every rank but the first is silent: four ranks writing one file
truncate and interleave each other.

Library code does not depend on any of this. A function that reports progress
takes a plain callable, `log=print` by default, and a `Logger` is callable —
`log("text")` writes the text verbatim — so only the commands, which own the
output, build one. Library code that wants a block header of its own without
importing a writer calls {func}`~poraque.utils.logging.header`, which returns
the `[NAME]` line as a string; the resource profile and the Materials Project
fetcher are written that way.
