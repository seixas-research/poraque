# Data sources and modular training

Poraquê trains on whatever mixture of directories you point it at, and trains
whichever of its two models the data can actually support. Those two facts are
what let a public archive of charge densities — hundreds of thousands of them,
and not a single kinetic energy density among them — become training data.

## Three layouts, one dataset

`data.train_paths` is a **list**, and each entry is detected independently:

```yaml
data:
  train_paths:
    - data/vasp             # local DFT runs
    - data/MP/chgcar        # a Materials Project download
  resolution: 32
```

| Layout | Recognised by | $V_\mathrm{ext}$ from | $\tau$ |
| --- | --- | --- | --- |
| `vasp` — a DFT calculation, or a directory of them | a `POSCAR` or `CONTCAR` | `POSCAR` + `POTCAR` tables, **exact** | read from `TAUCAR` when the run wrote one |
| `bulk` — standalone `CHGCAR` files, compressed or not | density files with no `POSCAR` | the density's own header, plus `potcar_dir` if set | never — no archive publishes it |
| `prepared` — per-material field directories | `EXTCAR`/`CHGCAR`/`TAUCAR` inside subdirectories | read from `EXTCAR` | read from `TAUCAR` |

Detection is by content, never by name, and `source: auto` is the default. Set
`source: bulk` (or a list, one name per path) to state it explicitly; a name
that disagrees with the directory is an error rather than an empty dataset.

Everything found is pooled, spectrally downsampled once into a shared cache,
and served through one `Dataset`. Identifiers that collide between archives are
prefixed with their directory, so two `struct_000`s never silently become one.

```{seealso}
{mod}`poraque.data.sources` for the source classes and how to add a layout;
{class}`poraque.data.dataset.MixedFieldDataset` for the `Dataset` itself.
```

## Training one model, not two

The two tasks are independent, and `task` selects them:

```yaml
task: ext2chg      # or chg2tau, or all
```

With `task: ext2chg` the `chg2tau` network is never constructed, never moved to
the device and never differentiated; no $\tau$ is read, and every
kinetic-energy term of the objective is inert by construction — the physics
losses are selected by task, so `von_weizsacker_weight` simply does not apply.
The checkpoint holds one model:

```python
>>> from poraque.ml import bundle_tasks
>>> bundle_tasks("models/poraque_models.pfno")
['ext2chg']
```

which loads exactly as a two-task bundle does. Asking such a bundle for
`chg2tau` — or handing it to the ASE calculator, which runs the whole chain
$V_\mathrm{ext}\to\rho\to\tau\to E$ — fails with a message saying so, because a
model that never saw a kinetic energy density cannot supply one.

**`TAUCAR` is optional everywhere.** A calculation directory that has a
`CHGCAR` and no `TAUCAR` is a first-class `ext2chg` dataset: it is ingested
without error and without warning, and the per-material log says `[no TAUCAR]`
rather than treating the absence as a fault. The check is per **material**, not
per directory, so one archive can hold runs of both kinds.

With `task: all` on data that carries no $\tau$ at all, the run reports

```text
SKIPPING chg2tau: no material under data/cache/mp/res32 has both CHGCAR and TAUCAR.
    Valence charge density -> kinetic energy density cannot be trained on data
    that does not carry TAUCAR.
```

and trains `ext2chg` normally. A task the data cannot supply is a fact about
the data, not a failure of the run.

## Why this matters: foundation models from public densities

The Materials Project publishes converged charge densities for a large fraction
of its ~150 000 materials. It publishes no kinetic energy densities, no
`OUTCAR`s alongside them, and no external potentials. Under a pipeline that
required all three, that archive is unusable. Under this one it is the largest
`ext2chg` training set available, because the two things it is missing are both
recoverable or unnecessary:

* the **structure** is inside the `CHGCAR` — a density file carries its own
  `POSCAR` in its first lines — so $V_\mathrm{ext}$ can be built from the
  density alone;
* the **valence charges** that construction needs are recovered from the
  densities themselves. A pseudopotential `CHGCAR` integrates to its cell's
  valence electron count, which is one linear equation per material in the
  per-element charges; a chemical space of any breadth over-determines them.
  On the Ag–Au–Pt set this returns 11, 11 and 10 — the `POTCAR` values, read
  off the data, from three small files.

So the recipe for a broad `ext2chg` model is:

```bash
poraque-mp --elements Ag Au Pt --estimate                    # size it first
poraque-mp --elements Ag Au Pt --output data/MP --max-size-mb 20
poraque-train --config configs/train_materialsproject.yaml
```

and to specialise it on your own chemistry afterwards, point `train_paths` at
both archives, or fine-tune (see {doc}`../fine_tuning/index`).

## The third missing piece: pseudopotentials

The structure and the valence charges are recoverable from the density. The
**pseudopotentials are not**, and they are what the exact external potential is
built from. This is the one gap the data cannot close by itself, and
`data.potcar_dir` is how you close it:

```yaml
data:
  train_paths: [data/MP]
  potcar_dir: /opt/vasp/potpaw_PBE
```

With the library, the archive supplies the structure and the library supplies
the pseudopotentials — together everything VASP's own `POTION` construction
needs — and $V_\mathrm{ext}$ is accurate to a relative $2\times10^{-5}$.
Without it, the Gaussian pseudo-ion model stands in. On the Ag–Au–Pt set the
two differ by **0.38 relative $L_2$**: different fields, not different
roundings of one. A species the library cannot serve warns once and falls back
on its own; the rest still get the exact potential. See
{ref}`the configuration reference <potcar-dir-and-the-gaussian-fallback>` for
the layouts recognised.

```{warning}
**Mixing archives can mix two definitions of $V_\mathrm{ext}$.** A calculation
directory has a `POTCAR`, so its potential is tabulated. A bulk archive with no
`potcar_dir` uses the Gaussian model. Train across both and the input field is
two different quantities under one name; the operator spends capacity
reconciling them, and any comparison against a model trained on tabulated
potentials has to say so.

The run warns when it happens, and the warning is keyed on the *construction*
rather than the layout — so setting `potcar_dir` makes both sources tabulated,
makes the mixture one quantity again, and silences the warning correctly rather
than merely suppressing it.
```

Without a library, an MP-only model learns *model potential* $\to$ *DFT
density*. That is a well-posed and self-consistent problem — inference builds
the very same potential — but it is not VASP's $V_\mathrm{ext}$, and it should
be stated wherever the model's numbers are.

## Fetching from the Materials Project

{class}`poraque.data.materials_project.MPDataFetcher` turns a chemical space
into a local dataset. Sizing is exact rather than modelled: charge densities are
objects in a public S3 bucket, so the fetcher resolves each material's static
task and issues a `HEAD` request, reading `Content-Length` without transferring
any payload.

```python
from poraque.data import MPDataFetcher

with MPDataFetcher(["Ag", "Au", "Pt"], outdir="data/MP",
                   band_gap=(0.0, 0.0),        # metals
                   num_sites=(1, 8)) as mp:
    mp.dry_run()                               # report only; writes nothing
    mp.run(max_size_mb=20)                     # resumable
```

On the command line `--estimate` is that dry run: it prints the file count and
the total transfer to the console and leaves **no file behind**, not even a
summary. Everything else is written under `--output` (equivalently `--outdir`),
which defaults to the current directory.

The API key is read from `api_key=`, then `$MP_API_KEY`, then a local `.env`,
then `~/.env`, so it never has to appear in a config or a shell history.

Downloads stay **gzipped**. Poraquê reads compressed volumetric files in place —
`.gz`, `.bz2`, `.xz` and `.zip`, streamed a line at a time so peak memory tracks
the parsed grid rather than the file — so expanding them buys nothing and costs
roughly a threefold storage multiplier.
