# Data sources and modular training

Poraquê trains on whatever mixture of directories you point it at, and trains
whichever of its two models the data can actually support. Those two facts are
what let a public archive of charge densities — hundreds of thousands of them,
and not a single kinetic energy density among them — become training data.

## One schema, whatever produced the data

`data.data_paths` is a **list of directories**, and every entry has the same
shape: subdirectories, one per material, each holding that material's
volumetric files.

```yaml
data:
  data_paths:
    - data/vasp/structures  # structure_0000/, structure_0001/, ...
    - data/MP               # mp-124/, mp-81/, ...
    - data/cache/res32      # a cache from an earlier run
  resolution: 32
```

Nothing in the config says which is which, because on disk they are the same
kind of thing. What differs is the **content** of a material's directory, and
content is read rather than declared:

| A material's directory holds | $V_\mathrm{ext}$ from | $\tau$ |
| --- | --- | --- |
| a full DFT run | its own inputs — `POTCAR` tables, **exact** | read from `TAUCAR` when the run wrote one |
| a density and nothing else | the density's own header, plus `potcar_dir` if set | absent — no archive publishes it |
| prepared fields | read from `EXTCAR` | read from `TAUCAR` |

**A material is a directory with a density in it.** The `CHGCAR` — text,
compressed, or a field inside `fields.h5` — carries the grid every other field
is placed on and the structure they are all built at, so it is the one thing
that qualifies a directory; inputs beside it are a bonus. A path holding a
**single** material's run directly — its files at the top level rather than
one level down — is read as that one material.

`TAUCAR` is optional **per material, not per directory**: one run in a tree may
have written one while its neighbour did not, and the one that did still
reaches `chg2tau` while both reach `ext2chg`. A task with no target anywhere is
skipped with a message rather than failing the run.

```{note}
Three keys used to answer this one question — `train_paths`, `root` and
`source` — and all three were removed on 2026-08-31. `source` is the one that
mattered: it asked the *config* to declare something the *directory* already
answers. A config using any of them fails and is told what to write instead.
```

Detection is by content, never by name. The one thing a config can still
state is `data.format` — `auto` (the default) detects the DFT code that wrote
each directory, `vasp` names it — and a name that disagrees with the directory
is an error rather than an empty dataset.

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
>>> bundle_tasks("models/poraque_models.poraque")
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
  On the Ag–Pt–Pt set this returns 11, 11 and 10 — the `POTCAR` values, read
  off the data, from three small files.

So the recipe for a broad `ext2chg` model is:

```bash
poraque-mp --elements Pt Pd Ni --estimate                    # size it first
poraque-mp --elements Pt Pd Ni --output data/MP --max-size-mb 20
poraque-train --config my_mp_run.yaml     # data_paths: [data/MP]
```

and to specialise it on your own chemistry afterwards, point `data_paths` at
both directories, or fine-tune (see {doc}`../fine_tuning/index`).

### Taking the archive whole

A chemical space is a set of elements. `--all` is the other question — every
material the index says has a charge density, whatever it is made of:

```bash
poraque-mp --all --estimate --sample 20      # what would 20 of them cost?
poraque-mp --all --sample 20 --output data/MP        # fetch exactly those 20

poraque-mp --all --num-sites 1 12 --max-size-mb 20 \
    --output data/MP --hdf5 --compression gzip
```

That cap has a second spelling for the scale this mode works at:
`--max-size-gb 0.02` is the same filter as `--max-size-mb 20`, a decimal factor
of 1000 apart, matching every size the report prints. One `CHGCAR` is an
MB-scale object and a whole-database download is a GB-scale one, and the unit
that reads naturally depends on which you have in mind. Passing both is
refused rather than resolved — two caps is a question about which one wins,
and no answer to it is worth guessing.

**It has to be written out.** Omitting `--elements` does not mean "everything";
a command line naming neither is refused, because that flag was required for
most of this tool's life and reading its absence as the whole database would
turn a forgotten argument into a multi-terabyte transfer.

That one flag is a single server-side query — `has_props=["charge_density"]` on
the summary endpoint — rather than one query per chemical system, and rather
than pulling every summary document in the database and discarding the ones
without a density. The search filters still apply, so `--all` is also how a
*subset* of the whole archive is chosen: the second command above is not
150 000 materials but the ones small enough to train on.

### `--hdf5` keeps everything, and VASP still cannot read it

`--hdf5` writes each density as a chunked, compressed field store rather than a
`CHGCAR`. It is a re-encoding, not a conversion — pymatgen already holds
$\rho\Omega$, which is the convention the store keeps — and it carries
**everything the text file does**: the magnetisation block of a spin-polarised
density, and the PAW augmentation records, the one-centre terms `ICHARG = 1`
will not restart without.

That was not true before 2026-09-01, and the way it was untrue is worth
knowing if you have stores on disk from before then. The store took the total
block and wrote no records, so one download produced two different datasets:
`data.spin: auto` resolves against what is on disk, so the same materials
trained a two-channel operator as `CHGCAR`s and a one-channel one as stores —
and no store could seed a VASP restart at all. **A store written before that
date has no records to give back**; re-fetch the material if you need them.

What no store can do is be opened by VASP, so a run needs the density written
back out:

```bash
poraque-vasp chgcar data/MP/mp-124/fields.h5 --output CHGCAR

# or straight into a deck, which converts rather than copies
poraque-vasp energy data/MP/mp-124/fields.h5 --like <run> --copy-density
```

Every mode of `poraque-vasp` takes a store where it takes a `CHGCAR`; address
one field as `fields.h5::CHGCAR` when the store holds several. A text source is
copied byte for byte rather than parsed and re-emitted — re-rendering a file
that was already correct can only lose digits or shift a column, and the
density block is a column-positional Fortran read.

### A sample is a selection, not just a sizing trick

`--sample N` draws N materials at random and **those become the working set**.
With `--estimate` it sizes exactly those N; without `--estimate` it downloads
them. Everything the run writes into `--output` is about that subset —
`summary.csv` and the CIFs included — so a sampled directory never carries an
index of a set it does not hold.

```{note}
Before 2026-09-01 `--sample` reached the dry run alone. Used without
`--estimate` it was **silently ignored**, and the command fetched the whole
matching set — with `--all`, the entire database.
```

Sizing changes with it. Over a chemical space every object is `HEAD`ed and the
total is exact; over the database that is tens of thousands of requests for a
number nobody needs to four digits. A sampled estimate measures its N exactly
— so `TOTAL DOWNLOAD` is a measurement of what a fetch would actually transfer,
not an extrapolation — and **also projects** to the whole set, as the sample
mean times the number advertised, on its own line and labelled as a projection.
Both are printed because they answer different questions, "what will this
fetch" and "what would all of it cost", and a report stating one of them gets
read as the other.

`--seed` decides *which* subset, and that is what makes a sampled download
resumable: the same seed selects the same materials, so an interrupted run
resumes onto the same subset rather than fetching N more beside it. Sizes are
also strongly right-tailed, so two draws of twenty can differ by tens of
percent, and a projection that moved on every run could not be planned against.

## The third missing piece: pseudopotentials

The structure and the valence charges are recoverable from the density. The
**pseudopotentials are not**, and they are what the exact external potential is
built from. This is the one gap the data cannot close by itself, and
`data.potcar_dir` is how you close it:

```yaml
data:
  data_paths: [data/MP]
  potcar_dir: /opt/vasp/potpaw_PBE
```

With the library, the archive supplies the structure and the library supplies
the pseudopotentials — together everything VASP's own `POTION` construction
needs — and $V_\mathrm{ext}$ is accurate to a relative $2\times10^{-5}$.
Without it, the Gaussian pseudo-ion model stands in. On the Ag–Pt–Pt set the
two differ by **0.38 relative $L_2$**: different fields, not different
roundings of one. A species the library cannot serve warns once and falls back
on its own; the rest still get the exact potential. See
{ref}`the configuration reference <potcar-dir-and-the-gaussian-fallback>` for
the layouts recognised.

```{warning}
**Mixing archives can mix two definitions of $V_\mathrm{ext}$.** A calculation
directory has a `POTCAR`, so its potential is tabulated. A material with no
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

with MPDataFetcher(["Pt", "Pd", "Ni"], outdir="data/MP",
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
