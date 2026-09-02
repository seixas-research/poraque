# Poraquê on four GPUs — the extension rename, and DDP over NCCL under Slurm

**From:** Claude Code on the Apple machine (Darwin 25.6.0, Apple Silicon, MPS,
**no CUDA at all**).
**To:** Claude Code on Santos Dumont (LNCC), `sequana_gpu`, Tesla V100.
**Subject:** the sequel to `CUDA_2.md`. Two changes: the checkpoint extension is
now `.poraque`, and Poraquê trains data-parallel over NCCL when a Slurm step
describes a group.

**Base:** `poraque 26.9.1` (`d5e6601`), on top of the `CUDA_2.md` work, which is
also still uncommitted. The changes are **unstaged in the working tree** on the
Apple machine — the standing instruction there is never to commit or push.

**Read this first, because it governs everything below.** No process group has
ever been formed. `init_process_group` has never been called, by anybody, on any
machine. The whole distributed path is written, linted, unit-tested (53 tests)
and dry-run against a *simulated* four-rank Slurm environment — but the machine
it was written on has no CUDA, and the module refuses Gloo on purpose (§6). So
everything up to the rendezvous is exercised and **nothing past it is**. You are
the first execution. §5 is the protocol, in the order that finds a problem
soonest.

---

## 0. Before anything else

### 0.1 The code is not on the remote

Forty-four files are modified and three are new — `src/poraque/ml/distributed.py`,
`tests/test_distributed.py` and `scripts/slurm/poraque_ddp.sbatch` — plus this
document and `CUDA_2.md`. None is committed. Ask the user to push, or `rsync`
the tree. Verify you have the right code with:

```bash
python -c "import poraque.ml.distributed as d; print(d.PORT_RANGE)"
python -c "from poraque.ml.config import BUNDLE_SUFFIX; print(BUNDLE_SUFFIX)"
```

`(20000, 29999)` and `.poraque`. An `ImportError` or `.pfno` means you are on
the `CUDA_2.md` state or earlier, and everything below is absent.

### 0.2 The install must be editable, or the tests lie

Unchanged from `CUDA_2.md` §0.2, and it matters more here because the new module
is a *new file* — a stale copy install will not merely run old code, it will
`ImportError`.

```bash
python -c "import poraque, os; print(os.path.dirname(poraque.__file__))"
```

If that is not the repository's `src/poraque`, either `pip install -e .` or
prefix everything with `PYTHONPATH=src`. Every verification quoted in this
document was run with `PYTHONPATH=src`.

### 0.3 Your existing checkpoints are all called `.pfno`, and they still load

The extension changed on 2026-09-02. **Nothing about the container changed** —
same `torch.save` payload, same `poraque-bundle-1` format tag, same keys. Only
the name moved, and `resolve_bundle_path` finds the old one:

```
  NOTE: models/x/x.poraque does not exist, but models/x/x.pfno does — using it.
        Poraque now writes x.poraque; rename the file to silence this.
```

Verified end to end against the shipped
`models/platinum_W16-M8-L3/platinum_W16-M8-L3.pfno`: asked for `.poraque`, got
the `.pfno`, loaded both tasks. **Do not rename your checkpoints**; there is no
reason to, and the notice is the system working.

### 0.4 The wheel, and the pre-flight

Unchanged: `torch==2.7.1+cu126` for a V100, and

```bash
export PYTHONNOUSERSITE=1
python -m poraque.ml.device --check || exit 1
```

The submission script in §2.6 runs that check itself, before `srun`.

---

## 1. The extension: `.pfno` → `.poraque`

### 1.1 What changed

A codebase-wide rename across 44 files — source, scripts, tests, `configs/`,
`README.md`, Sphinx, both LaTeX guides, `experiments/`, `.gitignore`.

The string is defined **once**, in `src/poraque/ml/config.py`:

```python
BUNDLE_SUFFIX = ".poraque"
```

It lives there rather than in `training.py` because `config.py` is the
**torch-free layer** and is where `TrainingConfig.checkpoint_path()` builds the
name; `training.py` imports it. That direction has no cycle and no second copy.
The name has now been changed twice — `.pth` → `.pfno` → `.poraque` — and each
previous time the cost was a default path somewhere still spelling the old one.

`BUNDLE_FILENAME` and `FINETUNED_BUNDLE_FILENAME` are now built from it, as is
`poraque_train.py`'s per-task safety copy and `checkpoint_path()`.

### 1.2 The legacy chain, and why it searches both ways

```python
LEGACY_BUNDLE_SUFFIXES = (".pfno", ".pth", ".pt")     # newest first
```

`resolve_bundle_path(path, log=None)` was a one-direction search: given
`x.poraque`, look for `x.pfno`, `x.pth`, `x.pt`. It now searches **every**
suffix Poraquê has used, including the current one, excluding whichever was
asked for.

The second direction is not symmetry for its own sake. A **LoRA checkpoint
records the absolute path of the base it adapts** — that is the price of a
7.6 kB file — and `state()["lora"]["base_checkpoint"]` written before the rename
names a `.pfno`. If the base is later renamed, that path is gone and the adapter
holds no weights of its own. `FieldOperator._from_lora_state` now routes through
`resolve_bundle_path` before its existence check.

Only the **extension** is ever guessed at. A different stem is a different
model and is never substituted; there is a test for that.

### 1.3 Committees

`scripts/poraque_committee.py:member_bundle` globbed `*.pfno` inside a member
directory. It now tries `BUNDLE_SUFFIX` first and the legacy suffixes after, so:

- a committee trained member-by-member across the rename is still one committee;
- a re-trained `.poraque` member wins over the stale `.pfno` it replaced;
- the "several checkpoints, none is the default" error still fires, per suffix.

### 1.4 What to check here

```bash
ls models/*/*.pfno                       # your existing models, untouched
poraque-inference <structure_dir>/ --models models/<name>/<name>.poraque
```

The second should print the NOTE from §0.3 and proceed. **It will then crash** —
see §8.1, which is a pre-existing bug unrelated to any of this.

---

## 2. Multi-GPU: what changed, file by file

### 2.1 `src/poraque/ml/distributed.py` — new, 644 lines

Everything about launching and joining a group. Public surface:

| symbol | what it does |
|---|---|
| `DistributedContext` | rank, local_rank, world_size, endpoint, launcher. **Falsy when disabled**, so call sites read `if context:` |
| `discover(requested="auto")` | Slurm first, then torchrun, then nothing. Never raises |
| `initialize(context, timeout_minutes=30)` | `set_device` then `init_process_group("nccl")`. Raises rather than falling back |
| `shutdown(context)` | `destroy_process_group`, called from a `finally` |
| `barrier(context)` | `dist.barrier(device_ids=[local_rank])`; no-op when disabled |
| `all_reduce_mean(value, context)` | the epoch loss, reduced on the device |
| `unwrap(model)` | the module inside a DDP wrapper |
| `describe(context)` | the log lines, including the Slurm variables actually present |
| `expand_nodelist(nodelist)` | `scontrol show hostnames`, with a pure-Python fallback |
| `default_master_port(job_id=None)` | a port in `PORT_RANGE = (20000, 29999)` from `SLURM_JOB_ID` |

**`context.device` is `cuda:<local_rank>`** when enabled and `"auto"` when not,
so it goes straight into `resolve_device` in both cases. Every rank on `cuda:0`
is four processes contending for one device, and it reports itself as a scaling
failure rather than as an error — that is why the ordinal comes from
`SLURM_LOCALID` and not from `auto`.

**`context.is_main` is `rank == 0`, and `True` when disabled.** That last part is
what makes every writer call site unconditional.

### 2.2 `src/poraque/ml/data.py` — `DistributedShapeBucketSampler`

This is the part that needed a design rather than a wiring.

`CUDA.md` item 5.1 was refused with a specific reason, and the reason was right:
`ShapeBucketSampler` is a **`batch_sampler`**, `DataLoader` accepts `sampler` *or*
`batch_sampler` and never both, so a plain `DistributedSampler` would have to
*replace* the bucketing. And a batch mixing 32³ with 40³ does not train badly —
it **raises** in `collate_fields`, because there is no padding anywhere in this
pipeline and the FFT is the reason.

The resolution is to distribute the **batches**, not the samples:

1. `ShapeBucketSampler._batches()` runs unchanged, on every rank. It is a pure
   function of `(seed, epoch)`, so all ranks build an **identical list** without
   communicating.
2. A real `torch.utils.data.distributed.DistributedSampler` is constructed over
   `range(n_batches)` and partitions **those indices**.
3. Each rank yields `batches[i % len(batches)] for i in partition`.

Properties, in the order they matter:

- **Equal counts per rank.** DDP all-reduces gradients inside each `backward()`.
  A rank that runs out of batches first leaves the others in a collective that
  never completes, and the job burns its allocation *in a hang* rather than
  failing. `DistributedSampler` pads to a multiple of the world size by wrapping
  to the front, and that is exactly the property being borrowed.
- **No batch mixes grid shapes.** The bucketing is untouched.
- **Unique, non-overlapping subsets**, exactly when the batch count divides the
  world size. When it does not, the padding duplicates up to `world_size - 1`
  batches — measured on the real Pt shape split (43 cells at 32³ + 11 at 40³,
  `batch_size` 10) that is 7 batches padded to 8, so each of four ranks gets 2
  and one batch is seen twice. Against a deadlock that is not a close call, and
  the test states it as a *bound* rather than pretending there is no overlap.
- **`set_epoch` forwards to both halves.** Missing either is silent rather than
  a crash: forget the bucket sampler and every epoch draws the same batches;
  forget the partition and every epoch sends the same batches to the same rank.

`make_dataloader` gained `distributed=None`. `None` and a disabled context are
the same thing.

### 2.3 `src/poraque/ml/training.py`

`train()` gained `distributed=None`. Six changes inside:

1. **Rank gating, once, at the top.** `if not context.is_main: verbose = False;
   checkpoint = None`. Four ranks calling `operator.save` on one path race on
   the same inode — which does not raise, it leaves a checkpoint that loads and
   holds a mixture.
2. **`_distributed_forward`** wraps the model in `DistributedDataParallel(
   device_ids=[local_rank], find_unused_parameters=False)` and returns the
   wrapper. It is **never assigned onto the operator**: DDP prefixes every
   `state_dict` key with `module.`, exactly as `torch.compile` prefixes them
   with `_orig_mod.`, and storing either makes every checkpoint the run writes
   unloadable everywhere else. Same discipline as `CUDA_2.md` §1's compile work,
   for the same reason.
3. **`_compiled_forward` gained `module=`**, and compiles **on top of** DDP.
   Dynamo's DDPOptimizer splits the graph at DDP's gradient-bucket boundaries so
   the all-reduces overlap the backward pass, and it can only do that if it can
   see the wrapper. Compiling inside DDP gives one graph with every all-reduce
   serialised after it. **This ordering is untested on hardware** — see §5.4.
4. **The training loss is all-reduced**, on the device, before the single
   `.item()` per epoch that `CUDA_2.md` §1 introduced. Without it each rank's
   `train loss` is over its own quarter of the data and the four numbers differ.
5. **Validation is deliberately not distributed.** Every rank evaluates the whole
   held-out set. That is redundant work — forward-only, on a fifth of the data —
   bought so that `best_error` and the early-stopping decision are identical on
   every rank *by construction* rather than by a reduction someone could forget.
   Ranks that disagreed about whether to `break` would leave the others in a
   collective nobody joins.
6. **A barrier after the loop**, so the non-writing ranks do not run ahead into
   the next task while rank 0 is still writing a checkpoint.

The "One GPU" paragraph in `train()`'s docstring is replaced by one stating the
three things that change under DDP: effective batch size, replicated validation,
reduced loss.

### 2.4 `src/poraque/ml/config.py`

```yaml
training:
  distributed: auto           # auto | off
  distributed_timeout: 30.0   # minutes
```

`auto` forms a group **when the environment already describes one**, and does
nothing otherwise. It **cannot invent ranks**: the launcher decides the topology
and this key decides only whether to believe it. `off` refuses a group inside a
multi-task allocation, which is how a scaling result is bisected.

`distributed_timeout` is long on purpose. The first collective happens after
every rank has read its prepared cache, and a cold Lustre read of a few hundred
densities is minutes; a timeout that fires there reports itself as a NCCL error
and sends the reader looking at the network.

### 2.5 `scripts/poraque_train.py`

- `discover_distributed` runs **before** anything else in `run()`, then
  `config.training.device = context.device` and `initialize_distributed`.
- **Non-main ranks are switched off once**, on that process's own copy of the
  config: `checkpoint`, `plot_figures`, `write_pdf_report`, `write_log`,
  `save_raw_plot_data` all `False`, and `output.log` / `output.json` emptied —
  those two are explicit-path overrides that bypass their own directory flag.
  Every writer downstream reads those flags, including ones added later.
- `Tee` gained `silent=`, given to every rank but the first.
- **The cache is built by rank 0 behind a barrier.** Four processes spectrally
  downsampling into one directory write the same files concurrently, and the
  loser of that race gets a truncated `CHGCAR` that **parses** — the format has
  no length field — into a field of the wrong shape. After the barrier the other
  ranks call `build_cache` too, which is a presence check: `build_field_cache`
  leaves materials already present alone.
- `loader_settings(config, distributed)` carries the context, so a k-fold fold
  cannot forget it and train on the whole dataset on every rank.
- `--distributed {auto,off}`.
- The launch is logged whether or not a group formed:

```
  ranks  : nccl rank 0/4 local_rank 0 on sdumont1234:24601 (launched by slurm)
             SLURM_JOB_ID = 4601
             SLURM_NTASKS = 4
           effective batch = 10 x 4 ranks = 40
```

and when none formed inside an allocation:

```
  ranks  : single device (no distributed group)
           a Slurm allocation is present but no group was formed; check --ntasks-per-node
           SLURM_NTASKS = 1
```

### 2.6 `scripts/slurm/poraque_ddp.sbatch` — new

```bash
#SBATCH --partition=sequana_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GPU. Not optional.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6

export PYTHONNOUSERSITE=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m poraque.ml.device --check || exit 1

srun --kill-on-bad-exit=1 \
     poraque-train --config "${CONFIG}" --device cuda \
                   --strict-device --distributed auto
```

Three of those are worth defending. `NCCL_SOCKET_IFNAME=ib0` is set explicitly
rather than left to NCCL's interface discovery, which on a multi-homed node can
select a management interface and produce a rendezvous that *times out* rather
than fails — **if `ib0` is not the right interface on `sequana_gpu`, this is the
first thing to change.** `OMP_NUM_THREADS` is per-task because four ranks each
spawning a thread per core oversubscribes the node fourfold, and the symptom is
a data pipeline that gets slower as GPUs are added. `--kill-on-bad-exit=1` so
one rank's exception takes the step down instead of leaving three in a
collective.

### 2.7 Slurm, with nothing hard-coded

| question | answered by |
|---|---|
| how many ranks | `SLURM_NTASKS` |
| which rank am I | `SLURM_PROCID` |
| which GPU | `SLURM_LOCALID`, falling back to `PROCID % SLURM_NTASKS_PER_NODE` |
| `MASTER_ADDR` | first host of `SLURM_STEP_NODELIST` |
| `MASTER_PORT` | `20000 + (SLURM_JOB_ID % 10000)` |

`SLURM_STEP_NODELIST` and not `SLURM_JOB_NODELIST`: the first is the nodes of
*this step*, the second the whole allocation, and they differ when a script runs
several steps over subsets of its nodes. The group being formed is the step's.

The nodelist parser handles `sdumont[1234-1236,1240]`, zero-padding (`node[01-03]`
→ `node01`, not `node1`), suffixes after the brackets, and commas inside brackets
not being separators. `scontrol show hostnames` is asked first and is right by
definition; the parser is the fallback for a login node or a container.

The port is derived so two jobs on one node cannot collide on a hard-coded
29500, every rank of one job derives the same number independently, and a
requeue keeps it.

**torchrun is consulted *after* Slurm.** A torchrun launch inside an allocation
carries both sets of variables, and its own describe the group it actually
formed; reversing the order would read the allocation's task count instead.

---

## 3. The new configuration surface

Added to what `CUDA_2.md` §2 listed:

| key | default | what it does |
|---|---|---|
| `training.distributed` | `auto` | `auto` \| `off` — DDP over NCCL when the launcher describes a group |
| `training.distributed_timeout` | `30.0` | minutes before a collective is declared failed |

CLI: `--distributed {auto,off}`.

Neither is in the cache fingerprint. Neither decides anything about what is on
disk, and a run that rebuilt its cache because it was given a second GPU would be
absurd.

---

## 4. What to verify first — none of this needs four GPUs

### 4.1 Sanity

```bash
PYTHONPATH=src pytest tests/ -q                  # expect 1811 passed, 98 skipped
PYTHONPATH=src pytest tests/test_distributed.py -v   # 53 tests
PYTHONPATH=src pytest tests/test_split.py -k Pfno -v # the extension fallback
ruff check --select F,E7,E9 src/ scripts/ tests/
```

The ten new test classes:

| class | what it pins |
|---|---|
| `TestTheSlurmNodelistIsExpandedWithoutSlurm` | the parser, including order — the *first* host is `MASTER_ADDR` |
| `TestTheRendezvousPortIsDerivedNotHardCoded` | in range, stable per job, differs between jobs |
| `TestNoLauncherMeansNoGroup` | bare process, one-task allocation, `off`, no CUDA |
| `TestTheSlurmTopologyIsRead` | which variable answers which question |
| `TestSlurmIsBelievedBeforeTorchrun` | the precedence, and why |
| `TestADisabledContextIsANoOp` | every collective reached unconditionally |
| `TestTheBatchesAreSplitNotTheSamples` | equal counts, no mixed shapes, coverage, the padding bound |
| `TestTheLoaderTakesTheContextUnconditionally` | `None` ≡ disabled context |
| `TestTheTrainingLoopSilencesEveryRankButTheFirst` | rank 0 writes; single-device path bit-identical |
| `TestTheSubmissionScriptLaunchesOneTaskPerGpu` | the sbatch file's task count matches its GPU count |

That last one is a test on a shell script, which is unusual and deliberate: the
one line that decides whether this feature does anything lives in a file nothing
else checks.

### 4.2 The single-device path is untouched

```bash
PYTHONPATH=src poraque-train --config configs/train.yaml --epochs 3 \
    --no-plots --no-report --device cuda --strict-device --distributed off
```

Measured on the Apple machine at CPU, 3 epochs, the shipped config: **ext2chg
0.0612, chg2tau 0.8518** — identical to the pre-change baseline. At 2 epochs
with `--distributed off`: 0.0412 / 0.9049. And in-process, `distributed=None`
against `DistributedContext()` gives `train_loss` equal to **1e-9**.

Anything that moves here is a regression in the single-GPU path, which is by far
the more important of the two.

### 4.3 The dry run, if you want to see the resolution without four GPUs

Monkeypatching `torch.cuda.is_available`/`device_count` and setting the Slurm
variables by hand resolves the whole topology. On the Apple machine, for
`SLURM_JOB_ID=8675309`, `SLURM_NTASKS=4`,
`SLURM_STEP_NODELIST=sdumont[1234-1235]`:

```
rank 0: nccl rank 0/4 local_rank 0 on sdumont1234:25309  device=cuda:0 is_main=True
rank 1: nccl rank 1/4 local_rank 1 on sdumont1234:25309  device=cuda:1 is_main=False
rank 2: nccl rank 2/4 local_rank 2 on sdumont1234:25309  device=cuda:2 is_main=False
rank 3: nccl rank 3/4 local_rank 3 on sdumont1234:25309  device=cuda:3 is_main=False
rendezvous: all four ranks agree on ('sdumont1234', 25309)
sampler   : batches/rank=[2,2,2,2] equal=True mixed_shapes=False covered=54/54
```

Worth repeating on the real login node, because it exercises `scontrol` rather
than the fallback parser and that is the branch that will actually run.

---

## 5. What needs the GPUs — in the order that finds a problem soonest

Everything here is unknown. Do not skip ahead.

### 5.1 Does the group come up at all

```bash
NCCL_DEBUG=INFO sbatch scripts/slurm/poraque_ddp.sbatch configs/train.yaml
```

with `training.epochs: 3` and `--no-plots --no-report`. Read, in this order:

1. the `ranks :` line — four ranks, `local_rank` 0–3, one address, one port;
2. whether `init_process_group` returns at all, or hangs;
3. whether the epoch table prints once and not four times;
4. whether exactly one `.poraque` is written.

**If it hangs at the rendezvous**, the first suspect is
`NCCL_SOCKET_IFNAME=ib0` (§2.6). Try unsetting it, then setting it to whatever
`ip -o link` says the fabric is called. The second suspect is the port: another
job may hold it; `default_master_port` is derived from the job id, so a
resubmission gets a different one for free.

**If the log prints four times**, `is_main` is not reaching `Tee`. **If four
checkpoints appear**, the `config.output.*` gating in `run()` did not take.

### 5.2 Does an epoch complete, and is the answer the same

Same config, `--distributed off` and `--distributed auto`, same seed, same
allocation. The validation error will **not** match, and that is expected rather
than a bug: the effective batch size is `batch_size × world_size`, so the two
runs are different optimisers. What must match is that both finish, both write
one bundle, and neither reports a `train loss` that is obviously a quarter of
the other's — that last would mean the all-reduce is not happening.

To compare the *same* optimisation, quarter `batch_size` for the four-rank run.

### 5.3 The number the whole thing is for

`seconds_per_epoch` from the metrics JSON, `off` against `auto`, 60 epochs, two
repetitions each. Read it against `CUDA_2.md`'s two standing facts, both of
which bear directly on what scaling is even available:

- **`bs=64` and `bs=128` are the same run.** `ShapeBucketSampler` caps a batch
  at the largest bucket, which held 70 of 115 structures; above ~56,
  `batch_size` means nothing. With four ranks the *per-rank* batch is what is
  capped, so this ceiling moves — which is itself worth checking.
- **Even with the cache on, utilisation peaked at 28 %.** 44.4 % of GPU time is
  elementwise. If the single-GPU run is 72 % idle, four of them may be too, and
  the honest possible outcome here is that DDP buys much less than 4× and the
  communication shows up as a new floor. Report the number you get, not the one
  the feature implies.

Also read `peak_vram_bytes` per rank: DDP holds a gradient bucket per rank, and
1.7 of 32 GiB single-device leaves plenty, but it is now measured rather than
assumed.

### 5.4 `torch.compile` on top of DDP

`CUDA_2.md` §4.1 left `training.compile` as the largest open question and it is
still open — now with a second variable. The ordering implemented is
`torch.compile(DDP(model))`, for DDPOptimizer's graph splitting. Run the
`CUDA_2.md` protocol (150 epochs × two repetitions; `false`, `dynamic=True`,
`mode="max-autotune"`; compile cost read off `seconds_per_epoch[0]`;
`TORCH_LOGS=recompiles`) **single-GPU first**, and only then with four. If the
19-distinct-shapes problem defeats `dynamic=True` on one GPU it will defeat it
on four, and mixing the two variables makes the result unreadable.

### 5.5 `find_unused_parameters=False`

Set on the assumption that every parameter of an FNO takes a gradient every
step — the spectral weights are complex and all modes are used. If DDP raises
*"Expected to have finished reduction in the prior iteration"*, that assumption
is wrong somewhere (the Pauli head on `chg2tau` is the place to look first), and
the fix is `True` at the cost of a graph traversal per iteration. Do not flip it
pre-emptively.

### 5.6 Multi-node

Untested, and the code does not distinguish it from single-node: `MASTER_ADDR`
is the first host of the step's nodelist whether that step spans one node or
four. `--nodes=2 --ntasks-per-node=4` should work. It is the last thing to try,
not the first, and there is no reason to expect a 54-structure dataset to want
it.

---

## 6. Deliberately not implemented, with the reason

So that none of these is re-proposed:

| item | why not |
|---|---|
| Gloo fallback | it would let a misconfigured job train distributed across 96 CPU cores at a fraction of one GPU's speed — the same silent waste `strict_device` exists to prevent. Without CUDA the group is **refused with a warning naming the cause** and the run continues on one device |
| `DataParallel` fallback | single-process multi-GPU is slower than one GPU for a model of a few megabytes, and it changes the effective batch size silently. Offering it would mean a laptop run and a cluster run differed in a way nothing recorded |
| distributing the **samples** | `DataLoader` takes `sampler` or `batch_sampler`, never both; replacing the bucketing makes `collate_fields` raise on the first mixed batch |
| distributing **validation** | every rank evaluating the whole set is redundant, and it is what makes the early-stopping decision identical across ranks without a reduction anyone can forget. Ranks disagreeing about when to `break` is a hang |
| distributing the **k-fold folds** | it is the better parallelisation — the folds are independent and need no communication — but it changes what the run produces per rank, and the version implemented can be compared against a single-device k-fold line for line |
| automatic learning-rate scaling | the effective batch size changes with the world size and nothing rescales for it. Linear scaling is a heuristic, this is a regression problem rather than a classification one, and silently changing the optimiser because a GPU was added is worse than stating the arithmetic. The run log prints it |
| `torchrun` as the documented launcher | `srun` is what SD uses and needs no extra process; torchrun is *supported*, not recommended |

---

## 7. Documentation

| where | what |
|---|---|
| `docs/source/installation.md` | new `(multi-gpu)=` section: the sbatch script, the one-task failure, the rendezvous derivation, the three consequences |
| `docs/source/configuration.md` | `(distributed)=` section, plus both keys in the `training` table |
| `latex/user_guide/sections/01_installation.tex` | `\label{sec:multigpu}`, replacing "One GPU" |
| `latex/user_guide/sections/06_configuration.tex` | `\label{sec:distributed}`, plus the table rows |
| `README.md` | "On several GPUs, under Slurm" |
| `configs/train_complete_and_commented.yaml` | both keys, with the `--ntasks-per-node` warning |
| `docs/source/fine_tuning/index.md`, `05b_fine_tuning.tex` | the `.pfno` → `.poraque` note, both directions of the search |
| `CLAUDE.md` | a "Multi-GPU" section, and pitfalls 22–25 |

Sphinx builds with **no warnings**; both LaTeX guides rebuild with no undefined
references in the final pass. The `.pfno` expansion "*Poraquê Fourier Neural
Operator*" was removed from the docs rather than transferred — it was the
expansion of the *old* name and would have become false.

---

## 8. Two problems found on the way, neither caused by this work

### 8.1 `poraque-inference` crashes on any spin-polarised model

`scripts/poraque_inference.py:665`:

```python
electrons = density.integrate()
# AttributeError: 'SpinDensity' object has no attribute 'integrate'
```

`operator.predict` returns a `SpinDensity` whenever `data.spin` resolved on —
which is **every model trained on the Pt data**. The crash lands after ext2chg
has already predicted successfully, so the model and the bundle are fine.

This is the **same bug** as the `np.asarray(density)` one fixed in
`physics/energy.py` on 2026-08-27 via `total_density`; that pass missed this
site. Pre-existing, found while verifying the extension rename, left unfixed
because it is outside that change's scope. It is a two-line fix and it blocks
the whole inference CLI on the current models. `CLAUDE.md` pitfall 24; pitfall 9
(a bundle must hold both tasks) sits behind it.

### 8.2 `models/platinum_W16-M8-L3/log/` was overwritten

A 3-epoch smoke test on the Apple machine ran with the repository's own
`output.root`. The metrics JSON, the log and the resolved config in that
directory now describe **3 epochs**, not the 400 that produced the `plots/`,
`report/` and `platinum_W16-M8-L3.pfno` beside them. `models/` is gitignored, so
there is nothing to restore from. The PDF reports are the 400-epoch ones and
still hold the real numbers. `CLAUDE.md` pitfall 25.

If you re-run that config on SD, you regenerate them — with the caveat that the
weights would be new too.

---

## 9. A note on method, since the reader is an agent

`CUDA_2.md` closed by saying that the item which won by a wide margin was a
parameter that already existed and was never passed. This one is the opposite
shape, and worth naming.

**The blocker recorded in `CUDA_2.md` §5 was correct and was still not a
reason to stop.** "A plain `DistributedSampler` does not compose with a
`batch_sampler`" is true, and the previous pass was right to refuse the feature
rather than ship something that raised on the first ragged batch. What made it
tractable was noticing that the *thing to distribute* had been assumed —
`DistributedSampler` distributes whatever it is given indices of, and giving it
the batch list rather than the sample list satisfies both constraints at once.
The refusal was good engineering; so was revisiting it with the constraint
written down.

The second thing worth carrying: **the padding is not a rounding detail.** It
would be very easy to read `drop_last=True` as the tidier choice and lose a few
batches per epoch. It would also deadlock the moment the batch count stopped
dividing the world size, which on this dataset is most of the time — and a
deadlock in a queue is a silent wall-clock loss, the same failure class as the
CPU fallback `strict_device` was built to stop. Both of the properties this
design is built around — equal counts, and a barrier before shared writes — are
there because the failure is a *hang* rather than an error, and a hang produces
no diagnosis at all.

And finally: **nothing here has been executed.** Every number in this document
is either from the CPU/MPS machine or from a simulation. Treat §5 as a list of
open questions with predictions attached, not as a report.
