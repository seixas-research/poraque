# Installation

Poraquê requires **Python 3.11 or newer**.

Every module is verified against the 3.11 grammar. There is no upper bound —
newer interpreters are supported.

```bash
git clone https://github.com/seixas-research/poraque.git
cd poraque
pip install -e .
```

This pulls in NumPy, SciPy, ASE, pandas, Matplotlib and PyTorch.

## Console commands

Installing also registers the console commands, which run from **any** directory once
the environment is active — no path to a script, no `cd` into the repository:

| Command | Does | Equivalent |
| --- | --- | --- |
| `poraque-train` | trains the operators | `python scripts/poraque_train.py` |
| `poraque-inference` | predicts a new structure | `python scripts/poraque_inference.py` |
| `poraque-committee` | ranks structures by ensemble disagreement | `python scripts/poraque_committee.py` |
| `poraque-vasp` | writes VASP inputs that read a prediction back — `bands`, `dos`, `energy`, and `chgcar` to write a field store out as a `CHGCAR` | `python scripts/poraque_vasp.py` |

Each is the `main()` of the corresponding script, so the two forms take
identical arguments and behave identically. Relative paths in a configuration
file (`data/vasp`, `models/`) are still resolved against the **working
directory**, so run these from the project root, or give absolute paths.

```bash
poraque-train --help
poraque-inference --help
poraque-committee --help
poraque-vasp --help
```

If the commands are not found after an upgrade, re-run `pip install -e .`:
entry points are registered at install time, not import time.

## Optional components

| Component | Needed for | Install |
| --- | --- | --- |
| CUDA build of PyTorch | NVIDIA GPUs | [see below](nvidia-gpus) |
| `pdflatex` / `latexmk` | automatic PDF reports | TeX Live or MacTeX |
| `sympy` | [symbolic distillation](symbolic/index.md) | `pip install -e ".[symbolic]"` |

The `symbolic` extra is one package now, and it is only for *reporting*: SymPy
renders a distilled expression as LaTeX and checks its asymptotic limits. The
search itself is `poraque.ml.gp` — genetic programming in NumPy, with SciPy
fitting the constants — and both of those are already hard dependencies, so
nothing about distillation needs a second install any more.

Until 2026-09-03 this extra carried PySR and, through it, a **Julia
toolchain** fetched on first use. That was the reason it was kept separate, and
removing it was an HPC decision rather than a scientific one: on a
supercomputer a first-use download is a network fetch from a compute node, a
writable depot on a filesystem that may be read-only or purged between jobs, a
precompilation pass per architecture, and a second language runtime inside an
MPI job.

Apple Silicon needs nothing extra: the Metal (MPS) backend ships with the
standard PyTorch wheel and is selected automatically.

(nvidia-gpus)=
## NVIDIA GPUs

`pyproject.toml` asks for `torch>=2.0` and deliberately says nothing about
CUDA: which build is right depends on the machine, and pinning one here would
break Apple Silicon, which wants the newest torch. But the choice has to be
made somewhere, because **the wrong one does not raise — it gives a silent CPU
run**.

Two things have to line up:

* **the driver**, which must be new enough for the wheel's CUDA runtime
  (`nvidia-smi` prints the highest CUDA version it supports, top right);
* **the GPU architecture**, which must appear in
  `torch.cuda.get_arch_list()`.

| GPU | capability | wheel |
| --- | --- | --- |
| P100 | sm_60 | `cu118` / `cu121` |
| **V100** | **sm_70** | **`cu126`** (CUDA 13 dropped Volta) |
| T4, RTX 20xx | sm_75 | `cu126` or newer |
| A100 | sm_80 | any wheel ≥ `cu118` |
| RTX 30xx | sm_86 | any wheel ≥ `cu118` |
| H100 | sm_90 | `cu121` or newer |
| B200, RTX 50xx | sm_100 / sm_120 | `cu128` or newer |

For a V100:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install poraque
```

The second row of that table is not a hypothetical. `pip install poraque` into
a clean environment on Santos Dumont brought `torch 2.13.0+cu130`, which failed
twice over: the driver (560.35.03, CUDA 12.6) was too old for a `cu130` runtime,
and `sm_70` is not in that build's `arch_list` at all, because CUDA 13 dropped
Maxwell, Pascal and Volta. Even with a newer driver the V100 would have had no
kernel to run — `no kernel image is available for execution on the device`, at
the first forward pass, after the queue time had been spent.

### Check before submitting anything to a queue

```bash
python -m poraque.ml.device --check --device cuda
```

It prints the torch build, its CUDA runtime, the architectures it carries
kernels for, and one line per GPU saying whether this build can use it — then
exits non-zero if the requested device is not usable. One line at the top of a
job script costs three seconds:

```bash
python -m poraque.ml.device --check --device cuda || exit 1
```

`--device` defaults to `auto`, and `--check` refuses a CPU under `auto` too —
a guard that cannot fail is not a guard, and this one could not: it resolved
to the CPU and exited 0 on any machine with a working CPU, which is every
machine. Naming the device you actually asked the scheduler for is still the
clearer form, because it is the one that also catches `cuda:2` on a node where
the job was given one GPU.

The same check belongs in the configuration, where it stops the run rather than
the script:

```yaml
training:
  device: cuda
  strict_device: true    # abort instead of falling back to the CPU
```

Without it, a job that cannot reach its GPU takes its place in the queue, warns
into a log nobody reads until afterwards, and trains on the CPU *inside* the
GPU allocation until the wall clock ends.

### `~/.local` outranks the active environment

A `torch` installed in `~/.local/lib/pythonX.Y/site-packages` **wins** over the
one in an active conda environment or venv. On a shared cluster that is easy to
acquire by accident and hard to notice: the environment a job activated is not
necessarily the one it ran.

```bash
export PYTHONNOUSERSITE=1
```

Poraquê's start-up banner prints torch's version, its CUDA build **and its
install directory** for exactly this reason, so a mismatch appears in the run's
own log instead of becoming a four-hour CPU run.

(multi-gpu)=
### Several GPUs, under Slurm

Poraquê trains data-parallel over NCCL when the launcher describes a group, and
on one device when it does not. It cannot invent ranks: **the launcher decides
the topology and `training.distributed` decides only whether to believe it.**
The shipped submission script is `scripts/slurm/poraque_ddp.sbatch`:

```bash
sbatch scripts/slurm/poraque_ddp.sbatch configs/train.yaml
```

The line that matters is `--ntasks-per-node`, which must equal the number of
GPUs:

```bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4          # one task per GPU. Not optional.
#SBATCH --gres=gpu:4

srun poraque-train --config configs/train.yaml --device cuda \
                   --strict-device --distributed auto
```

Requesting four GPUs and launching **one** task is the failure to watch for.
It is not an error — it is a perfectly good single-GPU run — but it leaves
`SLURM_NTASKS` at 1 and looks, from inside the process, exactly like the
single-GPU run somebody asked for. The run log therefore prints the Slurm
variables it saw:

```
  ranks  : nccl rank 0/4 local_rank 0 on sdumont1234:24601 (launched by slurm)
             SLURM_JOB_ID = 4601
             SLURM_NTASKS = 4
           effective batch = 10 x 4 ranks = 40
```

`MASTER_ADDR` is the first host of `SLURM_STEP_NODELIST` and `MASTER_PORT` is
derived from `SLURM_JOB_ID`, so nothing is hard-coded and two jobs sharing a
node cannot collide on a port. `torchrun` is understood as well, and is
consulted *after* Slurm: a `torchrun` launch inside an allocation carries both
sets of variables and its own describe the group it actually formed.

Three things to know before reading a scaling number:

- **The effective batch size is `batch_size` × the world size.** A four-rank
  run at `batch_size: 10` steps on 40 samples and is not the same optimiser as
  the one-rank run it is being compared against.
- **Rank 0 alone writes** the checkpoint, the metrics, the figures and the PDF,
  and alone prints. Four ranks opening one log truncate each other's.
- **`num_workers` gets worse, not better, under DDP.** Each rank is already a
  process with a field cache of its own. One rank per GPU, cache on, no
  workers — see [`data.cache_in_memory`](cache-in-memory).

To bisect a result, run the same allocation single-GPU with `--distributed off`
and compare `seconds_per_epoch` in the metrics JSON.

```{note}
NCCL only. Gloo would work on CPU and would be a way to test the plumbing, but
it would also let a misconfigured job train distributed across 96 CPU cores at
a fraction of one GPU's speed — the same silent waste `strict_device` exists to
prevent. Without CUDA the group is refused with a warning and the run continues
on one device.

There is no `DataParallel` fallback either. Single-process multi-GPU is slower
than one GPU for a model of a few megabytes, it changes the effective batch
size silently, and offering it would mean a laptop run and a cluster run
differed in a way nothing recorded.
```

## Verifying the installation

```bash
pytest
python -m poraque.ml.device
python -c "from poraque.ml.device import available_devices; print(available_devices())"
```

Without `--check` this reports and exits 0 whatever it finds, which is what a
verification step wants: a CPU-only machine is a working installation, not a
failure. `--check` is the queue guard, and it refuses a CPU deliberately.

The device list is ordered by preference, so the first entry is what
`device: auto` will select. `pytest -m "not gpu"` skips the tests that need an
accelerator, rather than reporting them as passes on a machine that has none.
