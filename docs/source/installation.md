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
| PySR + sympy | [symbolic distillation](symbolic/index.md) | `pip install -e ".[symbolic]"` |

The `symbolic` extra is kept separate because PySR carries a Julia toolchain,
which it downloads the first time a search runs. Everything else in the package
works without it.

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
python -m poraque.ml.device --check
```

It prints the torch build, its CUDA runtime, the architectures it carries
kernels for, and one line per GPU saying whether this build can use it — then
exits non-zero if the requested device is not usable. One line at the top of a
job script costs three seconds:

```bash
python -m poraque.ml.device --check || exit 1
```

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

### One GPU

Poraquê trains on a single device; request one. There is no
`DistributedDataParallel` path, so a job allocated four GPUs will see four and
use `cuda:0`. That is deliberate: for fields of this size the optimiser step is
tens of milliseconds on a model of a few megabytes, and the constraint is
getting data to it rather than the arithmetic — see
[`data.cache_in_memory`](configuration.md).

## Verifying the installation

```bash
pytest
python -m poraque.ml.device --check
python -c "from poraque.ml.device import available_devices; print(available_devices())"
```

The device list is ordered by preference, so the first entry is what
`device: auto` will select. `pytest -m "not gpu"` skips the tests that need an
accelerator, rather than reporting them as passes on a machine that has none.
