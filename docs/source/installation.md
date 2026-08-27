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
| `poraque-vasp` | writes VASP inputs that read a prediction back — `bands`, `dos`, `energy` | `python scripts/poraque_vasp.py` |

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
| CUDA build of PyTorch | NVIDIA GPUs | see [pytorch.org](https://pytorch.org) |
| `pdflatex` / `latexmk` | automatic PDF reports | TeX Live or MacTeX |
| PySR + sympy | [symbolic distillation](symbolic/index.md) | `pip install -e ".[symbolic]"` |

The `symbolic` extra is kept separate because PySR carries a Julia toolchain,
which it downloads the first time a search runs. Everything else in the package
works without it.

Apple Silicon needs nothing extra: the Metal (MPS) backend ships with the
standard PyTorch wheel and is selected automatically.

## Verifying the installation

```bash
pytest
python -c "from poraque.ml.device import available_devices; print(available_devices())"
```

The device list is ordered by preference, so the first entry is what
`device: auto` will select.
