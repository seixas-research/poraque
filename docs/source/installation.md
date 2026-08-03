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

## Optional components

| Component | Needed for | Install |
| --- | --- | --- |
| CUDA build of PyTorch | NVIDIA GPUs | see [pytorch.org](https://pytorch.org) |
| `pdflatex` / `latexmk` | automatic PDF reports | TeX Live or MacTeX |

Apple Silicon needs nothing extra: the Metal (MPS) backend ships with the
standard PyTorch wheel and is selected automatically.

## Verifying the installation

```bash
pytest
python -c "from poraque.ml.device import available_devices; print(available_devices())"
```

The device list is ordered by preference, so the first entry is what
`device: auto` will select.
