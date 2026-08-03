# Configuration

A run is defined by a YAML file with four sections mirroring the four things a
run needs. Command-line flags override individual entries, so one committed
config can be swept from the shell without being edited.

```bash
python scripts/train_fno.py --write-config configs/train_config.yaml
python scripts/train_fno.py --config configs/train_config.yaml --epochs 500
```

```{note}
Unknown keys are **rejected**, not ignored. A typo such as `learn_rate` would
otherwise be silently dropped and the run would quietly use the default,
producing results that do not match the file that appears to describe them.
```

The *resolved* configuration is written beside the results, recording what
actually ran, overrides included.

## `data`

| Key | Default | Meaning |
| --- | --- | --- |
| `root` | `data/vasp` | directory of per-material calculations |
| `resolution` | `32` | longest grid axis after spectral downsampling |
| `use_vasp_extcar` | `false` | read a reference `EXTCAR` instead of computing it |
| `gaussian_blur` | `null` | blur width in Å applied to the computed potential |
| `blur_method` | `spectral` | `spectral` or `ndimage` |

```{note}
`use_vasp_extcar` is off by default because standard VASP does not write that
file. Poraquê's reconstruction matches it to a relative $5\times10^{-5}$, so
the default costs essentially nothing.
```

## `model`

| Key | Default | Meaning |
| --- | --- | --- |
| `width` | `16` | channel width of the Fourier layers |
| `modes` | `8` | retained modes per axis (a *capacity* limit) |
| `n_layers` | `4` | number of Fourier layers |
| `mode_selection` | `fixed` | `fixed` index, or `physical` at constant $G_\max$ |
| `pauli_residual` | `false` | structural $\tau\ge\tau_\mathrm{vW}$ for `chg2tau` |

## `training`

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `universal` | one model on all structures, or `leave_one_out` |
| `holdout` | `null` | structures excluded from universal training |
| `epochs` | `200` | passes over the training set |
| `batch_size` | `4` | capped per shape bucket |
| `device` | `auto` | `auto`, `cuda`, `mps` or `cpu` |
| `physics` | all zero | weights of the physics-informed terms |

Physics weights default to zero, so the objective is the supervised baseline
until one is enabled deliberately.

## `output`

| Key | Default |
| --- | --- |
| `log`, `json` | `logs/…` |
| `checkpoint_dir` | `models` |
| `plot_dir` | `results/plots` |
| `report_dir` | `reports` |
