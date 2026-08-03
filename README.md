<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/poraque" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/assets/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/assets/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/assets/logo/logo_light.png" style="height: auto; width: auto; max-height: 100px; " alt="Poraquê logo">
  </picture>
</a>
</h1>

[![License: MIT](https://img.shields.io/github/license/seixas-research/poraque?color=green&style=for-the-badge)](LICENSE)

# Poraquê

Poraquê is a research code for electronic structure and machine learning on
3D scalar fields (external potential, charge density, kinetic energy density).

## Status

| Area | State |
| --- | --- |
| `poraque.fields` | 3D scalar fields (`EXTCAR`/`CHGCAR`/`TAUCAR`) on one shared grid |
| `poraque.fields.io` | Pluggable ingestion — VASP working; Quantum ESPRESSO and GPAW scaffolded |
| `poraque.ml` | Fourier Neural Operator pipeline, handles per-material grid shapes |
| Everything else | Legacy; being reorganized |

## Scripts

```bash
# Validate the external-potential model against reference VASP EXTCAR files
python scripts/validate_vasp_data.py --fit-sigma --form-factor

# Train the two neural operators (EXTCAR -> CHGCAR, CHGCAR -> TAUCAR)
python scripts/train_fno.py --resolution 32 --epochs 200
```

## Notes

- `plan/pi_fno.md` — roadmap for the physics-informed operator (PI-FNO).
- `plan/fno_physics.md` — how KS-DFT and OF-DFT relate to what the two models learn.

## Installation

```bash
git clone https://github.com/seixas-research/poraque.git
cd poraque
pip install -e .
```

## License

Open source under the [MIT License](LICENSE).
