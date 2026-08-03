<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/poraque" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/logo/logo_light.png" style="height: auto; width: auto; max-height: 100px; " alt="Poraquê logo">
  </picture>
</a>
</h1>

[![License: MIT](https://img.shields.io/github/license/seixas-research/poraque?color=green&style=for-the-badge)](LICENSE)

# Poraquê

> **🚧 Under structural refactoring.**
>
> The project is being restructured. The public API, module layout, and
> documentation are all in flux and **may change without notice**. This README is
> a placeholder and will be rewritten once the new architecture settles.

Poraquê is a research code for electronic structure and machine learning on
3D scalar fields (external potential, charge density, kinetic energy density).

## Status

| Area | State |
| --- | --- |
| `poraque.fields` | Under construction — 3D field descriptors and VASP-format I/O |
| `poraque.ml` | Under construction — Fourier Neural Operator pipeline |
| Everything else | Legacy; being reorganized |

## Installation

```bash
git clone https://github.com/seixas-research/poraque.git
cd poraque
pip install -e .
```

## License

Open source under the [MIT License](LICENSE).
