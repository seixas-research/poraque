# -*- coding: utf-8 -*-
# file: reporting.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Human-readable formatting of calculator output.

The helpers in this module turn the internal :class:`~poraque.core.Grid`,
:class:`~poraque.core.System`, the per-iteration convergence metrics, and the
final energy decomposition into clean blocks of text that the engines and the
ASE calculator print to standard output. Keeping all of the formatting here
means the scientific code never has to worry about layout, and the layout stays
consistent between the orbital-free and Kohn-Sham drivers.
"""

import numpy as np

from .units import BOHR_TO_ANGSTROM

try:  # ASE is a runtime dependency, but keep formatting import-safe.
    from ase.data import chemical_symbols
except Exception:  # pragma: no cover - exercised only without ASE installed
    chemical_symbols = None


_RULE = "=" * 70
_THIN = "-" * 70


def format_grid(grid):
    """
    Render the real-space / plane-wave grid parameters.

    Parameters
    ----------
    grid : Grid
        The grid to describe.

    Returns
    -------
    str
        A multi-line block listing the grid shape, spacing, cell volume, and the
        plane-wave cutoff used to build it (when available).
    """
    lines = [_RULE, "Plane-wave / real-space grid", _THIN]
    lines.append(f"  basis            : {getattr(grid, 'basis', 'plane waves')}")
    lines.append(f"  shape (Nx Ny Nz) : {grid.Nx} x {grid.Ny} x {grid.Nz} "
                 f"({grid.N} points)")
    h = np.asarray(grid.h, dtype=float)
    lines.append(f"  spacing (Bohr)   : {h[0]:.4f} {h[1]:.4f} {h[2]:.4f}")
    lines.append(f"  cell volume      : {grid.volume:.4f} Bohr^3")
    lines.append(f"  volume element   : {grid.volume_element:.6e} Bohr^3")
    if getattr(grid, "ecut", None) is not None:
        lines.append(f"  plane-wave Ecut  : {grid.ecut:.4f} Hartree "
                     f"({grid.ecut * 27.211386245988:.2f} eV)")
    else:
        lines.append("  plane-wave Ecut  : (set from explicit grid shape)")
    lines.append(_RULE)
    return "\n".join(lines)


def format_system(system):
    """
    Render the material structure: cell vectors, atoms, and boundary conditions.

    Parameters
    ----------
    system : System
        The atomic structure to describe (internal atomic units / Bohr).

    Returns
    -------
    str
        A multi-line block with the lattice vectors, periodic boundary
        conditions, electron count, and the fractional/Cartesian positions of
        every ion.
    """
    lines = [_RULE, "Material structure", _THIN]
    cell = np.asarray(system.cell, dtype=float)
    lines.append("  unit-cell vectors (Bohr | Angstrom):")
    for i, label in enumerate("abc"):
        b = cell[i]
        a = b * BOHR_TO_ANGSTROM
        lines.append(f"    {label} = [{b[0]:9.4f} {b[1]:9.4f} {b[2]:9.4f}] | "
                     f"[{a[0]:8.4f} {a[1]:8.4f} {a[2]:8.4f}]")
    lines.append(f"  periodic boundary conditions : {tuple(bool(p) for p in system.pbc)}")
    lines.append(f"  electrons (explicit)         : {system.electrons}")
    lines.append(f"  atoms                        : {len(system.atomic_numbers)}")
    lines.append(_THIN)
    lines.append("  #   El      x (Bohr)   y (Bohr)   z (Bohr)")
    for idx, (z, pos) in enumerate(zip(system.atomic_numbers, system.positions)):
        sym = chemical_symbols[int(z)] if chemical_symbols is not None else str(int(z))
        lines.append(f"  {idx:<3d} {sym:<3s}  {pos[0]:10.4f} {pos[1]:10.4f} {pos[2]:10.4f}")
    lines.append(_RULE)
    return "\n".join(lines)


def scf_header(method):
    """Return the header line for a self-consistent-field convergence table."""
    lines = [_RULE, f"{method} self-consistent field", _THIN,
             "  iter        total energy (Ha)        residual",
             _THIN]
    return "\n".join(lines)


def scf_step(iteration, energy, residual, extra=None):
    """
    Format one line of the SCF/minimization convergence table.

    Parameters
    ----------
    iteration : int
        Zero-based iteration index (printed one-based).
    energy : float
        Total energy at this step (Hartree).
    residual : float
        Convergence metric (density residual or gradient norm).
    extra : str, optional
        Additional trailing text (e.g. the chemical potential or step length).

    Returns
    -------
    str
        A single formatted table row.
    """
    line = f"  {iteration + 1:>4d}   {energy:>22.10f}   {residual:>14.3e}"
    if extra:
        line += f"   {extra}"
    return line


# The canonical order in which energy terms are reported, when present.
_DECOMP_ORDER = [
    "Kinetic",
    "External",
    "Hartree",
    "XC",
    "Nonlocal",
    "Ion-Ion",
]


def format_energy_decomposition(total_energy, components, converged=None,
                                iterations=None):
    """
    Render the final energy-accounting breakdown.

    The total energy is printed first, followed by every physical contribution:
    the (non-interacting) kinetic energy, the external/ionic potential energy,
    the Hartree (electrostatic) energy, the exchange-correlation energy, and any
    nonlocal pseudopotential or correction terms. Components not in the canonical
    order are appended afterwards so nothing is silently dropped.

    Parameters
    ----------
    total_energy : float
        Total energy (Hartree).
    components : dict
        Mapping of contribution name to energy (Hartree).
    converged : bool, optional
        Whether the calculation converged (printed when given).
    iterations : int, optional
        Number of iterations performed (printed when given).

    Returns
    -------
    str
        A multi-line, aligned energy table.
    """
    lines = [_RULE, "Energy decomposition (Hartree)", _THIN]
    lines.append(f"  {'Total energy':<34s} {total_energy:>22.10f}")
    lines.append(_THIN)

    shown = set()
    for key in _DECOMP_ORDER:
        if key in components:
            lines.append(f"  {key:<34s} {components[key]:>22.10f}")
            shown.add(key)
    for key, value in components.items():
        if key in shown:
            continue
        try:
            lines.append(f"  {key:<34s} {float(value):>22.10f}")
        except (TypeError, ValueError):
            continue

    if converged is not None or iterations is not None:
        lines.append(_THIN)
        if iterations is not None:
            lines.append(f"  iterations                         {iterations:>22d}")
        if converged is not None:
            lines.append(f"  converged                          {str(bool(converged)):>22s}")
    lines.append(_RULE)
    return "\n".join(lines)
