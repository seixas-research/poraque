# -*- coding: utf-8 -*-
# file: density_repair.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Sanity and repair of a predicted density before a Harris–Foulkes evaluation.

A Fourier neural operator returns a band-limited field, and a band-limited
field rings: the interstitial and the vacuum of a predicted
:math:`\rho(\mathbf r)` dip slightly below zero, and its integral drifts
from the valence count the pseudopotentials fix. Neither defect is benign in
a fixed-density functional. Every term is at least linear in
:math:`\rho`, the Hartree energy is quadratic, the exchange energy takes
:math:`\rho^{4/3}`, and the Harris–Foulkes expression is stationary only
*about* the self-consistent density, so an :math:`O(\delta\rho)` charge
defect enters the total energy at first order rather than second.

This module is the one place those two defects are inspected and, on
request, removed, so the three callers that evaluate a predicted density
(the local Gamma evaluator, the PAW Bloch solver and the VASP benchmark's
reference diagnostic) make the same correction and report it the same way.
The correction is deliberately minimal and fully recorded:

1. **positivity** — the physically non-negative quantities are floored. For
   a scalar density that is :math:`\rho` itself; for a spin density it is
   the pair of eigen-densities :math:`n_\pm = (\rho \pm |\mathbf m|)/2`,
   never :math:`\rho` and :math:`\mathbf m` separately, so the repaired
   spin-density matrix stays positive semidefinite and the local
   polarization :math:`\mathbf m/\rho` is untouched wherever nothing was
   clipped;
2. **charge conservation** — the integral of the floored density is
   compared with the exact valence count :math:`N = \sum_s n_s Z_s`
   taken from the POTCARs (or ``NELECT``), and the defect is reported in
   electrons and relative to :math:`N`;
3. **normalization** — a single global factor :math:`N/\int\rho` rescales
   the density, the magnetization and every on-site PAW occupancy record
   together. One factor, because any shape change would be a different
   prediction, not a repair; the records too, because in PAW the electron
   count is :math:`\int\tilde n` plus a term linear in the occupancies, and
   scaling one without the other would move charge between the grid and the
   spheres.

The factor is kept on the result because a global rescaling is not free: it
shifts the Hartree and exchange–correlation potentials and therefore the
eigenvalues and the Fermi level of the Hamiltonian built from the repaired
density. A caller that wants to know how much of its energy came from the
repair has the raw count, the clipped charge and the factor to say so.

:meth:`~poraque.fields.ChargeDensity.normalized` and
:meth:`~poraque.fields.SpinDensity.normalized` do the same floor-and-rescale
on field objects for the inference path; this module works on plain arrays,
carries the PAW records along, and returns the diagnostics the Harris path
has to log instead of writing them into field metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from ..fields import FieldGrid
from .harris import integrate

__all__ = [
    "DensityDiagnostics",
    "DensityRepair",
    "inspect_density",
    "repair_density",
    "require_valid_density",
]


def _magnitude(magnetization: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
    """|m| on the grid for a scalar (collinear) or three-component field."""
    if magnetization is None:
        return None
    m = np.asarray(magnetization, dtype=np.float64)
    if m.shape == shape:
        return np.abs(m)
    if m.shape == (3, *shape):
        return np.linalg.norm(m, axis=0)
    raise ValueError(
        f"Magnetization of shape {m.shape} does not fit a grid of shape {shape}: "
        "expected a scalar m_z or three Cartesian components"
    )


@dataclass(frozen=True)
class DensityDiagnostics:
    """What is wrong with a predicted density, measured without changing it.

    ``negative_electrons`` is the charge below the floor, integrated; for a
    spin density it is the charge the two eigen-densities carry below the
    floor, which is what the repair would add back. ``charge_defect`` is
    :math:`\\int\\rho - N` on the **raw** density, signed, so a prediction
    that leaks electrons reads negative.
    """

    electrons: float
    raw_electrons: float
    charge_defect: float
    relative_charge_defect: float
    negative_electrons: float
    negative_voxel_fraction: float
    minimum: float
    floor: float
    spin: bool

    @property
    def nonnegative(self) -> bool:
        return self.negative_electrons == 0.0 and self.minimum >= self.floor

    def normalized(self, rtol: float = 1e-6, atol: float = 1e-8) -> bool:
        return bool(np.isclose(self.raw_electrons, self.electrons, rtol=rtol, atol=atol))

    def problems(self) -> list[str]:
        """Human-readable list of the defects, empty when there are none."""
        out = []
        if not self.nonnegative:
            what = "eigen-densities (rho +/- |m|)/2" if self.spin else "density"
            out.append(
                f"{what} fall below {self.floor:g} on {100 * self.negative_voxel_fraction:.3f} % "
                f"of the voxels (minimum {self.minimum:.3e} e/A^3, "
                f"{self.negative_electrons:.3e} e below the floor)"
            )
        if not self.normalized():
            out.append(
                f"integral {self.raw_electrons:.6f} e against the valence count "
                f"{self.electrons:.6f} e ({self.charge_defect:+.3e} e, "
                f"{100 * self.relative_charge_defect:+.4f} %)"
            )
        return out

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DensityRepair:
    """A repaired density, its companions, and exactly what was done to them.

    ``charge`` integrates to ``electrons`` to machine precision.
    ``clipped_electrons`` is the charge the floor added before rescaling;
    ``normalization_factor`` the single scale applied afterwards to the
    density, the magnetization and the records alike. ``input_electrons``
    is the integral of ``charge``, kept as the check the caller will log.
    The raw arrays are not stored here; the caller keeps its own copies.
    """

    charge: np.ndarray
    magnetization: np.ndarray | None
    records: tuple[np.ndarray, ...]
    diagnostics: DensityDiagnostics
    clipped_electrons: float
    normalization_factor: float
    input_electrons: float
    repaired: bool = field(default=True)

    @classmethod
    def unrepaired(cls, charge: np.ndarray, magnetization: np.ndarray | None,
                   records: Sequence[np.ndarray], diagnostics: DensityDiagnostics) -> "DensityRepair":
        """The same record for a density used as predicted: factor 1, nothing clipped.

        Lets a caller log one dictionary whether or not it repaired, with
        ``repaired`` saying which.
        """
        return cls(charge=np.asarray(charge, dtype=np.float64),
                   magnetization=None if magnetization is None else np.asarray(magnetization, dtype=np.float64),
                   records=tuple(np.asarray(record, dtype=np.float64) for record in records),
                   diagnostics=diagnostics, clipped_electrons=0.0, normalization_factor=1.0,
                   input_electrons=diagnostics.raw_electrons, repaired=False)

    @property
    def electrons(self) -> float:
        return self.diagnostics.electrons

    @property
    def raw_electrons(self) -> float:
        return self.diagnostics.raw_electrons

    def as_dict(self) -> dict:
        """The scalar record a report stores; no arrays."""
        return {
            "repaired": self.repaired,
            "raw_electrons": self.raw_electrons,
            "input_electrons": self.input_electrons,
            "electrons": self.electrons,
            "charge_defect": self.diagnostics.charge_defect,
            "relative_charge_defect": self.diagnostics.relative_charge_defect,
            "negative_electrons": self.diagnostics.negative_electrons,
            "negative_voxel_fraction": self.diagnostics.negative_voxel_fraction,
            "minimum_raw": self.diagnostics.minimum,
            "floor": self.diagnostics.floor,
            "clipped_electrons": self.clipped_electrons,
            "normalization_factor": self.normalization_factor,
        }


def inspect_density(charge: np.ndarray, grid: FieldGrid, electrons: float,
                    magnetization: np.ndarray | None = None,
                    floor: float = 0.0) -> DensityDiagnostics:
    """Measure the positivity and charge defects of a density; change nothing.

    Parameters
    ----------
    charge
        :math:`\\rho` in electrons/Å³ on ``grid``.
    grid
        The mesh, for the quadrature weight.
    electrons
        The exact valence count :math:`N` the density should integrate to.
    magnetization
        Scalar :math:`m_z` or ``(3, *shape)`` Cartesian components; with it
        the positivity check is on :math:`(\\rho \\pm |\\mathbf m|)/2`.
    floor
        The smallest value admitted. Zero by default; a positive floor is
        what a functional that divides by :math:`\\rho` needs.
    """
    rho = np.asarray(charge, dtype=np.float64)
    if rho.shape != grid.shape or not np.isfinite(rho).all():
        raise ValueError("Density must be a finite array on the grid")
    if not np.isfinite(electrons) or electrons <= 0:
        raise ValueError("The valence electron count must be finite and positive")
    if not np.isfinite(floor) or floor < 0:
        raise ValueError("The positivity floor must be finite and non-negative")
    magnitude = _magnitude(magnetization, grid.shape)
    if magnitude is not None and not np.isfinite(magnitude).all():
        raise ValueError("Magnetization must be finite")
    raw = integrate(rho, grid)
    if magnitude is None:
        lowest = rho
        below = np.maximum(floor - rho, 0.0)
    else:
        plus, minus = (rho + magnitude) / 2, (rho - magnitude) / 2
        lowest = np.minimum(plus, minus)
        below = np.maximum(floor - plus, 0.0) + np.maximum(floor - minus, 0.0)
    return DensityDiagnostics(
        electrons=float(electrons),
        raw_electrons=raw,
        charge_defect=raw - float(electrons),
        relative_charge_defect=(raw - float(electrons)) / float(electrons),
        negative_electrons=integrate(below, grid),
        negative_voxel_fraction=float(np.count_nonzero(lowest < floor)) / lowest.size,
        minimum=float(lowest.min()),
        floor=float(floor),
        spin=magnitude is not None,
    )


def require_valid_density(charge: np.ndarray, grid: FieldGrid, electrons: float,
                          magnetization: np.ndarray | None = None,
                          floor: float = 0.0, what: str = "density") -> DensityDiagnostics:
    """Inspect, and raise naming every defect if there is one.

    This is the strict path: a caller that was not told to repair gets an
    error that says what the repair would have done, instead of a bare
    "nonnegative and normalized" from the evaluator three frames deeper.
    """
    diagnostics = inspect_density(charge, grid, electrons, magnetization, floor)
    problems = diagnostics.problems()
    if problems:
        raise ValueError(
            f"The {what} is not nonnegative and normalized: " + "; ".join(problems)
            + ". Enable the explicit density repair to floor and rescale it."
        )
    return diagnostics


def repair_density(charge: np.ndarray, grid: FieldGrid, electrons: float,
                   magnetization: np.ndarray | None = None,
                   records: Sequence[np.ndarray] = (),
                   floor: float = 0.0) -> DensityRepair:
    """Floor the physically non-negative densities, then rescale to ``electrons``.

    The inputs are not modified; the result holds new arrays. The three
    quantities scale by one factor: the density, the magnetization and
    every record in ``records`` (one array per atom, any shape).

    For a spin density the floor acts on :math:`n_\\pm = (\\rho \\pm |\\mathbf m|)/2`;
    the repaired :math:`\\rho` is :math:`n_+ + n_-` and the magnetization is
    rescaled pointwise by :math:`(n_+ - n_-)/|\\mathbf m|` so its direction
    is kept and :math:`|\\mathbf m| \\le \\rho` holds afterwards. Where
    :math:`|\\mathbf m| = 0` it stays zero.

    Raises
    ------
    ValueError
        If the floored density integrates to nothing, so there is no
        charge to rescale.
    """
    diagnostics = inspect_density(charge, grid, electrons, magnetization, floor)
    rho = np.array(charge, dtype=np.float64, copy=True)
    m = None if magnetization is None else np.array(magnetization, dtype=np.float64, copy=True)
    magnitude = _magnitude(m, grid.shape)
    if magnitude is None:
        repaired = np.maximum(rho, floor)
    else:
        plus = np.maximum((rho + magnitude) / 2, floor)
        minus = np.maximum((rho - magnitude) / 2, floor)
        repaired = plus + minus
        scale = np.divide(plus - minus, magnitude, out=np.zeros_like(magnitude),
                          where=magnitude > 0)
        m *= scale
    clipped = integrate(repaired - rho, grid)
    count = integrate(repaired, grid)
    if count <= 0:
        raise ValueError("The floored density integrates to no positive charge; nothing to normalize")
    factor = float(electrons) / count
    repaired *= factor
    if m is not None:
        m *= factor
    scaled = tuple(np.asarray(record, dtype=np.float64) * factor for record in records)
    return DensityRepair(
        charge=repaired,
        magnetization=m,
        records=scaled,
        diagnostics=diagnostics,
        clipped_electrons=clipped,
        normalization_factor=factor,
        input_electrons=integrate(repaired, grid),
    )
