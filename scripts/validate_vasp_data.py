#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: validate_vasp_data.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Validate Poraquê's field machinery against real VASP data.

For every ``struct*`` directory under ``data/vasp`` this script

1. reads the reference ``EXTCAR`` and adopts **its** grid as authoritative,
   so the comparison is strictly point-by-point;
2. recomputes the local external potential with
   :class:`~poraque.fields.ExternalPotential` and reports MAE, RMSE, relative
   :math:`L^2`, Pearson :math:`r` and the extreme values;
3. optionally fits the effective pseudo-ion width :math:`\sigma` and extracts
   the **empirical form factor** :math:`f(G)` by inverting the reference
   potential, which diagnoses *why* the two differ rather than only *how much*;
4. writes ``EXTCAR_NEW`` **only when the difference is significant** — if the
   agreement is within ``--write-tolerance`` the file is skipped, since a
   duplicate of ``EXTCAR`` is worse than useless;
5. runs an integrity check on ``TAUCAR`` (and ``CHGCAR``): grid agreement,
   finiteness, sign, and the integrated totals.

Usage
-----
::

    python scripts/validate_vasp_data.py
    python scripts/validate_vasp_data.py --root data/vasp --fit-sigma
    python scripts/validate_vasp_data.py --write-tolerance 0.02 --force-write
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poraque.fields import (  # noqa: E402
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    KineticEnergyDensity,
)
from poraque.fields.constants import COULOMB_CONSTANT_EV_ANGSTROM  # noqa: E402
from poraque.fields.external import _structure_factor  # noqa: E402
from poraque.fields.io import resolve_reader  # noqa: E402


# ===================================================================== #
# Comparison helpers
# ===================================================================== #
def compare_fields(predicted, reference):
    """
    Point-by-point comparison of two fields on the same grid.

    Both are compared *as stored*; the external potential already has zero
    cell-average by construction, so no alignment constant is removed. A
    separate ``offset`` entry reports the mean difference in case one of the
    two does carry a shift.

    Parameters
    ----------
    predicted, reference : numpy.ndarray
        Arrays of identical shape.

    Returns
    -------
    dict
        Error metrics in the units of the fields.
    """
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    difference = predicted - reference

    reference_norm = np.linalg.norm(reference)
    denominator = np.ptp(reference) or 1.0

    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / reference_norm)
        if reference_norm else float("nan"),
        "nrmse_range": float(np.sqrt(np.mean(difference ** 2)) / denominator),
        "pearson_r": float(np.corrcoef(predicted.ravel(), reference.ravel())[0, 1]),
        "offset": float(np.mean(difference)),
        "predicted_min": float(predicted.min()),
        "predicted_max": float(predicted.max()),
        "reference_min": float(reference.min()),
        "reference_max": float(reference.max()),
    }


def fit_effective_sigma(structure, grid, charges, reference, bounds=(0.15, 2.5)):
    """
    Find the Gaussian pseudo-ion width that best reproduces a reference field.

    Answers a question the raw error metrics cannot: is the disagreement a
    *shape* mismatch, or merely a badly chosen softening length? A small
    residual at the optimum means the Gaussian model is the right functional
    form and only its width was off.

    Parameters
    ----------
    structure : Structure
        Geometry.
    grid : FieldGrid
        Shared mesh.
    charges : dict
        ``{element: Z_val}``.
    reference : numpy.ndarray
        Reference potential on ``grid``.
    bounds : tuple of float, optional
        Search interval for :math:`\\sigma` in Å.

    Returns
    -------
    dict
        ``sigma``, ``relative_l2`` at the optimum, and the number of
        evaluations.
    """
    from scipy.optimize import minimize_scalar

    reference_norm = np.linalg.norm(reference)
    calls = {"n": 0}

    def objective(sigma):
        calls["n"] += 1
        elements = set(structure.elements)
        field = ExternalPotential.compute(
            structure, grid, charges,
            widths={element: float(sigma) for element in elements},
        )
        return float(np.linalg.norm(field.data - reference) / reference_norm)

    result = minimize_scalar(objective, bounds=bounds, method="bounded",
                             options={"xatol": 1e-3})
    return {"sigma": float(result.x), "relative_l2": float(result.fun),
            "evaluations": calls["n"]}


def empirical_form_factor(structure, grid, charges, reference, n_bins=60,
                          min_structure_factor=1e-3):
    r"""
    Recover the local pseudopotential form factor from a reference potential.

    Inverting the construction used by :class:`ExternalPotential`,

    .. math::

        f(\mathbf G) \;=\; -\,\frac{\Omega\, G^2\, V_{\rm ref}(\mathbf G)}
                                   {4\pi e^2\, Z\, S(\mathbf G)} ,

    which for a **single-species** cell is exact and needs no fitting. If the
    result depends only on :math:`|\mathbf G|` the reference is describable by
    an isotropic local pseudopotential — and the recovered curve *is* that
    pseudopotential's form factor, directly comparable with the Gaussian
    :math:`e^{-G^2\sigma^2/2}` the model assumes.

    Parameters
    ----------
    structure : Structure
        Geometry; must contain exactly one species.
    grid : FieldGrid
        Shared mesh.
    charges : dict
        ``{element: Z_val}``.
    reference : numpy.ndarray
        Reference potential on ``grid``.
    n_bins : int, optional
        Number of ``|G|`` bins.
    min_structure_factor : float, optional
        Discard modes where ``|S(G)|`` is too small to invert stably.

    Returns
    -------
    dict or None
        Binned ``g``, ``f_mean``, ``f_std``, ``count``, plus the isotropy
        diagnostic ``scatter``. ``None`` for multi-species cells.
    """
    if len(structure.symbols) != 1:
        return None

    element = structure.elements[0]
    charge = float(charges[element])

    reference_g = np.fft.fftn(reference) / grid.npoints    # Fourier coefficients
    structure_factor = _structure_factor(grid, structure.scaled_positions)
    g2 = grid.get_g2()

    prefactor = -4.0 * np.pi * COULOMB_CONSTANT_EV_ANGSTROM * charge / grid.volume
    usable = (g2 > 1e-8) & (np.abs(structure_factor) > min_structure_factor)

    form = np.real(
        reference_g[usable] * g2[usable]
        / (prefactor * structure_factor[usable])
    )
    magnitude = np.sqrt(g2[usable])

    edges = np.linspace(0.0, magnitude.max(), n_bins + 1)
    index = np.clip(np.digitize(magnitude, edges) - 1, 0, n_bins - 1)

    centres, means, deviations, counts = [], [], [], []
    for b in range(n_bins):
        selected = form[index == b]
        if selected.size < 4:
            continue
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        means.append(float(selected.mean()))
        deviations.append(float(selected.std()))
        counts.append(int(selected.size))

    means_array = np.asarray(means)
    deviations_array = np.asarray(deviations)
    significant = np.abs(means_array) > 0.05
    scatter = (float(np.mean(deviations_array[significant]
                             / np.abs(means_array[significant])))
               if significant.any() else float("nan"))

    return {
        "g": centres,
        "f_mean": means,
        "f_std": deviations,
        "count": counts,
        "scatter": scatter,
    }


def gaussian_sigma_from_form_factor(form_factor, g_max=6.0):
    r"""
    Least-squares :math:`\sigma` from :math:`\ln f = -G^2\sigma^2/2`.

    Fitting in the log domain over the low-``G`` region — where the form factor
    is well determined and closest to Gaussian — gives an independent estimate
    of the width to cross-check :func:`fit_effective_sigma`.

    Returns
    -------
    float or None
        Best-fit width in Å, or ``None`` if too few usable points.
    """
    g = np.asarray(form_factor["g"])
    f = np.asarray(form_factor["f_mean"])
    usable = (g > 0) & (g < g_max) & (f > 1e-3)
    if usable.sum() < 4:
        return None
    # ln f = -(sigma^2/2) G^2  ->  slope of ln f against G^2.
    slope = np.polyfit(g[usable] ** 2, np.log(f[usable]), 1)[0]
    return float(np.sqrt(-2.0 * slope)) if slope < 0 else None


# ===================================================================== #
# Per-structure driver
# ===================================================================== #
def validate_structure(directory, args):
    """Run every check for one calculation directory."""
    name = os.path.basename(os.path.normpath(directory))
    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
    report = {"name": name, "directory": directory}

    reader = resolve_reader(directory, args.code)
    structure = reader.read_structure(directory)
    parameters = reader.read_parameters(directory)
    pseudopotentials = reader.read_pseudopotentials(directory)
    charges = reader.valence_charges(directory)

    print(f"  code            : {reader.code}")
    print(f"  structure       : {structure.formula}  V = {structure.volume:.3f} A^3")
    print(f"  cutoff / prec   : {parameters.cutoff} eV / {parameters.precision}")
    for element, info in pseudopotentials.items():
        print(f"  pseudopotential : {element}  ZVAL = {info.valence_charge:g}  "
              f"RCORE = {info.core_radius:.4f} A")

    report.update({
        "formula": structure.formula,
        "volume": structure.volume,
        "natoms": structure.natoms,
        "cutoff": parameters.cutoff,
        "precision": parameters.precision,
        "valence_charges": charges,
    })

    # ---------------- reference EXTCAR ---------------- #
    extcar_path = reader.field_path(directory, "external")
    if not os.path.exists(extcar_path):
        print("  !! no reference EXTCAR; nothing to compare against")
        return report

    start = time.time()
    reference = ExternalPotential.read(extcar_path)
    grid = reference.grid
    print(f"  reference EXTCAR: shape {grid.shape}  read in {time.time() - start:.1f} s")
    print(f"                    mean {reference.mean():+.3e} eV   "
          f"range [{reference.data.min():.3f}, {reference.data.max():.3f}] eV")

    # A non-zero mean would mean a different G=0 convention from ours.
    report["reference_mean"] = reference.mean()
    report["grid_shape"] = list(grid.shape)

    derived = FieldGrid.from_parameters(structure, parameters, pseudopotentials)
    report["grid_from_encut"] = list(derived.shape)
    if derived.shape != grid.shape:
        print(f"  note: grid derived from ENCUT would be {derived.shape}; "
              f"using the reference grid {grid.shape} for the comparison")

    # ---------------- our EXTCAR ---------------- #
    start = time.time()
    computed = ExternalPotential.from_calculation(directory, code=reader.code,
                                                  grid=grid,
                                                  rcore_factor=args.rcore_factor)
    elapsed = time.time() - start
    sigma = computed.metadata["widths"]
    print(f"  computed EXTCAR : sigma = "
          + ", ".join(f"{k} {v:.4f} A" for k, v in sigma.items())
          + f"   in {elapsed:.2f} s")
    print(f"                    mean {computed.mean():+.3e} eV   "
          f"range [{computed.data.min():.3f}, {computed.data.max():.3f}] eV")

    metrics = compare_fields(computed.data, reference.data)
    report["default_model"] = {"widths": sigma, "metrics": metrics,
                               "seconds": elapsed}

    print("  --- comparison (default sigma = rcore_factor * RCORE) ---")
    for key in ("mae", "rmse", "max_abs", "relative_l2", "nrmse_range",
                "pearson_r", "offset"):
        print(f"      {key:<14s} {metrics[key]:+.6g}")

    best = computed
    best_metrics = metrics

    # ---------------- optional sigma fit ---------------- #
    if args.fit_sigma:
        start = time.time()
        fit = fit_effective_sigma(structure, grid, charges, reference.data)
        print(f"  --- best-fit Gaussian width ({fit['evaluations']} evaluations, "
              f"{time.time() - start:.1f} s) ---")
        print(f"      sigma*         {fit['sigma']:.4f} A")
        print(f"      relative_l2    {fit['relative_l2']:.6g}   "
              f"(default: {metrics['relative_l2']:.6g})")
        report["fitted_sigma"] = fit

        fitted = ExternalPotential.compute(
            structure, grid, charges,
            widths={e: fit["sigma"] for e in set(structure.elements)},
        )
        fitted_metrics = compare_fields(fitted.data, reference.data)
        report["fitted_model"] = {"sigma": fit["sigma"], "metrics": fitted_metrics}
        for key in ("mae", "rmse", "max_abs", "pearson_r"):
            print(f"      {key:<14s} {fitted_metrics[key]:+.6g}")
        if fitted_metrics["relative_l2"] < best_metrics["relative_l2"]:
            best, best_metrics = fitted, fitted_metrics

    # ---------------- form-factor diagnosis ---------------- #
    if args.form_factor:
        form = empirical_form_factor(structure, grid, charges, reference.data)
        if form is None:
            print("  --- form factor: skipped (multi-species cell) ---")
        else:
            report["form_factor"] = form
            print("  --- empirical form factor f(G) from the reference ---")
            print(f"      isotropy scatter (std/|mean| per |G| bin): {form['scatter']:.4f}")
            print(f"      {'|G| [1/A]':>10s} {'f_emp':>10s} {'gaussian':>10s}")
            width = float(np.mean(list(sigma.values())))
            for g, f in list(zip(form["g"], form["f_mean"]))[:12]:
                print(f"      {g:10.3f} {f:10.4f} "
                      f"{np.exp(-0.5 * g ** 2 * width ** 2):10.4f}")
            log_sigma = gaussian_sigma_from_form_factor(form)
            report["form_factor_sigma"] = log_sigma
            if log_sigma is not None:
                print(f"      sigma from log-slope fit: {log_sigma:.4f} A")

    # ---------------- write EXTCAR_NEW only if different ---------------- #
    target = os.path.join(directory, "EXTCAR_NEW")
    significant = best_metrics["relative_l2"] > args.write_tolerance
    report["write_tolerance"] = args.write_tolerance
    report["significant_difference"] = bool(significant)

    if significant or args.force_write:
        best.write(target, comment=f"{structure.formula}  EXTCAR_NEW "
                                   f"(poraque local external potential) [eV]")
        report["written"] = target
        print(f"  -> wrote {target}")
        print(f"     (relative L2 = {best_metrics['relative_l2']:.4f} exceeds the "
              f"{args.write_tolerance:.4f} tolerance)")
    else:
        report["written"] = None
        if os.path.exists(target):
            os.remove(target)
            print(f"  -> removed a stale {target}")
        print(f"  -> EXTCAR_NEW not written: agreement within tolerance "
              f"(relative L2 = {best_metrics['relative_l2']:.4f} "
              f"<= {args.write_tolerance:.4f})")

    # ---------------- TAUCAR / CHGCAR integrity ---------------- #
    report["fields"] = {}
    for kind, field_class, positive in (("density", ChargeDensity, True),
                                        ("kinetic", KineticEnergyDensity, True)):
        path = reader.field_path(directory, kind)
        label = os.path.basename(path)
        if not os.path.exists(path):
            print(f"  {label:<8s}: MISSING")
            report["fields"][kind] = {"status": "missing"}
            continue

        start = time.time()
        field = field_class.read(path, grid=grid)      # grid mismatch raises
        statistics = field.statistics()
        finite = bool(np.isfinite(field.data).all())
        negative = int(np.count_nonzero(field.data < 0))

        checks = {
            "status": "ok",
            "shape": list(field.shape),
            "grid_matches_extcar": True,
            "finite": finite,
            "negative_points": negative,
            "negative_fraction": negative / field.data.size,
            "seconds": time.time() - start,
            **statistics,
        }
        report["fields"][kind] = checks

        print(f"  {label:<8s}: shape {tuple(field.shape)}  grid OK  "
              f"finite {finite}  read {checks['seconds']:.1f} s")
        print(f"            min {statistics['min']:.6g}  max {statistics['max']:.6g}  "
              f"mean {statistics['mean']:.6g}")
        if kind == "density":
            print(f"            integral = {statistics['integral']:.4f} electrons "
                  f"(expected {sum(charges[e] * c for e, c in zip(structure.elements, structure.counts)):.1f})")
        else:
            print(f"            integral = {statistics['integral']:.4f} eV  (total T_s)")
        if positive and negative:
            print(f"            !! {negative} negative points "
                  f"({100 * negative / field.data.size:.3f}%)")

    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", default="data/vasp",
                        help="directory holding the struct* folders")
    parser.add_argument("--pattern", default="struct",
                        help="prefix identifying calculation folders")
    parser.add_argument("--code", default="auto",
                        help="DFT code name, or 'auto' to detect")
    parser.add_argument("--rcore-factor", type=float, default=0.5,
                        help="Gaussian width as a multiple of the core radius")
    parser.add_argument("--write-tolerance", type=float, default=0.01,
                        help="relative L2 below which EXTCAR_NEW is NOT written")
    parser.add_argument("--force-write", action="store_true",
                        help="write EXTCAR_NEW even when within tolerance")
    parser.add_argument("--fit-sigma", action="store_true",
                        help="fit the effective Gaussian width to the reference")
    parser.add_argument("--form-factor", action="store_true",
                        help="extract the empirical form factor f(G)")
    parser.add_argument("--json", default=None, help="write the full report to JSON")
    args = parser.parse_args(argv)

    directories = sorted(
        os.path.join(args.root, entry) for entry in os.listdir(args.root)
        if entry.startswith(args.pattern)
        and os.path.isdir(os.path.join(args.root, entry))
    )
    if not directories:
        parser.error(f"no {args.pattern}* directories under {args.root!r}")

    print(f"Validating {len(directories)} calculation(s) under {args.root!r}")
    reports = [validate_structure(directory, args) for directory in directories]

    # ------------------------------ summary ------------------------------ #
    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    header = f"{'structure':<14s} {'grid':>14s} {'rel L2':>9s} {'MAE [eV]':>10s} {'r':>8s}  EXTCAR_NEW"
    print(header)
    print("-" * len(header))
    for report in reports:
        model = report.get("fitted_model") or report.get("default_model")
        if model is None:
            print(f"{report['name']:<14s} {'-':>14s}")
            continue
        metrics = model["metrics"]
        print(f"{report['name']:<14s} {str(tuple(report['grid_shape'])):>14s} "
              f"{metrics['relative_l2']:9.4f} {metrics['mae']:10.4f} "
              f"{metrics['pearson_r']:8.4f}  "
              f"{'written' if report.get('written') else 'skipped (similar)'}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(reports, handle, indent=2, default=float)
        print(f"\nfull report -> {args.json}")

    return reports


if __name__ == "__main__":
    main()
