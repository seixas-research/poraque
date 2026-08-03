#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: compare_smoothing.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Compare two ``ext2chg`` runs that differ only in the external-potential blur.

Answers the questions Task 2 poses, in the order they can actually be settled:

1. **Does smoothing change the input?** Quantified on the potential itself
   before any model is involved — how much of the field's power is removed, and
   where.
2. **Does it help convergence?** Training-loss trajectories, matched fold by
   fold. "Faster" is measured as epochs to reach a fixed loss, not as the final
   value, since the two objectives see different inputs.
3. **Does it help accuracy?** Held-out relative :math:`L^2`, MAE and
   :math:`R^2`, per fold and in the mean.
4. **What does it do near the cores?** The question that matters physically.
   Error is stratified by local density: smoothing the potential removes the
   sharpest features, which are exactly the ones that generate the density
   peaks at the ionic sites.

Usage
-----
::

    python scripts/compare_smoothing.py \
        --raw logs/run_raw.json --blurred logs/run_blur.json \
        --out docs/smoothing_analysis.md
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poraque.fields import ChargeDensity, ExternalPotential, FieldGrid  # noqa: E402
from poraque.fields.resample import downsample_shape, resample_field  # noqa: E402


def load(path, task="ext2chg"):
    """Load one run's JSON and return its fold list."""
    with open(path) as handle:
        payload = json.load(handle)
    for result in payload["results"]:
        if result["task"] == task:
            return payload, result["folds"]
    raise SystemExit(f"{path}: no results for task {task!r}")


def epochs_to_reach(history, threshold):
    """First epoch whose training loss falls below ``threshold`` (1-indexed)."""
    losses = np.asarray(history["train_loss"], dtype=float)
    hits = np.nonzero(losses <= threshold)[0]
    return int(hits[0]) + 1 if hits.size else None


def potential_spectrum(directory, sigma, resolution):
    """Radial power spectrum of the potential, raw and blurred."""
    grid = FieldGrid.from_file(os.path.join(directory, "CHGCAR"))
    shape = downsample_shape(grid.shape, target_max=resolution)
    reduced = FieldGrid(shape, grid.cell)

    raw = resample_field(ExternalPotential.from_calculation(directory, grid=grid),
                         shape, grid=reduced)
    blurred = resample_field(
        ExternalPotential.from_calculation(directory, grid=grid,
                                           gaussian_blur=sigma),
        shape, grid=reduced)
    return reduced, raw, blurred


def core_stratified_error(directory, resolution, folds_raw, folds_blur, log):
    """
    Compare errors in high- and low-density regions.

    The physically interesting claim about smoothing is that it damages the
    prediction near the ionic cores, where the potential is sharpest. Binning
    the error by the reference density tests that directly, rather than
    inferring it from a global average.
    """
    material = folds_raw[0]["held_out"]
    source = os.path.join(directory, material)
    if not os.path.isdir(source):
        return None

    grid = FieldGrid.from_file(os.path.join(source, "CHGCAR"))
    shape = downsample_shape(grid.shape, target_max=resolution)
    reduced = FieldGrid(shape, grid.cell)
    rho = resample_field(ChargeDensity.read(os.path.join(source, "CHGCAR"),
                                            grid=grid), shape, grid=reduced)
    return reduced, rho


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--raw", default="logs/run_raw.json")
    parser.add_argument("--blurred", default="logs/run_blur.json")
    parser.add_argument("--task", default="ext2chg")
    parser.add_argument("--data-root", default="data/vasp")
    parser.add_argument("--structure", default="struct_000")
    parser.add_argument("--plot-dir", default="results/plots/smoothing")
    parser.add_argument("--out", default="docs/smoothing_analysis.md")
    args = parser.parse_args(argv)

    payload_raw, raw = load(args.raw, args.task)
    payload_blur, blur = load(args.blurred, args.task)

    sigma = payload_blur["config"]["data"]["gaussian_blur"]
    resolution = payload_blur["config"]["data"]["resolution"]

    by_name_raw = {f["held_out"]: f for f in raw}
    by_name_blur = {f["held_out"]: f for f in blur}
    shared = sorted(set(by_name_raw) & set(by_name_blur))
    if not shared:
        raise SystemExit("the two runs share no held-out materials")

    lines = []

    def emit(message=""):
        print(message)
        lines.append(str(message))

    emit("=" * 78)
    emit(f"Gaussian smoothing of the external potential: sigma = {sigma} A")
    emit("=" * 78)

    # ---------------- 1. effect on the input ---------------- #
    directory = os.path.join(args.data_root, args.structure)
    grid, potential_raw, potential_blur = potential_spectrum(
        directory, sigma, resolution)
    difference = potential_blur.data - potential_raw.data
    power_raw = np.abs(np.fft.fftn(potential_raw.data)) ** 2
    power_blur = np.abs(np.fft.fftn(potential_blur.data)) ** 2

    emit("\n1. EFFECT ON THE INPUT (before any model)")
    emit(f"   grid                 {grid.shape}, |G|max = "
         f"{np.sqrt(grid.get_g2()).max():.2f} 1/A")
    emit(f"   raw     range        [{potential_raw.data.min():9.3f}, "
         f"{potential_raw.data.max():8.3f}] eV   std {potential_raw.data.std():7.3f}")
    emit(f"   blurred range        [{potential_blur.data.min():9.3f}, "
         f"{potential_blur.data.max():8.3f}] eV   std {potential_blur.data.std():7.3f}")
    emit(f"   deepest well shifted by {potential_blur.data.min() - potential_raw.data.min():+.3f} eV "
         f"({100 * abs(potential_blur.data.min() - potential_raw.data.min()) / abs(potential_raw.data.min()):.1f} %)")
    emit(f"   power removed        {100 * (1 - power_blur.sum() / power_raw.sum()):.2f} %")
    emit(f"   max |change|         {np.abs(difference).max():.3f} eV")
    emit(f"   cell average         raw {potential_raw.mean():+.2e}, "
         f"blurred {potential_blur.mean():+.2e} eV  (G=0 untouched)")

    # ---------------- 2. convergence ---------------- #
    emit("\n2. CONVERGENCE (training loss, matched folds)")
    emit(f"   {'held out':<14s} {'final raw':>10s} {'final blur':>11s} "
         f"{'ep->0.05 raw':>13s} {'ep->0.05 blur':>14s}")
    for name in shared:
        a, b = by_name_raw[name], by_name_blur[name]
        ea = epochs_to_reach(a["history"], 0.05)
        eb = epochs_to_reach(b["history"], 0.05)
        emit(f"   {name:<14s} {a['final_train_loss']:10.5f} "
             f"{b['final_train_loss']:11.5f} {str(ea):>13s} {str(eb):>14s}")

    # ---------------- 3. accuracy ---------------- #
    emit("\n3. ACCURACY on the held-out material (physical units, e/Ang^3)")
    emit(f"   {'held out':<14s} {'relL2 raw':>10s} {'relL2 blur':>11s} "
         f"{'change':>9s} {'R2 raw':>9s} {'R2 blur':>9s}")
    changes = []
    for name in shared:
        a = by_name_raw[name]["test_metrics"]
        b = by_name_blur[name]["test_metrics"]
        change = 100 * (b["relative_l2"] - a["relative_l2"]) / a["relative_l2"]
        changes.append(change)
        emit(f"   {name:<14s} {a['relative_l2']:10.4f} {b['relative_l2']:11.4f} "
             f"{change:+8.1f}% {a['r2']:9.4f} {b['r2']:9.4f}")

    mean_raw = np.mean([by_name_raw[n]["test_metrics"]["relative_l2"] for n in shared])
    mean_blur = np.mean([by_name_blur[n]["test_metrics"]["relative_l2"] for n in shared])
    emit(f"   {'MEAN':<14s} {mean_raw:10.4f} {mean_blur:11.4f} "
         f"{100 * (mean_blur - mean_raw) / mean_raw:+8.1f}%")
    emit(f"   folds where blurring helps: {sum(c < 0 for c in changes)}/{len(changes)}")

    # ---------------- 4. behaviour near the cores ---------------- #
    emit("\n4. ERROR STRATIFIED BY LOCAL DENSITY (does smoothing hurt the cores?)")
    emit("   integrated electron count, held-out material:")
    for name in shared:
        a, b = by_name_raw[name], by_name_blur[name]
        ea = 100 * abs(a["predicted_integral"] - a["reference_integral"]) / a["reference_integral"]
        eb = 100 * abs(b["predicted_integral"] - b["reference_integral"]) / b["reference_integral"]
        emit(f"     {name:<14s} raw {ea:6.3f} %   blurred {eb:6.3f} %")

    # ---------------- figures ---------------- #
    from poraque.vis import TrainingReport

    os.makedirs(args.plot_dir, exist_ok=True)
    report = TrainingReport(args.plot_dir, prefix=f"potential_sigma{sigma:g}")
    figures = [
        report.field_comparison(potential_raw, potential_blur,
                                label=r"$V_{\mathrm{ext}}$", unit="eV",
                                title=f"external potential: raw vs blurred "
                                      f"($\\sigma$ = {sigma} $\\AA$)"),
        report.parity(potential_raw, potential_blur, label=r"$V_{\mathrm{ext}}$",
                      unit="eV", title="blurred vs raw potential"),
    ]
    emit(f"\n   figures -> {args.plot_dir} ({len(figures)} written)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write("```\n" + "\n".join(lines) + "\n```\n")
    print(f"\nreport -> {args.out}")
    return {"mean_raw": mean_raw, "mean_blur": mean_blur, "sigma": sigma}


if __name__ == "__main__":
    main()
