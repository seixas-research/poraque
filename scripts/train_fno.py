#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: train_fno.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Train and evaluate the two Fourier Neural Operators on the VASP dataset.

Tasks
-----
``ext2chg``
    ``EXTCAR`` -> ``CHGCAR``, the Hohenberg-Kohn map.
``chg2tau``
    ``CHGCAR`` -> ``TAUCAR``, the kinetic energy density functional.

Method notes
------------
**Downsampling is spectral.** The 128³ VASP grids are reduced by Fourier
truncation (:mod:`poraque.fields.resample`), which is the exact band-limited
projection for a plane-wave field: periodicity and the electron count survive
to machine precision. Interpolation would alias, break periodicity at the cell
boundary, and shift the integral. The reduced fields are written back in
``CHGCAR`` format so the *real* dataset pipeline is exercised, not a shortcut.

**Evaluation is leave-one-out.** With a handful of materials there is no
meaningful random split, so each material is held out in turn. Metrics are
reported in **physical units**, on the held-out material.

**Physics baselines are reported alongside.** A relative error means little in
isolation; for ``chg2tau`` the learned operator is compared against the
analytic Thomas-Fermi, von Weizsäcker and ``TF + vW/9`` functionals evaluated
on the same input, and for ``ext2chg`` against predicting the mean density.
Beating those baselines is the actual bar.

Usage
-----
::

    python scripts/train_fno.py
    python scripts/train_fno.py --resolution 32 --epochs 400
    python scripts/train_fno.py --task chg2tau --width 32 --modes 12
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from poraque.fields import (  # noqa: E402
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    KineticEnergyDensity,
    thomas_fermi_tau,
    von_weizsacker_tau,
)
from poraque.fields.io import resolve_reader  # noqa: E402
from poraque.fields.resample import downsample_shape, resample_field  # noqa: E402
from poraque.ml import (  # noqa: E402
    FieldOperator,
    FieldPairDataset,
    make_dataloader,
    train,
)
from poraque.ml.tasks import resolve_task  # noqa: E402

FIELD_CLASSES = {
    "EXTCAR": ExternalPotential,
    "CHGCAR": ChargeDensity,
    "TAUCAR": KineticEnergyDensity,
}


class Tee:
    """Write to the terminal and a log file at once."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.handle = open(path, "w")

    def __call__(self, message=""):
        print(message)
        self.handle.write(str(message) + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


# ===================================================================== #
# Cache construction
# ===================================================================== #
def build_cache(root, cache_root, resolution, log, pattern="struct", code="auto"):
    """
    Spectrally downsample every calculation and write a compact dataset.

    Returns
    -------
    str
        Path to the cache directory, laid out exactly like the source dataset
        so :class:`~poraque.ml.data.FieldPairDataset` can read it unchanged.
    """
    directories = sorted(
        os.path.join(root, entry) for entry in os.listdir(root)
        if entry.startswith(pattern) and os.path.isdir(os.path.join(root, entry))
    )
    if not directories:
        raise SystemExit(f"No {pattern}* directories under {root!r}.")

    target = os.path.join(cache_root, f"res{resolution}")
    log(f"Cache: {target}")

    for directory in directories:
        name = os.path.basename(os.path.normpath(directory))
        destination = os.path.join(target, name)
        reader = resolve_reader(directory, code)

        expected = [os.path.join(destination, f)
                    for f in ("EXTCAR", "CHGCAR", "TAUCAR")]
        if all(os.path.exists(path) for path in expected):
            shape = FieldGrid.from_file(expected[0]).shape
            log(f"  {name}: cached, grid {shape}")
            continue

        os.makedirs(destination, exist_ok=True)
        start = time.time()

        # The input field defines the shared grid for this material.
        source_grid = FieldGrid.from_file(reader.field_path(directory, "external"))
        reduced_shape = downsample_shape(source_grid.shape, target_max=resolution)
        reduced_grid = FieldGrid(reduced_shape, source_grid.cell,
                                 encut=source_grid.encut)

        summary, warnings = [], []
        for kind, filename in (("external", "EXTCAR"), ("density", "CHGCAR"),
                               ("kinetic", "TAUCAR")):
            path = reader.field_path(directory, kind)
            if not os.path.exists(path):
                raise SystemExit(f"{name}: missing {path}")
            field = FIELD_CLASSES[filename].read(path, grid=source_grid)
            reduced = resample_field(field, reduced_shape, grid=reduced_grid)
            reduced.write(os.path.join(destination, filename))
            summary.append(f"{filename} [{reduced.data.min():.3g}, "
                           f"{reduced.data.max():.3g}]")

            # rho and tau are non-negative, but band-limiting a field with
            # sharp core peaks rings (Gibbs) and can undershoot slightly. It is
            # an artefact of the truncation, not of the data, and it is worth
            # naming: it is why the dataset uses the sign-tolerant `asinh`
            # normalization rather than a logarithm.
            if filename in ("CHGCAR", "TAUCAR") and field.data.min() >= 0:
                negative = int(np.count_nonzero(reduced.data < 0))
                if negative:
                    warnings.append(
                        f"{filename}: {negative} of {reduced.data.size} points "
                        f"({100 * negative / reduced.data.size:.2f}%) went "
                        f"negative, min {reduced.data.min():.3g} "
                        f"(Gibbs ringing from band-limiting)"
                    )

        log(f"  {name}: {source_grid.shape} -> {reduced_shape} "
            f"in {time.time() - start:.1f} s   " + "  ".join(summary))
        for message in warnings:
            log(f"      note: {message}")

    return target


# ===================================================================== #
# Metrics
# ===================================================================== #
def metrics(prediction, target):
    """
    Error metrics in the physical units of the fields.

    Returns
    -------
    dict
        ``mse``, ``mae``, ``rmse``, ``relative_l2``, ``r2``, ``max_abs`` and
        ``nrmse_range``.
    """
    prediction = np.asarray(prediction, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    difference = prediction - target

    total = np.sum((target - target.mean()) ** 2)
    spread = np.ptp(target) or 1.0

    return {
        "mse": float(np.mean(difference ** 2)),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "max_abs": float(np.max(np.abs(difference))),
        "relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(target)),
        "nrmse_range": float(np.sqrt(np.mean(difference ** 2)) / spread),
        "r2": float(1.0 - np.sum(difference ** 2) / total) if total > 0 else float("nan"),
    }


def format_metrics(name, values, unit):
    """One aligned line per metric set."""
    return (f"    {name:<22s} MSE {values['mse']:11.5g}  MAE {values['mae']:10.5g}  "
            f"RMSE {values['rmse']:10.5g}  relL2 {values['relative_l2']:8.4f}  "
            f"R2 {values['r2']:8.4f}   [{unit}]")


def physics_baselines(task, source, target):
    """
    Analytic reference predictions for the same mapping.

    For ``chg2tau`` these are the classical orbital-free kinetic energy
    density functionals — the incumbents a learned KEDF must beat. For
    ``ext2chg`` no closed-form density exists, so the honest baseline is the
    constant mean density.

    Parameters
    ----------
    task : TaskSpec
    source, target : ScalarField
        Input and reference output for one material.

    Returns
    -------
    dict
        ``{label: prediction_array}``.
    """
    if task.name == "chg2tau":
        tf = thomas_fermi_tau(source.data)
        vw = von_weizsacker_tau(source.data, source.grid)
        return {
            "Thomas-Fermi": tf,
            "von Weizsacker": vw,
            "TF + vW/9": tf + vw / 9.0,
        }
    return {"mean density": np.full_like(target.data, target.data.mean())}


# ===================================================================== #
# Leave-one-out driver
# ===================================================================== #
def run_task(task_name, cache, args, log):
    """Train and evaluate one task with leave-one-out cross-validation."""
    task = resolve_task(task_name)
    log(f"\n{'=' * 78}")
    log(f"TASK  {task.name}:  {task.input_field} -> {task.target_field}")
    log(f"      {task.description}")
    log("=" * 78)

    dataset = FieldPairDataset(cache, task=task)
    shapes = dataset.shapes()
    log(f"  materials: {len(dataset)}   grids: {shapes}")
    log(f"  units: input [{task.input_unit}]  target [{task.target_unit}]")

    if len(dataset) < 2:
        log("  !! need at least 2 materials for leave-one-out; skipping")
        return None

    folds = []
    for held_out in range(len(dataset)):
        name = dataset.materials[held_out].identifier
        log(f"\n  --- fold {held_out + 1}/{len(dataset)}: hold out {name} ---")

        train_records = [m for i, m in enumerate(dataset.materials) if i != held_out]
        test_records = [dataset.materials[held_out]]

        train_set = FieldPairDataset(cache, task=task, materials=train_records)
        source_transform, target_transform = train_set.fit_transforms()
        test_set = FieldPairDataset(cache, task=task, materials=test_records,
                                    input_transform=source_transform,
                                    target_transform=target_transform)
        log(f"      train {[m.identifier for m in train_records]}  "
            f"test [{name}]")
        log(f"      transforms: in {source_transform}  out {target_transform}")

        # The tau = tau_vW + softplus head makes the Hoffmann-Ostenhof bound
        # structural. Its scale is fitted on the TRAINING split only; fitting
        # it on the held-out material would leak the answer.
        head = {}
        if args.pauli_head and task.name == "chg2tau":
            from poraque.ml import fit_pauli_scale, pauli_bound_violation

            head = {"pauli_residual": True,
                    "pauli_scale": fit_pauli_scale(train_set)}
            log(f"      head: tau = tau_vW[rho] + s*softplus(f)   "
                f"s = {head['pauli_scale']:.4f} eV/Ang^3 (fitted on train)")
            for entry in pauli_bound_violation(test_set):
                log(f"      reference bound check ({entry['material']}): "
                    f"{entry['violations']}/{entry['points']} points below "
                    f"tau_vW ({100 * entry['fraction']:.4f} %), "
                    f"tau_vW supplies {100 * entry['vw_fraction']:.1f} % of tau")

        torch.manual_seed(args.seed)
        operator = FieldOperator(
            task, width=args.width, modes=args.modes, n_layers=args.layers,
            projection_channels=args.projection,
            input_transform=source_transform, target_transform=target_transform,
            device=args.device, **head,
        )
        log(f"      model: {type(operator.model).__name__} width={args.width} "
            f"modes={args.modes} layers={args.layers}  "
            f"({operator.model.n_parameters():,} parameters)")

        start = time.time()
        history = train(operator, train_set, validation=test_set,
                        epochs=args.epochs, batch_size=1,
                        learning_rate=args.learning_rate, verbose=False,
                        seed=args.seed)
        elapsed = time.time() - start
        log(f"      trained {args.epochs} epochs in {elapsed:.1f} s   "
            f"loss {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")

        # ---------------- evaluation in physical units ---------------- #
        source, target = test_set.load_fields(0)
        prediction = operator.predict(source)
        test_metrics = metrics(prediction.data, target.data)

        train_source, train_target = train_set.load_fields(0)
        train_metrics = metrics(operator.predict(train_source).data,
                                train_target.data)

        log(f"      results on the HELD-OUT material ({name}):")
        log(format_metrics("FNO (test)", test_metrics, task.target_unit))
        log(format_metrics("FNO (train fit)", train_metrics, task.target_unit))
        for label, baseline in physics_baselines(task, source, target).items():
            log(format_metrics(f"baseline: {label}", metrics(baseline, target.data),
                               task.target_unit))

        # Integral quantities: what actually enters a total energy.
        predicted_integral = prediction.integrate()
        reference_integral = target.integrate()
        log(f"      integral: predicted {predicted_integral:12.4f}   "
            f"reference {reference_integral:12.4f}   "
            f"error {100 * abs(predicted_integral - reference_integral) / abs(reference_integral):.3f} %")

        # Verify the constraint on the actual prediction, whether or not the
        # head is enabled -- this is the number that shows a structural bound
        # doing its job versus a model that merely happens to respect it.
        constraint = None
        if task.name == "chg2tau":
            bound = von_weizsacker_tau(source.data, source.grid)
            deficit = prediction.data - bound
            violations = int(np.count_nonzero(deficit < -1e-6))
            constraint = {
                "violations": violations,
                "points": int(deficit.size),
                "fraction": violations / deficit.size,
                "worst_deficit": float(deficit.min()),
            }
            log(f"      constraint tau >= tau_vW: {violations}/{deficit.size} "
                f"violated ({100 * violations / deficit.size:.4f} %), "
                f"min margin {deficit.min():+.4g} eV/Ang^3")

        folds.append({
            "held_out": name,
            "train": [m.identifier for m in train_records],
            "test_metrics": test_metrics,
            "train_metrics": train_metrics,
            "baselines": {label: metrics(values, target.data) for label, values
                          in physics_baselines(task, source, target).items()},
            "predicted_integral": predicted_integral,
            "reference_integral": reference_integral,
            "constraint": constraint,
            "pauli_head": bool(head),
            "seconds": elapsed,
            "final_train_loss": history["train_loss"][-1],
        })

        if args.checkpoint_dir:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            path = os.path.join(args.checkpoint_dir, f"{task.name}_holdout_{name}.pt")
            operator.save(path)
            log(f"      checkpoint -> {path}")

    # ------------------------------ summary ------------------------------ #
    log(f"\n  --- {task.name}: leave-one-out summary ---")
    for key in ("mse", "mae", "rmse", "relative_l2", "r2"):
        values = [fold["test_metrics"][key] for fold in folds]
        log(f"      {key:<12s} mean {np.mean(values):12.5g}   "
            f"per fold {['%.5g' % v for v in values]}")

    return {"task": task.name, "folds": folds}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--root", default="data/vasp")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--pattern", default="struct")
    parser.add_argument("--code", default="auto")
    parser.add_argument("--task", default="all",
                        choices=["all", "ext2chg", "chg2tau"])
    parser.add_argument("--resolution", type=int, default=32,
                        help="longest grid axis after spectral downsampling")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--projection", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pauli-head", action="store_true",
                        help="for chg2tau, predict tau = tau_vW[rho] + "
                             "s*softplus(f) so tau >= tau_vW holds by "
                             "construction")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log", default="logs/fno_training.log")
    parser.add_argument("--json", default="logs/fno_training.json")
    args = parser.parse_args(argv)

    log = Tee(args.log)
    try:
        log("=" * 78)
        log("Poraque - Fourier Neural Operator training on VASP data")
        log("=" * 78)
        log(f"  torch {torch.__version__}   device "
            f"{args.device or ('cuda' if torch.cuda.is_available() else 'cpu')}")
        log(f"  source {args.root}   resolution {args.resolution}   "
            f"epochs {args.epochs}   seed {args.seed}")
        log("")

        cache = build_cache(args.root, args.cache, args.resolution, log,
                            pattern=args.pattern, code=args.code)

        names = ["ext2chg", "chg2tau"] if args.task == "all" else [args.task]
        results = [result for result in
                   (run_task(name, cache, args, log) for name in names)
                   if result is not None]

        log(f"\n{'=' * 78}\nOVERALL\n{'=' * 78}")
        log(f"  {'task':<12s} {'rel L2 (mean)':>14s} {'R2 (mean)':>12s} "
            f"{'MAE (mean)':>14s}")
        for result in results:
            folds = result["folds"]
            log(f"  {result['task']:<12s} "
                f"{np.mean([f['test_metrics']['relative_l2'] for f in folds]):14.4f} "
                f"{np.mean([f['test_metrics']['r2'] for f in folds]):12.4f} "
                f"{np.mean([f['test_metrics']['mae'] for f in folds]):14.5g}")

        n_materials = len(FieldPairDataset(cache, task=names[0]))
        log("")
        log(f"  NOTE: with only {n_materials} material(s) these numbers characterise")
        log("  the pipeline, not the science. Leave-one-out over a handful of")
        log("  related structures of one element measures interpolation between")
        log("  nearby geometries; it says nothing about transfer to new chemistry.")

        with open(args.json, "w") as handle:
            json.dump({"args": vars(args), "results": results}, handle,
                      indent=2, default=float)
        log(f"\n  log  -> {args.log}")
        log(f"  json -> {args.json}")
    finally:
        log.close()


if __name__ == "__main__":
    main()
