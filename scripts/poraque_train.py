#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_train.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Train and evaluate the Fourier Neural Operators on a DFT dataset.

The run is defined by a **YAML configuration file**; command-line flags
override individual entries, so one committed config can be swept from the
shell without being edited or copied. The resolved configuration is written
next to the results, recording exactly what ran.

Tasks
-----
``ext2chg``
    ``EXTCAR`` -> ``CHGCAR``, the Hohenberg-Kohn map.
``chg2tau``
    ``CHGCAR`` -> ``TAUCAR``, the kinetic energy density functional.

Method notes
------------
**Downsampling is spectral.** The native VASP grids are reduced by Fourier
truncation (:mod:`poraque.fields.resample`), the exact band-limited projection
for a plane-wave field: periodicity and the electron count survive to machine
precision. Interpolation would alias, break periodicity at the cell boundary
and shift the integral.

**One protocol, one variation.** A run trains a single model per task on a
train/validation split sized by ``data.valid_fraction`` (a fifth by default),
and reports metrics in **physical units** on the held-out structures. Setting
``enable_kfold`` swaps that for K-fold cross-validation, which is the only
other protocol; nothing else changes how training is organised.

**Physics baselines are reported alongside.** A relative error means little in
isolation, so ``chg2tau`` is compared against the analytic Thomas-Fermi, von
Weizsäcker and ``TF + vW/9`` functionals evaluated on the same input, and
``ext2chg`` against predicting the mean density. Beating those is the bar.

Usage
-----
Installed (``pip install -e .``), this is the ``poraque-train`` console command
and runs from any directory::

    poraque-train --write-config configs/train_config.yaml
    poraque-train --config configs/train_config.yaml
    poraque-train --config configs/train_config.yaml --epochs 500
    poraque-train --config configs/train_config.yaml --device mps
    poraque-train --config configs/train_config.yaml --kfold

Running this file directly — ``python scripts/poraque_train.py`` — is
equivalent, and needs nothing installed.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

# Run straight from a checkout, without installing, by preferring the in-tree
# package. Installed as the ``poraque-train`` console script this module sits in
# site-packages, that directory does not exist, and the installed package wins.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

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
    BUNDLE_FILENAME,
    FieldOperator,
    FieldPairDataset,
    save_bundle,
    train,
)
from poraque.ml.config import SAMPLE_CONFIG_HEADER, TrainingConfig  # noqa: E402
from poraque.ml.device import describe_device, resolve_device  # noqa: E402
from poraque.ml.losses import PhysicsInformedLoss  # noqa: E402
from poraque.ml.tasks import resolve_task  # noqa: E402

FIELD_CLASSES = {
    "EXTCAR": ExternalPotential,
    "CHGCAR": ChargeDensity,
    "TAUCAR": KineticEnergyDensity,
}

#: Display label and unit per field, for figures.
FIELD_LABELS = {
    "EXTCAR": (r"$V_{\mathrm{ext}}$", r"eV"),
    "CHGCAR": (r"$\rho$", r"e/$\AA^3$"),
    "TAUCAR": (r"$\tau$", r"eV/$\AA^3$"),
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
def build_cache(config, log):
    """
    Spectrally downsample every calculation and write a compact dataset.

    Returns
    -------
    str
        Cache directory, laid out like the source dataset so
        :class:`~poraque.ml.data.FieldPairDataset` reads it unchanged.
    """
    data = config.data
    directories = sorted(
        os.path.join(data.root, entry) for entry in os.listdir(data.root)
        if entry.startswith(data.pattern)
        and os.path.isdir(os.path.join(data.root, entry))
    )
    if not directories:
        raise SystemExit(f"No {data.pattern}* directories under {data.root!r}.")

    # The cache key encodes everything that changes the stored fields. Without
    # this, switching --gaussian-blur would silently reuse the previous cache
    # and the "comparison" would compare a model against itself.
    tag = f"res{data.resolution}"
    if data.gaussian_blur:
        tag += f"_blur{data.gaussian_blur:g}{data.blur_method[:4]}"
    target = os.path.join(data.cache, tag)
    log(f"Cache: {target}")

    for directory in directories:
        name = os.path.basename(os.path.normpath(directory))
        destination = os.path.join(target, name)
        reader = resolve_reader(directory, data.code)

        expected = [os.path.join(destination, f)
                    for f in ("EXTCAR", "CHGCAR", "TAUCAR")]
        if all(os.path.exists(path) for path in expected):
            log(f"  {name}: cached, grid {FieldGrid.from_file(expected[0]).shape}")
            continue

        os.makedirs(destination, exist_ok=True)
        start = time.time()

        # The shared grid comes from CHGCAR: the density is always present in a
        # standard VASP run, and every other field is placed on the same mesh.
        source_grid = FieldGrid.from_file(reader.field_path(directory, "density"))
        reduced_shape = downsample_shape(source_grid.shape,
                                         target_max=data.resolution)
        reduced_grid = FieldGrid(reduced_shape, source_grid.cell,
                                 encut=source_grid.encut)

        summary, warnings = [], []

        # ---- external potential ---------------------------------------- #
        # Always computed by poraque from POSCAR/INCAR/POTCAR, so the pipeline
        # works with any standard VASP distribution. Any EXTCAR present in the
        # source directory is ignored: the training input must be exactly what
        # ExternalPotential produces at inference time, when no such file
        # exists.
        potential = ExternalPotential.from_calculation(
            directory, code=reader.code, grid=source_grid,
            gaussian_blur=data.gaussian_blur,
            blur_method=data.blur_method,
        )
        origin = f"poraque/{potential.metadata.get('model', '?')}"
        if data.gaussian_blur:
            origin += f", blur {data.gaussian_blur} A ({data.blur_method})"

        reduced = resample_field(potential, reduced_shape, grid=reduced_grid)
        reduced.write(os.path.join(destination, "EXTCAR"))
        summary.append(f"EXTCAR [{reduced.data.min():.3g}, "
                       f"{reduced.data.max():.3g}] ({origin})")

        # ---- density and kinetic energy density ------------------------ #
        for kind, filename in (("density", "CHGCAR"), ("kinetic", "TAUCAR")):
            path = reader.field_path(directory, kind)
            if not os.path.exists(path):
                raise SystemExit(f"{name}: missing {path}")
            field = FIELD_CLASSES[filename].read(path, grid=source_grid)
            reduced = resample_field(field, reduced_shape, grid=reduced_grid)
            reduced.write(os.path.join(destination, filename))
            summary.append(f"{filename} [{reduced.data.min():.3g}, "
                           f"{reduced.data.max():.3g}]")

            # rho and tau are non-negative, but band-limiting a field with
            # sharp core peaks rings (Gibbs) and can undershoot slightly. That
            # is an artefact of the truncation, not of the data, and it is why
            # the dataset uses the sign-tolerant `asinh` normalization.
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
    """Error metrics in the physical units of the fields."""
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
        "r2": float(1.0 - np.sum(difference ** 2) / total) if total > 0
        else float("nan"),
    }


def format_metrics(name, values, unit):
    """One aligned line per metric set."""
    return (f"    {name:<22s} MSE {values['mse']:11.5g}  MAE {values['mae']:10.5g}  "
            f"RMSE {values['rmse']:10.5g}  relL2 {values['relative_l2']:8.4f}  "
            f"R2 {values['r2']:8.4f}   [{unit}]")


def physics_baselines(task, source, target):
    """Analytic reference predictions for the same mapping."""
    if task.name == "chg2tau":
        tf = thomas_fermi_tau(source.data)
        vw = von_weizsacker_tau(source.data, source.grid)
        return {"Thomas-Fermi": tf, "von Weizsacker": vw, "TF + vW/9": tf + vw / 9.0}
    return {"mean density": np.full_like(target.data, target.data.mean())}


def build_loss(config, task_name):
    """Assemble the objective from the ``training`` section of the config."""
    physics = dict(config.training.physics or {})
    return PhysicsInformedLoss(
        task=task_name,
        sobolev_weight=(config.training.sobolev_weight
                        if config.training.loss == "sobolev" else 0.0),
        electron_count_weight=physics.get("electron_count_weight", 0.0),
        positivity_weight=physics.get("positivity_weight", 0.0),
        von_weizsacker_weight=physics.get("von_weizsacker_weight", 0.0),
        euler_lagrange_weight=physics.get("euler_lagrange_weight", 0.0),
    )


# ===================================================================== #
# Leave-one-out driver
# ===================================================================== #
def build_operator(task, train_set, config, log):
    """Construct the operator, attaching the Pauli head when requested."""
    source_transform = train_set.input_transform
    target_transform = train_set.target_transform

    head = {}
    if config.model.pauli_residual and task.name == "chg2tau":
        from poraque.ml import fit_pauli_scale, pauli_bound_violation

        scale = (config.model.pauli_scale if config.model.pauli_scale
                 else fit_pauli_scale(train_set))
        head = {"pauli_residual": True, "pauli_scale": scale,
                "learn_pauli_scale": config.model.learn_pauli_scale}
        log(f"      head: tau = tau_vW[rho] + s*softplus(f)   s = {scale:.4f} eV/Ang^3")
        for entry in pauli_bound_violation(train_set):
            if entry["violations"]:
                log(f"      note: {entry['material']} violates tau >= tau_vW at "
                    f"{entry['violations']}/{entry['points']} points "
                    f"({100 * entry['fraction']:.4f} %)")

    # init_seed reaches FieldOperator, which isolates the draw from the global
    # stream; the manual_seed here keeps the ambient behaviour when it is unset.
    torch.manual_seed(config.training.seed)
    operator = FieldOperator(
        task, input_transform=source_transform, target_transform=target_transform,
        device=config.training.device,
        training_resolution=config.data.resolution,
        init_seed=config.training.init_seed,
        **config.model_kwargs(), **head,
    )
    log(f"      model: {type(operator.model).__name__} width={config.model.width} "
        f"modes={config.model.modes} layers={config.model.n_layers}  "
        f"({operator.model.n_parameters():,} parameters)")
    return operator


def resolve_validation_split(dataset, config):
    """
    Choose the structures held out for validation.

    The split is defined by one number, ``training.valid_fraction``: that share
    of the structures is drawn at random and kept back. ``0`` trains on every
    structure, at the cost of turning the reported metrics into a training fit.

    The draw is at the **structure level**: whole materials move together.
    Splitting voxels instead would put the same crystal on both sides, and
    because neighbouring voxels are strongly correlated the validation score
    would look excellent while saying nothing about a new material.

    Parameters
    ----------
    dataset : FieldPairDataset
        The full dataset, for its material list.
    config : TrainingConfig

    Returns
    -------
    tuple of (set of str, str)
        Held-out identifiers, and a human-readable description of the split —
        logged so the record shows the protocol, not only the result.
    """
    names = [m.identifier for m in dataset.materials]
    fraction = float(config.training.valid_fraction or 0.0)

    # Range-check before the zero case: a negative value is a typo, and
    # treating it as "no split" would silently change the protocol rather than
    # reporting the mistake.
    if not 0.0 <= fraction < 1.0:
        raise SystemExit(
            f"training.valid_fraction must lie in [0, 1), got {fraction}."
        )

    if fraction == 0.0:
        return set(), "none - trained on every structure, metrics are TRAINING FIT"
    if len(names) < 2:
        raise SystemExit(
            f"valid_fraction needs at least two structures to split; the "
            f"dataset has {len(names)}."
        )

    # Round to nearest, then clamp so neither side is empty: a fraction that
    # rounds to zero would silently train on everything while the config
    # claims a validation split.
    count = int(round(fraction * len(names)))
    count = max(1, min(count, len(names) - 1))

    seed = config.training.seed
    order = np.random.default_rng(seed).permutation(len(names))
    selected = {names[i] for i in order[:count]}
    return selected, (f"valid_fraction={fraction:g} -> {count}/{len(names)} "
                      f"structures, seed={seed}")


def evaluate_material(operator, dataset, index, task, log, label,
                      baselines=False):
    """
    Predict one material and report metrics against its reference field.

    Parameters
    ----------
    baselines : bool, optional
        Also report the analytic orbital-free functionals on the same input.
        A relative error means little in isolation: beating Thomas-Fermi and
        von Weizsäcker is the bar, and printing them beside the model is the
        only way the reader can tell whether it was cleared. Worth the extra
        lines on validation structures, noise on training ones.
    """
    source, target = dataset.load_fields(index)
    prediction = operator.predict(source)
    values = metrics(prediction.data, target.data)
    log(format_metrics(label, values, task.target_unit))

    if baselines:
        for name, reference in physics_baselines(task, source, target).items():
            log(format_metrics(f"  baseline: {name}",
                               metrics(reference, target.data),
                               task.target_unit))
    return prediction, target, values


def run_task(task_name, cache, config, log):
    r"""
    Train one model for ``task_name`` on a train/validation split.

    This is the ordinary path; K-fold cross-validation is the only variation on
    it. **One** model is fitted across all the training structures, never one
    per material: batches are drawn across structures — the sampler groups by
    grid shape and shuffles both within and across those groups — so a gradient
    step generally mixes several.

    ``training.valid_fraction`` holds back a fifth of the structures by
    default, so the reported score is genuinely held out and ``early_stopping``
    has something to watch. Set it to ``0`` when the goal is the final artefact
    trained on *all* the data — in which case the metrics become a **training
    fit** and carry no generalisation claim.
    """
    task = resolve_task(task_name)
    log(f"\n{'=' * 78}")
    log(f"TASK  {task.name}:  {task.input_field} -> {task.target_field}")
    log(f"      {task.description}")
    log("=" * 78)

    dataset = FieldPairDataset(cache, task=task)
    validation_names, split_origin = resolve_validation_split(dataset, config)

    train_records = [m for m in dataset.materials
                     if m.identifier not in validation_names]
    test_records = [m for m in dataset.materials
                    if m.identifier in validation_names]

    train_set = FieldPairDataset(cache, task=task, materials=train_records)
    source_transform, target_transform = train_set.fit_transforms()
    validation = (FieldPairDataset(cache, task=task, materials=test_records,
                                   input_transform=source_transform,
                                   target_transform=target_transform)
                  if test_records else None)

    shapes = train_set.shapes()
    buckets = {}
    for shape in shapes:
        buckets[tuple(shape)] = buckets.get(tuple(shape), 0) + 1

    log(f"  training structures : {len(train_set)}  "
        f"{[m.identifier for m in train_records]}")
    log(f"  validation          : "
        f"{sorted(validation_names) if validation_names else 'none'} "
        f"({split_origin})")
    log(f"  grid shapes         : {shapes}")
    log(f"  shape buckets       : "
        + ", ".join(f"{s}x{n}" for s, n in sorted(buckets.items())))
    log(f"  batch size          : {config.training.batch_size} "
        f"(capped per bucket; batches mix structures of equal shape)")
    log(f"  transforms          : in {source_transform}  out {target_transform}")

    operator = build_operator(task, train_set, config, log)

    # Early stopping needs something to watch. With no validation split the
    # library would warn -- rightly, for a caller who asked for both -- but here
    # the two are merely the shipped defaults, and a default configuration must
    # not warn. Say it plainly in the log instead.
    patience = config.training.early_stopping
    if patience and validation is None:
        log(f"  early stopping      : inactive ({patience} epochs requested, "
            f"but nothing is held out to measure improvement against)")
        patience = 0

    log(f"\n  progress (every {config.training.eval_epoch} epochs):")
    start = time.time()
    history = train(
        operator, train_set, validation=validation,
        epochs=config.training.epochs, batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        scheduler=config.training.scheduler, grad_clip=config.training.grad_clip,
        loss=build_loss(config, task.name), seed=config.training.seed,
        eval_every=config.training.eval_epoch, early_stopping=patience,
        log=log, verbose=True,
    )
    elapsed = time.time() - start
    log(f"\n  trained {len(history['train_loss'])}/{config.training.epochs} epochs in {elapsed:.1f} s   "
        f"loss {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")

    # ---------------- per-material evaluation ---------------- #
    label_text, unit = FIELD_LABELS[task.target_field]
    per_material, figures = {}, []
    report = None
    if config.output.plot_dir:
        from poraque.vis import TrainingReport

        report = TrainingReport(config.output.plot_dir, dpi=config.output.dpi,
                                fmt=config.output.plot_format,
                                prefix=f"{task.name}")
        figures.append(report.loss_curves(
            history, title=f"{task.name} ({len(train_set)} training structures)"))

    log(f"\n  per-structure results "
        f"({'TRAINING FIT' if not validation_names else 'train / validation'}):")
    for index in range(len(train_set)):
        name = train_records[index].identifier
        prediction, target, values = evaluate_material(
            operator, train_set, index, task, log, f"{name} (train)")
        per_material[name] = {"split": "train", "metrics": values,
                              "predicted_integral": prediction.integrate(),
                              "reference_integral": target.integrate()}
        if report is not None and index == 0:
            report.prefix = f"{task.name}_{name}"
            figures.append(report.field_comparison(
                target, prediction, label=label_text, unit=unit,
                log=(task.target_field in ("CHGCAR", "TAUCAR")),
                title=f"{task.name} · {name}"))
            figures.append(report.parity(
                target, prediction, label=label_text, unit=unit,
                log=(task.target_field in ("CHGCAR", "TAUCAR"))))

    if validation is not None:
        for index in range(len(validation)):
            name = test_records[index].identifier
            prediction, target, values = evaluate_material(
                operator, validation, index, task, log, f"{name} (VALIDATION)",
                baselines=True)
            per_material[name] = {"split": "validation", "metrics": values,
                                  "predicted_integral": prediction.integrate(),
                                  "reference_integral": target.integrate()}

    # ---------------- aggregate ---------------- #
    train_metrics = [v["metrics"] for v in per_material.values()
                     if v["split"] == "train"]
    log(f"\n  --- {task.name}: aggregate over {len(train_metrics)} training structures ---")
    for key in ("mse", "mae", "rmse", "relative_l2", "r2"):
        values = [m[key] for m in train_metrics]
        log(f"      {key:<12s} mean {np.mean(values):12.5g}   "
            f"min {np.min(values):11.5g}   max {np.max(values):11.5g}")

    if not validation_names:
        log("\n      NOTE: valid_fraction is 0, so nothing was held out and these")
        log("      are TRAINING-FIT numbers. They show the model can represent the")
        log("      data; they are not a generalisation estimate. Raise")
        log("      valid_fraction, or run --kfold, for that.")

    # ---------------- persist ---------------- #
    # The operator is handed back to main(), which writes every task into one
    # unified bundle after the last has trained. Writing per task would either
    # leave the file half-updated on a crash or retain a stale sibling from an
    # earlier run.
    checkpoint = None
    if figures:
        log(f"  figures         -> {config.output.plot_dir} ({len(figures)})")

    # ---------------- PDF report ---------------- #
    pdf = None
    if config.output.report_dir:
        from poraque.vis import ModelReport

        caveats = [
            f"{len(train_set)} structure(s), all of one element: nothing here "
            f"speaks to transfer across chemistry.",
        ]
        if not validation_names:
            caveats.append(
                "No structure was held out, so every number in the table is "
                "training fit, not a generalisation estimate."
            )
        reporter = ModelReport(config.output.report_dir)
        pdf = reporter.build(
            task=task.name, per_material=per_material, figures=figures,
            unit=task.target_unit, caveats=caveats,
            summary={
                "model": type(operator.model).__name__,
                "parameters": f"{operator.model.n_parameters():,}",
                "training structures": str(len(train_set)),
                "grid shapes": ", ".join(str(s) for s in sorted(buckets)),
                "epochs": str(config.training.epochs),
                "batch size": str(config.training.batch_size),
                "device": describe_device(operator.device),
                "training time": f"{elapsed:.1f} s",
                "final train loss": f"{history['train_loss'][-1]:.5f}",
            },
            configuration={f"{section}.{key}": str(value)
                           for section, values in config.to_dict().items()
                           if isinstance(values, dict)
                           for key, value in values.items()},
        )
        log(f"  PDF report      -> {pdf}")

    return {
        "task": task.name,
        "mode": "split",
        "report": pdf,
        "n_train": len(train_set),
        "train_structures": [m.identifier for m in train_records],
        "validation": sorted(validation_names),
        "grid_shapes": [list(s) for s in shapes],
        "per_material": per_material,
        "checkpoint": checkpoint,
        "figures": figures,
        "seconds": elapsed,
        "final_train_loss": history["train_loss"][-1],
        "history": {k: list(map(float, v)) for k, v in history.items()},
        # Not serialised into the JSON summary; main() pops it to build the
        # unified bundle once every task has trained.
        "operator": operator,
    }


def structure_level_folds(names, k, seed=0):
    r"""
    Partition **whole structures** into ``k`` validation groups.

    The split is at the structure level by construction: a fold is a set of
    material names, and every voxel of a material goes to the same side. A
    voxel-level split would place the same material in both training and
    validation, and the resulting score would measure interpolation *within* a
    material rather than transfer *to a new* one — the number would look
    excellent and mean nothing.

    Parameters
    ----------
    names : sequence of str
        Structure identifiers.
    k : int
        Requested number of folds; capped at ``len(names)``.
    seed : int, optional
        Shuffling seed, so the partition is reproducible.

    Returns
    -------
    list of list of str
        ``k`` disjoint groups whose union is ``names``.
    """
    names = list(names)
    k = max(2, min(int(k), len(names)))
    order = np.random.default_rng(seed).permutation(len(names))
    return [[names[i] for i in group]
            for group in np.array_split(order, k) if len(group)]


def run_task_kfold(task_name, cache, config, log):
    r"""
    K-fold cross-validation over structures.

    Each fold trains a fresh model on :math:`K-1` groups of structures and
    scores it on the held-out group, in physical units. The result is a
    *generalisation estimate* with a spread, not a deployable model: it fits
    *K* of them. Leave ``enable_kfold`` off for the artefact to ship.

    ``valid_fraction`` is ignored here — the folds define the splits.
    """
    task = resolve_task(task_name)
    dataset = FieldPairDataset(cache, task=task)
    names = [m.identifier for m in dataset.materials]
    folds = structure_level_folds(names, config.training.k_folds,
                                  config.training.seed)
    by_name = {m.identifier: m for m in dataset.materials}

    log(f"\n{'=' * 78}")
    log(f"TASK  {task.name}:  {task.input_field} -> {task.target_field}   "
        f"[{len(folds)}-FOLD CROSS-VALIDATION]")
    log(f"      {task.description}")
    log("=" * 78)
    log(f"  structures : {len(names)}")
    log(f"  folds      : {len(folds)} (requested {config.training.k_folds})")
    if len(folds) == len(names):
        log("               = leave-one-out, since k equals the structure count")
    for index, group in enumerate(folds, 1):
        log(f"     fold {index}: validate on {group}")
    log("  split is at the STRUCTURE level: no material appears on both sides")

    label_text, unit = FIELD_LABELS[task.target_field]
    records, figures = [], []

    for index, group in enumerate(folds, 1):
        log(f"\n  --- fold {index}/{len(folds)}: validate on {group} ---")
        train_records = [by_name[n] for n in names if n not in group]
        val_records = [by_name[n] for n in group]

        train_set = FieldPairDataset(cache, task=task, materials=train_records)
        source_transform, target_transform = train_set.fit_transforms()
        val_set = FieldPairDataset(cache, task=task, materials=val_records,
                                   input_transform=source_transform,
                                   target_transform=target_transform)
        log(f"      train on {len(train_set)}: {[m.identifier for m in train_records]}")

        operator = build_operator(task, train_set, config, log)
        start = time.time()
        history = train(
            operator, train_set, validation=val_set,
            epochs=config.training.epochs, batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            scheduler=config.training.scheduler,
            grad_clip=config.training.grad_clip,
            loss=build_loss(config, task.name), seed=config.training.seed,
            eval_every=config.training.eval_epoch,
            early_stopping=config.training.early_stopping,
            log=log, verbose=True,
        )
        elapsed = time.time() - start
        log(f"      trained {len(history['train_loss'])}/{config.training.epochs} epochs in {elapsed:.1f} s   "
            f"loss {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")

        for position in range(len(val_set)):
            name = val_records[position].identifier
            prediction, target, values = evaluate_material(
                operator, val_set, position, task, log, f"{name} (VALIDATION)",
                baselines=True)
            records.append({"fold": index, "material": name,
                            "split": f"fold {index}", "metrics": values,
                            "predicted_integral": prediction.integrate(),
                            "reference_integral": target.integrate()})
            if config.output.plot_dir and position == 0:
                from poraque.vis import TrainingReport

                report = TrainingReport(
                    config.output.plot_dir, dpi=config.output.dpi,
                    fmt=config.output.plot_format,
                    prefix=f"{task.name}_fold{index}_{name}")
                figures.append(report.loss_curves(
                    history, title=f"{task.name} · fold {index}"))
                figures.append(report.field_comparison(
                    target, prediction, label=label_text, unit=unit,
                    log=(task.target_field in ("CHGCAR", "TAUCAR")),
                    title=f"{task.name} · fold {index} · {name}"))

    # ---------------- aggregate across folds ---------------- #
    log(f"\n  --- {task.name}: {len(folds)}-fold summary "
        f"({len(records)} validation structures) ---")
    aggregate = {}
    for key in ("mse", "mae", "rmse", "relative_l2", "r2"):
        values = np.array([r["metrics"][key] for r in records], dtype=float)
        aggregate[key] = {"mean": float(values.mean()), "std": float(values.std()),
                          "min": float(values.min()), "max": float(values.max())}
        log(f"      {key:<12s} {values.mean():12.5g} +/- {values.std():<11.4g}"
            f"  [{values.min():.5g}, {values.max():.5g}]")

    log("\n      These ARE generalisation numbers: every score above comes from")
    log("      a model that never saw that structure. The spread across folds")
    log("      matters as much as the mean with a dataset this small.")

    # ---------------- consolidated report ---------------- #
    pdf = None
    if config.output.report_dir:
        from poraque.vis import ModelReport

        per_material = {r["material"]: {"split": r["split"],
                                        "metrics": r["metrics"]}
                        for r in records}
        reporter = ModelReport(config.output.report_dir)
        pdf = reporter.build(
            task=task.name, per_material=per_material, figures=figures,
            unit=task.target_unit,
            filename=f"{task.name}_kfold_report.pdf",
            summary={
                "protocol": f"{len(folds)}-fold cross-validation",
                "split level": "structure (whole materials held out)",
                "structures": str(len(names)),
                "validation scores": str(len(records)),
                "epochs per fold": str(config.training.epochs),
                "device": describe_device(resolve_device(config.training.device)),
                "mean relative L2": f"{aggregate['relative_l2']['mean']:.5g}"
                                    f" +/- {aggregate['relative_l2']['std']:.3g}",
                "mean MAE": f"{aggregate['mae']['mean']:.5g}"
                            f" +/- {aggregate['mae']['std']:.3g}",
                "mean MSE": f"{aggregate['mse']['mean']:.5g}"
                            f" +/- {aggregate['mse']['std']:.3g}",
            },
            caveats=[
                "Every score is from a model that never saw that structure, so "
                "these are generalisation estimates rather than training fit.",
                f"{len(names)} structures of a single element: this measures "
                f"transfer between geometries, not between chemistries.",
                "With few folds the spread is wide; quote the standard "
                "deviation alongside the mean.",
            ],
            configuration={f"{section}.{key}": str(value)
                           for section, values in config.to_dict().items()
                           if isinstance(values, dict)
                           for key, value in values.items()},
        )
        log(f"\n  consolidated report -> {pdf}")

    return {"task": task.name, "mode": "kfold", "n_folds": len(folds),
            "folds": [{"fold": i, "validate_on": g}
                      for i, g in enumerate(folds, 1)],
            "records": records, "aggregate": aggregate,
            "report": pdf, "figures": figures}


# ===================================================================== #
# Entry point
# ===================================================================== #
def build_parser():
    """Command-line interface. Every override defaults to ``None``.

    A ``None`` default is what lets the resolver distinguish "flag absent" from
    "flag set to a falsy value", so ``--grad-clip 0`` disables clipping instead
    of being mistaken for an unset flag.
    """
    parser = argparse.ArgumentParser(
        description="Train Poraque's Fourier Neural Operators from a YAML config.",
    )
    parser.add_argument("--config", default=None,
                        help="YAML configuration file (defaults are used if omitted)")
    parser.add_argument("--write-config", metavar="PATH", default=None,
                        help="write a sample configuration to PATH and exit")

    parser.add_argument("--task", default=None, choices=["all", "ext2chg", "chg2tau"])
    group = parser.add_argument_group("data overrides")
    group.add_argument("--root", dest="data.root", default=None)
    group.add_argument("--cache", dest="data.cache", default=None)
    group.add_argument("--pattern", dest="data.pattern", default=None)
    group.add_argument("--code", dest="data.code", default=None)
    group.add_argument("--resolution", dest="data.resolution", type=int, default=None)
    group.add_argument("--gaussian-blur", dest="data.gaussian_blur", type=float,
                       default=None,
                       help="Gaussian blur width in Angstrom for the computed "
                            "external potential")
    group.add_argument("--blur-method", dest="data.blur_method", default=None,
                       choices=["spectral", "ndimage"])

    group = parser.add_argument_group("model overrides")
    group.add_argument("--width", dest="model.width", type=int, default=None)
    group.add_argument("--modes", dest="model.modes", type=int, default=None)
    group.add_argument("--layers", dest="model.n_layers", type=int, default=None)
    group.add_argument("--projection", dest="model.projection_channels",
                       type=int, default=None)
    group.add_argument("--pauli-head", dest="model.pauli_residual",
                       action="store_const", const=True, default=None,
                       help="tau = tau_vW[rho] + s*softplus(f) for chg2tau")
    group.add_argument("--no-pauli-head", dest="model.pauli_residual",
                       action="store_const", const=False, default=None)

    group = parser.add_argument_group("training overrides")
    group.add_argument("--epochs", dest="training.epochs", type=int, default=None)
    group.add_argument("--batch-size", dest="training.batch_size", type=int,
                       default=None)
    group.add_argument("--learning-rate", dest="training.learning_rate",
                       type=float, default=None)
    group.add_argument("--valid-fraction", dest="training.valid_fraction",
                       type=float, default=None, metavar="F",
                       help="fraction of structures held out for validation "
                            "(default 0.2); 0 trains on every structure and "
                            "reports a training fit")
    group.add_argument("--kfold", dest="training.enable_kfold",
                       action="store_const", const=True, default=None,
                       help="run K-fold cross-validation instead, ignoring "
                            "--valid-fraction")
    group.add_argument("--k-folds", dest="training.k_folds", type=int,
                       default=None, help="number of folds (capped at the "
                                          "number of structures; equal to it "
                                          "is leave-one-out)")
    group.add_argument("--eval-epoch", dest="training.eval_epoch", type=int,
                       default=None, metavar="N",
                       help="evaluate and log every N epochs (default 10)")
    group.add_argument("--early-stopping", dest="training.early_stopping",
                       type=int, default=None, metavar="N",
                       help="stop after N epochs without validation "
                            "improvement and restore the best weights "
                            "(default 50); 0 disables. Needs a validation "
                            "split")
    group.add_argument("--seed", dest="training.seed", type=int, default=None,
                       help="seeds the split, the folds and the batch order")
    group.add_argument("--init-seed", dest="training.init_seed", type=int,
                       default=None, metavar="N",
                       help="seed the weight initialisation only, leaving the "
                            "data pipeline on --seed. Vary it across runs to "
                            "build a query-by-committee ensemble")
    group.add_argument("--device", dest="training.device", default=None,
                       help="auto | cuda | mps | cpu")

    group = parser.add_argument_group("output overrides")
    group.add_argument("--log", dest="output.log", default=None)
    group.add_argument("--json", dest="output.json", default=None)
    group.add_argument("--checkpoint-dir", dest="output.checkpoint_dir", default=None)
    group.add_argument("--plot-dir", dest="output.plot_dir", default=None)
    group.add_argument("--report-dir", dest="output.report_dir", default=None)
    group.add_argument("--no-plots", action="store_true",
                       help="skip figure generation")
    return parser


def run(argv=None):
    """Parse ``argv``, train every requested task, and return the results."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.write_config:
        os.makedirs(os.path.dirname(args.write_config) or ".", exist_ok=True)
        with open(args.write_config, "w") as handle:
            handle.write(SAMPLE_CONFIG_HEADER)
            handle.write(TrainingConfig().to_yaml())
        print(f"Sample configuration written to {args.write_config}")
        return None

    config = (TrainingConfig.from_yaml(args.config) if args.config
              else TrainingConfig())
    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("config", "write_config", "no_plots")}
    config.apply_overrides(overrides)
    if args.no_plots:
        config.output.plot_dir = None

    log = Tee(config.output.log)
    try:
        device = resolve_device(config.training.device)
        log("=" * 78)
        log("Poraque - Fourier Neural Operator training")
        log("=" * 78)
        log(f"  torch {torch.__version__}")
        log(f"  device: {describe_device(device)}  (requested "
            f"{config.training.device!r})")
        log(f"  config: {args.config or '<built-in defaults>'}")
        log("")
        for line in config.describe().splitlines():
            log(f"  {line}")
        log("")

        cache = build_cache(config, log)
        names = (["ext2chg", "chg2tau"] if config.task == "all" else [config.task])
        # One protocol, one variation: K-fold cross-validation.
        driver = run_task_kfold if config.training.enable_kfold else run_task
        results = [result for result in
                   (driver(name, cache, config, log) for name in names)
                   if result is not None]

        # ---------------- unified checkpoint ---------------- #
        # One file for the whole chain, written after every task has trained,
        # so the two halves cannot drift apart or be mixed across runs.
        operators = {r["task"]: r.pop("operator") for r in results
                     if r.get("operator") is not None}
        bundle = None
        if operators and config.output.checkpoint_dir:
            bundle = os.path.join(config.output.checkpoint_dir,
                                  BUNDLE_FILENAME)
            save_bundle(bundle, operators, metadata={
                "structures": sorted({name for r in results
                                      for name in r.get("train_structures", [])}),
                "resolution": config.data.resolution,
                "epochs": config.training.epochs,
            })
            for result in results:
                result["checkpoint"] = bundle
            log(f"\n  models -> {bundle}  ({', '.join(sorted(operators))})")
            if len(operators) < 2:
                log("  NOTE: the bundle holds one task. Run with task: all to")
                log("  produce a complete pipeline in a single file.")

        log(f"\n{'=' * 78}\nOVERALL\n{'=' * 78}")
        log(f"  {'task':<12s} {'rel L2 (mean)':>14s} {'R2 (mean)':>12s} "
            f"{'MAE (mean)':>14s}   basis")
        for result in results:
            if result.get("mode") == "kfold":
                values = [r["metrics"] for r in result["records"]]
                basis = (f"{result['n_folds']}-fold CV, "
                         f"{len(values)} validation structures")
            else:
                values = [v["metrics"] for v in result["per_material"].values()]
                basis = (f"training fit, {result['n_train']} structures"
                         if not result["validation"] else
                         f"train + {len(result['validation'])} validation")
            log(f"  {result['task']:<12s} "
                f"{np.mean([m['relative_l2'] for m in values]):14.4f} "
                f"{np.mean([m['r2'] for m in values]):12.4f} "
                f"{np.mean([m['mae'] for m in values]):14.5g}   {basis}")

        n_materials = len(FieldPairDataset(cache, task=names[0]))
        log("")
        log(f"  NOTE: {n_materials} material(s), all of one element. These numbers")
        log("  characterise the pipeline, not the science: they say nothing about")
        log("  transfer to new chemistry.")
        if any(r.get("mode") == "split" and not r.get("validation")
               for r in results):
            log("  For the runs marked 'training fit' above, no structure was held")
            log("  out, so those are not generalisation estimates at all.")

        # Archive the resolved config beside the results, so the run is
        # reproducible even if the source config is later edited.
        resolved = os.path.splitext(config.output.json)[0] + "_config.yaml"
        os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
        config.to_yaml(resolved)

        with open(config.output.json, "w") as handle:
            json.dump({"config": config.to_dict(), "device": str(device),
                       "results": results}, handle, indent=2, default=float)
        log(f"\n  log             -> {config.output.log}")
        log(f"  metrics         -> {config.output.json}")
        log(f"  resolved config -> {resolved}")
        if config.output.plot_dir:
            log(f"  figures         -> {config.output.plot_dir}")
        return results
    finally:
        log.close()


def main(argv=None):
    """Console entry point for ``poraque-train``.

    Returns a process exit status, because the ``[project.scripts]`` wrapper
    calls ``sys.exit(main())`` and would treat any other object as an error
    message. :func:`run` returns the result records themselves.
    """
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
