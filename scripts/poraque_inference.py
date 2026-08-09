#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_inference.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Predict the electronic structure of a new crystal from its geometry alone.

The pipeline chains the two learned operators into what is, in effect, a
complete orbital-free DFT calculation:

.. code-block:: text

    POSCAR + INCAR + POTCAR
              |
              |  ExternalPotential  (analytic, no network)
              v
           EXTCAR   ---- Model 1 (ext2chg) ---->  CHGCAR / CHG
                                                       |
                                                       |  Model 2 (chg2tau)
                                                       v
                                                    TAUCAR

Only the geometry and the pseudopotentials are required: no wavefunctions, no
SCF cycle, no prior DFT run on this structure. The first step is *analytic* —
:class:`~poraque.fields.ExternalPotential` builds the local ionic potential in
closed form — and only the two field-to-field maps are learned.

Every output is written in ``CHGCAR`` format, so the predictions are read by
any tool that handles VASP volumetric files.

Both operators are read from a **single unified checkpoint**,
``models/poraque_models.pfno``, written by ``poraque-train``. One file for the
whole chain means the two halves cannot be copied separately or mixed across
training runs.

Grid selection
--------------
All three fields must share one mesh. Its shape is resolved in this order:

1. ``--grid`` given explicitly;
2. ``--like`` pointing at an existing volumetric file, whose grid is adopted —
   the right choice when the prediction must be compared point-by-point with a
   reference calculation;
3. ``--resolution``, which sizes a grid of that many points along the longest
   axis while preserving the cell's aspect ratio;
4. otherwise a cutoff and precision, through the ``ENCUT``/``PREC`` rule of
   :meth:`~poraque.fields.FieldGrid.from_parameters`. Those come from
   ``--from-incar`` when it is given, and otherwise from ``--encut`` (default
   :data:`DEFAULT_ENCUT` eV) with ``--prec-accurate``.

   ``--from-incar`` takes precedence over both flags rather than merging with
   them: a run described by an input file should reproduce that file's grid,
   and a flag silently modifying it would make the two disagree while
   appearing to agree. Anything overridden is named in the log.

.. warning::
   An FNO is resolution-flexible but not resolution-*indifferent*. Evaluating
   far from the grid density a model was trained on extrapolates in a way no
   metric here can detect, so the script warns when the requested grid departs
   markedly from the checkpoint's training resolution.

Usage
-----
Installed (``pip install -e .``), this is the ``poraque-inference`` console
command and runs from any directory::

    poraque-inference new_structure/ --output predictions/new

    # explicit bundle, and a coarser grid
    poraque-inference new_structure/ \
        --models models/poraque_models.pfno --encut 300

    # match an existing calculation's grid, for a direct comparison
    poraque-inference run/ --like run/CHGCAR --compare

    # a CHGCAR VASP will restart from with ICHARG=1: the PAW augmentation
    # records are copied from the reference, since no grid-based model can
    # produce the one-centre terms
    poraque-inference run/ --like run/CHGCAR --add-paw

Running this file directly — ``python scripts/poraque_inference.py`` — is
equivalent, and needs nothing installed.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

# Run straight from a checkout, without installing, by preferring the in-tree
# package. Installed as the ``poraque-inference`` console script this module
# sits in site-packages, that directory does not exist, and the installed
# package wins.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

import torch  # noqa: E402

from poraque import banner  # noqa: E402
from poraque.fields import (  # noqa: E402
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    KineticEnergyDensity,
    von_weizsacker_tau,
)
from poraque.fields.io import resolve_reader  # noqa: E402
from poraque.fields.resample import downsample_shape  # noqa: E402
from poraque.ml import (  # noqa: E402
    BUNDLE_FILENAME,
    resolve_bundle_path,
    infer_backbone_kwargs,
    load_bundle,
    read_bundle,
)
from poraque.ml.device import describe_device, resolve_device  # noqa: E402

#: Plane-wave cutoff (eV) that sizes the inference grid by default.
#:
#: Fixed rather than inherited from the structure's own ``INCAR``, because a
#: genuinely new structure has no prior calculation to inherit from, and the
#: same geometry must give the same grid either way.
#:
#: At the default ``PREC=Normal`` this puts a ~12 Å cell on 32^3 -- exactly the
#: ``resolution: 32`` the shipped models were trained at -- so the default
#: evaluates them where they were fitted rather than extrapolating.
#: ``--prec-accurate`` switches to VASP's factor-of-two rule and gives 42^3.
DEFAULT_ENCUT = 200.0


# ===================================================================== #
# Checkpoint handling
# ===================================================================== #
def load_operator(bundle, task, device):
    """
    Load one stage of the pipeline from the unified checkpoint.

    The architecture is inferred from the stored tensors, so no hyper-parameter
    has to be repeated on the command line — a mismatch there is otherwise a
    silent source of nonsense predictions.

    Parameters
    ----------
    bundle : str
        Path to ``poraque_models.pfno``.
    task : str
        ``"ext2chg"`` or ``"chg2tau"``.
    device : torch.device or str
        Target device.

    Returns
    -------
    tuple of (FieldOperator, dict)
        The operator and a small description of what was loaded.
    """
    try:
        operator = load_bundle(bundle, task, device=device)
    except (KeyError, ValueError) as error:
        raise SystemExit(f"{bundle}: {error}") from error

    state = read_bundle(bundle)[task]
    info = {
        "path": str(bundle),
        "task": task,
        "pauli_residual": bool(state.get("pauli_residual", False)),
        "pauli_scale": state.get("pauli_scale"),
        "hyperparameters": infer_backbone_kwargs(state["model_state"]),
        "training_resolution": state.get("training_resolution"),
    }
    return operator, info


# ===================================================================== #
# Grid resolution
# ===================================================================== #
#: Files a reference PAW augmentation record may be taken from, best first.
#: ``CHG`` is listed because it shares the layout, but it never carries the
#: records in practice -- VASP writes them only to ``CHGCAR``.
AUGMENTATION_SOURCES = ("CHGCAR", "AECCAR0", "CHG")


def collect_augmentation(directory, structure, shape, log):
    r"""
    Take the PAW one-centre records from a reference calculation.

    The model predicts :math:`\rho` on the plane-wave grid. VASP's
    ``ICHARG=1`` also wants the augmentation occupancies — the part of the
    density inside the PAW spheres, which is not representable on that grid at
    all and which no grid-based model can produce. They have to be borrowed.

    Returns
    -------
    list of str or None
        The records, or ``None`` when none could be used. Every rejection is
        explained rather than silently yielding a file VASP would refuse.
    """
    from poraque.fields.vasp.volumetric import (
        count_augmentation_records,
        read_augmentation,
    )

    log("\n        --add-paw: looking for reference augmentation records")
    for name in AUGMENTATION_SOURCES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        try:
            reference_shape, block = read_augmentation(path)
        except (OSError, ValueError, IndexError) as error:
            log(f"        !! {name}: could not be parsed ({error})")
            continue
        if not block:
            log(f"        {name}: no augmentation records (not a PAW CHGCAR)")
            continue

        records = count_augmentation_records(block)
        atoms = int(sum(structure.counts))
        if records != atoms:
            log(f"        !! {name}: {records} augmentation records for "
                f"{atoms} atoms — the reference is a different structure, "
                f"so its on-site occupancies do not correspond to these "
                f"positions. Skipping.")
            continue

        log(f"        {name}: {records} records for {atoms} atoms "
            f"({len(block)} lines)")
        # The records are on-site and grid-independent, so a mismatch here is
        # not an error in them -- but VASP reads the density on the grid the
        # file declares, and ICHARG=1 wants that to be the run's own FFT grid.
        if tuple(reference_shape) != tuple(shape):
            log(f"        !! the prediction is on {tuple(shape)} and the "
                f"reference on {tuple(reference_shape)}. The records are "
                f"still valid, but VASP expects the CHGCAR grid to match "
                f"NGXF/NGYF/NGZF — rerun with --like {path} to match it.")
        return block

    return None


def augmentation_from_bundle(bundle, structure, log):
    r"""
    Fall back to the per-element table the model carries.

    Averaged over the training calculations at training time, so a structure
    with no reference of its own can still be written as an ``ICHARG=1``
    restart. It is an approximation to on-site terms that are properly a
    property of the converged wavefunctions, and the log says so.

    Returns
    -------
    list of str or None
    """
    from poraque.fields.vasp.augmentation import records_for_structure
    from poraque.ml import read_bundle

    try:
        metadata = read_bundle(bundle).get("metadata") or {}
    except (OSError, KeyError, ValueError):
        return None
    reference = metadata.get("paw_reference") or {}
    if not reference:
        return None

    lines, missing = records_for_structure(structure, reference)
    if missing:
        log(f"        !! the bundle's PAW reference covers "
            f"{sorted(reference)} but this structure also contains "
            f"{missing} — no records were written, because a file with them "
            f"for some atoms and not others is worse than one with none.")
        return None

    covered = ", ".join(
        f"{element} (averaged over {entry['atoms']} atoms in "
        f"{entry['structures']} structures)"
        for element, entry in sorted(reference.items()))
    log(f"        using the bundle's PAW reference: {covered}")
    log("        !! these are AVERAGED on-site terms, not this structure's. "
        "They are a starting guess for ICHARG=1, not a converged on-site "
        "density — about 9 % RMS from the truth on the reference dataset.")
    return lines


def resolve_augmentation(directory, bundle, structure, shape, log):
    """
    A reference calculation if there is one, the bundle's table otherwise.

    A real calculation beside the structure always wins: its records are that
    system's, where the bundle's are an average over other systems.
    """
    block = collect_augmentation(directory, structure, shape, log)
    if block:
        return block

    block = augmentation_from_bundle(bundle, structure, log)
    if block:
        return block

    raise SystemExit(
        f"--add-paw found no PAW augmentation records to use.\n"
        f"None of {list(AUGMENTATION_SOURCES)} in {directory!r} carries any, "
        f"and {bundle} has no stored per-element reference either.\n"
        f"The records are the one-centre part of the density, inside the PAW "
        f"spheres; they are not representable on the plane-wave grid, so no "
        f"model predicts them. Either run VASP once on this geometry "
        f"(LCHARG=.TRUE.) and point at that directory, or retrain so the "
        f"bundle carries a reference, or drop --add-paw and use the file for "
        f"visualisation rather than as an ICHARG=1 restart.")


def read_reference(path, field_class, grid):
    """
    Read a reference field and bring it onto ``grid``.

    A reference calculation is normally stored on its native, much finer mesh,
    while the prediction lives on whatever grid inference used. Comparing them
    requires putting both on one mesh, and the correct operator for that is a
    **Fourier truncation**, not interpolation: it is the exact band-limited
    projection of a plane-wave field, and it preserves the cell average — hence
    the electron count — to machine precision.

    Returns
    -------
    ScalarField
        The reference, on ``grid``.
    """
    from poraque.fields.resample import resample_field

    native = FieldGrid.from_file(path)
    field = field_class.read(path, grid=native)
    if tuple(native.shape) == tuple(grid.shape):
        return field
    return resample_field(field, grid.shape, grid=grid)


#: ``PREC`` values that ask for a wrap-around-free density grid.
ACCURATE_PRECISIONS = ("accurate", "high")


def resolve_cutoff_settings(args, log):
    r"""
    Settle ``ENCUT`` and ``PREC`` from the flags and any supplied ``INCAR``.

    ``--from-incar`` wins outright over ``--encut`` and ``--prec-accurate``:
    a run described by an input file should reproduce that file's grid, and a
    flag silently modifying it would make the two disagree while appearing to
    agree. Anything overridden is named in the log rather than dropped
    quietly.

    Returns
    -------
    tuple of (float, str, str)
        Cutoff in eV, precision, and a phrase describing where they came from.
    """
    manual_prec = "accurate" if args.prec_accurate else "normal"
    if not args.from_incar:
        return float(args.encut), manual_prec, "--encut"

    from poraque.fields.vasp.incar import Incar

    if not os.path.exists(args.from_incar):
        raise SystemExit(f"--from-incar {args.from_incar!r} does not exist.")
    try:
        incar = Incar.from_file(args.from_incar)
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"--from-incar {args.from_incar!r} could not be parsed: {error}"
        ) from error

    cutoff = incar.get_float("ENCUT")
    precision = str(incar.get("PREC", "normal")).strip().lower()
    if cutoff is None:
        raise SystemExit(
            f"{args.from_incar} sets no ENCUT, so it cannot size a grid. "
            f"Add one, or use --encut instead.")

    overridden = []
    if args.encut != DEFAULT_ENCUT:
        overridden.append(f"--encut {args.encut:g}")
    if args.prec_accurate and precision not in ACCURATE_PRECISIONS:
        overridden.append("--prec-accurate")
    if overridden:
        log(f"  !! --from-incar overrides {', '.join(overridden)}: the grid "
            f"comes from {args.from_incar} alone.")

    return cutoff, precision, f"from {args.from_incar}"


def vasp_native_grid(structure, args, log):
    r"""
    The grid VASP itself would build for this cell, cutoff and precision.

    :meth:`~poraque.fields.FieldGrid.from_encut` sizes a grid the way a
    plane-wave code *should*; this reproduces what VASP actually does, which is
    not the same thing. VASP rounds the **coarse** grid to an FFT-friendly
    size and only then doubles it for the density, so a 27-atom gold cell at
    450 eV gets 128 points where rounding the density size in one step gives
    64 — a factor of two, and a ``CHGCAR`` VASP would refuse on a restart.

    Reproduced from ``main.F``; validated against every reference calculation
    in this project, 17 of 17 exact.

    Returns
    -------
    FieldGrid
    """
    from poraque.fields.vasp.fftgrid import vasp_grid_shapes

    cutoff, precision, source = resolve_cutoff_settings(args, log)
    coarse, fine = vasp_grid_shapes(structure.cell, cutoff, prec=precision)
    log(f"  grid: {fine} (--to-vasp: VASP's own rule at ENCUT={cutoff:g} eV, "
        f"PREC={precision} ({source}); its coarse grid would be {coarse})")
    return FieldGrid(fine, structure.cell, encut=cutoff, prec=precision)


def resolve_grid(structure, parameters, pseudopotentials, args, log):
    """Choose the shared grid for every predicted field."""
    if getattr(args, "to_vasp", False):
        # Ahead of --resolution and the cutoff path: the point of the flag is
        # a grid VASP will accept, and anything that reshapes it defeats that.
        # --grid and --like stay in front, since both name a shape outright.
        if not (args.grid or args.like):
            return vasp_native_grid(structure, args, log)
        log("  note: --to-vasp is ignored; --grid/--like set the shape "
            "explicitly.")
    if args.grid:
        shape = tuple(int(n) for n in args.grid)
        log(f"  grid: {shape} (explicit --grid)")
        return FieldGrid(shape, structure.cell)

    if args.like:
        grid = FieldGrid.from_file(args.like)
        log(f"  grid: {grid.shape} (adopted from {args.like})")
        if not np.allclose(grid.cell, structure.cell, atol=1e-4):
            warnings.warn(
                f"{args.like} has a different cell from the structure; using "
                f"the structure's cell with the reference grid shape.",
                RuntimeWarning,
            )
            grid = FieldGrid(grid.shape, structure.cell)
        return grid

    # The default cutoff is Poraque's own (DEFAULT_ENCUT), not the one the
    # reference calculation happened to use: inference must produce the same
    # grid for the same geometry whether or not an INCAR is present, since a
    # new structure generally has no prior calculation to inherit from.
    # `--from-incar` is the deliberate exception, and says so in the log.
    cutoff, precision, source = resolve_cutoff_settings(args, log)
    derived = FieldGrid.from_parameters(structure, parameters, pseudopotentials,
                                        encut=cutoff, prec=precision)
    if args.resolution:
        shape = downsample_shape(derived.shape, target_max=args.resolution)
        log(f"  grid: {shape} (--resolution {args.resolution}; "
            f"ENCUT={cutoff:g} eV at PREC={precision} would give "
            f"{derived.shape})")
        return FieldGrid(shape, structure.cell, encut=derived.encut)

    stated = parameters.cutoff if parameters else None
    origin = f"ENCUT={cutoff:g} eV, PREC={derived.prec} ({source})"
    if stated and abs(float(stated) - float(cutoff)) > 1e-6:
        origin += f"; the calculation's own ENCUT is {stated:g} eV"
    log(f"  grid: {derived.shape} ({origin})")
    return derived


def warn_on_resolution_shift(grid, info, log):
    """Warn when the evaluation grid is far from the training resolution."""
    trained = info.get("training_resolution")
    if not trained:
        return
    longest = max(grid.shape)
    if not (0.5 * trained <= longest <= 2.0 * trained):
        log(f"  !! WARNING: this model was trained at resolution ~{trained} but "
            f"is being evaluated at {longest}. An FNO is resolution-flexible, "
            f"not resolution-indifferent; treat the result with suspicion.")


# ===================================================================== #
# Pipeline
# ===================================================================== #
def run(args, log):
    """Execute the geometry -> EXTCAR -> CHGCAR -> TAUCAR pipeline."""
    device = resolve_device(args.device)
    log("=" * 78)
    log("Poraque - FNO inference: crystal geometry -> charge and kinetic densities")
    log("=" * 78)
    log(f"  torch {torch.__version__}   device: {describe_device(device)}")
    log(f"  structure directory: {args.directory}")
    # Only on CPU, because that is the only place it is consulted. Printed
    # because a silent optimisation is one nobody can tell is missing: a run
    # that falls back to PyTorch is 2-3x slower at cache resolutions and looks
    # exactly the same otherwise.
    if device.type == "cpu":
        from poraque.ml.backend import describe as describe_backend

        log(f"  {describe_backend()}")

    # ---------------- 1. geometry and pseudopotentials ---------------- #
    reader = resolve_reader(args.directory, args.code)
    structure = reader.read_structure(args.directory)
    parameters = reader.read_parameters(args.directory)
    pseudopotentials = reader.read_pseudopotentials(args.directory)

    log(f"  code: {reader.code}")
    log(f"  structure: {structure.formula}  {structure.natoms} atoms  "
        f"V = {structure.volume:.3f} A^3")
    for element, entry in pseudopotentials.items():
        log(f"  pseudopotential: {element}  ZVAL = {entry.valence_charge:g}  "
            f"RCORE = {entry.core_radius:.4f} A")
    if not pseudopotentials and not args.zval:
        raise SystemExit(
            "No pseudopotential file found and no --zval given; the valence "
            "charges are required to build the external potential."
        )

    zval = None
    if args.zval:
        zval = {}
        for item in args.zval:
            element, _, value = item.partition("=")
            zval[element.strip()] = float(value)
        log(f"  valence-charge overrides: {zval}")

    grid = resolve_grid(structure, parameters, pseudopotentials, args, log)
    os.makedirs(args.output, exist_ok=True)

    results = {
        "directory": os.path.abspath(args.directory),
        "formula": structure.formula,
        "natoms": structure.natoms,
        "volume": structure.volume,
        "grid": list(grid.shape),
        "device": str(device),
        "outputs": {},
    }

    # ---------------- 2. analytic external potential ---------------- #
    start = time.time()
    potential = ExternalPotential.from_calculation(
        args.directory, code=reader.code, grid=grid,
        rcore_factor=args.rcore_factor, sigma=args.sigma, zval=zval,
    )
    log(f"\n  [1/3] EXTCAR  (analytic, no network)   {time.time() - start:.2f} s")
    model_used = potential.metadata.get("model", "unknown")
    if model_used == "potcar":
        log("        model: tabulated POTCAR local potential (exact; "
            "reproduces VASP EXTCAR)")
        log("        PSGMAX = " + ", ".join(
            f"{k} {v:.3f} 1/A"
            for k, v in potential.metadata.get("psgmax", {}).items()))
    else:
        log(f"        model: {model_used}   sigma = " + ", ".join(
            f"{k} {v:.4f} A"
            for k, v in potential.metadata.get("widths", {}).items()))
    log(f"        range [{potential.data.min():.4f}, {potential.data.max():.4f}] eV"
        f"   mean {potential.mean():+.3e} eV")

    extcar = os.path.join(args.output, "EXTCAR")
    potential.write(extcar)
    results["outputs"]["EXTCAR"] = extcar
    log(f"        -> {extcar}")

    # ---------------- 3. Model 1: EXTCAR -> CHGCAR ---------------- #
    operator, info = load_operator(args.models, "ext2chg", device)
    log(f"\n  [2/3] CHGCAR  via {os.path.basename(args.models)} [ext2chg]")
    log(f"        model: {type(operator.model).__name__}  "
        f"{info['hyperparameters']}")
    warn_on_resolution_shift(grid, info, log)

    start = time.time()
    density = operator.predict(potential)
    log(f"        predicted in {time.time() - start:.2f} s")

    electrons = density.integrate()
    expected = sum(
        (zval or {}).get(element,
                         pseudopotentials[element].valence_charge
                         if element in pseudopotentials else 0.0) * count
        for element, count in zip(structure.elements, structure.counts)
    )
    log(f"        range [{density.data.min():.5f}, {density.data.max():.5f}] e/A^3")
    log(f"        integral = {electrons:.4f} electrons"
        + (f"   (expected {expected:.1f}, error "
           f"{100 * abs(electrons - expected) / expected:.3f} %)"
           if expected else ""))
    negative = int(np.count_nonzero(density.data < 0))
    if negative:
        log(f"        !! {negative} negative voxels "
            f"({100 * negative / density.data.size:.4f} %), min "
            f"{density.data.min():.4g} e/A^3")

    # The number of valence electrons is fixed by the pseudopotentials, so it
    # is a constraint rather than something to predict. VASP checks it on read
    # (`BRMIX` in broyden.F, tolerance 1e-5 relative) and, when it disagrees,
    # forces the total by shifting the G=0 component alone -- which dumps the
    # entire discrepancy into a uniform background instead of onto the atoms.
    # Scaling multiplicatively here fixes the total while preserving the
    # predicted shape and its non-negativity, which is a far better restart.
    results["electrons_raw"] = electrons
    results["normalized"] = False
    if args.normalize and expected and electrons > 0:
        factor = expected / electrons
        density.data = density.data * factor
        results["normalized"] = True
        results["normalization_factor"] = factor
        log(f"        normalized to {expected:.4f} electrons "
            f"(x {factor:.6f}); pass --no-normalize to keep the raw "
            f"prediction")
        # Quote the deficit on the same basis as the line above, so the two
        # percentages agree instead of differing by their denominator.
        deficit = abs(electrons - expected) / expected
        if deficit > 0.05:
            log(f"        !! the raw prediction was {100 * deficit:.1f} % "
                f"off. Renormalizing fixes the total charge, not the shape "
                f"-- an error this large means the structure is far from the "
                f"training set and the density is unreliable however it is "
                f"scaled.")
        electrons = density.integrate()

    augmentation = None
    if getattr(args, "add_paw", False):
        augmentation = resolve_augmentation(args.directory, args.models,
                                            structure, grid.shape, log)
        results["paw_augmentation"] = {
            "records": len(augmentation) if augmentation else 0,
        }

    chgcar = os.path.join(args.output, "CHGCAR")
    density.write(chgcar, augmentation=augmentation)
    results["outputs"]["CHGCAR"] = chgcar
    log(f"        -> {chgcar}")

    if args.write_chg:
        chg = os.path.join(args.output, "CHG")
        density.write(chg, columns=10, width=11, decimals=5)
        results["outputs"]["CHG"] = chg
        log(f"        -> {chg}  (coarse CHG formatting)")

    results["electrons"] = electrons
    results["electrons_expected"] = expected or None

    # ---------------- 4. Model 2: CHGCAR -> TAUCAR ---------------- #
    operator2, info2 = load_operator(args.models, "chg2tau", device)
    log(f"\n  [3/3] TAUCAR  via {os.path.basename(args.models)} [chg2tau]")
    log(f"        model: {type(operator2.model).__name__}  "
        f"{info2['hyperparameters']}"
        + ("   [tau = tau_vW + softplus head]" if info2["pauli_residual"] else ""))
    warn_on_resolution_shift(grid, info2, log)

    start = time.time()
    tau = operator2.predict(density)
    log(f"        predicted in {time.time() - start:.2f} s")
    log(f"        range [{tau.data.min():.4f}, {tau.data.max():.4f}] eV/A^3")
    log(f"        integral = {tau.integrate():.4f} eV  (total T_s)")

    # The Hoffmann-Ostenhof bound is a theorem; check it even when the head
    # guarantees it, because it also validates the chained density.
    bound = von_weizsacker_tau(density.data, grid)
    deficit = tau.data - bound
    violations = int(np.count_nonzero(deficit < -1e-6))
    log(f"        constraint tau >= tau_vW[rho]: {violations}/{deficit.size} "
        f"violated ({100 * violations / deficit.size:.4f} %), min margin "
        f"{deficit.min():+.4g} eV/A^3")
    results["kinetic_energy"] = tau.integrate()
    results["constraint"] = {"violations": violations,
                             "fraction": violations / deficit.size,
                             "worst_deficit": float(deficit.min())}

    taucar = os.path.join(args.output, "TAUCAR")
    tau.write(taucar)
    results["outputs"]["TAUCAR"] = taucar
    log(f"        -> {taucar}")

    # ---------------- 4b. Hartree potential, solved not predicted ------- #
    # Poisson's equation is exact and costs two FFTs, so there is no third
    # operator here and no error introduced beyond whatever the density
    # already carries.
    if args.write_locpot:
        from poraque.fields import HartreePotential

        log("\n  [extra] LOCPOT  via Poisson's equation (no model)")
        hartree = HartreePotential.from_density(density)
        combined = args.locpot_total
        field = (hartree.total_with(potential) if combined else hartree)
        log(f"        v_H range [{hartree.data.min():.4f}, "
            f"{hartree.data.max():.4f}] eV")
        if combined:
            log("        writing v_H + V_ext, as a plain VASP LOCPOT holds")
            log(f"        total range [{field.data.min():.4f}, "
                f"{field.data.max():.4f}] eV")

        locpot = os.path.join(args.output, "LOCPOT")
        field.write(locpot)
        results["outputs"]["LOCPOT"] = locpot
        results["hartree"] = {
            "min": float(hartree.data.min()),
            "max": float(hartree.data.max()),
            "includes_external": bool(combined),
        }
        log(f"        -> {locpot}")

    # ---------------- 5. optional comparison ---------------- #
    if args.compare:
        log(f"\n  --- comparison against reference files in {args.directory} ---")
        results["comparison"] = {}
        for filename, field_class, predicted in (
            ("EXTCAR", ExternalPotential, potential),
            ("CHGCAR", ChargeDensity, density),
            ("TAUCAR", KineticEnergyDensity, tau),
        ):
            path = os.path.join(args.directory, filename)
            if not os.path.exists(path):
                log(f"      {filename:<8s} no reference file")
                continue
            reference = read_reference(path, field_class, grid)
            difference = predicted.data - reference.data
            entry = {
                "relative_l2": float(np.linalg.norm(difference)
                                     / np.linalg.norm(reference.data)),
                "mae": float(np.mean(np.abs(difference))),
                "rmse": float(np.sqrt(np.mean(difference ** 2))),
                "pearson_r": float(np.corrcoef(predicted.data.ravel(),
                                               reference.data.ravel())[0, 1]),
            }
            results["comparison"][filename] = entry
            log(f"      {filename:<8s} relative L2 {entry['relative_l2']:8.4f}   "
                f"MAE {entry['mae']:10.5g}   r {entry['pearson_r']:7.4f}")

    # ---------------- 6. energy decomposition ---------------- #
    if args.functional != "skip":
        log(f"\n  --- energy components (xc: {args.functional}) ---")
        from poraque.physics import EnergyCalculator

        pscore = None
        potcar_path = os.path.join(args.directory, "POTCAR")
        if os.path.exists(potcar_path):
            from poraque.fields.vasp.potcar import Potcar

            entries = Potcar.from_file(potcar_path, parse_tables=True)
            pscore = {e.element: e.pscore for e in entries
                      if e.pscore is not None}

        energy = EnergyCalculator.from_potential(
            potential, pscore=pscore, functional=args.functional)
        components = energy.compute(density, tau, potential)
        for line in str(components).splitlines():
            log(f"    {line}")
        results["energy"] = components.as_dict()

        log("\n    NOTE: a pseudo-valence energy. The PAW one-centre terms")
        log("    are absent, so this is not a DFT total energy, and on the")
        log("    current models the error on energy differences exceeds the")
        log("    differences themselves. See docs/source/energy/index.md.")

    # ---------------- 7. optional figures ---------------- #
    if args.plot_dir:
        from poraque.vis import TrainingReport

        report = TrainingReport(args.plot_dir, dpi=args.dpi,
                                prefix=os.path.basename(
                                    os.path.normpath(args.directory)))
        produced = []
        for filename, field_class, predicted, label, unit, positive in (
            ("CHGCAR", ChargeDensity, density, r"$\rho$", r"e/$\AA^3$", True),
            ("TAUCAR", KineticEnergyDensity, tau, r"$\tau$", r"eV/$\AA^3$", True),
        ):
            path = os.path.join(args.directory, filename)
            if os.path.exists(path):
                reference = read_reference(path, field_class, grid)
                report.prefix = (f"{os.path.basename(os.path.normpath(args.directory))}"
                                 f"_{filename}")
                produced.append(report.field_comparison(
                    reference, predicted, label=label, unit=unit, log=positive,
                    title=f"predicted vs reference {filename}"))
                produced.append(report.parity(reference, predicted, label=label,
                                              unit=unit, log=positive))
        results["figures"] = produced
        log(f"\n  figures: {len(produced)} written to {args.plot_dir}"
            if produced else
            "\n  figures: none (no reference fields to compare against)")

    log(f"\n{'=' * 78}")
    log("Predicted fields are in CHGCAR format.")
    log("=" * 78)
    return results


def build_parser():
    """
    The command-line interface, as a parser.

    Separated from :func:`predict` so the flags can be inspected without
    running an inference — which needs a trained bundle, a POTCAR and a grid,
    none of which a test of the interface should require.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Predict CHGCAR and TAUCAR for a new structure from its "
                    "geometry, using the trained Fourier Neural Operators.",
    )
    parser.add_argument("directory",
                        help="directory with POSCAR/INCAR/POTCAR of the new structure")
    parser.add_argument("--models", "-m",
                        default=os.path.join("models", BUNDLE_FILENAME),
                        help=f"unified checkpoint holding both operators "
                             f"(default: models/{BUNDLE_FILENAME})")
    parser.add_argument("--output", "-o", default="predictions",
                        help="directory for the predicted volumetric files")

    parser.add_argument("--code", default="auto", help="DFT code, or 'auto'")
    parser.add_argument("--encut", type=float, default=DEFAULT_ENCUT,
                        metavar="EV",
                        help=f"plane-wave cutoff in eV setting the grid "
                             f"resolution (default: {DEFAULT_ENCUT:g}). Used "
                             f"unless --grid, --like or --resolution is given, "
                             f"and IGNORED when --from-incar is given")
    parser.add_argument("--prec-accurate", action="store_true",
                        help="size the grid with VASP's PREC=Accurate rule "
                             "(density grid twice the wavefunction cutoff, "
                             "wrap-around free) instead of the cheaper "
                             "PREC=Normal 3/2 rule. IGNORED when --from-incar "
                             "is given")
    parser.add_argument("--from-incar", metavar="INCAR", default=None,
                        help="take ENCUT and PREC from this INCAR. TAKES "
                             "PRECEDENCE over --encut and --prec-accurate, "
                             "which are then ignored and reported as "
                             "overridden")
    parser.add_argument("--to-vasp", action="store_true",
                        help="size the grid with VASP's own rule rather than "
                             "the generic plane-wave one, so the CHGCAR "
                             "matches the NGXF/NGYF/NGZF a VASP run would "
                             "build and can seed ICHARG=1. Combine with "
                             "--from-incar to follow a specific input file; "
                             "overridden by --grid and --like, and it "
                             "supersedes --resolution")
    parser.add_argument("--grid", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                        help="explicit grid shape (overrides --encut)")
    parser.add_argument("--like", metavar="FILE",
                        help="adopt the grid of an existing volumetric file")
    parser.add_argument("--resolution", type=int,
                        help="longest grid axis, preserving the cell aspect ratio")
    parser.add_argument("--rcore-factor", type=float, default=0.5,
                        help="Gaussian width as a multiple of the core radius")
    parser.add_argument("--sigma", type=float, default=None,
                        help="explicit Gaussian width in Angstrom for every species")
    parser.add_argument("--zval", nargs="*", metavar="EL=CHARGE",
                        help="valence-charge overrides, e.g. Au=11")

    parser.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    parser.add_argument("--write-chg", action="store_true",
                        help="also write a CHG-formatted copy of the density")
    parser.add_argument("--write-locpot", action="store_true",
                        help="also solve Poisson's equation for the Hartree "
                             "potential and write it as LOCPOT. Exact, not "
                             "predicted: v_H(G) = 4 pi e^2 rho(G) / G^2")
    parser.add_argument("--locpot-total", action="store_true",
                        help="make --write-locpot emit v_H + V_ext, the total "
                             "local potential a plain VASP LOCPOT holds, "
                             "rather than the Hartree term alone")
    parser.add_argument("--no-normalize", dest="normalize",
                        action="store_false", default=True,
                        help="do not rescale the predicted density to the "
                             "electron count implied by the pseudopotentials. "
                             "The count is an exact constraint and VASP "
                             "requires the CHGCAR to satisfy it to 1e-5 "
                             "relative, so the rescaling is on by default; "
                             "turn it off to inspect the raw prediction")
    parser.add_argument("--add-paw", action="store_true",
                        help="append the PAW augmentation records from a "
                             "reference CHGCAR in the input directory, which "
                             "VASP requires to restart from the predicted "
                             "density with ICHARG=1")
    parser.add_argument("--compare", action="store_true",
                        help="compare against reference files in the input directory")
    parser.add_argument("--functional", default="pbe",
                        choices=["pbe", "lda", "pbe-x", "lda-x", "none", "skip"],
                        help="exchange-correlation approximation for the energy "
                             "decomposition (default: pbe, matching the "
                             "PAW_PBE reference data); 'skip' omits the energy "
                             "report entirely")
    parser.add_argument("--plot-dir", default=None,
                        help="write comparison figures to this directory")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--json", default=None, help="write a JSON summary")
    return parser


def predict(argv=None):
    """Parse ``argv``, run the prediction, and return the result records."""
    args = build_parser().parse_args(argv)

    lines = []

    def log(message=""):
        print(message)
        lines.append(str(message))

    args.models = resolve_bundle_path(args.models, log)
    results = run(args, log)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2, default=float)
        print(f"\nJSON summary -> {args.json}")

    log_path = os.path.join(args.output, "inference.log")
    with open(log_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"log          -> {log_path}")
    return results


def main(argv=None):
    """Console entry point for ``poraque-inference``.

    Returns a process exit status, because the ``[project.scripts]`` wrapper
    calls ``sys.exit(main())`` and would treat any other object as an error
    message. :func:`predict` returns the result records themselves.
    """
    banner()
    predict(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
