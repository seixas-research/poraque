#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_committee.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Query by committee: rank structures by how much the ensemble disagrees.

Reference data costs a plane-wave DFT run, so the question worth answering is
which structure to compute next. This scores every structure in the dataset by
the **Jensen-Shannon divergence** across an ensemble whose members differ only
in their weight initialisation, and orders them.

.. code-block:: text

    N models, same data, different init_seed
              |
              |  each predicts rho (or tau) for every structure
              v
    JSD across members, per structure  ->  ranking  ->  compute the top one

Why JSD
-------
The charge density is a probability density up to the electron count, which is
fixed by the pseudopotentials, so information-theoretic distances apply
directly. For an ensemble the right form is the divergence about the mean,

.. math::

    \mathrm{JSD} = \frac1K\sum_k D_{\rm KL}(p_k \,\|\, \bar p),

which is the mutual information between the prediction and the member index —
the quantity active learning maximises. It is symmetric, bounded by
:math:`\ln K`, and finite wherever the members are, none of which a raw
:math:`D_{\rm KL}` gives.

The :math:`L^2` spread and the spread of the integrated quantity are reported
beside it. They measure different things: JSD is normalised per member and so
is blind to a common rescaling, which is exactly the electron-count drift the
integral spread captures.

Validating the measure
----------------------
A disagreement measure is worth nothing until it is shown to *rank* correctly.
With ``--against``, the ranking is correlated against a set of reference errors
— for instance the cross-validated errors from a previous ``--kfold`` run — and
the Spearman coefficient is reported. Read Spearman: active learning consumes
an ordering.

Usage
-----
Installed (``pip install -e .``), this is the ``poraque-committee`` console
command and runs from any directory::

    # 1. train the members (same seed, different init_seed)
    for s in 0 1 2 3; do
      poraque-train --config configs/train_config.yaml \
          --init-seed $s --valid-fraction 0 \
          --checkpoint-dir models/committee_$s \
          --log logs/committee_$s.log --json logs/committee_$s.json \
          --no-plots --report-dir ""
    done

    # 2. rank the structures, and check the ranking against known errors
    poraque-committee --models "models/committee_*" \
        --task ext2chg --against logs/kfold12b.json

Running this file directly — ``python scripts/poraque_committee.py`` — is
equivalent, and needs nothing installed.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

# Run straight from a checkout, without installing, by preferring the in-tree
# package. Installed as the ``poraque-committee`` console script this module
# sits in site-packages, that directory does not exist, and the installed
# package wins.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from poraque.fields import ChargeDensity, FieldGrid  # noqa: E402
from poraque.ml import (  # noqa: E402
    BUNDLE_FILENAME,
    Committee,
    disagreement_error_correlation,
)
from poraque.ml.data import discover_materials  # noqa: E402
from poraque.ml.device import describe_device, resolve_device  # noqa: E402
from poraque.ml.tasks import resolve_task  # noqa: E402


def resolve_bundles(pattern):
    """Expand ``--models`` into one bundle path per member."""
    matches = sorted(glob.glob(pattern))
    paths = []
    for entry in matches:
        candidate = (os.path.join(entry, BUNDLE_FILENAME)
                     if os.path.isdir(entry) else entry)
        if os.path.isfile(candidate):
            paths.append(candidate)
    if len(paths) < 2:
        raise SystemExit(
            f"{pattern!r} matched {len(paths)} checkpoint(s); a committee "
            f"needs at least two. Train members with poraque-train --init-seed."
        )
    return paths


def reference_errors(path, task):
    """``{structure: relative_l2}`` from a previous run's JSON summary."""
    with open(path) as handle:
        payload = json.load(handle)

    errors = {}
    for result in payload.get("results", []):
        if result.get("task") != task:
            continue
        for record in result.get("records", []):
            errors[record["material"]] = record["metrics"]["relative_l2"]
        for name, entry in (result.get("per_material") or {}).items():
            errors[name] = entry["metrics"]["relative_l2"]
    return errors


def rank(argv=None):
    """Parse ``argv``, score every structure, and return the ranked records."""
    parser = argparse.ArgumentParser(
        description="Rank structures by committee disagreement (JSD).")
    parser.add_argument("--models", default="models/committee_*",
                        help="glob matching one directory or bundle per member")
    parser.add_argument("--task", default="ext2chg",
                        choices=["ext2chg", "chg2tau"])
    parser.add_argument("--cache", default="data/cache/res32",
                        help="cached dataset the members were trained on")
    parser.add_argument("--against", metavar="JSON", default=None,
                        help="a previous run's JSON, to correlate the ranking "
                             "against its measured errors")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", default=None, help="write a JSON summary")
    args = parser.parse_args(argv)

    task = resolve_task(args.task)
    device = resolve_device(args.device)
    paths = resolve_bundles(args.models)
    committee = Committee.from_bundles(paths, task.name, device=device)

    print("=" * 78)
    print(f"Query by committee - {task.name}: {task.input_field} -> "
          f"{task.target_field}")
    print("=" * 78)
    print(f"  members    : {len(committee)}  (init_seed "
          f"{committee.init_seeds})")
    print(f"  device     : {describe_device(device)}")
    print(f"  cache      : {args.cache}")

    materials = discover_materials(args.cache)
    if not materials:
        raise SystemExit(f"no materials under {args.cache!r}.")

    records = []
    for material in materials:
        grid = FieldGrid.from_file(os.path.join(material.directory,
                                                task.input_field))
        source = _read(material.directory, task.input_field, grid)
        target = _read(material.directory, task.target_field, grid)
        scored = committee.disagreement(source, reference=target.data)
        records.append({
            "material": material.identifier,
            "jsd": scored["jsd"],
            "jsd_normalised": scored["jsd_normalised"],
            "relative": scored["relative"],
            "integral_relative": scored["integral_relative"],
            "error": scored["error"],
            "ratio": scored["ratio"],
        })

    order = sorted(records, key=lambda r: -r["jsd"])
    print(f"\n  ranked by Jensen-Shannon divergence (most uncertain first):")
    print(f"    {'structure':<14s} {'JSD':>10s} {'JSD/lnK':>9s} "
          f"{'L2 spread':>10s} {'int spread':>11s} {'error':>9s}")
    print("    " + "-" * 68)
    for record in order:
        print(f"    {record['material']:<14s} {record['jsd']:10.3e} "
              f"{record['jsd_normalised']:9.4f} {record['relative']:10.4f} "
              f"{record['integral_relative']:11.4f} {record['error']:9.4f}")

    # ---------------- does the ranking mean anything? ---------------- #
    print(f"\n  calibration (committee spread vs its own error):")
    for key, label in (("jsd", "JSD"), ("relative", "L2 spread")):
        scored = [{"relative": r[key], "error": r["error"]} for r in records]
        stats = disagreement_error_correlation(scored)
        print(f"    {label:<11s} Spearman {stats['spearman']:+.3f}   "
              f"Pearson {stats['pearson']:+.3f}")

    if args.against:
        truth = reference_errors(args.against, task.name)
        shared = [r for r in records if r["material"] in truth]
        if len(shared) >= 3:
            print(f"\n  against measured errors in "
                  f"{os.path.basename(args.against)} ({len(shared)} structures):")
            for key, label in (("jsd", "JSD"), ("relative", "L2 spread")):
                scored = [{"relative": r[key],
                           "error": truth[r["material"]]} for r in shared]
                stats = disagreement_error_correlation(scored)
                print(f"    {label:<11s} Spearman {stats['spearman']:+.3f}   "
                      f"Pearson {stats['pearson']:+.3f}")
            print("\n    Spearman is the one to read: active learning consumes")
            print("    an ordering, not a calibrated magnitude.")
        else:
            print(f"\n  !! only {len(shared)} structures in common with "
                  f"{args.against}; need 3 to correlate.")

    print("\n" + "=" * 78)
    print("  Members differ only in initialisation, so this is optimisation")
    print("  variance -- a lower bound on the error, not a calibrated bar.")
    print("  Every member saw the same structures, so all of them can be")
    print("  confidently wrong about chemistry the dataset omits.")
    print("=" * 78)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump({"task": task.name, "members": paths,
                       "init_seeds": committee.init_seeds,
                       "records": records}, handle, indent=2, default=float)
        print(f"\n  summary -> {args.json}")
    return records


def _read(directory, filename, grid):
    """Read one cached field onto ``grid``."""
    from poraque.fields import ExternalPotential, KineticEnergyDensity

    classes = {"EXTCAR": ExternalPotential, "CHGCAR": ChargeDensity,
               "TAUCAR": KineticEnergyDensity}
    return classes[filename].read(os.path.join(directory, filename), grid=grid)


def main(argv=None):
    """Console entry point for ``poraque-committee``.

    Returns a process exit status, because the ``[project.scripts]`` wrapper
    calls ``sys.exit(main())`` and would treat any other object as an error
    message. :func:`rank` returns the scored records themselves.
    """
    rank(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
