#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_committee.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
**Calibration.** Does committee disagreement actually predict error?

.. rubric:: This command, or ``poraque-active-learning``?

They are the two halves of one loop and they are not interchangeable. The
difference is the data each one runs on, and it decides everything else:

==================  ==========================  ==============================
                    ``poraque-committee``       ``poraque-active-learning``
==================  ==========================  ==============================
runs on             a **labelled** dataset      an **unlabelled** pool
                    (``--cache``): inputs       (``--pool``): inputs only,
                    *and* targets               targets not computed yet
answers             "is this measure worth      "which structures should I
                    trusting?"                  compute next?"
produces            a Spearman correlation      a ranking, and a transfer of
                    between disagreement and    the top K into the training
                    the *known* error           set
costs               nothing — the DFT is        the DFT runs it selects
                    already done
==================  ==========================  ==============================

Run this one **first**. Disagreement is only a proxy for error, and until the
correlation is measured on data where the error is known, a ranking built from
it is a guess with a decimal point. Once the Spearman coefficient says the
ordering is sound, ``poraque-active-learning`` is what spends the budget.

Both share the same committee, the same Jensen-Shannon divergence and the same
ranking table (:func:`poraque.ml.active_learning.format_ranking`), which is why
the two outputs look alike — they differ in what the number is *for*.

.. rubric:: What it measures

Scores every structure by the **Jensen-Shannon divergence** across an ensemble
whose members differ only in their weight initialisation, and orders them.

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

    # 1. train the members (same seed, different init_seed). One --name per
    #    member is all it takes: everything each run writes lands in
    #    models/committee_<s>/, weights and log together.
    for s in 0 1 2 3; do
      poraque-train --config configs/train_config.yaml \
          --init-seed $s --name committee_$s \
          --valid-fraction 0 --no-plots --no-report
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
from poraque.ml.active_learning import format_ranking  # noqa: E402
from poraque.ml.data import discover_materials  # noqa: E402
from poraque.ml.device import describe_device, resolve_device  # noqa: E402
from poraque.ml.tasks import resolve_task  # noqa: E402


def member_bundle(entry):
    """
    The checkpoint inside one member directory, whatever the run named it.

    ``poraque-train`` writes ``<output.root>/<name>/<name>.pfno``, and
    ``task.name`` is exactly the key a user is told to set so two runs cannot
    overwrite each other. Looking only for :data:`BUNDLE_FILENAME` therefore
    finds the members of a *default* run and none of the members of a named
    one — and reports it as "no committee here", which sends the user back to
    train the members they already trained.

    Returns
    -------
    str or None
        Path to the bundle, or ``None`` when the directory holds no checkpoint.

    Raises
    ------
    SystemExit
        When a directory holds several checkpoints and none is the default
        name, since picking one arbitrarily would silently build a committee
        out of unrelated models.
    """
    if not os.path.isdir(entry):
        return entry if os.path.isfile(entry) else None

    default = os.path.join(entry, BUNDLE_FILENAME)
    if os.path.isfile(default):
        return default

    found = sorted(glob.glob(os.path.join(entry, "*.pfno")))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise SystemExit(
            f"{entry} holds {len(found)} checkpoints "
            f"({', '.join(os.path.basename(p) for p in found)}) and none is "
            f"the default {BUNDLE_FILENAME}. Point --models at the files "
            f"themselves so the members are unambiguous."
        )
    return None


def resolve_bundles(pattern):
    """Expand ``--models`` into one bundle path per member."""
    paths = [bundle for bundle in
             (member_bundle(entry) for entry in sorted(glob.glob(pattern)))
             if bundle is not None]
    if len(paths) < 2:
        raise SystemExit(
            f"{pattern!r} matched {len(paths)} checkpoint(s); a committee "
            f"needs at least two. Train members with poraque-train "
            f"--init-seed S --name committee_S, one per seed."
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
        description="CALIBRATION: does committee disagreement predict error? "
                    "Correlates the two on data where the error is known.",
        epilog="Runs on a LABELLED dataset (--cache): inputs AND targets, so "
               "the true error is available to correlate against. Its sibling "
               "poraque-active-learning runs on an unlabelled pool and spends "
               "a DFT budget on the ranking. Run this one first -- read the "
               "Spearman coefficient before trusting that ordering.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
    print("CALIBRATION: does disagreement predict error on data where the")
    print("error is known? Read the Spearman coefficient below.")
    print("(To spend a DFT budget on a ranking, that is")
    print(" poraque-active-learning, on an unlabelled pool.)")
    print("=" * 78)
    print(f"  members    : {len(committee)}  (init_seed "
          f"{committee.init_seeds})")
    print(f"  device     : {describe_device(device)}")
    print(f"  cache      : {args.cache}")

    # This task's two fields, not the default triple. Asking for TAUCAR as
    # well made an ext2chg dataset -- which by definition has no TAUCAR --
    # report as "no materials", so the very data the committee was trained on
    # could not be ranked.
    materials = discover_materials(args.cache, required=task.required_files)
    if not materials:
        raise SystemExit(
            f"no materials under {args.cache!r} carry both "
            f"{task.input_field} and {task.target_field}, which "
            f"{task.name} needs.")

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

    # `disagreement` returns jsd=None by design when the members predict a
    # signed field -- which is exactly what an under-trained density model
    # does. Sorting on that raised `bad operand type for unary -: NoneType`
    # and lost the whole ranking, including the structures that did score.
    ranked = [record for record in records if record["jsd"] is not None]
    unscored = [record for record in records if record["jsd"] is None]

    print(f"\n  ranked by Jensen-Shannon divergence (most uncertain first): "
          f"{len(ranked)} of {len(records)}")
    if ranked:
        # The same table poraque-active-learning prints, from the same
        # definition: it is the same measure over the same committee, and two
        # hand-written tables had drifted apart in width, precision and column
        # name.
        print(format_ranking(ranked))
    if unscored:
        print(f"\n    {len(unscored)} structure(s) have no JSD: the members "
              f"predict a signed field there, which is")
        print("    not a density and has no divergence. The L2 spread below "
              "still applies to them.")
        for record in unscored:
            print(f"      {record['material']:<18s} L2 spread "
                  f"{record['relative']:.4f}")
    if not ranked:
        print("\n  no structure could be ranked by JSD; train the members "
              "further before")
        print("  spending a DFT budget on this ordering.")

    # ---------------- does the ranking mean anything? ---------------- #
    print("\n  calibration (committee spread vs its own error):")
    for key, label in (("jsd", "JSD"), ("relative", "L2 spread")):
        scored = [{"relative": r[key], "error": r["error"]} for r in records
                  if r[key] is not None and r["error"] is not None]
        if len(scored) < 3:
            print(f"    {label:<11s} needs 3 scored structures, has "
                  f"{len(scored)}")
            continue
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
                           "error": truth[r["material"]]} for r in shared
                          if r[key] is not None]
                if len(scored) < 3:
                    print(f"    {label:<11s} needs 3 scored structures, has "
                          f"{len(scored)}")
                    continue
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
