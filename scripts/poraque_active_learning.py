#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_active_learning.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
**Selection.** Which unlabelled structures should the next DFT runs be spent on?

.. rubric:: This command, or ``poraque-committee``?

They are the two halves of one loop and they are not interchangeable. The
difference is the data each one runs on, and it decides everything else:

==================  ==============================  ==========================
                    ``poraque-active-learning``     ``poraque-committee``
==================  ==============================  ==========================
runs on             an **unlabelled** pool          a **labelled** dataset
                    (``--pool``): inputs only,      (``--cache``): inputs
                    targets not computed yet        *and* targets
answers             "which structures should I      "is this measure worth
                    compute next?"                  trusting?"
produces            a ranking, and a transfer of    a Spearman correlation
                    the top K into the training     between disagreement and
                    set                             the *known* error
costs               the DFT runs it selects         nothing — the DFT is
                                                    already done
==================  ==============================  ==========================

Run ``poraque-committee`` **first**. This command turns a disagreement measure
into a spending decision, and until that measure has been correlated against
known errors the ordering it produces is a guess with a decimal point.

Both share the same committee, the same Jensen-Shannon divergence and the same
ranking table (:func:`~poraque.ml.active_learning.format_ranking`), which is
why the two outputs look alike — they differ in what the number is *for*.

.. rubric:: What it does

A reference structure costs a plane-wave DFT run. Labelling at random spends
that budget uniformly over structures the model already predicts well. This
spends it where the committee is least determined, ranked by the
**Jensen-Shannon divergence** across members that differ only in their weight
initialisation.

.. code-block:: text

    M members  ->  predict rho for every candidate in the pool
                          |
                   normalise to probability densities
                          |
        JSD = (1/M) sum_i D_KL( rho_i || rho_bar )     one number per candidate
                          |
                   rank  ->  top K  ->  move into the training set  ->  retrain

One invocation is **one round**. The loop is closed outside it, because the
step between two rounds is a training run:

.. code-block:: bash

    # round 1 -- rank only, and look at the spread before spending anything
    poraque-active-learning --models "models/committee_*" --task ext2chg \
        --pool data/pool --select 5

    # commit to it
    poraque-active-learning --models "models/committee_*" --task ext2chg \
        --pool data/pool --select 5 --train data/cache/res32 \
        --promote move --json logs/active_round1.json

    # ... run DFT on the five new structures, then retrain the members ...
    for s in 0 1 2 3; do
      poraque-train --config configs/train_config.yaml \
          --init-seed $s --name committee_$s
    done

Without ``--promote`` nothing on disk is touched: the run scores, ranks and
reports. That is the default because moving directories between a candidate
pool and a training set is not something a scoring run should do by accident.

Read the spread before the ranking
----------------------------------
The ``max/min`` line of the JSD summary is the one that says whether the round
is worth anything. A pool whose candidates all score alike carries no ranking
information, and selecting from it is random sampling with extra steps ---
however confident the ordering looks.

The measure itself is only worth what a calibration check says it is worth.
``poraque-committee --against`` correlates the same disagreement against known
errors; Spearman is the coefficient to read, since active learning consumes an
ordering.

Running this file directly --- ``python scripts/poraque_active_learning.py`` ---
is equivalent to the installed console command, and needs nothing installed.
"""

import argparse
import json
import os
import sys

# Run straight from a checkout, without installing, by preferring the in-tree
# package. Installed as the ``poraque-active-learning`` console script this
# module sits in site-packages, that directory does not exist, and the installed
# package wins.
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if os.path.isdir(_SRC):
    sys.path.insert(0, _SRC)

from poraque.ml import Committee  # noqa: E402
from poraque.ml.active_learning import (  # noqa: E402
    TRANSFERS,
    format_ranking,
    format_statistics,
    promote,
    round_to_dict,
    run_round,
)
from poraque.ml.device import describe_device, resolve_device  # noqa: E402
from poraque.ml.tasks import resolve_task  # noqa: E402


def existing_identifiers(directory):
    """Structure names already in ``directory``, so none is selected twice."""
    if not directory or not os.path.isdir(directory):
        return set()
    return {entry for entry in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, entry))}


def build_parser():
    """The command line, as its own function so the tests can exercise it."""
    parser = argparse.ArgumentParser(
        description="SELECTION: which unlabelled structures to compute next. "
                    "Ranks a pool by committee disagreement (JSD) and takes "
                    "the top K.",
        epilog="Runs on an UNLABELLED pool (--pool): input fields only, "
               "targets not yet computed. Its sibling poraque-committee runs "
               "on LABELLED data and answers a different question -- whether "
               "the disagreement measure predicts error at all. Run that one "
               "first; this one spends the DFT budget.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="models/committee_*",
                        help="glob matching one directory or bundle per member")
    parser.add_argument("--task", default="ext2chg",
                        choices=["ext2chg", "chg2tau"])
    parser.add_argument("--pool", default="data/pool",
                        help="directory of unlabelled candidates; each needs "
                             "only the task's input field")
    parser.add_argument("--train", default=None,
                        help="training-set root the selection is moved into, "
                             "and whose existing structures are excluded from "
                             "the pool")
    parser.add_argument("--select", type=int, default=5, metavar="K",
                        help="how many structures this round buys")
    parser.add_argument("--promote", choices=list(TRANSFERS), default=None,
                        help="transfer the selection into --train. Omitted, "
                             "the run only scores and ranks.")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --promote, report the transfers without "
                             "performing them")
    parser.add_argument("--batch-size", type=int, default=4, metavar="N",
                        help="candidates per chunk; caps peak memory, since "
                             "the accelerator's cached blocks are released "
                             "after each chunk")
    parser.add_argument("--residency", choices=["all", "one"], default="all",
                        help="'one' keeps a single member on the accelerator "
                             "at a time, so peak memory is set by the largest "
                             "member rather than by the size of the committee")
    parser.add_argument("--round", type=int, default=1, metavar="N",
                        help="round number, for the log and the summary")
    parser.add_argument("--show", type=int, default=20, metavar="N",
                        help="rows of the ranking to print; 0 for all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--json", default=None, help="write a JSON summary")
    return parser


def select(argv=None):
    """Parse ``argv``, run one round, and return its result dict."""
    args = build_parser().parse_args(argv)

    from poraque_committee import resolve_bundles

    task = resolve_task(args.task)
    device = resolve_device(args.device)
    paths = resolve_bundles(args.models)
    committee = Committee.from_bundles(paths, task.name, device=device)

    print("=" * 78)
    print(f"Active learning round {args.round} - {task.name}: "
          f"{task.input_field} -> {task.target_field}")
    print("SELECTION: choosing which unlabelled structures to compute next.")
    print("(To ask whether this measure predicts error at all, that is")
    print(" poraque-committee, on labelled data.)")
    print("=" * 78)
    print(f"  members    : {len(committee)}  (init_seed "
          f"{committee.init_seeds})")
    print(f"  device     : {describe_device(device)}")
    print(f"  pool       : {args.pool}")
    print(f"  training   : {args.train or '(not given; ranking only)'}")
    print(f"  selecting  : top {args.select} by Jensen-Shannon divergence")
    print()

    already = existing_identifiers(args.train)
    if already:
        print(f"  excluding {len(already)} structure(s) already in "
              f"{args.train}")

    # Scored and ranked first, with no `train_root`, so the whole report is on
    # screen before anything on disk moves. `run_round` can do both in one call
    # and does for a programmatic caller; here the ordering of the output is
    # what a reader needs to judge the round before it is committed to.
    result = run_round(
        committee, args.pool, task,
        n_select=args.select,
        batch_size=args.batch_size,
        residency=args.residency,
        exclude=already,
        log=print,
    )

    print(f"\n  active-learning metrics, round {args.round}:")
    print(format_statistics(result["statistics"]))

    print("\n  ranked by Jensen-Shannon divergence (most uncertain first):")
    print(format_ranking(result["records"],
                         limit=None if args.show == 0 else args.show))

    print(f"\n  selected {len(result['selection'])} structure(s):")
    for record in result["selection"]:
        print(f"    {record['material']:<18s} JSD {record['jsd']:.4e}"
              f"   ({record['jsd_normalised']:.4f} of ln M)")

    if not args.promote:
        print("\n  nothing was moved: pass --promote {move,copy,symlink} "
              "together")
        print("  with --train to transfer the selection into the training set.")
    elif not args.train:
        raise SystemExit("--promote needs --train: there is nowhere to "
                         "transfer the selection to.")
    else:
        verb = "would transfer" if args.dry_run else "transferring"
        print(f"\n  {verb} the selection into {args.train}:")
        result["promoted"] = promote(result["selection"], args.train,
                                     mode=args.promote, dry_run=args.dry_run,
                                     log=print)
        if args.dry_run:
            print("\n  --dry-run: nothing was moved.")

    print("\n" + "=" * 78)
    print("  Members differ only in initialisation, so this ranks by")
    print("  optimisation variance -- a lower bound on the error, not a")
    print("  calibrated bar. Validate the ordering with")
    print("  `poraque-committee --against` before spending a DFT budget on it.")
    print("=" * 78)

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        payload = round_to_dict(result)
        payload["round"] = args.round
        payload["members"] = paths
        payload["init_seeds"] = committee.init_seeds
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2, default=float)
        print(f"\n  summary -> {args.json}")
    return result


def main(argv=None):
    """Console entry point for ``poraque-active-learning``.

    Returns a process exit status, because the ``[project.scripts]`` wrapper
    calls ``sys.exit(main())`` and would treat any other object as an error
    message. :func:`select` returns the round itself.
    """
    select(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
