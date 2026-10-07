#!/usr/bin/env python
# -*- coding: utf-8 -*-
# file: poraque_active_learning.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
**Selection.** Which unlabeled structures should the next DFT runs be spent on?

.. rubric:: This command, or ``poraque-committee``?

They are the two halves of one loop and they are not interchangeable. The
difference is the data each one runs on, and it decides everything else:

==================  ==============================  ==========================
                    ``poraque-active-learning``     ``poraque-committee``
==================  ==============================  ==========================
runs on             an **unlabeled** pool          a **labeled** dataset
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

A reference structure costs a plane-wave DFT run. Labeling at random spends
that budget uniformly over structures the model already predicts well. This
spends it where the committee is least determined, ranked by the
**Jensen-Shannon divergence** across members that differ only in their weight
initialization.

.. code-block:: text

    M members  ->  predict rho for every candidate in the pool
                          |
                   normalize to probability densities
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

    # commit to it (--train is the tagged cache a training run built)
    poraque-active-learning --models "models/committee_*" --task ext2chg \
        --pool data/pool --select 5 --train data/cache/res32_potcar \
        --promote move --json logs/active_round1.json

    # ... run DFT on the five new structures, then retrain the members ...
    for s in 0 1 2 3; do
      poraque-train --config configs/train.yaml \
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
from poraque.utils.logging import INDENT, Logger, header  # noqa: E402


def existing_identifiers(directory):
    """Structure names already in ``directory``, so none is selected twice."""
    if not directory or not os.path.isdir(directory):
        return set()
    return {entry for entry in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, entry))}


def _section(log, name):
    """
    Open the block ``name`` on ``log``, whatever ``log`` is.

    :func:`report_round` takes any callable, so a test can hand it
    ``list.append`` and read back what it said; :func:`main` hands it a
    :class:`~poraque.utils.logging.Logger`. A plain callable gets exactly what
    :meth:`Logger.section` would have written --- a blank line and the header
    --- so the layout does not depend on who is listening.
    """
    if isinstance(log, Logger):
        log.section(name)
    else:
        log("")
        log(header(name))


def report_round(result, log=print, round_number=1, show=20):
    """
    What a scored round found, before anything on disk moves.

    Three blocks: ``[STATISTICS]``, the spread of the disagreement over the
    pool, which says whether the round carries any ranking at all;
    ``[RANKING]``, the pool most uncertain first; and ``[SELECTION]``, what
    the round buys.

    Parameters
    ----------
    result : dict
        As :func:`~poraque.ml.active_learning.run_round` returns it.
    log : callable, optional
    round_number : int, optional
    show : int, optional
        Rows of the ranking; ``0`` for all of them.
    """
    _section(log, "statistics")
    log(f"  active-learning metrics, round {round_number}:")
    log(format_statistics(result["statistics"], indent=INDENT))

    _section(log, "ranking")
    log("  ranked by Jensen-Shannon divergence (most uncertain first):")
    log(format_ranking(result["records"], limit=None if show == 0 else show,
                       indent=INDENT))

    _section(log, "selection")
    log(f"  selected {len(result['selection'])} structure(s):")
    for record in result["selection"]:
        log(f"    {record['material']:<18s} JSD {record['jsd']:.4e}"
            f"   ({record['jsd_normalized']:.4f} of ln M)")


def build_parser():
    """The command line, as its own function so the tests can exercise it."""
    parser = argparse.ArgumentParser(
        description="SELECTION: which unlabeled structures to compute next. "
                    "Ranks a pool by committee disagreement (JSD) and takes "
                    "the top K.",
        epilog="Runs on an UNLABELED pool (--pool): input fields only, "
               "targets not yet computed. Its sibling poraque-committee runs "
               "on LABELED data and answers a different question -- whether "
               "the disagreement measure predicts error at all. Run that one "
               "first; this one spends the DFT budget.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", default="models/committee_*",
                        help="glob matching one directory or bundle per member")
    parser.add_argument("--task", default="ext2chg",
                        choices=["ext2chg", "chg2tau"])
    parser.add_argument("--pool", default="data/pool",
                        help="directory of unlabeled candidates; each needs "
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


def select(argv=None, log=None):
    """
    Parse ``argv``, run one round, and return its result dict.

    ``log`` is the :class:`~poraque.utils.logging.Logger` :func:`main` built;
    called directly, the command builds its own, on the terminal.
    """
    log = log if log is not None else Logger()
    args = build_parser().parse_args(argv)
    if args.promote and not args.train:
        # Checked before any scoring: this mistake is knowable from the
        # arguments alone, and finding out after the full committee pass
        # wastes the whole run.
        raise SystemExit("--promote needs --train: there is nowhere to "
                         "transfer the selection to.")

    from poraque_committee import resolve_bundles

    task = resolve_task(args.task)
    device = resolve_device(args.device)
    paths = resolve_bundles(args.models)
    committee = Committee.from_bundles(paths, task.name, device=device)

    log.title(f"Active learning round {args.round} - {task.name}: "
              f"{task.input_field} -> {task.target_field}")
    log.section("purpose")
    log.text("SELECTION: choosing which unlabeled structures to compute next.")
    log.text("(To ask whether this measure predicts error at all, that is")
    log.text(" poraque-committee, on labeled data.)")
    log.section("committee")
    log.fields({"members": f"{len(committee)}  (init_seed "
                           f"{committee.init_seeds})",
                "device": describe_device(device),
                "pool": args.pool,
                "training": args.train or "(not given; ranking only)",
                "selecting": f"top {args.select} by Jensen-Shannon divergence"})

    already = existing_identifiers(args.train)
    if already:
        log.text(f"excluding {len(already)} structure(s) already in "
                 f"{args.train}")

    # Scored and ranked first, with no `train_root`, so the whole report is on
    # screen before anything on disk moves. `run_round` can do both in one call
    # and does for a programmatic caller; here the ordering of the output is
    # what a reader needs to judge the round before it is committed to.
    log.section("scoring")
    result = run_round(
        committee, args.pool, task,
        n_select=args.select,
        batch_size=args.batch_size,
        residency=args.residency,
        exclude=already,
        log=log,
    )

    report_round(result, log, round_number=args.round, show=args.show)

    log.section("promotion")
    if not args.promote:
        log.text("nothing was moved: pass --promote {move,copy,symlink} "
                 "together")
        log.text("with --train to transfer the selection into the training "
                 "set.")
    else:
        # --promote without --train was rejected at argument time.
        verb = "would transfer" if args.dry_run else "transferring"
        log.text(f"{verb} the selection into {args.train}:")
        result["promoted"] = promote(result["selection"], args.train,
                                     mode=args.promote, dry_run=args.dry_run,
                                     log=log)
        if args.dry_run:
            log.text("--dry-run: nothing was moved.")

    log.section("caveats")
    log.text("Members differ only in initialization, so this ranks by")
    log.text("optimization variance -- a lower bound on the error, not a")
    log.text("calibrated bar. Validate the ordering with")
    log.text("`poraque-committee --against` before spending a DFT budget on "
             "it.")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        payload = round_to_dict(result)
        payload["round"] = args.round
        payload["members"] = paths
        payload["init_seeds"] = committee.init_seeds
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2, default=float)
        log.section("output files")
        log.fields({"summary": args.json})
    return result


def main(argv=None):
    """Console entry point for ``poraque-active-learning``.

    Returns a process exit status, because the ``[project.scripts]`` wrapper
    calls ``sys.exit(main())`` and would treat any other object as an error
    message. :func:`select` returns the round itself.
    """
    log = Logger()
    log.banner()
    select(argv, log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
