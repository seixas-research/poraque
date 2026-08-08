# -*- coding: utf-8 -*-
# file: active_learning.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Active learning: choose the next DFT calculations by committee disagreement.

A reference structure costs a plane-wave run. Labelling an unlabelled pool at
random spends that budget uniformly over structures the model already predicts
well. Query by committee spends it where the model is *least determined*:

.. code-block:: text

    M members (same data, different init_seed)
              |
              |  predict rho for every candidate in the unlabelled pool
              v
      normalise each to a probability density
              |
              v
    JSD = (1/M) sum_i D_KL( rho_i || rho_bar )     one number per candidate
              |
              v
       rank, take the top K, label them, retrain

The measure
-----------
Each member's prediction is first normalised to unit integral, which is what
makes a Kullback-Leibler divergence between them defined at all, and then

.. math::

    \bar\rho = \frac1M\sum_{i=1}^M \rho_i,
    \qquad
    \mathrm{JSD} = \frac1M\sum_{i=1}^M D_{\rm KL}\!\left(\rho_i\,\|\,\bar\rho\right).

This is the generalised Jensen-Shannon divergence, and it is exactly the mutual
information between the prediction and the member index — the quantity active
learning maximises. It is symmetric, finite wherever the members are, and
bounded by :math:`\ln M`, none of which a raw :math:`D_{\rm KL}` gives.
:func:`~poraque.ml.committee.jensen_shannon_spread` computes it; this module is
the loop around it.

.. important::

   Normalising discards magnitude, so the JSD measures **shape** disagreement
   and is blind to a common rescaling. The electron-count spread it throws away
   is reported beside it by
   :func:`~poraque.ml.committee.committee_integrals`, and both are recorded on
   every candidate. Ranking on the JSD alone is a deliberate choice — it is the
   information-theoretic quantity — not an oversight.

.. warning::

   Members that differ only in initialisation explore optimisation variance,
   which is a lower bound on the true error rather than a calibrated
   uncertainty. Validate the ranking before trusting a round of it:
   ``poraque-committee --against`` correlates the same measure against known
   errors, and Spearman is the coefficient to read.

Memory
------
Naively this holds :math:`M` copies of a :math:`128^3` field, plus :math:`M`
models on the accelerator, for every candidate at once. Three things keep that
bounded, all in :func:`score_pool`:

* every prediction runs under :func:`torch.no_grad`, so no autograd graph is
  built (:meth:`~poraque.ml.training.FieldOperator.predict` is already
  decorated, and the loop asserts it rather than assuming it);
* the pool is walked in **chunks**, and the accelerator's cached blocks are
  released at the end of each — a scored candidate keeps a handful of floats,
  never a grid;
* with ``residency="one"`` only a single member is resident on the accelerator
  at a time, so peak memory is set by the largest member rather than by the
  size of the committee.
"""

import os
import shutil

import numpy as np

from .committee import (
    committee_integrals,
    committee_spread,
    jensen_shannon_spread,
)
from .device import empty_cache
from .tasks import resolve_task

#: How a selected candidate is transferred into the training set.
#:
#: ``"move"``
#:     Rename it out of the pool. The pool then holds exactly the structures
#:     still unlabelled, which is the invariant the next round depends on.
#: ``"copy"``
#:     Duplicate it, leaving the pool untouched. Costs disk and lets the same
#:     candidate be selected again next round.
#: ``"symlink"``
#:     Link it. Cheap, and keeps one copy on disk, but a training set of
#:     symlinks breaks if the pool is later cleaned up.
TRANSFERS = ("move", "copy", "symlink")


# ===================================================================== #
# The pool
# ===================================================================== #
def discover_pool(root, task, exclude=()):
    r"""
    Find the unlabelled candidates under ``root``.

    "Unlabelled" is the whole point: a candidate needs only the task's **input**
    field, since the target is what the DFT run would produce and has not been
    computed yet. That is what separates this from
    :func:`~poraque.ml.data.discover_materials`, which requires a complete
    pair and would silently skip every genuine candidate.

    Parameters
    ----------
    root : str or pathlib.Path
        Directory holding one subdirectory per candidate.
    task : str or TaskSpec
        ``"ext2chg"`` or ``"chg2tau"``; selects which field must be present.
    exclude : container of str, optional
        Identifiers to skip — normally the structures already in the training
        set, so a candidate cannot be selected twice.

    Returns
    -------
    list of MaterialRecord
        Sorted by identifier, so a round is reproducible.
    """
    from .data import discover_materials

    task = resolve_task(task)
    records = discover_materials(root, required=(task.input_field,))
    return [record for record in records if record.identifier not in exclude]


def load_input(record, task):
    """Read one candidate's input field, on its own grid."""
    from ..fields import FieldGrid
    from .data import FIELD_CLASSES

    task = resolve_task(task)
    path = os.path.join(record.directory, task.input_field)
    return FIELD_CLASSES[task.input_field].read(path,
                                                grid=FieldGrid.from_file(path))


# ===================================================================== #
# Scoring
# ===================================================================== #
class _Resident:
    """
    Hold one member on the accelerator for the length of a ``with`` block.

    A committee of :math:`M` operators resident at once costs :math:`M` times
    the parameters *and* :math:`M` times the transient activations of a 3D FFT.
    Cycling members through the device instead caps the first at one member,
    which is the difference between fitting and not on a small card.

    The operator's own :attr:`device` is deliberately **not** touched: it is
    what :meth:`~poraque.ml.training.FieldOperator.predict` builds its input
    tensors on, and inside the block the weights are on exactly that device, so
    the two agree where it matters. Outside the block the weights are parked on
    the CPU and the operator is not usable until :func:`_restore` puts them
    back — which is why both public entry points guarantee it in a ``finally``.
    """

    def __init__(self, operator, active):
        self.operator = operator
        self.active = bool(active)

    def __enter__(self):
        if self.active:
            self.operator.model.to(self.operator.device)
        return self.operator

    def __exit__(self, *_):
        if self.active:
            self.operator.model.to("cpu")
            empty_cache(self.operator.device)
        return False


def _park(committee, cycling):
    """
    Move the whole committee to the CPU before a sweep begins.

    Without this the cycling saves nothing on the first candidate: the members
    arrive already resident, and :class:`_Resident` only evicts one *after* it
    has been used, so the peak is still the whole committee — which is the
    moment an accelerator runs out of memory.
    """
    if cycling:
        for operator in committee.operators:
            operator.model.to("cpu")
            empty_cache(operator.device)


def _restore(committee, cycling):
    """Put every member back on its own device, whatever went wrong."""
    if cycling:
        for operator in committee.operators:
            operator.model.to(operator.device)


def _predict_all(committee, field, cycling):
    """
    Every member's prediction for ``field``, as detached numpy arrays.

    Two memory guarantees live here, and they are the reason this is not just
    ``committee.predict``:

    * ``no_grad`` rather than trusting the decorator on
      :meth:`~poraque.ml.training.FieldOperator.predict` — this is the loop
      where a retained graph would be :math:`M` times a 3D field per candidate,
      and the guarantee is cheap to state at the point that depends on it;
    * each :class:`~poraque.fields.ScalarField` is reduced to its array and
      dropped immediately, since the wrapper also holds the grid and the
      structure and :math:`M` of those per candidate is the leak.
    """
    import torch

    values = []
    with torch.no_grad():
        for operator in committee.operators:
            with _Resident(operator, cycling) as member:
                prediction = member.predict(field)
            values.append(np.asarray(prediction.data, dtype=float))
            del prediction
    return values


def _reduce(values, grid):
    """Turn M predicted grids into the handful of scalars that are kept."""
    from .committee import _mostly_positive

    spread = committee_spread(values)
    record = {
        "n_members": int(spread["n_members"]),
        "relative": float(spread["relative"]),
        "mean_spread": float(spread["mean_spread"]),
        "max_spread": float(spread["max_spread"]),
        "integral_relative": float(committee_integrals(values, grid)["relative"]),
        "jsd": None,
        "jsd_normalised": None,
        "clipped": 0,
    }

    if _mostly_positive(values):
        divergence = jensen_shannon_spread(values, grid)
        record["jsd"] = float(divergence["jsd"])
        record["jsd_normalised"] = float(divergence["normalised"])
        record["clipped"] = int(divergence["clipped"])
    return record


def score_candidate(committee, field, residency="all"):
    r"""
    Score one candidate: how much do the members disagree about it?

    Parameters
    ----------
    committee : Committee
        At least two operators sharing a task.
    field : ScalarField
        The **input** field to predict from.
    residency : {"all", "one"}, optional
        Whether every member stays on the accelerator or is cycled through it
        one at a time. See :class:`_Resident`. The committee is left on the
        devices it was found on either way.

    Returns
    -------
    dict
        ``jsd`` (nats), ``jsd_normalised`` (:math:`\mathrm{JSD}/\ln M`, in
        :math:`[0,1]`), ``relative`` (the :math:`L^2` spread),
        ``integral_relative`` (the spread of the electron count, which the JSD
        deliberately discards), ``clipped`` and ``n_members``.

        ``jsd`` is ``None`` when the members predict a field that is negative
        over much of the cell — an untrained operator does, and flooring that
        into a probability density would produce a number that looks
        meaningful. The :math:`L^2` measures still apply and are still
        returned.

    Notes
    -----
    Only scalars survive the call. The pointwise divergence field is genuinely
    useful — it says *where* the members disagree — but keeping one per
    candidate is what turns a pool sweep into a memory problem, so a caller
    that wants it should ask
    :func:`~poraque.ml.committee.jensen_shannon_spread` for a single structure.
    """
    cycling = residency == "one"
    _park(committee, cycling)
    try:
        values = _predict_all(committee, field, cycling)
    finally:
        _restore(committee, cycling)
    return _reduce(values, field.grid)


def score_pool(committee, records, task, batch_size=4, residency="all",
               log=None):
    r"""
    Score every candidate in the pool, a chunk at a time.

    Parameters
    ----------
    committee : Committee
    records : sequence of MaterialRecord
        The pool, from :func:`discover_pool`.
    task : str or TaskSpec
    batch_size : int, optional
        Candidates per chunk. Each chunk is scored, reduced to scalars and then
        followed by a release of the accelerator's cached blocks, so peak
        memory is set by the chunk rather than by the pool. It is a *memory*
        knob, not a throughput one — the members still predict one structure at
        a time, because candidates differ in grid shape and cannot be batched
        into one tensor.
    residency : {"all", "one"}, optional
        Passed to :func:`score_candidate`.
    log : callable, optional
        Progress sink, called once per chunk.

    Returns
    -------
    list of dict
        One record per candidate, each carrying ``material`` and ``directory``
        alongside the fields of :func:`score_candidate`. In pool order; use
        :func:`select_top_k` to rank.

    Raises
    ------
    ValueError
        If ``batch_size`` is not positive, which would loop forever.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")

    emit = log if log is not None else (lambda *_: None)
    task = resolve_task(task)
    device = committee.operators[0].device
    cycling = residency == "one"

    # The park-and-restore is hoisted out of the sweep rather than delegated to
    # score_candidate: done per candidate it would move every member back onto
    # the accelerator between structures, which is the cost the cycling exists
    # to avoid.
    scored = []
    _park(committee, cycling)
    try:
        for start in range(0, len(records), batch_size):
            chunk = records[start:start + batch_size]
            for record in chunk:
                field = load_input(record, task)
                entry = _reduce(_predict_all(committee, field, cycling),
                                field.grid)
                entry["material"] = record.identifier
                entry["directory"] = record.directory
                scored.append(entry)
                del field
            # Once per chunk, not once per structure: releasing the allocator's
            # blocks forces a resynchronisation, and doing it per structure
            # costs more than the fragmentation it avoids.
            empty_cache(device)
            emit(f"    scored {min(start + batch_size, len(records))}"
                 f"/{len(records)} candidates")
    finally:
        _restore(committee, cycling)
    return scored


# ===================================================================== #
# Selection
# ===================================================================== #
def jsd_statistics(records):
    """
    Minimum, maximum and mean JSD over a scored pool.

    The spread between the minimum and the maximum is the part worth reading:
    a pool whose candidates all score alike carries no ranking information, and
    a round run on it selects arbitrarily however confident the numbers look.

    Parameters
    ----------
    records : sequence of dict
        As returned by :func:`score_pool`.

    Returns
    -------
    dict
        ``n``, ``n_scored``, ``min``, ``max``, ``mean``, ``median``, ``std``
        and ``spread`` (max / min). Every statistic is ``nan`` when no
        candidate could be scored, rather than a number computed from nothing.
    """
    values = np.array([record["jsd"] for record in records
                       if record.get("jsd") is not None], dtype=float)
    summary = {"n": len(records), "n_scored": int(values.size)}
    if not values.size:
        return {**summary, **{key: float("nan") for key in
                              ("min", "max", "mean", "median", "std", "spread")}}
    return {
        **summary,
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "spread": float(values.max() / values.min()) if values.min() > 0
        else float("inf"),
    }


def select_top_k(records, k):
    """
    The ``k`` candidates the committee disagrees about most.

    Candidates whose JSD could not be computed are excluded rather than sorted
    to one end: they carry no ranking information, and letting them fill a
    selection would spend DFT runs on the fact that the measure did not apply.

    Parameters
    ----------
    records : sequence of dict
        As returned by :func:`score_pool`.
    k : int
        How many to select. A ``k`` larger than the pool returns the pool.

    Returns
    -------
    list of dict
        Highest JSD first.
    """
    ranked = sorted((r for r in records if r.get("jsd") is not None),
                    key=lambda record: -record["jsd"])
    return ranked[:max(int(k), 0)]


def promote(selection, destination, mode="move", dry_run=False, log=None):
    """
    Transfer the selected candidates into the training set.

    Parameters
    ----------
    selection : sequence of dict
        Scored records, from :func:`select_top_k`.
    destination : str or pathlib.Path
        Training-set root. Created if it does not exist.
    mode : {"move", "copy", "symlink"}, optional
        See :data:`TRANSFERS`.
    dry_run : bool, optional
        Resolve and report every transfer without performing it.
    log : callable, optional
        Progress sink.

    Returns
    -------
    list of dict
        ``{"material", "source", "destination", "transferred"}`` per candidate.
        ``transferred`` is ``False`` for one already present at the
        destination, which is skipped rather than overwritten — a half-written
        training structure is worse than a missing one, and a candidate already
        promoted is the ordinary way this happens.

    Raises
    ------
    ValueError
        On an unknown ``mode``.
    """
    if mode not in TRANSFERS:
        raise ValueError(f"Unknown transfer mode {mode!r}; expected one of "
                         f"{list(TRANSFERS)}.")
    emit = log if log is not None else (lambda *_: None)
    destination = str(destination)
    if not dry_run:
        os.makedirs(destination, exist_ok=True)

    moved = []
    for record in selection:
        source = record["directory"]
        target = os.path.join(destination, record["material"])
        entry = {"material": record["material"], "source": source,
                 "destination": target, "transferred": False}

        if os.path.exists(target):
            emit(f"    {record['material']:<16s} already at {target}; skipped")
            moved.append(entry)
            continue
        if dry_run:
            emit(f"    {record['material']:<16s} would {mode} -> {target}")
            moved.append(entry)
            continue

        if mode == "move":
            shutil.move(source, target)
        elif mode == "copy":
            shutil.copytree(source, target)
        else:
            os.symlink(os.path.abspath(source), target)
        entry["transferred"] = True
        emit(f"    {record['material']:<16s} {mode} -> {target}")
        moved.append(entry)
    return moved


# ===================================================================== #
# One round
# ===================================================================== #
def run_round(committee, pool_root, task, n_select=5, train_root=None,
              batch_size=4, residency="all", mode="move", dry_run=True,
              exclude=(), log=None):
    """
    Score a pool, rank it, and optionally promote the top-``K``.

    Parameters
    ----------
    committee : Committee
    pool_root : str
        Directory of unlabelled candidates.
    task : str or TaskSpec
    n_select : int, optional
        How many structures this round buys.
    train_root : str, optional
        Where the selection is transferred to. ``None`` scores and ranks
        without touching the filesystem.
    batch_size, residency : optional
        Passed to :func:`score_pool`.
    mode : str, optional
        Passed to :func:`promote`.
    dry_run : bool, optional
        Report the transfers without performing them. **Defaults to true**:
        moving structures between directories is not something a scoring run
        should do by accident.
    exclude : container of str, optional
        Identifiers to leave out of the pool.
    log : callable, optional
        Progress sink.

    Returns
    -------
    dict
        ``task``, ``pool``, ``n_members``, ``statistics``, ``records``,
        ``selection`` and ``promoted``.

    Raises
    ------
    ValueError
        When the pool is empty, or when nothing in it could be scored — either
        is a configuration error that would otherwise be reported as a
        successful round that selected nothing.
    """
    emit = log if log is not None else (lambda *_: None)
    task = resolve_task(task)

    records = discover_pool(pool_root, task, exclude=exclude)
    if not records:
        raise ValueError(
            f"no candidates under {pool_root!r} carrying a "
            f"{task.input_field}; an unlabelled pool needs only the input "
            f"field, so this is a path or a layout problem.")

    emit(f"  scoring {len(records)} candidates with {len(committee)} members "
         f"(chunks of {batch_size}, residency {residency})")
    scored = score_pool(committee, records, task, batch_size=batch_size,
                        residency=residency, log=log)

    statistics = jsd_statistics(scored)
    if not statistics["n_scored"]:
        raise ValueError(
            "no candidate could be scored: every member predicted a field "
            "that is negative over much of the cell, which is not a "
            "probability density. Untrained or badly trained members do this.")

    selection = select_top_k(scored, n_select)
    promoted = []
    if train_root is not None and selection:
        promoted = promote(selection, train_root, mode=mode, dry_run=dry_run,
                           log=log)

    return {
        "task": task.name,
        "pool": str(pool_root),
        "n_members": len(committee),
        "statistics": statistics,
        "records": scored,
        "selection": selection,
        "promoted": promoted,
    }


def format_statistics(statistics, indent="    "):
    """
    The round's JSD statistics as terminal text.

    Returns
    -------
    str
    """
    lines = [
        f"{indent}candidates scored : {statistics['n_scored']} "
        f"of {statistics['n']}",
    ]
    if not statistics["n_scored"]:
        return "\n".join(lines + [f"{indent}JSD               : unavailable"])
    lines += [
        f"{indent}JSD  min          : {statistics['min']:.6e}",
        f"{indent}     max          : {statistics['max']:.6e}",
        f"{indent}     mean         : {statistics['mean']:.6e}",
        f"{indent}     median       : {statistics['median']:.6e}",
        f"{indent}     std          : {statistics['std']:.6e}",
        f"{indent}     max/min      : {statistics['spread']:.2f}"
        f"   (a flat pool carries no ranking)",
    ]
    return "\n".join(lines)


#: Columns of the disagreement ranking, as ``(heading, key, width, format)``.
#:
#: One definition, used by both ``poraque-active-learning`` and
#: ``poraque-committee``, because the two print the same measure over the same
#: committee and had drifted into two tables: different column widths, a
#: different precision on the same number, and the normalisation labelled
#: ``JSD/lnM`` in one and ``JSD/lnK`` in the other. ``M`` is the member count
#: throughout; ``K`` is reserved for the size of the top-K selection, which is
#: a different number that appears in the same output.
RANKING_COLUMNS = (
    ("structure", "material", 18, "<18s"),
    ("JSD", "jsd", 12, "12.4e"),
    ("JSD/lnM", "jsd_normalised", 9, "9.4f"),
    ("L2 spread", "relative", 10, "10.4f"),
    ("int spread", "integral_relative", 11, "11.4f"),
    ("error", "error", 9, "9.4f"),
)


def format_ranking(records, limit=None, indent="    ", columns=None):
    """
    The ranked pool as a terminal table, most uncertain first.

    The heading and its rule are both derived from ``columns``, so the rule
    cannot end up a character short of the heading it underlines — which is
    what a hand-counted ``"-" * 63`` had done.

    Parameters
    ----------
    records : sequence of dict
    limit : int, optional
        Show only this many rows.
    indent : str, optional
    columns : sequence, optional
        Subset of :data:`RANKING_COLUMNS` to print. Defaults to every column
        the records actually carry, so a pool with no reference field simply
        omits ``error`` rather than printing a column of blanks.

    Returns
    -------
    str
    """
    ranked = sorted((r for r in records if r.get("jsd") is not None),
                    key=lambda record: -record["jsd"])
    if limit is not None:
        ranked = ranked[:int(limit)]

    if columns is None:
        present = {key for record in ranked for key, value in record.items()
                   if value is not None}
        columns = [column for column in RANKING_COLUMNS
                   if column[1] in present or column[1] == "material"]

    heading = " ".join(f"{title:>{width}s}" if spec[0] != "<"
                       else f"{title:<{width}s}"
                       for title, _, width, spec in columns)
    lines = [indent + heading, indent + "-" * len(heading)]
    for record in ranked:
        lines.append(indent + " ".join(
            f"{record[key]:{spec}}" for _, key, _, spec in columns))
    return "\n".join(lines)


def round_to_dict(result):
    """
    Serialisable form of a :func:`run_round` result.

    Returns
    -------
    dict
    """
    return {
        "task": result["task"],
        "pool": result["pool"],
        "n_members": result["n_members"],
        "statistics": result["statistics"],
        "selection": [record["material"] for record in result["selection"]],
        "promoted": result["promoted"],
        "records": [{key: value for key, value in record.items()
                     if not isinstance(value, np.ndarray)}
                    for record in result["records"]],
    }
