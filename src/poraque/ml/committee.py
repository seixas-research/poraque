# -*- coding: utf-8 -*-
# file: committee.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Query by committee: disagreement among independently initialised operators.

An ensemble of operators that share their data and architecture and differ only
in ``init_seed`` gives, at every point of the grid, a *spread* of predictions.
Where the members agree the answer is determined by the data; where they
diverge it is determined by the initialisation, which is another way of saying
the data did not pin it down.

The point is **active learning**. Reference data here costs a plane-wave DFT
run, so the question that matters is which structure to compute next. A
disagreement measure answers it — provided the measure is validated first.

What makes this setup unusual
-----------------------------
The predictions are **3D fields**, not scalars, so the disagreement is itself a
field. That is more informative than a number, and this module keeps both:

``spread``
    :math:`\sigma(\mathbf r)`, the pointwise standard deviation across members.
    It says *where* the committee is unsure — near the ionic cores, in the
    interstitial region — which a scalar cannot.
``relative``
    :math:`\|\sigma\| / \|\bar f\|`, a scalar on the same footing as the
    relative :math:`L^2` the models are scored with, so the two are directly
    comparable.
``integral_spread``
    Standard deviation of :math:`\int f\,d^3r` across members: the electron
    count for ``ext2chg``, the kinetic energy for ``chg2tau``. These are the
    quantities the energy is built from, so their spread is the one that
    propagates into a predicted energy.

Reading the number honestly
---------------------------
.. warning::

   Members that differ only in initialisation explore *optimisation* variance.
   That is a lower bound on the true error, not a calibrated uncertainty:
   deep ensembles are known to be over-confident, and every member here shares
   the same training structures, so none of them can know about chemistry the
   dataset omits.

   The measure is therefore only worth what a calibration check says it is
   worth — see :func:`disagreement_error_correlation`. Use it to *rank*
   candidates, not as an error bar.
"""

import numpy as np


def committee_spread(predictions, reference=None):
    r"""
    Pointwise and aggregate disagreement across committee members.

    Parameters
    ----------
    predictions : sequence of ScalarField or array_like
        One prediction per member, all on the same grid.
    reference : ScalarField or array_like, optional
        Ground truth. When given, the true error of the committee mean is
        returned alongside, which is what a calibration check needs.

    Returns
    -------
    dict
        ``mean`` and ``spread`` fields, the scalars ``relative``,
        ``max_spread`` and ``mean_spread``, and — with a ``reference`` —
        ``error`` (relative :math:`L^2` of the committee mean) and ``ratio``
        (``relative / error``, which is 1 for a perfectly calibrated
        committee and below 1 when it is over-confident).

    Raises
    ------
    ValueError
        With fewer than two members, since a spread over one sample is zero
        and would read as perfect confidence.
    """
    stack = np.stack([np.asarray(p, dtype=float) for p in predictions])
    if stack.shape[0] < 2:
        raise ValueError(
            f"a committee needs at least two members, got {stack.shape[0]}; "
            f"the spread over one is identically zero and would read as "
            f"perfect confidence."
        )

    mean = stack.mean(axis=0)
    # ddof=1: the members are a sample of the initialisations that could have
    # been drawn, not the population of them.
    spread = stack.std(axis=0, ddof=1)

    norm = np.linalg.norm(mean)
    result = {
        "n_members": int(stack.shape[0]),
        "mean": mean,
        "spread": spread,
        "relative": float(np.linalg.norm(spread) / norm) if norm else float("inf"),
        "mean_spread": float(spread.mean()),
        "max_spread": float(spread.max()),
    }

    if reference is not None:
        truth = np.asarray(reference, dtype=float)
        denominator = np.linalg.norm(truth)
        error = float(np.linalg.norm(mean - truth) / denominator)
        result["error"] = error
        result["ratio"] = result["relative"] / error if error else float("inf")
    return result


def committee_integrals(predictions, grid):
    r"""
    Spread of :math:`\int f\,d^3r` across the committee.

    For ``ext2chg`` this is the electron count and for ``chg2tau`` the kinetic
    energy — the integrated quantities the total energy is assembled from. A
    committee can agree closely pointwise and still disagree on the integral,
    or the reverse, so this is not implied by :func:`committee_spread`.

    Parameters
    ----------
    predictions : sequence of ScalarField or array_like
    grid : FieldGrid

    Returns
    -------
    dict
        ``values`` (per member), ``mean``, ``spread`` and ``relative``.
    """
    values = np.array([grid.integrate(np.asarray(p, dtype=float))
                       for p in predictions], dtype=float)
    mean = float(values.mean())
    spread = float(values.std(ddof=1))
    return {
        "values": values,
        "mean": mean,
        "spread": spread,
        "relative": spread / abs(mean) if mean else float("inf"),
    }


def probability_density(field, grid, clip=True):
    r"""
    Rescale a scalar field into a **probability density** on the cell.

    The Jensen-Shannon divergence is defined between probability
    distributions, and a charge density is not one: it integrates to the
    valence electron count :math:`N`, not to one. Both requirements are met
    here, and neither is optional — :math:`\ln` of a negative number is not
    defined, and an unnormalised pair would report a divergence that mostly
    measures the difference in :math:`N`.

    .. math::

        p(\mathbf r) = \frac{\max(\rho(\mathbf r),\,0)}
                            {\int \max(\rho,\,0)\,d^3r},
        \qquad \int p\,d^3r = 1 .

    Parameters
    ----------
    field : ScalarField or array_like
        A field that is non-negative up to band-limiting artefacts.
    grid : FieldGrid
        Supplies the volume element: the normalisation is an **integral**, not
        a sum, so a field on a 32³ mesh and the same field on a 64³ mesh
        normalise to the same distribution.
    clip : bool, optional
        Clamp negatives to zero first. Band-limiting a density with sharp core
        peaks rings (Gibbs), so a resampled or predicted field dips slightly
        below zero; that is an artefact of the truncation and clamping it is
        the right response. Pass ``False`` to have a genuinely signed field
        raise rather than be quietly turned into a distribution it is not.

    Returns
    -------
    tuple of (numpy.ndarray, int)
        The distribution, integrating to exactly 1, and how many voxels the
        clamp touched.

    Raises
    ------
    ValueError
        When the field integrates to zero — there is no distribution to form —
        or when ``clip`` is off and it goes negative.
    """
    values = np.asarray(field, dtype=float)

    clipped = int((values < 0.0).sum())
    if clipped and not clip:
        raise ValueError(
            f"{clipped} of {values.size} voxels are negative, so this field is "
            f"not a density and has no probability distribution. Pass "
            f"clip=True only if the negatives are band-limiting artefacts.")
    if clipped:
        values = np.clip(values, 0.0, None)

    total = float(values.sum()) * grid.volume_element
    if total <= 0.0:
        raise ValueError(
            "the field integrates to zero after clamping, so it cannot be "
            "normalised into a probability density.")

    # Divide by the integral, not by the sum: the result must integrate to one
    # against the same volume element the divergence is integrated with.
    return values / total, clipped


def _kl_integrand(p, q):
    r"""
    :math:`p\ln(p/q)`, with the :math:`p = 0` limit taken as zero.

    Both distributions have empty regions once negatives are clamped, and
    ``0 * log(0/q)`` is ``nan`` in floating point rather than the ``0`` the
    limit gives. Evaluating the logarithm only where ``p`` is positive is
    exact, not a regularisation: no floor is introduced and nothing is
    perturbed.
    """
    support = p > 0.0
    out = np.zeros_like(p)
    out[support] = p[support] * np.log(p[support] / q[support])
    return out


def jensen_shannon_divergence(prediction, reference, grid, clip=True):
    r"""
    Jensen-Shannon divergence between a predicted and a reference density.

    .. math::

        \mathrm{JSD}(p\,\|\,q) = \tfrac12 D_{\rm KL}(p\,\|\,m)
                               + \tfrac12 D_{\rm KL}(q\,\|\,m),
        \qquad m = \tfrac12 (p + q),

    with both fields first turned into probability densities by
    :func:`probability_density`. Unlike a raw :math:`D_{\rm KL}` it is
    symmetric, and it is finite even where one distribution vanishes and the
    other does not — which a density on a grid does, in the interstitial
    region, all the time.

    It measures **shape**, not magnitude: a prediction carrying 5 % too much
    charge but distributed identically scores zero. Read it beside the
    electron-count error, never instead of it — that division of labour is
    what makes it worth reporting at all, since the relative :math:`L^2`
    confounds the two.

    Parameters
    ----------
    prediction, reference : ScalarField or array_like
        Non-negative fields on the same grid.
    grid : FieldGrid
        Supplies the volume element.
    clip : bool, optional
        Clamp band-limiting undershoot to zero; see
        :func:`probability_density`.

    Returns
    -------
    dict
        ``jsd`` in nats, ``normalised`` = ``jsd / ln 2`` (in :math:`[0, 1]`,
        saturating when the two share no support), ``pointwise`` — the
        integrand, showing *where* the shapes differ — and ``clipped``, the
        number of voxels the clamp touched in either field.

    Examples
    --------
    >>> import numpy as np
    >>> from poraque.fields import FieldGrid
    >>> from poraque.ml.committee import jensen_shannon_divergence
    >>> grid = FieldGrid((8, 8, 8), np.eye(3) * 4.0)
    >>> rho = np.random.default_rng(0).random(grid.shape) + 0.1
    >>> result = jensen_shannon_divergence(rho, 2.5 * rho, grid)
    >>> bool(result["jsd"] < 1e-12)      # blind to a common rescaling
    True
    """
    p, clipped_p = probability_density(prediction, grid, clip=clip)
    q, clipped_q = probability_density(reference, grid, clip=clip)
    if p.shape != q.shape:
        raise ValueError(
            f"prediction and reference must share a grid, got shapes "
            f"{p.shape} and {q.shape}.")

    mean = 0.5 * (p + q)
    pointwise = 0.5 * (_kl_integrand(p, mean) + _kl_integrand(q, mean))
    jsd = float(pointwise.sum() * grid.volume_element)

    bound = float(np.log(2.0))
    return {
        "jsd": jsd,
        "normalised": jsd / bound,
        "bound": bound,
        "pointwise": pointwise,
        "clipped": clipped_p + clipped_q,
    }


def jensen_shannon_spread(predictions, grid, floor=1e-12):
    r"""
    Information-theoretic disagreement: the Jensen-Shannon divergence.

    The charge density is *already* a probability density up to a constant —
    :math:`\rho/N` integrates to one, and :math:`N` is fixed by the
    pseudopotentials — so information-theoretic distances apply to it directly.
    This is the same footing on which Hirshfeld partitioning is derived, where
    the stockholder weights are the ones minimising the KL divergence to a
    promolecular reference.

    For a *committee* the right quantity is not a pairwise KL but the
    divergence of the members about their mean,

    .. math::

        \mathrm{JSD} = \frac1M\sum_m D_{\rm KL}\!\left(p_m \,\|\, \bar p\right),
        \qquad \bar p = \frac1M\sum_m p_m ,

    which is exactly the mutual information between the prediction and the
    member index — the quantity active learning maximises. Unlike a raw
    :math:`D_{\rm KL}`, it is symmetric, finite whenever the members are, and
    bounded by :math:`\ln M`, so ``normalised`` lands in :math:`[0, 1]`.

    Parameters
    ----------
    predictions : sequence of ScalarField or array_like
        One **non-negative** field per member, on the same grid.
    grid : FieldGrid
        Supplies the volume element; the divergence is an integral, not a sum.
    floor : float, optional
        Retained for callers that pass it; ignored. Negatives are clamped to
        **zero** by :func:`probability_density` and the :math:`p\ln p` limit is
        taken exactly, so a positive floor is no longer needed to keep the
        logarithm defined — and a floor perturbs every normalisation slightly,
        which this does not.

    Returns
    -------
    dict
        ``jsd`` in nats, ``normalised`` = ``jsd / ln M``, ``pointwise`` (the
        integrand, a field showing where the members disagree in information
        terms) and ``clipped`` — how many voxels the clamp touched.

    Notes
    -----
    Each member is turned into a probability density first
    (:func:`probability_density`), so this measures **shape** disagreement and
    is deliberately blind to a common rescaling. That is a division of labour,
    not an oversight: pair it with :func:`committee_integrals`, which measures
    exactly the magnitude this discards. Either alone is a partial picture of a
    committee whose electron count is known to drift.

    See Also
    --------
    jensen_shannon_divergence : the same quantity between a *prediction and its
        reference*, which is an accuracy measure rather than a spread.

    .. warning::

       Not applicable to :math:`V_{\rm ext}`, which is signed — a third of its
       voxels are negative and its cell average is zero by construction, so it
       is not a density in any sense and the divergence is undefined rather
       than merely awkward. For :math:`\tau` the divergence is computable but
       normalisation discards :math:`T_{\rm s}`, which is the part the energy
       needs.

       The logarithm also weights the low-density tails far more heavily than
       the relative :math:`L^2` does. That is the intended behaviour — a 10 %
       error is a 10 % error wherever it sits — but it means the two measures
       can disagree about which structure is worst, and the tails contribute
       little to the energy. Report both.
    """
    if len(predictions) < 2:
        raise ValueError(
            f"a committee needs at least two members, got {len(predictions)}."
        )

    # Each member normalised to unit integral: the divergence is defined
    # between probability densities, and the electron count is handled
    # separately by committee_integrals.
    normalised = [probability_density(p, grid) for p in predictions]
    members = np.stack([p for p, _ in normalised])
    clipped = sum(count for _, count in normalised)

    mean = members.mean(axis=0)
    # p log(p / p_bar), averaged over members and integrated over the cell.
    pointwise = np.mean([_kl_integrand(member, mean) for member in members],
                        axis=0)
    jsd = float(pointwise.sum() * grid.volume_element)

    bound = np.log(members.shape[0])
    return {
        "n_members": int(members.shape[0]),
        "jsd": jsd,
        "normalised": jsd / bound if bound else 0.0,
        "bound": float(bound),
        "pointwise": pointwise,
        "clipped": clipped,
    }


class Committee:
    r"""
    An ensemble of operators that differ only in their initialisation.

    The members share their data, architecture and batch order, so what they
    disagree about is what the data failed to determine. :meth:`disagreement`
    reports that, headed by the **Jensen-Shannon divergence** — the measure
    with an information-theoretic reading rather than merely a numerical one,
    and the one that equals the mutual information between the prediction and
    the member index.

    Parameters
    ----------
    operators : sequence of FieldOperator
        At least two, all for the same task.

    Raises
    ------
    ValueError
        With fewer than two members, or when they disagree about the task —
        a "committee" mixing ``ext2chg`` and ``chg2tau`` would produce a
        number with no meaning at all.

    Examples
    --------
    >>> committee = Committee.from_bundles(                    # doctest: +SKIP
    ...     sorted(glob("models/committee_*/poraque_models.poraque")), "ext2chg")
    >>> committee.disagreement(potential)["jsd"]               # doctest: +SKIP
    """

    def __init__(self, operators):
        self.operators = list(operators)
        if len(self.operators) < 2:
            raise ValueError(
                f"a committee needs at least two members, got "
                f"{len(self.operators)}."
            )
        tasks = {op.task.name for op in self.operators}
        if len(tasks) > 1:
            raise ValueError(
                f"committee members must share a task, got {sorted(tasks)}."
            )
        self.task = self.operators[0].task

        seeds = [op.init_seed for op in self.operators]
        if len(set(seeds)) < len(seeds):
            import warnings

            warnings.warn(
                f"committee members share init_seed values {seeds}; members "
                f"initialised identically are not independent and their "
                f"agreement is guaranteed rather than measured.",
                RuntimeWarning, stacklevel=2,
            )
        self.init_seeds = seeds

    @classmethod
    def from_bundles(cls, paths, task, device=None):
        """
        Load one task's operator from each of several unified checkpoints.

        Parameters
        ----------
        paths : sequence of str
            One ``poraque_models.poraque`` per member.
        task : str
            ``"ext2chg"`` or ``"chg2tau"``.
        device : str or torch.device, optional
        """
        from .training import load_bundle

        return cls([load_bundle(path, task, device=device) for path in paths])

    def __len__(self):
        return len(self.operators)

    def predict(self, field):
        """Every member's prediction for ``field``, in physical units."""
        return [operator.predict(field) for operator in self.operators]

    def disagreement(self, field, reference=None):
        r"""
        Score the committee on one input.

        Returns the Jensen-Shannon divergence as the headline number, with the
        :math:`L^2` spread and the spread of the integrated quantity alongside
        — the three measure different things and no one of them implies the
        others.

        Parameters
        ----------
        field : ScalarField
            The input to predict from.
        reference : ScalarField or array_like, optional
            Ground truth for the *output*, enabling the calibration fields.

        Returns
        -------
        dict
            ``jsd`` and ``jsd_normalised`` (the headline), ``relative`` (the
            :math:`L^2` spread), ``integral_relative``, the ``mean`` and
            ``pointwise`` fields, and — with a reference — ``error`` and
            ``ratio``.

        Notes
        -----
        The divergence needs a non-negative field. Both learned outputs
        (:math:`\rho`, :math:`\tau`) qualify; were a committee ever built over
        a signed field, ``jsd`` would be ``None`` and only the :math:`L^2`
        measures reported.
        """
        predictions = self.predict(field)
        values = [p.data for p in predictions]
        grid = field.grid

        result = committee_spread(values, reference=reference)
        result["integral_relative"] = committee_integrals(values, grid)["relative"]
        result["predictions"] = predictions
        result["task"] = self.task.name

        if min(v.min() for v in values) < 0 and not _mostly_positive(values):
            # A signed field is not a density; say so rather than floor it into
            # one and report a number that looks meaningful.
            result["jsd"] = None
            result["jsd_normalised"] = None
            return result

        divergence = jensen_shannon_spread(values, grid)
        result["jsd"] = divergence["jsd"]
        result["jsd_normalised"] = divergence["normalised"]
        result["jsd_pointwise"] = divergence["pointwise"]
        result["jsd_clipped"] = divergence["clipped"]
        return result


def _mostly_positive(values, tolerance=0.01):
    """
    True when negatives are rare enough to be band-limiting artefacts.

    A predicted density rings slightly below zero from Gibbs oscillations; that
    is an artefact worth flooring. A field that is negative over a third of the
    cell is a different kind of object and must not be silently normalised into
    a probability density.
    """
    return all((np.asarray(v) < 0).mean() < tolerance for v in values)


def disagreement_error_correlation(records):
    r"""
    Does disagreement predict error? The only question that matters.

    A disagreement measure earns its keep by *ranking* candidates: the
    structure the committee argues about most should be the one it gets most
    wrong. This computes both the linear and the rank correlation between the
    two over a set of structures.

    Parameters
    ----------
    records : sequence of dict
        Each with ``relative`` (committee disagreement) and ``error`` (true
        relative :math:`L^2` of the committee mean), as produced by
        :func:`committee_spread` given a reference.

    Returns
    -------
    dict
        ``pearson``, ``spearman``, ``n`` and ``calibration`` — the mean
        ``relative / error``, below 1 when the committee is over-confident.

    Notes
    -----
    Spearman is the one to read. Active learning consumes an *ordering*, and a
    measure can rank perfectly while being badly scaled; a strong Pearson with
    a weak Spearman is the opposite and is not useful.
    """
    disagreement = np.array([r["relative"] for r in records], dtype=float)
    error = np.array([r["error"] for r in records], dtype=float)
    if disagreement.size < 3:
        raise ValueError(
            f"need at least three structures to correlate, got "
            f"{disagreement.size}."
        )

    def rank(values):
        order = values.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks

    def correlate(a, b):
        a, b = a - a.mean(), b - b.mean()
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / denominator) if denominator else float("nan")

    return {
        "n": int(disagreement.size),
        "pearson": correlate(disagreement, error),
        "spearman": correlate(rank(disagreement), rank(error)),
        "calibration": float(np.mean(disagreement / np.where(error, error, np.nan))),
    }
