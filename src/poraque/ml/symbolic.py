# -*- coding: utf-8 -*-
# file: symbolic.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Symbolic distillation: read a closed-form functional out of a trained operator.

A Fourier Neural Operator reproduces :math:`\rho \mapsto \tau` to a few percent
and explains nothing. Symbolic regression searches short algebraic expressions
for one that reproduces the same mapping, trading accuracy for a formula that
can be read, published, and checked against known physics.

.. code-block:: text

    trained FNO  --predict-->  tau(r) on a grid
                                    |
    rho(r) --> features: rho, p = |grad rho| / (2 k_F rho)
                              q = lap rho  / (4 k_F^2 rho)
                                    |
                       symbolic regression (native GP)
                                    |
                             tau = f(rho, p, q)

What this can and cannot find
-----------------------------
The features are evaluated **at a point**, so the hypothesis space is exactly
the semi-local functionals. The non-local part of what the operator learned
cannot be expressed in it *at all*, and shows up as irreducible residual.

That makes a poor fit informative rather than disappointing: it puts a number
on how much of the learned map is not semi-local. A near-perfect fit would say
the operator found nothing a GGA-level functional could not have.

Two known answers sit inside the search space and are the calibration for any
result:

.. math::

    \tau_{\rm TF} = C_{\rm TF}\,\rho^{5/3},
    \qquad
    \tau_{\rm vW} = \frac{|\nabla\rho|^2}{8\rho}.

In the reduced variables these read :math:`F = 1` and :math:`F = 5p^2/3`. If a
run cannot recover them on data generated from them, the *search* is
misconfigured, not the physics — :func:`build_features` is deliberately exact
for both so that check is available.

Vacuum
------
:math:`p` and :math:`q` divide by :math:`\rho`, which decays exponentially into
vacuum. Denominators are clamped at :data:`DEFAULT_EPSILON` and voxels at or
below it are dropped: in vacuum the reduced variables are ratios of two
vanishing numbers, so they carry noise at a plausible magnitude rather than
information, and a fit corrupted that way looks healthy.

Units
-----
Features and targets are built in **atomic units** (:math:`\rho` in
:math:`e/a_0^3`, :math:`\tau` in :math:`E_h/a_0^3`), not the eV/Å³ the files
carry. In atomic units the coefficients the search must discover are order
unity — :math:`C_{\rm TF} = 2.871`, the von Weizsäcker :math:`1/8` — whereas in
eV/Å³ they are arbitrary decimals that a genetic search wastes its budget
approximating.

Engine
------
:func:`poraque.ml.gp.native_engine` is the backend: genetic programming over
expression trees, in NumPy, with SciPy fitting the constants of the final
front. Both are already hard dependencies, so distillation needs nothing that
training does not — deliberately. A second language runtime (a Julia-based
searcher, say) is faster, and on a supercomputer it is also a network fetch
from a compute node, a writable depot on a filesystem that may be purged
between jobs, a precompilation pass per architecture and a second runtime
inside an MPI job, for minutes of work at the end of a run that took hours.

**The physics is enforced as fitness, not as a filter applied afterwards**:
every candidate is scored on the data term plus the constraint penalties, over
the same probe points with the same weights and the same penalty ceiling, so a
constrained ``loss`` is comparable between runs. Expressions come back in
Python spelling, with ``**`` for exponentiation, so a distilled functional is
something a reader can paste into Python unchanged.

The engine is *injected* rather than imported by :class:`SymbolicDistiller`,
so a different backend remains a parameter and not a rewrite.
"""

from dataclasses import dataclass, field

import numpy as np

from ..fields.constants import (
    BOHR_TO_ANGSTROM,
    C_TF,
    HARTREE_TO_EV,
)

#: Operator alphabets a run defaults to when the config leaves them unset.
DEFAULT_UNARY = ("exp", "log", "sqrt", "abs")
DEFAULT_BINARY = ("+", "-", "*", "/", "^")

#: Feature schemes understood by :func:`build_features`. These select the
#: *input* variables; :data:`TEMPLATES` selects how the target is factorised.
FEATURE_SCHEMES = ("gga", "reduced", "raw", "enhancement")

#: Target factorisations.
#:
#: ``"pauli"``
#:     The **Pauli enhancement factor**, ``F = (tau - tau_vW) / tau_TF``. This
#:     is the well-posed object of orbital-free DFT: ``tau_P = tau - tau_vW``
#:     is non-negative by Hoffmann-Ostenhof, so the search cannot spend its
#:     budget on a quantity that has to cancel against a larger one.
#: ``"thomas_fermi"``
#:     The plain ratio ``F = tau / tau_TF``. Simpler, and the older form.
#: ``"none"``
#:     Fit the target directly.
TEMPLATES = ("none", "thomas_fermi", "pauli")

#: ``features: enhancement`` predates the split between inputs and target
#: factorisation, when one name selected both. It is kept as an alias so an
#: existing config keeps working and keeps meaning exactly what it did: the
#: Pauli factor on ``(p, q)``.
#:
#: It therefore **overrides** any ``template`` set beside it, which is why the
#: explicit pair is the better spelling in a new config -- ``features:
#: reduced`` with ``template: pauli`` says the same thing and cannot silently
#: ignore half of itself.
_FEATURE_ALIASES = {"enhancement": ("reduced", "pauli")}

#: Density below which a voxel is vacuum: dropped, and used to clamp every
#: denominator. In atomic units (:math:`e/a_0^3`).
DEFAULT_EPSILON = 1e-8

#: eV/Å³ per Hartree/Bohr³.
_HA_PER_BOHR3_TO_EV_PER_ANG3 = HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3


@dataclass
class FeatureTable:
    r"""
    Design matrix for the search.

    Attributes
    ----------
    features : numpy.ndarray
        ``(n_points, n_features)``.
    target : numpy.ndarray
        ``(n_points,)``.
    feature_names : list of str
        Column names, used verbatim as the engine's variable names and so
        carried into the printed expression.
    target_name : str
        Name of the fitted quantity.
    scheme : str
        Which of :data:`FEATURE_SCHEMES` produced it.
    units : str
        Human-readable unit note for the report.
    source : str
        Where the fitted values came from — ``"model"`` or ``"reference"``.
        Carried on the table rather than read back off the config, so a table
        built by hand cannot be reported as something it is not.
    template : str
        Which of :data:`TEMPLATES` factorised the target.
    density : numpy.ndarray or None
        :math:`\rho` in atomic units for every retained voxel, kept even when
        it is not a feature. Reconstructing :math:`\tau` from an enhancement
        factor needs :math:`\tau_{\rm TF}`, and that needs the density — which
        the ``reduced`` scheme does not otherwise pass on.
    physical_target : numpy.ndarray or None
        The target in its physical form (:math:`\tau` in eV/Å³), before any
        template divided it. What a parity plot has to be drawn against.
    tau_vw : numpy.ndarray or None
        :math:`\tau_{\rm vW}` in atomic units, retained so the ``pauli``
        template can be undone: reconstructing :math:`\tau` needs the term
        that was subtracted, not just the one that was divided out.
    """

    features: np.ndarray
    target: np.ndarray
    feature_names: list
    target_name: str
    scheme: str
    units: str
    source: str = ""
    template: str = "none"
    density: np.ndarray = None
    physical_target: np.ndarray = None
    tau_vw: np.ndarray = None

    def __len__(self):
        return int(self.features.shape[0])


@dataclass
class SymbolicResult:
    """
    Outcome of one search.

    Attributes
    ----------
    expression : str
        Best expression, in the engine's own notation.
    latex : str
        The same expression as LaTeX, for the report. Falls back to the plain
        form when it cannot be parsed.
    complexity : int
        Node count of the chosen expression.
    loss : float
        The engine's reported loss for it.
    r2 : float
        Coefficient of determination against the fitted target, computed here
        rather than taken from the engine so it means the same thing across
        backends.
    relative_l2 : float
        The error measure the rest of the package quotes, for comparability
        with the operator's own score.
    pareto : list of dict
        The accuracy/complexity front: every candidate the engine kept, as
        ``{"complexity", "loss", "expression"}``. The front is the result —
        a single expression hides the trade that produced it.
    """

    expression: str
    latex: str
    complexity: int
    loss: float
    r2: float
    relative_l2: float
    pareto: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)
    target_name: str = ""
    scheme: str = ""
    units: str = ""
    target: str = ""
    n_samples: int = 0
    engine: str = ""
    limits: dict = field(default_factory=dict)
    compliant_expressions: list = field(default_factory=list)
    #: Which physical constraints were part of the *objective*, as opposed to
    #: checked afterwards. Empty when the search was unconstrained. Reported
    #: because a limit that cannot be expressed in the chosen variables is
    #: simply absent, and "enforced" would otherwise be assumed.
    constraints_enforced: list = field(default_factory=list)
    template: str = "none"
    full_expression: str = ""
    full_latex: str = ""
    parity_plot: str = None
    validation: dict = field(default_factory=dict)
    #: The same scoring as :attr:`validation`, but on the voxels the search was
    #: fitted to. Present even when nothing was held out, so a parity plot can
    #: always be drawn; read it as a training fit, not a generalisation score.
    fitted: dict = field(default_factory=dict)

    #: The Pareto knee: the front entry nearest the ideal corner once
    #: complexity and log-loss are rescaled to [0, 1]. Reported beside the
    #: lowest-loss expression because the two answer different questions --
    #: "what fits best" and "what is worth its length" -- and the second is
    #: usually the one a reader wants to quote.
    knee: dict = field(default_factory=dict)
    #: The knee expression scored the same way as :attr:`fitted` /
    #: :attr:`validation`, so its parity plot is drawn on identical data.
    knee_fitted: dict = field(default_factory=dict)
    knee_validation: dict = field(default_factory=dict)
    knee_parity_plot: str = None
    pareto_plot: str = None

    def knee_expression(self):
        """The knee's expression, falling back to the best one."""
        return self.knee.get("expression") or self.expression

    def summary(self):
        """Multi-line text block, for the log and the terminal."""
        lines = [
            f"  expression   : {self.full_expression or self.expression}",]
        if self.template != "none":
            lines.append(f"  fitted        : {self.target_name} = "
                         f"{self.expression}   (template: {self.template})")
        lines += [
            f"  variables    : {', '.join(self.feature_names)}  [{self.units}]",
            f"  complexity   : {self.complexity} nodes",
            f"  fit          : R2 {self.r2:.4f}   relative L2 {self.relative_l2:.4f}",
            f"  fitted on    : {self.n_samples} voxels of the "
            f"{'operator prediction' if self.target == 'model' else 'DFT reference'}",
        ]
        if self.validation:
            lines.append(
                f"  held out     : relative L2 "
                f"{self.validation.get('relative_l2', float('nan')):.4f}   "
                f"R2 {self.validation.get('r2', float('nan')):.4f}   "
                f"on {self.validation.get('n_points', 0)} voxels vs DFT")
        lines.append(
            "  constraints  : " +
            (", ".join(self.constraints_enforced) + " (penalised in-loop)"
             if self.constraints_enforced
             else "none in-loop; the limits below are checked after the search"))
        if self.limits:
            lines.append(f"  limits       : {self.limits.get('badge', '--/--')}"
                         f"   (score {self.limits.get('score', 0.0):.1f} of 1)")
            for key in ("thomas_fermi", "von_weizsacker"):
                check = self.limits.get(key) or {}
                if check.get("detail"):
                    lines.append(f"      {key:<16s} {check['detail']}")
        if self.pareto:
            lines.append("  accuracy/complexity front "
                         "(TF/vW = asymptotic limits satisfied):")
            lines.append(f"      {'nodes':>5s}  {'loss':>12s}  {'limits':>7s}"
                         f"  expression")
            for entry in self.pareto:
                badge = (entry.get("limits") or {}).get("badge", "--/--")
                lines.append(f"      {entry['complexity']:5d}  "
                             f"{entry['loss']:12.5g}  {badge:>7s}  "
                             f"{entry['expression']}")
        if self.compliant_expressions:
            lines.append(f"  {len(self.compliant_expressions)} of "
                         f"{len(self.pareto)} candidates satisfy BOTH limits; "
                         f"the simplest is:")
            lines.append(f"      {self.compliant_expressions[0]}")
        elif self.pareto:
            lines.append("  no candidate satisfies both limits — the front is "
                         "numerically good and physically incomplete.")
        return "\n".join(lines)


# ===================================================================== #
# Features
# ===================================================================== #
def spectral_laplacian(field, grid, length_unit="angstrom"):
    r"""
    Laplacian of a periodic field via FFT, :math:`\nabla^2 f \to -|G|^2 \hat f`.

    One transform pair rather than the six a gradient-of-gradient would cost,
    and exact for band-limited fields for the same reason
    :func:`~poraque.fields.density.spectral_gradient` is.

    Parameters
    ----------
    field : array_like
        Real array of shape ``grid.shape``.
    grid : FieldGrid
        Mesh supplying the reciprocal vectors (Å⁻¹).
    length_unit : {"angstrom", "bohr"}, optional
        Unit of the differentiation variable in the result.

    Returns
    -------
    numpy.ndarray
    """
    scale = 1.0 if length_unit == "angstrom" else BOHR_TO_ANGSTROM
    g_squared = sum((component * scale) ** 2
                    for component in grid.get_g_vectors())
    transformed = np.fft.fftn(np.asarray(field, dtype=float))
    return np.real(np.fft.ifftn(-g_squared * transformed))


def resolve_scheme(scheme, template="none"):
    """
    Split a feature scheme into ``(inputs, template)``, expanding aliases.

    Returns
    -------
    tuple of (str, str)
    """
    if scheme in _FEATURE_ALIASES:
        return _FEATURE_ALIASES[scheme]
    if scheme not in FEATURE_SCHEMES:
        raise ValueError(f"Unknown feature scheme {scheme!r}; "
                         f"expected one of {list(FEATURE_SCHEMES)}.")
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template {template!r}; "
                         f"expected one of {list(TEMPLATES)}.")
    return scheme, template


def build_features(density, target, grid, scheme="gga",
                   epsilon=DEFAULT_EPSILON, template="none"):
    r"""
    Assemble the design matrix from a density and a target field.

    The default variables are the ones a semi-local kinetic functional is
    actually written in — the density together with its **dimensionless**
    reduced derivatives,

    .. math::

        k_F = (3\pi^2\rho)^{1/3},
        \qquad
        p = \frac{|\nabla\rho|}{2 k_F \rho},
        \qquad
        q = \frac{\nabla^2\rho}{4 k_F^2 \rho}.

    :math:`p` and :math:`q` are the GGA and meta-GGA variables. They are
    dimensionless, and invariant under the coordinate scaling
    :math:`\rho_\lambda(\mathbf r) = \lambda^3\rho(\lambda\mathbf r)` that
    fixes :math:`T_s` — so a functional expressed in them holds at every
    density scale, rather than having to be rediscovered at each. Feeding the
    raw :math:`|\nabla\rho|` and :math:`\nabla^2\rho` instead makes every
    coefficient carry units, which a genetic search spends its budget
    approximating.

    Derivatives are spectral — :math:`\nabla \to i\mathbf{G}`,
    :math:`\nabla^2 \to -|\mathbf{G}|^2` — which is exact for the band-limited
    periodic fields a plane-wave grid carries, unlike any finite-difference
    stencil.

    Parameters
    ----------
    density : ChargeDensity or numpy.ndarray
        :math:`\rho` in e/Å³, as stored.
    target : KineticEnergyDensity or numpy.ndarray
        :math:`\tau` in eV/Å³, as stored — either the operator's prediction or
        the DFT reference.
    grid : FieldGrid
        Supplies the reciprocal vectors for the spectral derivatives.
    scheme : {"gga", "reduced", "raw", "enhancement"}, optional
        The **input variables**; ``template`` separately selects how the target
        is factorised.

        ``"gga"`` gives :math:`(\rho, p, q)`.

        ``"reduced"`` gives :math:`(p, q)` alone. Paired with
        ``template="pauli"`` this is the form the literature writes kinetic
        functionals in — everything the search must find is then order unity,
        and :math:`\rho` is dropped because a dimensionless :math:`F` cannot
        depend on it: offering it would only give the search a way to fit the
        particular densities in the dataset.

        ``"raw"`` gives :math:`(\rho, |\nabla\rho|, \nabla^2\rho)`, all in
        atomic units — dimensional, and kept only for checking the reduced
        forms against something unprocessed.

        ``"enhancement"`` is an alias for ``"reduced"`` with
        ``template="pauli"``, and **overrides** any ``template`` passed
        alongside it.
    epsilon : float, optional
        Vacuum threshold in atomic units (:math:`e/a_0^3`). Used twice, and
        both uses are needed:

        1. Every denominator is clamped at it, so :math:`k_F`, :math:`p` and
           :math:`q` are finite even where the density underflows.
        2. Voxels with :math:`\rho \le \epsilon` are then **dropped**. Vacuum
           carries no information about the functional — :math:`p` and
           :math:`q` there are ratios of two numerically tiny numbers, so they
           are noise with a plausible magnitude, which is worse than a gap.

        The mask also removes the slightly negative voxels that band-limiting
        leaves around the core peaks (Gibbs ringing), where a fractional power
        of :math:`\rho` would otherwise be complex.

    Returns
    -------
    FeatureTable
    """
    scheme, template = resolve_scheme(scheme, template)

    from ..fields.density import spectral_gradient

    rho_ang = np.asarray(getattr(density, "data", density), dtype=float)
    tau_ang = np.asarray(getattr(target, "data", target), dtype=float)
    if rho_ang.shape != tau_ang.shape:
        raise ValueError("density and target must share a grid: "
                         f"{rho_ang.shape} against {tau_ang.shape}.")

    # -> atomic units, where the constants worth finding are order unity.
    rho = rho_ang * BOHR_TO_ANGSTROM ** 3
    tau = tau_ang / _HA_PER_BOHR3_TO_EV_PER_ANG3

    gradient = spectral_gradient(rho, grid, length_unit="bohr")
    grad_norm = np.sqrt(sum(component ** 2 for component in gradient))
    laplacian = spectral_laplacian(rho, grid, length_unit="bohr")

    epsilon = float(epsilon)
    # Clamp first, mask second. Clamping alone would keep vacuum voxels whose
    # p and q are noise; masking alone would still evaluate the ratios on the
    # full grid and raise or emit NaN on the way there.
    rho_safe = np.clip(rho, epsilon, None)
    k_f = (3.0 * np.pi ** 2 * rho_safe) ** (1.0 / 3.0)
    p = grad_norm / (2.0 * k_f * rho_safe)
    q = laplacian / (4.0 * k_f ** 2 * rho_safe)

    keep = rho > epsilon
    if not keep.any():
        raise ValueError(
            f"Every voxel is at or below the vacuum threshold "
            f"epsilon={epsilon:g} e/a0^3; nothing to fit.")

    columns, names, units = {
        "gga": ([rho, p, q], ["rho", "p", "q"],
                "rho in e/a0^3; p and q dimensionless"),
        "reduced": ([p, q], ["p", "q"], "dimensionless"),
        "raw": ([rho, grad_norm, laplacian], ["rho", "grad_rho", "lap_rho"],
                "atomic units"),
    }[scheme]

    # The template divides the target by the part of the physics already known,
    # so the search works on what is not. Thomas-Fermi supplies the density
    # scaling exactly; without it the search spends its budget rediscovering
    # rho^(5/3) and every constant it finds carries units.
    tau_tf = C_TF * rho_safe ** (5.0 / 3.0)
    # von Weizsacker on the grid, from its definition. Equivalent to
    # (5/3) p^2 tau_TF -- a test pins the two against each other -- but written
    # out so the quantity being subtracted is the one the name says.
    tau_vw = grad_norm ** 2 / (8.0 * rho_safe)

    if template == "pauli":
        # The Pauli term is what a kinetic functional actually has to model:
        # tau_vW is known in closed form, so leaving it in the target means
        # fitting a quantity that is mostly already known, and near the vW
        # limit means fitting a near-cancellation.
        fitted, target_name = (tau - tau_vw) / tau_tf, "F"
        units = f"{units}; F = (tau - tau_vW) / tau_TF, dimensionless"
    elif template == "thomas_fermi":
        fitted, target_name = tau / tau_tf, "F"
        units = f"{units}; F = tau / tau_TF, dimensionless"
    else:
        fitted, target_name = tau, "tau"
        units = f"{units}; tau in Ha/a0^3"

    return FeatureTable(
        features=np.column_stack([column[keep] for column in columns]),
        target=fitted[keep], feature_names=names, target_name=target_name,
        scheme=scheme, units=units, template=template,
        density=rho[keep], physical_target=tau_ang[keep],
        tau_vw=tau_vw[keep])


def sample_rows(table, n_samples, seed=0):
    """
    Take a random subset of the rows, without replacement.

    The search cost is linear in the row count and a single 32³ structure is
    already 32 768 voxels, so the whole dataset is never handed over whole.
    """
    total = len(table)
    if n_samples <= 0 or n_samples >= total:
        return table
    picked = np.random.default_rng(seed).choice(total, n_samples, replace=False)
    return FeatureTable(
        features=table.features[picked], target=table.target[picked],
        feature_names=list(table.feature_names),
        target_name=table.target_name, scheme=table.scheme, units=table.units,
        source=table.source, template=table.template,
        density=None if table.density is None else table.density[picked],
        physical_target=(None if table.physical_target is None
                         else table.physical_target[picked]),
        tau_vw=None if table.tau_vw is None else table.tau_vw[picked])


def stack_tables(tables):
    """Concatenate per-structure tables into one design matrix."""
    tables = [table for table in tables if len(table)]
    if not tables:
        raise ValueError("No usable voxels in any structure.")
    first = tables[0]
    if any(table.feature_names != first.feature_names for table in tables):
        raise ValueError("Cannot stack tables built with different schemes.")
    def join(attribute):
        parts = [getattr(t, attribute) for t in tables]
        return None if any(p is None for p in parts) else np.concatenate(parts)

    return FeatureTable(
        features=np.concatenate([t.features for t in tables]),
        target=np.concatenate([t.target for t in tables]),
        feature_names=list(first.feature_names),
        target_name=first.target_name, scheme=first.scheme, units=first.units,
        source=first.source, template=first.template,
        density=join("density"), physical_target=join("physical_target"),
        tau_vw=join("tau_vw"))


# ===================================================================== #
# Physical constraints, enforced inside the search
# ===================================================================== #
#: Reduced gradient standing in for :math:`p\to\infty` in the von Weizsäcker
#: probe. Large enough that no smooth interpolating form is still in its
#: crossover region, small enough that :math:`p^2` and :math:`p^{8/3}` stay far
#: inside the range of a 32-bit float. The search itself runs in float64, but
#: the value is left where it was: it is a stand-in for infinity, and the one
#: that has been used on this project's own data.
DEFAULT_P_INFINITY = 1e6

#: Densities the probes are swept over when :math:`\rho` is itself a feature
#: (the ``gga`` scheme). The limits are statements about *every* density, so a
#: candidate that satisfies them at one value and not another has not satisfied
#: them; sweeping turns that into a penalty rather than a lucky pass. The same
#: three values are used by the post-hoc checker in :func:`_evaluate`.
PROBE_DENSITIES = (0.05, 0.2, 0.8)

#: Ceiling on any single penalty term, and the value charged when a probe
#: cannot be evaluated at all.
#:
#: A candidate that diverges at :math:`p = 10^6` scores :math:`\sim10^{12}`
#: there, and one built from ``exp`` overflows to infinity. Either would swamp
#: the accumulator and make every unphysical candidate *equally* unphysical —
#: including in 32-bit arithmetic, where the sum stops resolving the data term
#: entirely. Clamping keeps the objective finite and ordered while leaving any
#: violator decisively worse than a data term of order one.
PENALTY_CEILING = 1e6

#: Default penalty weights, four orders of magnitude above a converged data
#: term. These are constraints, not regularisers: the intent is that no
#: accuracy gain can buy a violation.
DEFAULT_PHYSICS_WEIGHTS = {
    "positivity": 1.0e2,
    "thomas_fermi": 1.0e2,
    "von_weizsacker": 1.0e2,
}

#: Data terms the physics penalties are added to.
DATA_LOSSES = {"mse": "abs2", "mae": "abs"}


def symbolic_physics(config):
    """
    The ``symbolic.physics`` block, with every key guaranteed present.

    Reads a mapping when the config carries one and falls back to
    :data:`DEFAULT_PHYSICS_WEIGHTS` and :data:`DEFAULT_P_INFINITY` otherwise,
    so a partial block -- ``physics: {enable: false}`` -- is a valid config
    rather than a ``KeyError`` two hundred lines into a run.

    Parameters
    ----------
    config : SymbolicConfig or object
        Anything carrying a ``physics`` mapping.

    Returns
    -------
    dict
        ``enable`` plus the three weights and ``p_infinity``.
    """
    defaults = {
        "enable": True,
        "positivity_weight": DEFAULT_PHYSICS_WEIGHTS["positivity"],
        "thomas_fermi_weight": DEFAULT_PHYSICS_WEIGHTS["thomas_fermi"],
        "von_weizsacker_weight": DEFAULT_PHYSICS_WEIGHTS["von_weizsacker"],
        "p_infinity": DEFAULT_P_INFINITY,
    }
    supplied = getattr(config, "physics", None)
    if isinstance(supplied, dict):
        defaults.update({k: v for k, v in supplied.items() if v is not None})
    return defaults


def physics_probes(feature_names, template, p_infinity=DEFAULT_P_INFINITY,
                   densities=PROBE_DENSITIES):
    r"""
    The synthetic points the asymptotic limits are enforced at.

    The two limits that pin a kinetic functional are statements about the
    enhancement factor at *specific* points of feature space — not about the
    training data — so they cannot be expressed as an elementwise loss over the
    batch. Each probe here is one such point, together with the value the
    candidate must take there.

    Thomas-Fermi
        :math:`F(p=0, q=0) = 1`. A uniform electron gas has no density
        variation, so the functional must collapse onto Thomas-Fermi exactly.
    von Weizsäcker
        :math:`F(p\to\infty, q=0) \to 0` for the Pauli factor, since
        :math:`\tau \to \tau_{\rm vW}` where the density varies rapidly. Under
        the ``thomas_fermi`` template the same statement reads
        :math:`F \to 5p^2/3`, which at :math:`p = 10^6` is a number of order
        :math:`10^{12}`; that probe is therefore compared **relatively**, since
        an absolute tolerance there is finer than the float it is measured in.

    Both are undefined unless the fitted quantity is an enhancement factor and
    :math:`p` and :math:`q` are among the variables, so the ``none`` template
    and the ``raw`` scheme return no probes — positivity still applies to them,
    and is enforced on the batch instead.

    Parameters
    ----------
    feature_names : sequence of str
        The search's variables, in column order.
    template : str
        One of :data:`TEMPLATES`.
    p_infinity : float, optional
        Reduced gradient standing in for the von Weizsäcker limit.
    densities : sequence of float, optional
        Values of :math:`\rho` to sweep when it is a feature.

    Returns
    -------
    list of dict
        ``{"limit", "point", "target", "scale"}``. ``point`` maps every feature
        name to its value; the penalty is ``|F(point) - target| / scale``.
    """
    names = list(feature_names)
    # Every probe has to give a value to *every* variable, so the variable set
    # must be one this function knows how to fix. That is the two reduced
    # derivatives, optionally with the density; anything else -- the `raw`
    # scheme, a hand-built table -- has no probe and keeps positivity alone.
    if template == "none" or not {"p", "q"}.issubset(names) \
            or not set(names).issubset({"rho", "p", "q"}):
        return []

    sweep = tuple(densities) if "rho" in names else (None,)
    probes = []
    for density in sweep:
        base = {} if density is None else {"rho": float(density)}
        probes.append({
            "limit": "thomas_fermi",
            "point": {**base, "p": 0.0, "q": 0.0},
            "target": 1.0,
            "scale": 1.0,
        })
        if template == "pauli":
            reference = 0.0
        else:
            # tau_vW / tau_TF = 5 p^2 / 3 exactly; under the thomas_fermi
            # template that is what F must approach rather than zero.
            reference = 5.0 / 3.0 * float(p_infinity) ** 2
        probes.append({
            "limit": "von_weizsacker",
            "point": {**base, "p": float(p_infinity), "q": 0.0},
            "target": reference,
            # Absolute for the Pauli factor, whose target is zero and for which
            # a relative deviation is undefined; relative for the other, whose
            # target is too large to compare absolutely in 32-bit arithmetic.
            "scale": 1.0 if reference == 0.0 else abs(reference),
        })
    return probes


# ===================================================================== #
# Engine
# ===================================================================== #
def physics_constraints(feature_names, template, weights=None,
                        p_infinity=DEFAULT_P_INFINITY,
                        densities=PROBE_DENSITIES, ceiling=PENALTY_CEILING):
    r"""
    The physics the search is scored against, as data the engine can read.

    Filtering candidates *after* a run only reports how few of them were
    physical. This makes the constraints part of the fitness instead, so the
    populations never spend their budget on forms that were going to be
    discarded:

    .. math::

        \mathcal{L} = \underbrace{\frac1n\sum_i \ell(F_i - y_i)}_{\text{data}}
        \;+\; w_{+}\,\frac1n\sum_i \min(F_i, 0)^2
        \;+\; \sum_{\rm probes} w_\ell\,
              \frac{|F(\mathbf x_\ell) - t_\ell|}{s_\ell} .

    The first penalty is evaluated on the batch, where negative predictions are
    the whole violation: :math:`\tau \ge 0` always, and
    :math:`\tau - \tau_{\rm vW}\ge0` by Hoffmann-Ostenhof, so a negative value
    is unphysical under every template. The rest are evaluated on the synthetic
    points of :func:`physics_probes`, which is why the constraints cannot be an
    elementwise loss: an elementwise loss receives only ``(prediction,
    target)`` pairs and can never evaluate a candidate anywhere the data does
    not already sit.

    Returned as a plain mapping rather than as code for the engine, which is
    what lets :mod:`poraque.ml.gp` evaluate the identical expression without
    importing anything from this module — the two would otherwise be a cycle.

    Parameters
    ----------
    feature_names : sequence of str
        The search's variables, in the column order the engine will receive.
    template : str
        One of :data:`TEMPLATES`; selects which limits are meaningful.
    weights : dict, optional
        Overrides for :data:`DEFAULT_PHYSICS_WEIGHTS`, keyed by
        ``"positivity"``, ``"thomas_fermi"`` and ``"von_weizsacker"``.
    p_infinity : float, optional
        Reduced gradient standing in for the von Weizsäcker limit.
    densities : sequence of float, optional
        Passed to :func:`physics_probes`.
    ceiling : float, optional
        Clamp on each penalty term; see :data:`PENALTY_CEILING`.

    Returns
    -------
    tuple of (dict, list of str)
        The constraint block — ``probes``, ``weights`` and ``ceiling`` — and
        the names of the constraints it actually enforces, which is what the
        run should report: a limit that cannot be expressed in the chosen
        variables is silently absent otherwise.

    Notes
    -----
    The reported ``loss`` of every candidate is this **constrained** objective,
    so it is not comparable with an unconstrained run's. The :math:`R^2` and
    relative :math:`L^2` on :class:`SymbolicResult` are computed in Python from
    the expression alone and are unaffected.

    Examples
    --------
    >>> block, enforced = physics_constraints(["p", "q"], "pauli")
    >>> enforced
    ['positivity', 'thomas_fermi', 'von_weizsacker']
    >>> len(block["probes"])
    2
    """
    scored = dict(DEFAULT_PHYSICS_WEIGHTS)
    scored.update(weights or {})

    probes = physics_probes(list(feature_names), template,
                            p_infinity=p_infinity, densities=densities)
    enforced = ["positivity"]
    enforced += sorted({probe["limit"] for probe in probes},
                       key=["thomas_fermi", "von_weizsacker"].index)
    return ({"probes": probes, "weights": scored, "ceiling": float(ceiling)},
            enforced)


#: Decimal places kept in a reported expression.
#:
#: A search returns constants at full float precision --
#: ``0.33333334326744079589843750`` is a real example -- and three of those in
#: one formula overrun the width of a PDF page. Three places is what the
#: numbers are worth: the search is stochastic, its constants move in the third
#: place between seeds, and printing twenty digits states a precision the
#: method does not have.
REPORT_DECIMALS = 3


def round_expression(expression, decimals=REPORT_DECIMALS,
                     feature_names=()):
    """
    Round every numeric constant in an expression for display.

    Parameters
    ----------
    expression : str
        In the engine's notation.
    decimals : int, optional
    feature_names : sequence of str, optional
        Symbols to keep as symbols rather than let SymPy invent functions from.

    Returns
    -------
    str
        The rounded expression, or the input unchanged when it cannot be
        parsed -- a display convenience must never lose the result.
    """
    try:
        import sympy
    except ImportError:                        # pragma: no cover
        return str(expression)
    try:
        symbols = {name: sympy.Symbol(name) for name in feature_names}
        parsed = sympy.sympify(str(expression), locals=symbols)
        rounded = parsed.xreplace(
            {value: sympy.Float(round(float(value), decimals), decimals + 2)
             for value in parsed.atoms(sympy.Float)})
        return str(rounded)
    except (sympy.SympifyError, TypeError, SyntaxError, AttributeError,
            ValueError, OverflowError):
        return str(expression)


def pareto_knee(front):
    r"""
    The knee of an accuracy/complexity front, by distance to the ideal point.

    The front states a trade and refuses to resolve it; the knee is where
    resolving it costs least. Both axes are rescaled to :math:`[0, 1]` across
    the front -- they are a node count and a loss, with no common unit, so a
    distance is only meaningful once they are made comparable -- and the knee
    is the candidate nearest the ideal corner :math:`(0, 0)`:

    .. math::

        d_i = \sqrt{\hat c_i^{\,2} + \hat{\mathcal{L}}_i^{\,2}}, \qquad
        \hat x_i = \frac{x_i - \min x}{\max x - \min x} .

    The loss is compared on a **logarithmic** scale. It spans orders of
    magnitude across a front, and on a linear scale every candidate but the
    most accurate one collapses onto :math:`\hat{\mathcal{L}} \approx 1`, which
    makes the knee the shortest expression regardless of what it costs.

    Parameters
    ----------
    front : sequence of dict
        Entries with ``complexity`` and ``loss``.

    Returns
    -------
    dict
        The chosen entry, with ``distance`` and ``knee=True`` added, plus
        ``distance`` written onto every entry of ``front`` in place. ``{}``
        for an empty front.

    Notes
    -----
    A knee is a heuristic, not a theorem. With one candidate it is that
    candidate; where the front is a straight line every point is equidistant
    and the first wins. It is a defensible default, not an answer to
    "which expression is *right*" -- that is what the asymptotic limits are
    for.
    """
    entries = [dict(entry) for entry in front
               if entry.get("complexity") is not None
               and entry.get("loss") is not None
               and np.isfinite(entry["loss"])]
    if not entries:
        return {}

    complexity = np.array([float(e["complexity"]) for e in entries])
    # log10, so a front spanning decades is compared on the scale it varies on.
    loss = np.log10(np.maximum([float(e["loss"]) for e in entries], 1e-300))

    def unit(values):
        span = values.max() - values.min()
        return (np.zeros_like(values) if span <= 0
                else (values - values.min()) / span)

    distance = np.hypot(unit(complexity), unit(loss))
    for entry, value in zip(entries, distance):
        entry["distance"] = float(value)

    # Written back so the report's table can show the column it ranked on.
    by_key = {(e["complexity"], e["loss"]): e["distance"] for e in entries}
    for entry in front:
        key = (entry.get("complexity"), entry.get("loss"))
        if key in by_key:
            entry["distance"] = by_key[key]

    chosen = dict(entries[int(np.argmin(distance))])
    chosen["knee"] = True
    return chosen


def expression_to_latex(expression, feature_names=(),
                        decimals=REPORT_DECIMALS):
    """
    Render an expression as LaTeX, falling back to a verbatim box.

    SymPy is used rather than string surgery so that precedence and grouping
    survive. A failure here must not lose the result: an expression that cannot
    be parsed is still the answer, and is passed through as monospace text.

    Constants are rounded to ``decimals`` places first. A search returns them
    at full float precision, and three such constants in one formula overrun
    the width of a PDF page; ``None`` keeps them verbatim.
    """
    if decimals is not None:
        expression = round_expression(expression, decimals, feature_names)
    try:
        import sympy
    except ImportError:  # pragma: no cover - sympy ships with the package deps
        return rf"\texttt{{{expression}}}"

    try:
        symbols = {name: sympy.Symbol(name) for name in feature_names}
        return sympy.latex(sympy.sympify(str(expression), locals=symbols))
    except (sympy.SympifyError, TypeError, SyntaxError, AttributeError):
        escaped = str(expression).replace("\\", r"\textbackslash{}")
        return rf"\texttt{{{escaped}}}"


# ===================================================================== #
# Physical asymptotic limits
# ===================================================================== #
@dataclass
class LimitCheck:
    """
    One asymptotic limit, tested on a candidate expression.

    Attributes
    ----------
    name : str
        ``"thomas_fermi"`` or ``"von_weizsacker"``.
    value : float or None
        The limit that was found: :math:`F(0,0)` for Thomas-Fermi, and the
        coefficient of :math:`5p^2/3` for von Weizsäcker. ``None`` when it
        could not be determined.
    passes : bool
        Whether ``value`` is 1 to within the tolerance.
    method : str
        ``"analytic"`` (SymPy), ``"numeric"`` (probed), or ``"undetermined"``.
    detail : str
        One line for the report, including *why* when it fails.
    """

    name: str
    value: float = None
    passes: bool = False
    method: str = "undetermined"
    detail: str = ""


@dataclass
class AsymptoticCompliance:
    r"""
    Whether an expression obeys the two limits that pin a kinetic functional.

    Both are statements about the Pauli enhancement factor
    :math:`F = (\tau - \tau_{\rm vW})/\tau_{\rm TF}`:

    ``thomas_fermi``
        :math:`F(0,0) = 1`. A uniform electron gas has no density variation, so
        the functional must collapse to Thomas-Fermi exactly.
    ``von_weizsacker``
        :math:`F \to 5p^2/3` as :math:`p \to \infty`. Where the density varies
        rapidly — a single orbital, an exponential tail — the exact answer is
        von Weizsäcker.

    Neither known functional passes both: Thomas-Fermi fails the second and von
    Weizsäcker fails the first. That is the point. An expression passing both
    interpolates between the two regimes, which is what a usable semi-local
    kinetic functional has to do.

    Attributes
    ----------
    bounded_at_infinity : bool
        Whether :math:`F` tends to *any* finite limit as :math:`p\to\infty`.
        Reported separately from whether that limit is zero: a functional that
        settles on a finite non-zero constant is qualitatively different from
        one that diverges, and only the first is a candidate for repair.
    score : float
        Fraction of the two limits satisfied — 0, 0.5 or 1.
    """

    thomas_fermi: LimitCheck
    von_weizsacker: LimitCheck
    bounded_at_infinity: bool = False
    score: float = 0.0

    @property
    def passes(self):
        """Both limits satisfied."""
        return self.thomas_fermi.passes and self.von_weizsacker.passes

    def badge(self):
        """Compact ``TF``/``vW`` indicator for a console table."""
        return (f"{'TF' if self.thomas_fermi.passes else '--'}"
                f"/{'vW' if self.von_weizsacker.passes else '--'}")


def pauli_form(expression, feature_names, scheme, template="none"):
    r"""
    Rewrite a candidate as the Pauli factor
    :math:`F = (\tau - \tau_{\rm vW})/\tau_{\rm TF}`.

    The asymptotic limits are statements about that one quantity, so every
    combination is converted to it and a single checker serves all of them:

    - ``template="pauli"`` already fits it; returned as is.
    - ``template="thomas_fermi"`` fits :math:`\tau/\tau_{\rm TF}`, so
      :math:`\tau_{\rm vW}/\tau_{\rm TF} = 5p^2/3` is subtracted.
    - ``template="none"`` fits :math:`\tau`, so it is divided by
      :math:`C_{\rm TF}\rho^{5/3}` and then reduced the same way. Under
      ``scheme="raw"`` the dimensional derivatives are substituted first —
      :math:`|\nabla\rho| = 2k_F\rho\,p` and
      :math:`\nabla^2\rho = 4k_F^2\rho\,q`.

    Returns
    -------
    tuple of (sympy.Expr, dict)
        The expression and the symbol table it was built with. Symbols are
        created here and passed to ``sympify``: parsing without them yields
        *different* ``Symbol`` objects, and a limit taken against those
        silently returns the expression unchanged.
    """
    import sympy

    symbols = {"p": sympy.Symbol("p", positive=True),
               "q": sympy.Symbol("q", real=True),
               "rho": sympy.Symbol("rho", positive=True)}
    for name in feature_names:
        symbols.setdefault(name, sympy.Symbol(name, real=True))

    parsed = sympy.sympify(str(expression), locals=symbols)
    p, q, rho = symbols["p"], symbols["q"], symbols["rho"]

    scheme, template = resolve_scheme(scheme, template)
    # tau_vW / tau_TF = 5 p^2 / 3 exactly, which is what lets every template be
    # brought to one convention without needing the density.
    vw_ratio = sympy.Rational(5, 3) * p ** 2

    if template == "pauli":
        return parsed, symbols                 # already the Pauli factor
    if template == "thomas_fermi":
        return parsed - vw_ratio, symbols      # F = tau/tau_TF -> F - 5p^2/3

    if scheme == "raw":
        k_f = (3 * sympy.pi ** 2 * rho) ** sympy.Rational(1, 3)
        parsed = parsed.subs({
            symbols.get("grad_rho", sympy.Symbol("grad_rho")): 2 * k_f * rho * p,
            symbols.get("lap_rho", sympy.Symbol("lap_rho")): 4 * k_f ** 2 * rho * q,
        })

    return (parsed / (sympy.Float(C_TF) * rho ** sympy.Rational(5, 3))
            - vw_ratio), symbols


def check_asymptotic_limits(expression, feature_names, scheme,
                            tolerance=0.05, template="none"):
    r"""
    Test a candidate against the Thomas-Fermi and von Weizsäcker limits.

    Analytic first, via :func:`sympy.limit`. A genetic search produces deeply
    nested expressions that SymPy can fail or stall on, so a numerical probe is
    the fallback: :math:`F` evaluated at successively smaller :math:`p, q` and
    at successively larger :math:`p`, accepted only when the values converge.
    Which route was taken is recorded, because an analytic limit is a proof and
    a numerical one is evidence.

    Parameters
    ----------
    expression : str
        Candidate, in the engine's notation.
    feature_names : sequence of str
        Its variables.
    scheme : str
        One of :data:`FEATURE_SCHEMES`; selects the conversion to :math:`F`.
    tolerance : float, optional
        Relative tolerance on "equals 1".

    Returns
    -------
    AsymptoticCompliance
    """
    import sympy

    try:
        enhancement, symbols = pauli_form(expression, feature_names,
                                          scheme, template)
    except Exception:                                   # noqa: BLE001
        unparsed = LimitCheck("thomas_fermi",
                              detail="expression could not be parsed")
        return AsymptoticCompliance(
            unparsed, LimitCheck("von_weizsacker",
                                 detail="expression could not be parsed"))

    p, q = symbols["p"], symbols["q"]
    thomas_fermi = _limit_to(
        "thomas_fermi", expected=1.0, tolerance=tolerance,
        analytic=lambda: sympy.limit(sympy.limit(enhancement, p, 0), q, 0),
        numeric=lambda scale: _evaluate(enhancement, {p: scale, q: scale}),
        probes=(1e-3, 1e-5, 1e-7),
        target="F(0,0) = 1")

    von_weizsacker = _limit_to(
        "von_weizsacker", expected=0.0, tolerance=tolerance,
        analytic=lambda: sympy.limit(enhancement, p, sympy.oo),
        numeric=lambda scale: _evaluate(enhancement, {p: scale, q: 0.0}),
        probes=(1e3, 1e4, 1e5),
        target="F -> 0 as p -> infinity")

    bounded = (von_weizsacker.value is not None
               and np.isfinite(von_weizsacker.value))
    score = 0.5 * (thomas_fermi.passes + von_weizsacker.passes)
    return AsymptoticCompliance(thomas_fermi, von_weizsacker,
                                bounded_at_infinity=bool(bounded), score=score)


def _limit_to(name, expected, tolerance, analytic, numeric, probes, target):
    """
    Resolve one limit, analytically if possible and numerically if not.

    ``expected`` is compared on an **absolute** tolerance. A relative one would
    be meaningless for the von Weizsacker limit, whose target is zero.
    """
    value, method = None, "undetermined"
    try:
        result = analytic()
        leftover = getattr(result, "free_symbols", set())
        if leftover:
            return LimitCheck(
                name, None, False, "analytic",
                f"limit still depends on {sorted(str(s) for s in leftover)} — "
                f"a functional must satisfy {target} for every density")
        candidate = complex(result)
        if abs(candidate.imag) < 1e-12 and np.isfinite(candidate.real):
            value, method = float(candidate.real), "analytic"
    except Exception:                                   # noqa: BLE001
        value = None

    if value is None:
        # Converged probe: three shrinking (or growing) evaluations that agree
        # to 1% are a limit; anything else is a value that has not settled.
        samples = [numeric(scale) for scale in probes]
        usable = [s for s in samples if s is not None and np.isfinite(s)]
        if len(usable) == len(probes):
            spread = max(usable) - min(usable)
            reference = max(abs(usable[-1]), 1e-12)
            if spread / reference < 1e-2:
                value, method = float(usable[-1]), "numeric"

    if value is None:
        return LimitCheck(name, None, False, "undetermined",
                          f"could not resolve {target}")

    passes = abs(value - expected) <= tolerance
    verdict = "satisfied" if passes else "violated"
    return LimitCheck(name, value, passes, method,
                      f"{target}: found {value:.4g} ({verdict}, {method})")


def _evaluate(expression, substitutions):
    """Numeric value of ``expression`` at a point, or ``None``."""
    import sympy

    try:
        # Any symbol left unfixed -- `rho` in the gga scheme -- is swept, and
        # the point is accepted only if the value does not depend on it: the
        # limits must hold at every density, not at one convenient one.
        free = expression.free_symbols - set(substitutions)
        densities = [0.05, 0.2, 0.8] if free else [None]
        values = []
        for density in densities:
            point = dict(substitutions)
            for symbol in free:
                point[symbol] = density
            with np.errstate(all="ignore"):
                values.append(float(sympy.N(expression.subs(point))))
        if not all(np.isfinite(v) for v in values):
            return None
        if len(values) > 1:
            spread = max(values) - min(values)
            if spread / max(abs(values[-1]), 1e-12) > 1e-6:
                return None                 # genuinely density-dependent
        return values[0]
    except Exception:                                   # noqa: BLE001
        return None


class SymbolicDistiller:
    """
    Search for a closed-form expression reproducing a fitted mapping.

    Parameters
    ----------
    config : SymbolicConfig, optional
        Supplies every search parameter. Omit it to take the defaults.
    engine : callable, optional
        ``(features, target, feature_names, parameters) -> front``. Defaults to
        :func:`~poraque.ml.gp.native_engine`. Injected rather than imported
        so the pipeline can be exercised against a stub, and so a different
        backend stays a parameter rather than a rewrite.

    Examples
    --------
    >>> distiller = SymbolicDistiller()                       # doctest: +SKIP
    >>> result = distiller.fit(table)                         # doctest: +SKIP
    >>> print(result.expression)                              # doctest: +SKIP
    """

    def __init__(self, config=None, engine=None, limit_tolerance=0.05):
        from .config import SymbolicConfig

        self.config = config if config is not None else SymbolicConfig()
        if engine is None:
            from .gp import native_engine

            engine = native_engine
        self.engine = engine
        self.limit_tolerance = float(limit_tolerance)

    def parameters(self, table=None):
        """
        Search settings as a plain dict, as handed to the engine.

        Parameters
        ----------
        table : FeatureTable, optional
            The design matrix the search will run on. It is what selects the
            physical constraints: which limits are even expressible depends on
            the variables and the template, neither of which the config knows
            on its own. Without it no ``physics`` block is built, and the
            engine scores on the data term alone.

        Returns
        -------
        dict
            Engine settings, plus ``physics`` (the probes, weights and
            penalty ceiling, or ``None``) and ``constraints_enforced`` (the
            constraint names that block actually imposes).
        """
        config = self.config
        unary = list(config.unary_operations or DEFAULT_UNARY)
        binary = list(config.binary_operations or DEFAULT_BINARY)

        # `symbolic.physics`, never `training.physics_informed`: that one
        # constrains
        # the neural operator over voxels, this constrains a candidate algebraic
        # expression over probe points. Two of the key names are the same in
        # both blocks and mean different things, which is why they are nested
        # separately rather than sharing a prefix.
        physics = symbolic_physics(config)
        objective, enforced = None, []
        if table is not None and physics.get("enable", False):
            objective, enforced = physics_constraints(
                table.feature_names, table.template,
                weights={
                    "positivity": float(physics["positivity_weight"]),
                    "thomas_fermi": float(physics["thomas_fermi_weight"]),
                    "von_weizsacker": float(physics["von_weizsacker_weight"]),
                },
                p_infinity=float(physics["p_infinity"]),
            )
        return {
            "physics": objective,
            "constraints_enforced": enforced,
            "data_loss": str(config.data_loss),
            "unary_operations": unary,
            "binary_operations": binary,
            "iterations": int(config.iterations),
            "population_size": int(config.population_size),
            "populations": int(config.populations),
            "max_size": int(config.max_size),
            "max_depth": int(config.max_depth),
            "parsimony": float(config.parsimony),
            "seed": int(config.seed),
            "deterministic": bool(getattr(config, "deterministic", False)),
            # A binary operator's limits are a (left, right) pair; YAML can
            # only give a list, so they are converted here rather than asking
            # the user to write something YAML cannot express.
            "constraints": {
                str(name): (tuple(limit) if isinstance(limit, (list, tuple))
                            else limit)
                for name, limit in (getattr(config, "constraints", None)
                                    or {}).items()
            },
        }

    def fit(self, table):
        """
        Run the search over ``table`` and score the winner.

        Returns
        -------
        SymbolicResult

        Raises
        ------
        ValueError
            If the engine returns no candidate at all — a silent empty result
            would otherwise be reported as a successful distillation.
        """
        parameters = self.parameters(table)
        front = self.engine(table.features, table.target,
                            list(table.feature_names), parameters)
        if not front:
            raise ValueError("The symbolic engine returned no expression.")

        front = sorted(front, key=lambda entry: entry["complexity"])

        # Every candidate is checked, not only the winner: the most accurate
        # expression is frequently the least physical, and a slightly worse one
        # that obeys both limits is usually the better functional.
        for entry in front:
            compliance = check_asymptotic_limits(
                entry["expression"], table.feature_names, table.scheme,
                tolerance=self.limit_tolerance, template=table.template)
            entry["limits"] = _compliance_to_dict(compliance)

        best = min(front, key=lambda entry: entry["loss"])
        compliant = [entry for entry in front if entry["limits"]["passes"]]

        predicted = self.evaluate(best["expression"], table)
        return SymbolicResult(
            expression=best["expression"],
            latex=expression_to_latex(best["expression"], table.feature_names),
            complexity=int(best["complexity"]),
            loss=float(best["loss"]),
            r2=_r2(predicted, table.target),
            relative_l2=_relative_l2(predicted, table.target),
            pareto=front,
            # Computed here so the front and its knee cannot disagree: the
            # distance column is written back onto `front` in the same call.
            knee=pareto_knee(front),
            feature_names=list(table.feature_names),
            target_name=table.target_name,
            scheme=table.scheme,
            units=table.units,
            target=(table.source
                    or getattr(self.config, "target", "")),
            n_samples=len(table),
            engine=getattr(self.engine, "__name__", str(self.engine)),
            limits=best["limits"],
            compliant_expressions=[entry["expression"] for entry in compliant],
            template=table.template,
            full_expression=full_expression(best["expression"], table.template,
                                            table.target_name),
            full_latex=full_latex(
                expression_to_latex(best["expression"], table.feature_names),
                table.template),
            constraints_enforced=list(parameters["constraints_enforced"]),
        )

    @staticmethod
    def evaluate(expression, table):
        """
        Evaluate an expression on a table's features.

        Scoring is done here, on the same rows, rather than trusting the
        engine's own loss: engines differ in what they report (mean square,
        mean absolute, normalised) and the numbers in the report have to be
        comparable with the operator's.

        Returns
        -------
        numpy.ndarray or None
            ``None`` when the expression cannot be parsed or evaluated, which
            leaves the fit metrics as NaN instead of inventing them.
        """
        try:
            import sympy
        except ImportError:  # pragma: no cover
            return None

        try:
            symbols = [sympy.Symbol(name) for name in table.feature_names]
            parsed = sympy.sympify(str(expression),
                                   locals={s.name: s for s in symbols})
            function = sympy.lambdify(symbols, parsed, "numpy")
            columns = [table.features[:, i] for i in range(len(symbols))]
            with np.errstate(all="ignore"):
                values = np.asarray(function(*columns), dtype=float)
            if values.ndim == 0:                      # a constant expression
                values = np.full(table.target.shape, float(values))
            return values
        except Exception:                             # noqa: BLE001
            # Any parse or evaluation failure means "no score", never a score
            # computed from something other than the reported expression.
            return None


def reconstruct_tau(values, table):
    r"""
    Undo the template, returning :math:`\tau` in the units the files carry.

    An expression fitted under ``template: thomas_fermi`` predicts the
    enhancement factor, not :math:`\tau`. Comparing that against a DFT
    :math:`\tau` without multiplying :math:`\tau_{\rm TF}` back in would put
    two different quantities on one axis and call the result a parity plot.

    Parameters
    ----------
    values : numpy.ndarray
        What the expression returned, in the target's fitted form.
    table : FeatureTable
        Supplies the template and the density it needs.

    Returns
    -------
    numpy.ndarray or None
        :math:`\tau` in eV/Å³, or ``None`` when the density needed to rebuild
        it was not retained.
    """
    if values is None:
        return None
    if table.template == "none":
        return np.asarray(values) * _HA_PER_BOHR3_TO_EV_PER_ANG3
    if table.density is None:
        return None

    tau_tf = C_TF * np.clip(table.density, DEFAULT_EPSILON, None) ** (5.0 / 3.0)
    tau = np.asarray(values) * tau_tf
    if table.template == "pauli":
        # tau = tau_vW + tau_TF * F. The subtracted term has to be added back,
        # not just the divided one -- reporting tau_TF * F alone would be the
        # Pauli term mislabelled as the whole kinetic energy density.
        if table.tau_vw is None:
            return None
        tau = tau + table.tau_vw
    return tau * _HA_PER_BOHR3_TO_EV_PER_ANG3


def full_expression(expression, template, target_name="tau"):
    """
    Rebuild the complete formula by folding the template back in.

    The left-hand side is whatever the reconstruction *yields*, not what was
    fitted: under a template the search returns ``F``, but multiplying
    :math:`\\tau_{\\rm TF}` back in gives :math:`\\tau`, and labelling that
    line ``F =`` would state an identity that is false.
    """
    if template == "pauli":
        return f"tau = tau_vW + C_TF * rho^(5/3) * ({expression})"
    if template == "thomas_fermi":
        return f"tau = C_TF * rho^(5/3) * ({expression})"
    return f"{target_name} = {expression}"


def full_latex(latex, template):
    r"""The same, as LaTeX: :math:`\tau = C_{\rm TF}\rho^{5/3}\,(F)`."""
    if template == "pauli":
        return (r"\tau = \tau_{\mathrm{vW}} + C_{\mathrm{TF}}\,\rho^{5/3}"
                r"\left(" + latex + r"\right)")
    if template == "thomas_fermi":
        return (r"\tau = C_{\mathrm{TF}}\,\rho^{5/3}\left(" + latex +
                r"\right)")
    return latex


def result_to_dict(result):
    """
    Serialisable form of a :class:`SymbolicResult`.

    The scored sections -- ``fitted``, ``validation`` and their knee
    counterparts -- carry the two voxel arrays a parity plot is drawn from.
    They belong in a figure, not in a JSON summary: thousands of floats no
    reader consults, and which ``json.dump`` cannot encode anyway.

    Every dict-valued field is stripped, not a named list of them. A hand-kept
    list is exactly what broke when the knee sections were added: they carried
    the same arrays, were not on the list, and the run died at the very end
    with ``TypeError: only 0-dimensional arrays can be converted to Python
    scalars`` -- after the training, the report and the figures were done.
    """
    from dataclasses import asdict

    payload = asdict(result)
    for key, section in payload.items():
        if isinstance(section, dict):
            payload[key] = {name: value for name, value in section.items()
                            if not isinstance(value, np.ndarray)}
    return payload


def _compliance_to_dict(compliance):
    """Flatten an :class:`AsymptoticCompliance` for JSON and the report."""
    from dataclasses import asdict

    payload = asdict(compliance)
    payload["passes"] = compliance.passes
    payload["badge"] = compliance.badge()
    return payload


def _r2(predicted, reference):
    """Coefficient of determination, NaN when the prediction is unusable."""
    if predicted is None:
        return float("nan")
    finite = np.isfinite(predicted)
    if not finite.all():
        return float("nan")
    total = np.sum((reference - reference.mean()) ** 2)
    if total <= 0:
        return float("nan")
    return float(1.0 - np.sum((predicted - reference) ** 2) / total)


def _relative_l2(predicted, reference):
    """Relative L2, on the same convention as the rest of the package."""
    if predicted is None:
        return float("nan")
    finite = np.isfinite(predicted)
    if not finite.all():
        return float("nan")
    denominator = np.linalg.norm(reference)
    if denominator <= 0:
        return float("nan")
    return float(np.linalg.norm(predicted - reference) / denominator)


# ===================================================================== #
# Driver
# ===================================================================== #
def distill_dataset(dataset, config, operator=None, log=None, engine=None,
                    validation=None):
    r"""
    Build the design matrix over a dataset and search it.

    Parameters
    ----------
    dataset : FieldPairDataset
        Supplies :math:`(\rho, \tau)` pairs; only ``chg2tau`` is meaningful,
        since a functional of the density is what is being sought.
    config : SymbolicConfig
        Search settings, including which target to fit.
    operator : FieldOperator, optional
        Required when ``config.target == "model"`` — the point of distillation
        is to fit what the network predicts, not the data it was trained on.
    log : callable, optional
        Progress sink.
    engine : callable, optional
        Passed to :class:`SymbolicDistiller`.

    Returns
    -------
    SymbolicResult
    """
    emit = log if log is not None else (lambda *_: None)
    if config.target not in ("model", "reference"):
        raise ValueError(f"Unknown symbolic target {config.target!r}; "
                         f"expected 'model' or 'reference'.")
    if config.target == "model" and operator is None:
        raise ValueError("target='model' distils the trained operator, so an "
                         "operator is required. Use target='reference' to fit "
                         "the DFT data instead.")

    scheme, template = resolve_scheme(config.features,
                                      getattr(config, "template", "none"))

    def tabulate(source_dataset, fit_target):
        """One stacked table over a dataset, fitting the requested target."""
        built = []
        for index in range(len(source_dataset)):
            source, reference = source_dataset.load_fields(index)
            values = (operator.predict(source) if fit_target == "model"
                      else reference)
            entry = build_features(
                source, values, source.grid, scheme=scheme, template=template,
                epsilon=getattr(config, "epsilon", DEFAULT_EPSILON))
            entry.source = fit_target
            built.append(entry)
        return stack_tables(built)

    table = tabulate(dataset, config.target)
    emit(f"  built {len(table)} usable voxels from {len(dataset)} structure(s)"
         f"  [features: {scheme}, template: {template}, "
         f"target: {config.target}]")
    table = sample_rows(table, config.n_samples, seed=config.seed)
    emit(f"  searching on {len(table)} sampled voxels "
         f"({config.iterations} iterations)")

    distiller = SymbolicDistiller(config, engine=engine)
    enforced = distiller.parameters(table)["constraints_enforced"]
    if enforced:
        emit(f"  physical constraints penalised inside the search: "
             f"{', '.join(enforced)}  "
             f"[{config.data_loss} data term, "
             f"p_inf = {symbolic_physics(config)['p_infinity']:g}]")
        missing = {"thomas_fermi", "von_weizsacker"} - set(enforced)
        if missing:
            # Said out loud rather than left to be inferred: under `template:
            # none` the fitted quantity is tau, and under `features: raw` there
            # is no p to take a limit in, so these are not merely switched off
            # -- they cannot be written down in the variables being searched.
            emit(f"  not expressible in these variables, checked afterwards "
                 f"only: {', '.join(sorted(missing))}")
    else:
        emit("  physical constraints are not penalised in the search "
             "(symbolic.physics.enable is off); limits checked afterwards")

    result = distiller.fit(table)

    # Always scored on the data it was fitted to, so a parity plot can be drawn
    # even for a run with no validation split. It is a weaker statement than the
    # held-out score and is labelled as such wherever it is used, but "no plot"
    # is not a better answer than "a plot that says which data it is".
    result.fitted = _score_on(result.expression, table)
    # The knee is scored on exactly the same voxels, so the two parity plots
    # differ in the expression and in nothing else.
    knee = result.knee_expression()
    if knee != result.expression:
        result.knee_fitted = _score_on(knee, table)

    # Score the winner on the held-out structures, against the DFT reference
    # rather than against whatever was fitted: the question a parity plot
    # answers is how the formula does against ground truth, and when
    # `target: model` was fitted the two are not the same thing.
    if validation is not None and len(validation):
        held_out = sample_rows(tabulate(validation, "reference"),
                               config.n_samples, seed=config.seed)
        result.validation = _score_on(result.expression, held_out)
        if knee != result.expression:
            result.knee_validation = _score_on(knee, held_out)
        emit(f"  validated on {len(held_out)} held-out voxels: "
             f"relative L2 {result.validation.get('relative_l2', float('nan')):.4f}")

    if result.knee:
        emit(f"  pareto knee  : complexity {result.knee['complexity']} nodes, "
             f"loss {result.knee['loss']:.4g}  "
             f"(lowest loss: {result.complexity} nodes, {result.loss:.4g})")
        if knee == result.expression:
            emit("                 the knee IS the lowest-loss expression; "
                 "nothing was traded away")
    return result


def _score_on(expression, table):
    """
    Evaluate an expression on a table and score it in physical units.

    Returns
    -------
    dict
        ``reference`` and ``predicted`` arrays of :math:`\\tau` in eV/Å³ plus
        the usual metrics, or an empty dict when the expression could not be
        evaluated there.
    """
    values = SymbolicDistiller.evaluate(expression, table)
    predicted = reconstruct_tau(values, table)
    reference = table.physical_target
    if predicted is None or reference is None:
        return {}
    finite = np.isfinite(predicted) & np.isfinite(reference)
    if not finite.any():
        return {}
    predicted, reference = predicted[finite], reference[finite]
    return {
        "n_points": int(finite.sum()),
        "relative_l2": _relative_l2(predicted, reference),
        "r2": _r2(predicted, reference),
        "mae": float(np.mean(np.abs(predicted - reference))),
        "predicted": predicted,
        "reference": reference,
    }
