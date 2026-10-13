# -*- coding: utf-8 -*-
# file: symbolic_config.py
"""
The ``symbolic`` section of the training configuration, as it stood when
symbolic distillation was archived out of the package (2026-10-13).
Copied verbatim from ``poraque/ml/config.py``; it still leans on
``split_enable_block`` and ``_yaml_literal`` from there.
"""

from dataclasses import dataclass, field

from poraque.ml.config import split_enable_block, _yaml_literal  # noqa: F401


#: ``symbolic.physics`` settings: the weight of each constraint on the
#: **symbolic search**. See :attr:`SymbolicConfig.physics` for why two of these
#: names collide with :data:`PHYSICS_INFORMED_KEYS` and must not be merged.
SYMBOLIC_PHYSICS_KEYS = (
    "positivity_weight",
    "thomas_fermi_weight",
    "von_weizsacker_weight",
    "p_infinity",
)



@dataclass
class SymbolicConfig:
    r"""
    Symbolic distillation: fit a closed-form expression to what was learned.

    A Fourier Neural Operator is accurate and opaque. Symbolic regression
    searches the space of short algebraic expressions for one that reproduces
    the same mapping, trading a little accuracy for something that can be read,
    published and checked against known physics.

    .. warning::
       The features are **semi-local** — :math:`\rho` and its derivatives at a
       point — so the search can only recover a semi-local functional. The
       non-local part of what the operator learned is outside the hypothesis
       space by construction and will show up as irreducible residual, not as a
       failure of the search. A poor fit is therefore evidence *about the
       physics*, not only about the run.

    Attributes
    ----------
    enable : bool
        Run the search after training. Off by default: it is minutes to hours
        of CPU on top of the fit, and it needs an extra dependency.

        Spelled ``enable`` since 26.9.8, matching ``fine_tuning.enable`` and
        the ``enable`` inside every other optional block. It was
        ``enable_symbolic_distillation``, which restated its own section in its
        own name — ``symbolic.enable_symbolic_distillation`` — and was the
        longest key in the schema, wide enough to set the configuration
        table's column width in the PDF report.
    target : str
        ``"model"`` distils the trained operator's own predictions — what the
        network learned, faithful to it including its errors. ``"reference"``
        fits the DFT data directly, which is plain symbolic regression against
        ground truth and answers a different question.
    features : str
        The **input variables**. ``template`` is the independent second knob
        and selects what is fitted *against* them.

        ``"gga"`` gives the density together with its dimensionless reduced
        derivatives — the reduced gradient
        :math:`p = |\nabla\rho|/(2k_F\rho)` and reduced Laplacian
        :math:`q = \nabla^2\rho/(4k_F^2\rho)`, with
        :math:`k_F = (3\pi^2\rho)^{1/3}`. These are the GGA and meta-GGA
        variables a semi-local kinetic functional is written in.

        ``"reduced"`` gives :math:`(p, q)` alone. With ``template: pauli``
        this is the form the literature uses, in which Thomas-Fermi is
        :math:`F = 1` and von Weizsäcker is :math:`F = 0`, and every constant
        to be found is order unity. :math:`\rho` is dropped on purpose:
        :math:`p` and :math:`q` are invariant under the coordinate scaling
        that fixes :math:`T_s`, so a dimensionless :math:`F` cannot depend on
        the density, and offering it only gives the search a way to fit the
        particular densities in the dataset.

        ``"raw"`` gives :math:`(\rho, |\nabla\rho|, \nabla^2\rho)`:
        dimensional, kept for checking the reduced forms against something
        unprocessed.

        ``"enhancement"`` is a kept alias for ``"reduced"`` with
        ``template: pauli``, and **overrides** whatever ``template`` is set
        beside it. Prefer the explicit pair in a new config.
    template : str
        Factorization of the target before the search sees it.

        ``"pauli"`` fits the **Pauli enhancement factor**

        .. math::

            F = \frac{\tau - \tau_{\rm vW}}{\tau_{\rm TF}},
            \qquad
            \tau = \tau_{\rm vW} + \tau_{\rm TF}\,F,

        which is the well-posed target of orbital-free DFT:
        :math:`\tau_{\rm vW} = |\nabla\rho|^2/8\rho` is known in closed form,
        and :math:`\tau - \tau_{\rm vW} \ge 0` by Hoffmann-Ostenhof. Leaving
        :math:`\tau_{\rm vW}` in the target means fitting a quantity that is
        mostly already known, and near the von Weizsäcker limit it means
        fitting a near-cancellation between two large numbers.

        ``"thomas_fermi"`` fits the plain ratio :math:`F = \tau/\tau_{\rm TF}`.
        ``"none"`` fits the target directly.

        A template *gives away* the part of the physics that is already known,
        so the search works on the part that is not — and every constant it
        must find becomes order unity.
    epsilon : float
        Vacuum threshold in atomic units. Denominators are clamped at it and
        voxels at or below it are dropped, because :math:`p` and :math:`q` in
        vacuum are ratios of two vanishing numbers — noise with a plausible
        magnitude, which corrupts a fit more quietly than a gap would.
    constraints : dict
        Per-operator limits on argument complexity, passed to the engine.
        ``{"^": [-1, 1]}`` by default: the base is unconstrained (``-1``) and
        the **exponent** is held to complexity 1.

        Unconstrained exponents are the main source of nonsense in a power
        operator — a fractional power of a negative quantity leaves the reals,
        and an exponent that is itself a subtree is unreadable and almost never
        physical. Real functionals have simple exponents: ``5/3``, ``4/3``,
        ``2``.
    physics : dict
        Physics constraints on the **symbolic search**, and nothing else.

        .. important::

           This is not ``training.physics_informed``. That block
           constrains the *neural
           operator* — charge conservation, positivity of a predicted density,
           the Euler-Lagrange residual — and its terms are added to a training
           loss over voxels. This block constrains a *candidate algebraic
           expression* for the Pauli enhancement factor, and its terms are
           added to a symbolic-regression fitness over probe points.

           They are separate objectives on separate objects, evaluated at
           different times by different engines. Two of the names collide
           (``positivity_weight``, ``von_weizsacker_weight``) and mean
           different things in each, which is exactly why they live in
           separate blocks rather than sharing a prefix.

        ``enable``
            Penalize violations **inside** the evolutionary loop rather than
            filtering the front afterwards. On by default.

        Filtering after a run only measures how few candidates were physical;
        by then the populations have already spent their budget converging on
        forms that must be discarded. With this on, the engine's objective
        becomes

        .. math::

            \mathcal{L} = \mathcal{L}_{\rm data}
            + w_{+}\,\frac1n\sum_i \min(F_i, 0)^2
            + \sum_{\ell} w_\ell\,\frac{|F(\mathbf x_\ell) - t_\ell|}{s_\ell},

        with the second term evaluated on the batch and the third on synthetic
        probe points — :math:`(p, q) = (0, 0)` for Thomas-Fermi and
        :math:`(p_\infty, 0)` for von Weizsäcker.

        Which terms are active depends on what the run can express, and the
        log says which:

        - **positivity** always. :math:`\tau \ge 0`, and
          :math:`\tau - \tau_{\rm vW} \ge 0` by Hoffmann-Ostenhof, so a
          negative prediction is unphysical under every template.
        - **both limits** under ``template: pauli`` or
          ``template: thomas_fermi``, provided :math:`p` and :math:`q` are
          among the variables.
        - **neither limit** under ``template: none``, where the fitted
          quantity is :math:`\tau` rather than an enhancement factor, or under
          ``features: raw``, where there is no :math:`p` to take a limit in.
          They are still checked after the search, as they always were.

        .. note::
           The ``loss`` column of the reported front is then this
           **constrained** objective and is not comparable with an
           unconstrained run's. The :math:`R^2` and relative :math:`L^2` are
           computed separately from the expression itself and are unaffected.

        ``positivity_weight``, ``thomas_fermi_weight``, ``von_weizsacker_weight``
            Penalty weights, four orders of magnitude above a converged data
            term by default. These are constraints, not regularizers: the
            intent is that no accuracy gain can buy a violation. Each limit
            carries its weight once, however many probe points express it.

        ``p_infinity``
            The reduced gradient standing in for :math:`p \to \infty` in the
            von Weizsäcker probe. Large enough that no smooth interpolating
            form is still in its crossover region, and small enough that
            :math:`p^{8/3}` stays far inside the range of the 32-bit float the
            engine searches in.
    data_loss : {"mse", "mae"}
        The unpenalized part of that objective — the data term, not a
        constraint, which is why it stays here rather than under ``physics``.
        ``mae`` is the more robust choice on a density whose tails span orders
        of magnitude.
    unary_operations, binary_operations : list of str
        The operator alphabet handed to the engine. Keep it small: the search
        space grows combinatorially, and an operator that cannot appear in the
        answer only dilutes the population. ``/`` and ``log`` are the usual
        sources of singularities on a density that approaches zero.
    iterations : int
        Search iterations. The single biggest quality/time knob.
    population_size, populations : int
        Individuals per population, and how many run in parallel.
    max_depth, max_size : int
        Ceilings on expression depth and node count — the parsimony that keeps
        the result readable rather than a fitted polynomial in disguise.
    parsimony : float
        Penalty per node in the fitness. Higher gives shorter, worse-fitting
        expressions.
    n_samples : int
        Voxels sampled across the dataset. A single 32³ structure is already
        32 768 points and the search cost is linear in them, so the default
        trades a negligible amount of statistics for a tractable run.
    seed : int
        Seeds the sampling always, and the search only under
        ``deterministic``.
    deterministic : bool
        Make the search reproducible. The engine requires **serial**
        evaluation for this, so it costs the parallelism across populations —
        roughly a factor of ``populations`` in wall-clock.

        Off by default because a single search is one sample of a stochastic
        process either way, and reading one expression as *the* answer is the
        mistake this setting can otherwise encourage. Turn it on to reproduce
        a specific result, not to make the result more trustworthy.

        With it off, ``seed`` is deliberately **not** passed to the engine: a
        seed that cannot deliver reproducibility is a promise the run does not
        keep, and the engine warns about exactly that.
    """

    enable: bool = False
    target: str = "model"
    features: str = "gga"
    template: str = "none"
    epsilon: float = 1e-8
    constraints: dict = field(default_factory=lambda: {"^": [-1, 1]})
    data_loss: str = "mse"
    physics: dict = field(default_factory=lambda: {
        "enable": True,
        "positivity_weight": 1.0e2,
        "thomas_fermi_weight": 1.0e2,
        "von_weizsacker_weight": 1.0e2,
        "p_infinity": 1.0e6,
    })
    unary_operations: list = field(
        default_factory=lambda: ["exp", "log", "sqrt", "abs"])
    binary_operations: list = field(
        default_factory=lambda: ["+", "-", "*", "/", "^"])
    iterations: int = 40
    population_size: int = 33
    populations: int = 15
    max_depth: int = 10
    max_size: int = 30
    parsimony: float = 0.0032
    n_samples: int = 4000
    seed: int = 0
    deterministic: bool = False


