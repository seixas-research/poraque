# -*- coding: utf-8 -*-
# file: config.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
YAML configuration for neural-operator training runs.

A training run is defined by four groups of settings — where the data is, how
the operator is shaped, how it is optimised, and where results go — and
:class:`TrainingConfig` mirrors that structure exactly. Keeping the run
definition in a file rather than in a shell history makes an experiment
reproducible and diffable: the config is written next to the results, so a run
can always be repeated verbatim.

Precedence, highest first:

1. explicit command-line flags,
2. values in the YAML file,
3. the defaults declared here.

That ordering is what allows a single committed config to be swept over one
parameter from the shell (``--epochs 500``) without editing or copying it.

Unknown keys are rejected rather than ignored: a typo like ``learn_rate`` would
otherwise be silently dropped and the run would quietly use the default,
producing results that do not match the file that appears to describe them.
"""

from dataclasses import asdict, dataclass, field, fields

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the environment
    yaml = None


def _require_yaml():
    if yaml is None:  # pragma: no cover
        raise ImportError(
            "Reading or writing YAML configs requires PyYAML: pip install pyyaml"
        )


@dataclass
class TaskConfig:
    """
    What is trained, and what the run's artefacts are called.

    Attributes
    ----------
    type : str
        ``"ext2chg"``, ``"chg2tau"`` or ``"all"``.
    name : str
        Identifier for **this run's outputs**, and the stem of every file it
        writes::

            models/<name>.pfno              the weights
            reports/<name>_report.pdf       the PDF report
            results/plots/<name>/           the figures

        Two runs that differ in anything worth keeping — a chemical space, a
        resolution, a set of physics weights — should differ in ``name``, or
        the second silently overwrites the first. The default reproduces the
        historical filename, so a config written before this key existed
        writes exactly where it always did.

    Notes
    -----
    ``task`` also accepts the bare string it used to be::

        task: ext2chg                       # same as {type: ext2chg}

    which is read as ``type`` with the default ``name``.
    """

    type: str = "all"
    # Matches poraque.ml.training.BUNDLE_FILENAME without its suffix, so an
    # unnamed run writes the file every earlier version wrote.
    name: str = "poraque_models"

    def names(self):
        """
        The task names to train, expanding ``"all"``.

        Returns
        -------
        list of str
        """
        return (["ext2chg", "chg2tau"] if self.type == "all"
                else [str(self.type)])


@dataclass
class DataConfig:
    """
    Where the fields live and how they are prepared.

    Attributes
    ----------
    train_paths : list of str or None
        The dataset, as a **list of directories**, which may mix layouts::

            data:
              train_paths:
                - data/vasp             # local DFT runs
                - data/MP/chgcar        # a Materials Project download

        Each entry is auto-detected (see ``source``) and everything found is
        pooled into one training set. ``null`` falls back to ``root``, so a
        config written before this key existed keeps working unchanged.

        .. warning::

           Mixing a calculation archive with a bulk density archive mixes two
           *definitions* of the external potential — the tabulated local
           pseudopotential and the Gaussian pseudo-ion model, which differ by
           roughly 0.1 relative :math:`L_2`. The run warns when it happens. It
           is a legitimate trade (far more data, a fuzzier input) but never a
           good accident.
    root : str
        Single dataset directory, used when ``train_paths`` is ``null``.
    source : str
        Layout of each path, or ``"auto"`` (the default) to detect it.

        ``"vasp"`` is the classic layout: one calculation directory per
        material, selected by ``pattern``, each holding the inputs and
        volumetric outputs of a run. The external potential is computed from
        those inputs; ``TAUCAR`` is used where a run wrote one and simply not
        offered where it did not.

        ``"bulk"`` is an archive of standalone ``CHGCAR`` files, compressed or
        not — what the Materials Project ships. The potential is built from the
        structure each density carries in its own header, exactly when
        ``potcar_dir`` supplies the pseudopotentials and by the Gaussian
        pseudo-ion model otherwise. ``chg2tau`` is not trainable on such an
        archive: no public archive publishes :math:`\\tau`.

        ``"prepared"`` is a directory of per-material ``EXTCAR``/``CHGCAR``/
        ``TAUCAR`` folders — a cache from an earlier run, read as it stands.

        A single name applies to every path; pass a list to set them
        individually.
    cache : str
        Where spectrally downsampled copies are written.
    pattern : str
        Prefix identifying calculation folders inside a ``vasp`` path. Ignored
        by the other layouts.
    code : str
        DFT code name, or ``"auto"`` to detect it.
    potcar_dir : str or None
        A ``POTCAR`` **library** — one subdirectory per pseudopotential, as
        VASP ships them (``<potcar_dir>/Ag/POTCAR``, optionally ``.gz``/``.Z``;
        a flat ``POTCAR.Ag`` layout is also recognised).

        It matters most for a Materials Project download, which publishes a
        structure and a density and no pseudopotentials. Supply the library and
        the external potential is built from VASP's **tabulated local
        potential**, reproducing a reference ``EXTCAR`` to a relative
        :math:`2\\times10^{-5}`; omit it and the **Gaussian pseudo-ion model**
        stands in, whose residual against that reference is of order
        :math:`0.1` relative :math:`L_2`.

        The same setting also rescues a local run whose ``POTCAR`` was stripped
        for licensing. A run that has its own ``POTCAR`` ignores this entirely.

        Choose the library that generated the data. The Materials Project uses
        the VASP PBE set (``PAW_PBE``); pointing at ``PAW_LDA`` would build a
        potential the densities were never computed from, which is worse than
        the Gaussian fallback because it looks exact.

        Missing or unreadable entries fall back to the Gaussian model **with a
        warning** rather than failing the run: it is a quality difference, not
        an outage.
    sigma : float or None
        Gaussian pseudo-ion width in Å, used only where the Gaussian model is
        reached — i.e. where no ``POTCAR`` and no ``potcar_dir`` supply a
        tabulated potential. ``null`` derives it per species from the
        pseudopotential core radius where one is known, and falls back to
        :data:`poraque.fields.external.DEFAULT_SIGMA` otherwise. Set it
        explicitly when a reference ``EXTCAR`` allows it to be fitted.
    resolution : int
        Longest grid axis after spectral downsampling. The reduction is a
        Fourier truncation, exact for band-limited plane-wave fields.
    gaussian_blur : float or None
        Width in Å of a Gaussian blur applied to the computed external
        potential. ``null`` disables it.
    blur_method : str
        ``"spectral"`` (exact, isotropic on any cell) or ``"ndimage"``
        (:func:`scipy.ndimage.gaussian_filter` with ``mode="wrap"``, which
        blurs along grid axes and is therefore anisotropic in Cartesian space
        unless the cell is orthogonal).
    spin : str or bool
        Whether the densities are spin-polarised (``ISPIN = 2``), in which case
        a ``CHGCAR`` carries two channels — the total density and the
        magnetisation density — and the operator is built with two channels to
        match. ``"auto"`` (the default) reads the answer off the data, since a
        spin-polarised ``CHGCAR`` has a second grid block and a collinear one
        does not. Set ``true`` or ``false`` to require one, turning a
        mislabelled dataset into an error rather than a silent reinterpretation.
    """

    train_paths: list = None
    root: str = "data/vasp"
    source: str = "auto"
    cache: str = "data/cache"
    pattern: str = "struct"
    code: str = "auto"
    resolution: int = 32
    potcar_dir: str = None
    sigma: float = None
    gaussian_blur: float = None
    blur_method: str = "spectral"
    spin: str = "auto"

    def paths(self):
        """
        The directories to train on, however they were specified.

        ``train_paths`` wins when it is set; ``root`` is the single-path
        fallback that keeps every config written before it existed working.

        Returns
        -------
        list of str
        """
        if not self.train_paths:
            return [self.root]
        if isinstance(self.train_paths, str):
            return [self.train_paths]
        return [str(path) for path in self.train_paths]

    def formats(self):
        """
        The layout of each path: one name, one per path, or ``"auto"``.

        Returns
        -------
        str or list of str
            Handed straight to :func:`~poraque.data.sources.resolve_source`.
        """
        if isinstance(self.source, (list, tuple)):
            return [str(name) for name in self.source]
        return str(self.source)


@dataclass
class ModelConfig:
    """
    Architecture of the Fourier Neural Operator.

    Attributes
    ----------
    width : int
        Channel width of the Fourier layers.
    modes : int
        Retained Fourier modes per axis. This is a *capacity* limit: on a grid
        too coarse to supply them, fewer are used automatically.
    n_layers : int
        Number of Fourier layers.
    projection_channels : int
        Hidden width of the output projection.
    activation : str
        ``gelu``, ``relu``, ``silu`` or ``tanh``.
    use_coordinates : bool
        Append fractional-coordinate channels.
    cell_conditioning : bool
        Condition the layers on lattice descriptors (FiLM).
    embedding_dim : int
        Width of the cell embedding.
    mode_selection : str
        ``"fixed"`` truncates at a constant mode index; ``"physical"``
        truncates at a constant wavevector ``g_max``, so every material
        contributes the same band of physics. Prefer the latter when cell sizes
        vary widely.
    g_max : float or None
        Cutoff wavevector in Å⁻¹, required by ``mode_selection: physical``.
    pauli_residual : bool
        For ``chg2tau``, predict ``tau = tau_vW[rho] + s*softplus(f)`` so the
        Hoffmann-Ostenhof bound holds by construction.
    pauli_scale : float or None
        Initial Pauli-term scale in eV/Å³; ``null`` fits it from the training
        split.
    learn_pauli_scale : bool
        Optimise the scale alongside the backbone.
    """

    width: int = 16
    modes: int = 8
    n_layers: int = 4
    projection_channels: int = 48
    activation: str = "gelu"
    use_coordinates: bool = True
    cell_conditioning: bool = True
    embedding_dim: int = 32
    mode_selection: str = "fixed"
    g_max: float = None
    pauli_residual: bool = False
    pauli_scale: float = None
    learn_pauli_scale: bool = True


@dataclass
class TrainingConfig_:
    """
    Optimisation settings.

    Attributes
    ----------
    epochs : int
        Passes over the training set.
    batch_size : int
        Maximum samples per batch *within one grid-shape bucket*; materials of
        different shape are never mixed.
    learning_rate, weight_decay : float
        AdamW hyper-parameters.
    scheduler : str or None
        ``"cosine"`` or ``null``.
    grad_clip : float
        Global gradient-norm clip; ``0`` disables.
    valid_fraction : float
        Fraction of structures held out for validation, drawn by shuffling the
        structure list with ``seed``. Defaults to ``0.2``, so an ordinary run
        reports a genuine held-out score rather than a training fit, and
        ``early_stopping`` has something to watch.

        Set it to ``0`` to train on **every** structure. That is the right
        choice for the final deployable artefact — it uses all the data — but
        its metrics are then a training fit and carry no generalisation claim,
        and early stopping cannot act.

        The split is at the **structure level**, like ``k_folds``: whole
        materials move together. At least one structure is always kept on each
        side, so a non-zero fraction on a two-structure dataset gives 1 + 1
        rather than an empty validation set.

        Ignored when ``enable_kfold`` is set, which supplies its own splits.
    eval_epoch : int
        Evaluate and log every this many epochs. Validation is *only* computed
        on those epochs, so raising it on a large validation set is a genuine
        speed-up rather than only a quieter log.
    early_stopping : int
        Stop after this many epochs without an improvement in the **validation**
        error, and restore the best weights seen. ``0`` disables it and always
        runs the full ``epochs``.

        Requires a validation split (``valid_fraction > 0``): with
        nothing held out there is only the training loss, which falls
        monotonically by construction and so can never signal that training
        should stop. The run says so rather than appearing to be protected.

        Counted in epochs, but only *checked* on the epochs where validation is
        computed, so a patience shorter than ``eval_epoch`` behaves like
        ``eval_epoch``.
    enable_kfold : bool
        Run K-fold cross-validation instead of a single fit. This is the only
        variation on the training protocol: every other run trains once on the
        ``valid_fraction`` split.

        It answers a different question — whether the architecture generalises
        — and produces *K* models rather than one to deploy.
    k_folds : int
        Number of folds. Splitting is at the **structure level**: each fold
        holds out whole materials, never a subset of voxels from a material
        that also appears in training. A voxel-level split would let the model
        see the same material in both halves and would report a
        wildly optimistic score that says nothing about transfer to a new
        material.

        Capped at the number of structures; ``k_folds`` equal to that count is
        leave-one-out.
    seed : int
        Seed for the data pipeline: the validation draw, the fold partition and
        the batch order. Also seeds the weight initialisation unless
        ``init_seed`` overrides it.
    init_seed : int or None
        Seed for the **weight initialisation only**, leaving everything else on
        ``seed``. ``null`` means "use ``seed``".

        Separating the two is what makes a *query-by-committee* ensemble
        interpretable: train N models that share ``seed`` and differ only in
        ``init_seed``, and their disagreement isolates the spread of
        optimisation outcomes instead of confounding it with a reshuffled
        dataset.
    device : str
        ``"auto"`` (CUDA, then Apple MPS, then CPU), or an explicit backend.
    loss : str
        ``"relative_l2"`` or ``"sobolev"``.
    sobolev_weight : float
        Weight of the gradient term when ``loss: sobolev``.
    physics : dict
        Weights of the physics-informed terms; all default to zero, so the
        objective is the supervised baseline until one is enabled deliberately.

        ``electron_count_weight`` is the **charge-conservation** weight, and
        the one to reach for first on ``ext2chg``:

        .. math::

            \\mathcal{L}_{N} = \\left\\langle
            \\left(\\frac{\\int\\hat\\rho\\,d^3r - \\int\\rho\\,d^3r}
                        {\\int\\rho\\,d^3r}\\right)^{2}\\right\\rangle ,

        the squared relative error between the integral of the predicted
        density and the integral of the reference — the valence electron
        count. It costs one reduction per batch and pins the single global
        degree of freedom a pointwise regression loss controls worst, since a
        relative :math:`L^2` is indifferent to a percent of charge spread
        thinly across the cell while a total energy is not.

        ``0.1`` puts the constraint an order of magnitude below the data term,
        which is the intended balance: the physics guides, the data decides.
    """

    epochs: int = 300
    batch_size: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    valid_fraction: float = 0.2
    enable_kfold: bool = False
    k_folds: int = 5
    eval_epoch: int = 10
    early_stopping: int = 100
    seed: int = 0
    init_seed: int = None
    device: str = "auto"
    loss: str = "relative_l2"
    sobolev_weight: float = 0.1
    physics: dict = field(default_factory=lambda: {
        "electron_count_weight": 0.0,
        "positivity_weight": 0.0,
        "von_weizsacker_weight": 0.0,
        "euler_lagrange_weight": 0.0,
    })


@dataclass
class OutputConfig:
    """
    Where artefacts are written.

    Attributes
    ----------
    log : str
        Plain-text training log.
    json : str
        Machine-readable metrics.
    checkpoint_dir : str or None
        Directory for model weights; ``null`` disables checkpointing.
    plot_dir : str or None
        Directory for the figures produced by
        :class:`~poraque.vis.TrainingReport`; ``null`` disables plotting.
    report_dir : str or None
        Directory for the automatically generated PDF report; ``null``
        disables it.
    plot_format : str
        Image format for saved figures.
    dpi : int
        Raster resolution for saved figures.
    """

    log: str = "logs/fno_training.log"
    json: str = "logs/fno_training.json"
    checkpoint_dir: str = "models"
    plot_dir: str = "results/plots"
    report_dir: str = "reports"
    plot_format: str = "png"
    dpi: int = 160


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
    enable_symbolic_distillation : bool
        Run the search after training. Off by default: it is minutes to hours
        of CPU on top of the fit, and it needs an extra dependency.
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
        Factorisation of the target before the search sees it.

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
    physics_constraints : bool
        Penalise violations of the physics **inside** the evolutionary loop,
        rather than filtering the front afterwards. On by default.

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
    data_loss : {"mse", "mae"}
        The unpenalised part of that objective. ``mae`` is the more robust
        choice on a density whose tails span orders of magnitude.
    positivity_weight, thomas_fermi_weight, von_weizsacker_weight : float
        Penalty weights, four orders of magnitude above a converged data term
        by default. These are constraints, not regularisers: the intent is that
        no accuracy gain can buy a violation. Each limit carries its weight
        once however many probe points express it.
    p_infinity : float
        The reduced gradient standing in for :math:`p \to \infty` in the von
        Weizsäcker probe. Large enough that no smooth interpolating form is
        still in its crossover region, and small enough that :math:`p^{8/3}`
        stays far inside the range of the 32-bit float the engine searches in.
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

    enable_symbolic_distillation: bool = False
    target: str = "model"
    features: str = "gga"
    template: str = "none"
    epsilon: float = 1e-8
    constraints: dict = field(default_factory=lambda: {"^": [-1, 1]})
    physics_constraints: bool = True
    data_loss: str = "mse"
    positivity_weight: float = 1.0e2
    thomas_fermi_weight: float = 1.0e2
    von_weizsacker_weight: float = 1.0e2
    p_infinity: float = 1.0e6
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


@dataclass
class FineTuningConfig:
    r"""
    Adapt a trained operator to a narrower class of materials.

    A model fitted across a broad dataset is a good *starting point* for a
    specific family, and a poor substitute for one. Fine-tuning continues that
    fit on the smaller set at a much lower learning rate, keeping what the base
    model learned about the general map while letting it specialise.

    .. important::
       The **normalizations come from the checkpoint**, not from the new
       dataset. Refitting them would rescale the network's inputs out from
       under weights that were trained against the old scale, which destroys
       most of what the pre-training bought. It also means the new data must
       fall in a comparable range: fine-tuning on a family whose densities are
       an order of magnitude away is transfer learning in name only.

    Attributes
    ----------
    enable : bool
        Start from ``pretrained_checkpoint`` instead of a fresh
        initialisation.
    pretrained_checkpoint : str
        Bundle to start from. The **architecture is read from its tensors**,
        so the ``model`` section of this config is ignored for the shape of
        the network — a remembered hyper-parameter that disagreed with the
        weights could only load mismatched tensors.
    learning_rate : float
        Optimiser step for the fine-tune, replacing ``training.learning_rate``.
        Much smaller by default: the base learning rate would walk the weights
        away from the solution being adapted before the small dataset could
        constrain them.
    freeze_lifting_layers : bool
        Hold the input lifting layer fixed and train the rest. The lifting map
        is the most general part of the network — it embeds the input field
        before any operator acts — so it is the part least in need of
        specialisation, and freezing it leaves fewer parameters for a small
        dataset to overfit.

        The projection head is deliberately *not* frozen: it decodes to
        physical units, which is exactly what differs between material
        families.
    """

    enable: bool = False
    pretrained_checkpoint: str = "models/poraque_models.pfno"
    learning_rate: float = 1e-5
    freeze_lifting_layers: bool = False


@dataclass
class TrainingConfig:
    """
    Complete definition of a training run.

    Parameters
    ----------
    task : TaskConfig
        What to train, and the name every output file is built from.
    data, model, training, output, symbolic : dataclass
        The five setting groups.

    Examples
    --------
    >>> config = TrainingConfig.from_yaml("configs/train_config.yaml")  # doctest: +SKIP
    >>> config.model.width                                              # doctest: +SKIP
    16
    """

    task: TaskConfig = field(default_factory=TaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig_ = field(default_factory=TrainingConfig_)
    output: OutputConfig = field(default_factory=OutputConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)

    # Order matters twice: it is the order the run header prints, and the order
    # a bare (undotted) override is resolved against.
    _SECTIONS = {"task": TaskConfig, "data": DataConfig, "model": ModelConfig,
                 "training": TrainingConfig_, "output": OutputConfig,
                 "symbolic": SymbolicConfig, "fine_tuning": FineTuningConfig}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _task_section(value):
        """
        Read ``task`` in either spelling.

        It began as a bare string and became a section when the run gained a
        ``name``. Both are accepted permanently rather than transitionally:
        ``task: ext2chg`` is the shorter and clearer form when the defaults for
        everything else will do, and every config written before the section
        existed keeps working unchanged.
        """
        if value is None:
            return {}
        if isinstance(value, str):
            return {"type": value}
        return dict(value)

    @classmethod
    def from_dict(cls, mapping):
        """
        Build a config from a nested mapping, validating every key.

        Parameters
        ----------
        mapping : dict
            Parsed YAML contents.

        Returns
        -------
        TrainingConfig

        Raises
        ------
        ValueError
            On an unknown top-level key or an unknown key inside a section.
            Silent acceptance would let a typo change nothing while appearing
            to change something.
        """
        mapping = dict(mapping or {})
        known = set(cls._SECTIONS)
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"Unknown configuration section(s): {unknown}. "
                f"Expected any of {sorted(known)}."
            )
        mapping["task"] = cls._task_section(mapping.get("task"))

        sections = {}
        for name, section_class in cls._SECTIONS.items():
            values = dict(mapping.get(name) or {})
            valid = {f.name for f in fields(section_class)}
            bad = sorted(set(values) - valid)
            if bad:
                raise ValueError(
                    f"Unknown key(s) in section '{name}': {bad}. "
                    f"Valid keys: {sorted(valid)}."
                )
            sections[name] = section_class(**values)

        return cls(**sections)

    @classmethod
    def from_yaml(cls, path):
        """
        Read a configuration from a YAML file.

        Parameters
        ----------
        path : str or pathlib.Path

        Returns
        -------
        TrainingConfig
        """
        _require_yaml()
        with open(path, "r") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self):
        """Nested plain-``dict`` representation."""
        return asdict(self)

    def to_yaml(self, path=None):
        """
        Serialise to YAML.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            When given, write to this file. The resolved config is worth
            saving beside the results: it records what actually ran, including
            command-line overrides.

        Returns
        -------
        str
            The YAML text.
        """
        _require_yaml()
        text = yaml.safe_dump(self.to_dict(), sort_keys=False,
                              default_flow_style=False)
        if path is not None:
            with open(path, "w") as handle:
                handle.write(text)
        return text

    # ------------------------------------------------------------------ #
    # Overrides
    # ------------------------------------------------------------------ #
    def apply_overrides(self, overrides):
        """
        Apply command-line overrides in place.

        Only non-``None`` values are applied, which is how "flag was not given"
        is distinguished from "flag was given a falsy value" — important for
        booleans and zeros.

        Parameters
        ----------
        overrides : dict
            ``{"section.key": value}`` or ``{"key": value}`` for the top level.
            Bare keys are resolved against the sections, first match wins.

        Returns
        -------
        TrainingConfig
            ``self``, for chaining.
        """
        for key, value in (overrides or {}).items():
            if value is None:
                continue
            if "." in key:
                section_name, attribute = key.split(".", 1)
                section = getattr(self, section_name)
                if not hasattr(section, attribute):
                    raise ValueError(f"Unknown override {key!r}.")
                setattr(section, attribute, value)
            elif key == "task":
                # `--task ext2chg`, and any caller still passing the bare
                # string the key used to hold.
                self.task.type = value
            else:
                for section_name in self._SECTIONS:
                    section = getattr(self, section_name)
                    if hasattr(section, key):
                        setattr(section, key, value)
                        break
                else:
                    raise ValueError(f"Unknown override {key!r}.")
        return self

    # ------------------------------------------------------------------ #
    # Consumers
    # ------------------------------------------------------------------ #
    def model_kwargs(self):
        """
        Keyword arguments for :class:`~poraque.ml.fno.FNO3d`.

        The Pauli-head settings are excluded: they configure
        :class:`~poraque.ml.training.FieldOperator`, not the backbone.
        """
        excluded = {"pauli_residual", "pauli_scale", "learn_pauli_scale"}
        return {f.name: getattr(self.model, f.name)
                for f in fields(self.model) if f.name not in excluded}

    #: How a value is written back out in :meth:`describe`. The header echoes a
    #: YAML file, so it uses YAML's spellings: a reader comparing the two
    #: should not have to translate ``None`` into ``null`` in their head.
    _LITERALS = {None: "null", True: "true", False: "false"}

    @classmethod
    def _describe_value(cls, value):
        """One setting, spelled as the YAML file spells it."""
        if isinstance(value, bool) or value is None:
            return cls._LITERALS[value]
        if isinstance(value, (list, tuple)):
            return "[]" if not value else ", ".join(str(item) for item in value)
        return str(value)

    @classmethod
    def _describe_section(cls, mapping, indent):
        """
        One ``key = value`` per line, with the ``=`` signs aligned.

        Recursive, because ``training.physics`` is itself a mapping and a
        nested dict rendered as a repr is exactly the unreadable run this
        avoids.
        """
        pad = " " * indent
        scalars = [key for key, value in mapping.items()
                   if not (isinstance(value, dict) and value)]
        width = max([len(str(key)) for key in scalars] or [0])

        lines = []
        for key, value in mapping.items():
            if isinstance(value, dict) and value:
                lines.append(f"{pad}{key}:")
                lines.extend(cls._describe_section(value, indent + 2))
            else:
                lines.append(f"{pad}{str(key):<{width}s} = "
                             f"{cls._describe_value(value)}")
        return lines

    def describe(self):
        """
        Multi-line human-readable summary, for the log header.

        **One setting per line**, indented under its section, with the ``=``
        signs aligned within each. The run header is where a reader checks
        what is switched on before committing hours of GPU to it, and a
        comma-separated section wrapped across the terminal at whatever column
        it happened to reach: the settings that mattered were wherever the
        wrapping put them, and ``training.physics`` was 116 characters of
        ``{'electron_count_weight': 0.0, ...}`` riding on the end of one.

        Values are written in YAML's spelling — ``null``, ``true``, ``false``
        — so the header and the file it came from read the same.

        Returns
        -------
        str

        Examples
        --------
        >>> print(TrainingConfig().describe())            # doctest: +SKIP
        task:
          type = all
          name = poraque_models
        data:
          train_paths = null
          root        = data/vasp
          ...
        """
        lines = []
        for name in self._SECTIONS:
            section = getattr(self, name)
            lines.append(f"{name}:")
            lines.extend(self._describe_section(
                {entry.name: getattr(section, entry.name)
                 for entry in fields(section)}, indent=2))
        return "\n".join(lines)


#: Written by ``poraque-train --write-config``.
SAMPLE_CONFIG_HEADER = """\
# =====================================================================
# Poraque - Fourier Neural Operator training configuration
#
#   poraque-train --config configs/train_config.yaml
#
# Command-line flags override these values, so one committed config can
# be swept from the shell without editing it:
#
#   poraque-train --config configs/train_config.yaml --epochs 500
#
# Equivalent, from a checkout with nothing installed:
#
#   python scripts/poraque_train.py --config configs/train_config.yaml
# =====================================================================
"""
