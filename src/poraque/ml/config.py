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
class DataConfig:
    """
    Where the fields live and how they are prepared.

    Attributes
    ----------
    root : str
        Directory holding the per-material calculation folders.
    cache : str
        Where spectrally downsampled copies are written.
    pattern : str
        Prefix identifying calculation folders inside ``root``.
    code : str
        DFT code name, or ``"auto"`` to detect it.
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
    """

    root: str = "data/vasp"
    cache: str = "data/cache"
    pattern: str = "struct"
    code: str = "auto"
    resolution: int = 32
    gaussian_blur: float = None
    blur_method: str = "spectral"


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
        ``"gga"`` regresses :math:`\tau` on the density together with its
        dimensionless reduced derivatives — the reduced gradient
        :math:`p = |\nabla\rho|/(2k_F\rho)` and reduced Laplacian
        :math:`q = \nabla^2\rho/(4k_F^2\rho)`, with
        :math:`k_F = (3\pi^2\rho)^{1/3}`. These are the GGA and meta-GGA
        variables a semi-local kinetic functional is written in.

        ``"enhancement"`` drops :math:`\rho` and fits
        :math:`F = \tau/\tau_{\rm TF}` on :math:`(p, q)` alone — the form the
        literature uses, in which Thomas-Fermi is :math:`F = 1` and von
        Weizsäcker is :math:`F = 5p^2/3`, and every constant to be found is
        order unity.

        ``"raw"`` regresses :math:`\tau` on
        :math:`(\rho, |\nabla\rho|, \nabla^2\rho)`: dimensional, kept for
        checking the reduced forms against something unprocessed.
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
    task : str
        ``"ext2chg"``, ``"chg2tau"`` or ``"all"``.
    data, model, training, output, symbolic : dataclass
        The five setting groups.

    Examples
    --------
    >>> config = TrainingConfig.from_yaml("configs/train_config.yaml")  # doctest: +SKIP
    >>> config.model.width                                              # doctest: +SKIP
    16
    """

    task: str = "all"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig_ = field(default_factory=TrainingConfig_)
    output: OutputConfig = field(default_factory=OutputConfig)
    symbolic: SymbolicConfig = field(default_factory=SymbolicConfig)
    fine_tuning: FineTuningConfig = field(default_factory=FineTuningConfig)

    _SECTIONS = {"data": DataConfig, "model": ModelConfig,
                 "training": TrainingConfig_, "output": OutputConfig,
                 "symbolic": SymbolicConfig, "fine_tuning": FineTuningConfig}

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
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
        known = set(cls._SECTIONS) | {"task"}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"Unknown configuration section(s): {unknown}. "
                f"Expected any of {sorted(known)}."
            )

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

        return cls(task=mapping.get("task", "all"), **sections)

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
                self.task = value
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

    def describe(self):
        """
        Multi-line human-readable summary, for the log header.

        Nested mappings are broken out one entry per line instead of being
        inlined as a repr. ``training.physics`` is 116 characters of
        ``{'electron_count_weight': 0.0, ...}``, which wrapped across the
        terminal and buried the four weights that actually select the
        objective -- the run header is where a reader checks what is switched
        on, so those have to be readable at a glance.
        """
        lines = [f"task: {self.task}"]
        for name in self._SECTIONS:
            section = getattr(self, name)
            inline, nested = [], []
            for entry in fields(section):
                value = getattr(section, entry.name)
                if isinstance(value, dict) and value:
                    nested.append((entry.name, value))
                else:
                    inline.append(f"{entry.name}={value}")
            lines.append(f"{name}: {', '.join(inline)}")
            for key, mapping in nested:
                lines.append(f"  {key}:")
                width = max(len(str(inner)) for inner in mapping)
                for inner, setting in mapping.items():
                    lines.append(f"    {str(inner):<{width}s} = {setting}")
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
