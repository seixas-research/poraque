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
    train_fraction : float
        Share of materials used for training when ``split`` is ``"random"``.
    split : str
        ``"leave_one_out"`` (default; the honest choice for small datasets) or
        ``"random"``.
    seed : int
        Seed for the split.
    """

    root: str = "data/vasp"
    cache: str = "data/cache"
    pattern: str = "struct"
    code: str = "auto"
    resolution: int = 32
    train_fraction: float = 0.8
    split: str = "leave_one_out"
    seed: int = 0


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
    seed : int
        Seed for weight initialisation and batching.
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

    epochs: int = 200
    batch_size: int = 1
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    seed: int = 0
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
    plot_format : str
        Image format for saved figures.
    dpi : int
        Raster resolution for saved figures.
    """

    log: str = "logs/fno_training.log"
    json: str = "logs/fno_training.json"
    checkpoint_dir: str = "models"
    plot_dir: str = "results/plots"
    plot_format: str = "png"
    dpi: int = 160


@dataclass
class TrainingConfig:
    """
    Complete definition of a training run.

    Parameters
    ----------
    task : str
        ``"ext2chg"``, ``"chg2tau"`` or ``"all"``.
    data, model, training, output : dataclass
        The four setting groups.

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

    _SECTIONS = {"data": DataConfig, "model": ModelConfig,
                 "training": TrainingConfig_, "output": OutputConfig}

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
        """Multi-line human-readable summary, for the log header."""
        lines = [f"task: {self.task}"]
        for name in self._SECTIONS:
            section = getattr(self, name)
            body = ", ".join(f"{f.name}={getattr(section, f.name)}"
                             for f in fields(section))
            lines.append(f"{name}: {body}")
        return "\n".join(lines)


#: Written by ``scripts/train_fno.py --write-config``.
SAMPLE_CONFIG_HEADER = """\
# =====================================================================
# Poraque - Fourier Neural Operator training configuration
#
#   python scripts/train_fno.py --config configs/train_config.yaml
#
# Command-line flags override these values, so one committed config can
# be swept from the shell without editing it:
#
#   python scripts/train_fno.py --config configs/train_config.yaml --epochs 500
# =====================================================================
"""
