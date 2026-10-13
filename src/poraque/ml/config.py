# -*- coding: utf-8 -*-
# file: config.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
YAML configuration for neural-operator training runs.

A run is described by eight sections, each about one thing:

.. code-block:: yaml

    run:          # what the run is called, where it writes, its seed
    data:         # where the materials are, and how they are cached
    ext2chg:      # the density operator: V_ext -> pseudo-density + occupancies
    chg2tau:      # the kinetic operator: pseudo-density + occupancies -> tau
    training:     # the optimization, shared by both operators
    fine_tuning:  # start from a trained checkpoint (off unless present)
    runtime:      # the hardware: device, processes, distribution
    report:       # figures and the PDF

**An operator is trained when its section is present.** ``ext2chg`` and
``chg2tau`` are the two tasks a user names; each carries its own ``model``,
because the two maps need different networks and one shared block forced
the larger one into a corner of the other (``model.paw.width`` beside
``model.width``). Turning a task's ``occupancies`` off trains the grid alone
-- internally the ``ext2pseudo`` or ``pseudo2tau`` task -- so the four
:class:`~poraque.ml.tasks.TaskSpec` names survive as the vocabulary of the
code and the checkpoint, while a configuration speaks of two.

**A feature that can be switched off is a block whose presence is the
switch.** ``false`` turns it off, ``true`` or an empty key turns it on with
its defaults, a mapping turns it on with those settings; leaving the key out
takes the default, which the comment beside each one in
``configs/train_complete_and_commented.yaml`` names. There is no ``enable``
key anywhere, so no block can be switched off while still listing settings
that read as if they applied.

Precedence, highest first: command-line flags, the YAML file, the defaults
declared here. Unknown keys raise rather than being ignored: a typo would
otherwise leave the default in force under a file that appears to say
otherwise. Keys and sections of the previous schema raise too, and name
where their setting went.
"""

from dataclasses import dataclass, field, fields

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the environment
    yaml = None


def _require_yaml():
    if yaml is None:
        raise ImportError("Reading or writing YAML configs requires PyYAML: "
                          "pip install pyyaml")


#: Suffix of a trained checkpoint.
BUNDLE_SUFFIX = ".poraque"

#: Assumed when ``data.path`` is unset: this project's own Pt cells. It names
#: the *structures* subdirectory rather than ``data/vasp``, because a material
#: is one subdirectory of the path and no deeper, and the isolated atom beside
#: it is the reference for the baseline, never a training sample.
DEFAULT_DATA_PATH = "data/vasp/structures"


# ---------------------------------------------------------------------- #
# The previous schema, refused by name
# ---------------------------------------------------------------------- #
#: Top-level sections that no longer exist, and what replaced them.
RETIRED_SECTIONS = {
    "task": "the task is chosen by which operator sections are present: "
            "write an `ext2chg:` and/or a `chg2tau:` section; the run's "
            "name is `run.name`",
    "model": "every operator carries its own network: `ext2chg.model` and "
             "`chg2tau.model`",
    "output": "`run.output` (the parent directory, null writes nothing) and "
              "the `report` section (figures, pdf, format, dpi, raw_data)",
    "symbolic": "symbolic distillation was archived to "
                "experiments/kan_symbolic on 2026-10-13",
}

#: Keys that moved or went, keyed by ``(section, key)`` because the same word
#: can be gone from one section and live in another.
RETIRED_KEYS = {
    ("data", "data_paths"): "data.path (one directory of materials)",
    ("data", "cache"): "data.cache.dir",
    ("data", "resolution"): "data.grid.resolution",
    ("data", "max_spacing"): "data.grid.max_spacing",
    ("data", "storage"): "data.cache.storage (text | hdf5)",
    ("data", "compression"): "data.cache.compression",
    ("data", "compression_level"): "nothing -- gzip runs at level 4",
    ("data", "cache_in_memory"): "data.cache.in_memory",
    ("data", "cache_workers"): "runtime.workers",
    ("data", "atomic_reference"): "data.atoms",
    ("data", "delta_density"): "ext2chg.target (delta | full)",
    ("data", "target_scale"): "nothing -- the asinh scale is fitted",
    ("data", "pattern"): "nothing -- every subdirectory is a material",
    ("data", "format"): "nothing -- the files are read as VASP's",
    ("data", "precision"): "nothing -- fields are held in float64",
    ("data", "xc"): "nothing -- training never read it",
    ("data", "sigma"): "nothing -- the Gaussian width is derived per species",
    ("data", "gaussian_blur"): "nothing",
    ("data", "blur_method"): "nothing",
    ("data", "strict_potcar"): "nothing -- what the library served is "
                               "recorded in the cache",
    ("data", "paw_source"): "nothing -- the isolated atom's records are used",
    ("training", "learning_rate"): "training.optimizer.learning_rate",
    ("training", "weight_decay"): "training.optimizer.weight_decay",
    ("training", "scheduler"): "nothing -- the schedule is cosine",
    ("training", "grad_clip"): "nothing -- the gradient norm is clipped at 1",
    ("training", "valid_fraction"): "training.validation.fraction",
    ("training", "eval_epoch"): "training.validation.every",
    ("training", "early_stopping"): "training.validation.patience",
    ("training", "enable_kfold"): "training.validation.kfold (a number of "
                                  "folds, or null)",
    ("training", "k_folds"): "training.validation.kfold",
    ("training", "seed"): "run.seed",
    ("training", "init_seed"): "nothing -- run.seed seeds the weights too",
    ("training", "device"): "runtime.device",
    ("training", "strict_device"): "runtime.strict_device",
    ("training", "distributed"): "runtime.distributed",
    ("training", "distributed_timeout"): "nothing -- 30 minutes",
    ("training", "num_workers"): "runtime.loader_workers",
    ("training", "pin_memory"): "nothing -- pinned on CUDA, not elsewhere",
    ("training", "tf32"): "runtime.tf32",
    ("training", "sobolev_weight"): "training.loss.sobolev_weight",
    ("training", "spin_loss"): "training.spin.mode",
    ("training", "magnetization_weight"): "training.spin.magnetization_weight",
    ("training", "magnetization_floor"): "training.spin.magnetization_floor",
    ("training", "physics_informed"): "training.physics (the weights alone; "
                                      "all zero switches it off)",
    ("training", "compile"): "nothing -- torch.compile was measured on the "
                             "FNO backbone and removed",
    ("fine_tuning", "enable"): "the section's presence (fine_tuning: false "
                               "turns it off)",
    ("fine_tuning", "pretrained_checkpoint"): "fine_tuning.checkpoint",
    ("fine_tuning", "freeze_lifting_layers"): "fine_tuning.freeze_lifting",
    ("fine_tuning", "use_lora"): "fine_tuning.lora (a block)",
    ("fine_tuning", "lora_rank"): "fine_tuning.lora.rank",
    ("fine_tuning", "lora_alpha"): "fine_tuning.lora.alpha",
    ("fine_tuning", "lora_dropout"): "fine_tuning.lora.dropout",
}
for _task in ("ext2chg", "chg2tau"):
    RETIRED_KEYS.update({
        (f"{_task}.model", "n_layers"): f"{_task}.model.layers",
        (f"{_task}.model", "mode_selection"):
            f"{_task}.model.band (null keeps `modes` per axis)",
        (f"{_task}.model", "g_max"): f"{_task}.model.band",
        (f"{_task}.model", "embedding_dim"):
            f"{_task}.model.cell_conditioning (the width, or false)",
        (f"{_task}.model", "use_coordinates"):
            "nothing -- coordinates are appended exactly when the model is "
            "not equivariant",
        (f"{_task}.model", "projection_activation"):
            "nothing -- the read-out follows `activation`",
        (f"{_task}.model", "kan_setup"):
            "nothing -- the KAN activations were archived to "
            "experiments/kan_symbolic; use silu, gelu, relu or tanh",
        (f"{_task}.model.equivariant", "spherical_cutoff"):
            "nothing -- the retained modes are always a ball",
    })


def retired_replacement(section, key):
    """What to write instead of a retired key, or ``None`` if merely unknown."""
    return RETIRED_KEYS.get((section, key))


# ---------------------------------------------------------------------- #
# The machinery: sections, blocks, parsing and dumping
# ---------------------------------------------------------------------- #
def section(cls, **defaults):
    """A sub-section that is always there, with these defaults."""
    return field(default_factory=lambda: cls(**defaults),
                 metadata={"section": cls, "defaults": defaults})


def block(cls, on=False, **defaults):
    """A switchable block: ``None`` when off, an instance of ``cls`` when on."""
    return field(default_factory=(lambda: cls(**defaults)) if on
                 else (lambda: None),
                 metadata={"block": cls, "defaults": defaults})


def _yaml_literal(value):
    return {None: "null", True: "true", False: "false"}.get(value, repr(value))


def _parse(cls, mapping, path, defaults=None):
    """Build ``cls`` from a mapping, recursing into sections and blocks."""
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, dict):
        raise ValueError(f"{path} must be a block of settings, not "
                         f"{_yaml_literal(mapping)}.")
    known = {entry.name: entry for entry in fields(cls)}
    unknown = sorted(set(mapping) - set(known))
    if unknown:
        moved = [f"'{key}' -> {retired_replacement(path, key)}"
                 for key in unknown if retired_replacement(path, key)]
        raise ValueError(
            f"Unknown key(s) in {path}: {unknown}. "
            + (f"Replaced: {'; '.join(moved)}. " if moved else "")
            + f"Valid keys: {sorted(known)}.")
    values = dict(defaults or {})
    for key, value in mapping.items():
        values[key] = _value(known[key], value, f"{path}.{key}")
    return cls(**values)


def _value(entry, value, path):
    """One field's value, parsed according to what the field is."""
    meta = entry.metadata
    if "section" in meta:
        return _parse(meta["section"], value, path, meta["defaults"])
    if "block" in meta:
        if value is False:
            return None
        if value is True or value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be false, true or a block of "
                             f"settings, not {_yaml_literal(value)}.")
        return _parse(meta["block"], value, path, meta["defaults"])
    return value


def _dump(instance):
    """The mapping :func:`_parse` reads back to an equal instance."""
    out = {}
    for entry in fields(instance):
        value = getattr(instance, entry.name)
        if "section" in entry.metadata:
            out[entry.name] = _dump(value)
        elif "block" in entry.metadata:
            out[entry.name] = False if value is None else _dump(value)
        elif isinstance(value, tuple):
            out[entry.name] = list(value)
        else:
            out[entry.name] = value
    return out


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value, minimum=1):
    return isinstance(value, int) and not isinstance(value, bool) \
        and value >= minimum


def _choice(value, allowed, path):
    _check(value in allowed, f"{path} is {_yaml_literal(value)}; expected one "
                             f"of {list(allowed)}.")


# ---------------------------------------------------------------------- #
# run, data
# ---------------------------------------------------------------------- #
@dataclass
class RunConfig:
    """
    What the run is called, where it writes, and its seed.

    Everything the run writes goes in ``<output>/<name>/``: the checkpoint
    ``<name>.poraque``, ``log/``, ``plots/`` and ``report/``. ``output: null``
    writes nothing at all. ``seed`` draws the validation split, the folds,
    the batch order and the initial weights.
    """

    name: str = "poraque_models"
    output: str = "models"
    seed: int = 42

    def validate(self, path="run"):
        _check(isinstance(self.name, str) and self.name,
               f"{path}.name must be a non-empty string.")
        _check(_is_count(self.seed, 0), f"{path}.seed must be a non-negative "
                                        f"integer.")


@dataclass
class GridConfig:
    """
    The grid every field is cached on.

    ``resolution`` caps the longest axis after spectral downsampling;
    ``max_spacing`` (Å) refines any axis left coarser than it, never past the
    native grid -- for long thin cells, whose short axes would otherwise be
    cached at a fraction of the resolution of the long one.
    """

    resolution: int = 32
    max_spacing: float = None


@dataclass
class CacheConfig:
    """
    Where the downsampled fields are written, and how.

    ``storage: text`` writes a ``CHGCAR``-format file per field, ``hdf5`` one
    chunked ``fields.h5`` per material (``compression: gzip | lzf`` there
    only). ``in_memory`` keeps decoded fields and normalized samples in RAM
    between epochs; ``auto`` does so under a 4 GiB estimate.
    """

    dir: str = "data/cache"
    storage: str = "text"
    compression: str = None
    in_memory: object = "auto"


@dataclass
class DataConfig:
    """
    The dataset: one directory, one subdirectory per material.

    A material is a directory with a density in it; inputs beside it (POSCAR,
    POTCAR, ...) make the external potential exact, a ``TAUCAR`` makes it a
    ``chg2tau`` sample. ``potcar_dir`` is the library used where the material
    ships no POTCAR; ``atoms`` the isolated-atom reference (the δ-density
    baseline and the PAW records); ``spin`` whether the magnetization is kept.
    """

    path: str = DEFAULT_DATA_PATH
    potcar_dir: str = None
    atoms: str = "data/vasp/isolated_atoms"
    spin: object = "auto"
    grid: GridConfig = section(GridConfig)
    cache: CacheConfig = section(CacheConfig)

    #: Settings of the cache builder that are no longer choices.
    FIXED = {"pattern": "", "format": "auto", "sigma": None,
             "gaussian_blur": None, "blur_method": "spectral",
             "strict_potcar": False, "precision": "float64",
             "paw_source": "atomic", "compression_level": 4}

    def paths(self):
        """``[path]``: the cache builder takes a list."""
        return [str(self.path or DEFAULT_DATA_PATH)]

    @property
    def storage(self):
        """The cache builder's name for the layout (``files`` or ``hdf5``)."""
        return "files" if self.cache.storage == "text" else self.cache.storage

    def validate(self, path="data"):
        _check(isinstance(self.path, str) and self.path,
               f"{path}.path must be one directory.")
        _choice(self.spin, ("auto", True, False), f"{path}.spin")
        _check(_is_count(self.grid.resolution, 4),
               f"{path}.grid.resolution must be an integer of at least 4.")
        _check(self.grid.max_spacing is None
               or (_is_number(self.grid.max_spacing)
                   and self.grid.max_spacing > 0),
               f"{path}.grid.max_spacing must be null or a positive length "
               f"in Ang.")
        _choice(self.cache.storage, ("text", "hdf5"), f"{path}.cache.storage")
        _choice(self.cache.compression, (None, "gzip", "lzf"),
                f"{path}.cache.compression")
        _check(self.cache.compression is None or self.cache.storage == "hdf5",
               f"{path}.cache.compression applies to storage: hdf5 only; a "
               f"text cache is not compressed.")
        _choice(self.cache.in_memory, ("auto", True, False),
                f"{path}.cache.in_memory")


# ---------------------------------------------------------------------- #
# The operators
# ---------------------------------------------------------------------- #
#: Nonlinearities a model may use (``poraque.ml.fno.ACTIVATIONS``).
ACTIVATION_NAMES = ("silu", "gelu", "relu", "tanh")


@dataclass
class EquivariantConfig:
    """
    The radial spectral layer: a kernel that depends on :math:`|G|` alone.

    ``n_radial`` basis functions spanning ``g_basis`` (1/Å; ``auto`` reads the
    median band off the training split). On, the operator commutes with every
    rotation and inversion; the retained modes are a ball and no coordinate
    channels are appended, both because the property fails otherwise.
    """

    n_radial: int = 16
    g_basis: object = "auto"


@dataclass
class ModelConfig:
    r"""
    One operator's Fourier neural operator.

    ``width`` channels through ``layers`` Fourier blocks, each keeping up to
    ``modes`` modes per axis. ``band`` (1/Å) keeps the same physical band in
    every cell instead, still capped at ``modes``; ``null`` keeps ``modes``
    per axis whatever the cell. ``projection_channels`` is the read-out's
    hidden width, ``activation`` the nonlinearity of every layer and of the
    read-out. ``cell_conditioning`` is the width of the lattice embedding
    every layer is modulated by (FiLM), or ``false``. ``precision`` is what
    the operator computes in.

    The three that set the size
    ---------------------------
    ``width``, ``modes`` and ``projection_channels`` are the settings that
    decide how big the operator is and what it can represent. They are easy to
    confuse because all three are "some number of channels", so it is worth
    being precise about what each one counts.

    Written out with every index, and matching
    :meth:`~poraque.ml.fno.FNO3d.forward` rather than the textbook schematic:

    **Lift** — the only place :math:`C_{\rm in}` appears. The ``+3`` are the
    fractional-coordinate channels, present when the model is not
    ``equivariant``:

    .. math::

        v^{(0)}_o(\mathbf r) = \sum_{i=1}^{C_{\rm in}+3}
            W^{\rm lift}_{oi}\, x_i(\mathbf r) + b^{\rm lift}_o ,
        \qquad o = 1 \dots C

    **Layer** :math:`\ell = 1 \dots L`:

    .. math::

        \hat v^{(\ell-1)}_i(\mathbf k)
            &= \mathcal{F}_{\mathbf r \to \mathbf k}
               \big[ v^{(\ell-1)}_i \big](\mathbf k) \\[2pt]
        \hat z^{(\ell)}_o(\mathbf k)
            &= \begin{cases}
               \displaystyle\sum_{i=1}^{C}
                 R^{(\ell,\,c(\mathbf k))}_{oi,\,k_1k_2k_3}\,
                 \hat v^{(\ell-1)}_i(\mathbf k)
                 & |k_1| < m_1,\ |k_2| < m_2,\ 0 \le k_3 < m_3 \\[4pt]
               0 & \text{otherwise}
               \end{cases} \\[2pt]
        z^{(\ell)}_o(\mathbf r)
            &= \mathcal{F}^{-1}_{\mathbf k \to \mathbf r}
               \big[ \hat z^{(\ell)}_o \big](\mathbf r) \\[2pt]
        y^{(\ell)}_o(\mathbf r)
            &= \mathrm{GroupNorm}\Big( z^{(\ell)}_o(\mathbf r)
               + \sum_{i=1}^{C} W^{(\ell)}_{oi}\, v^{(\ell-1)}_i(\mathbf r)
               + b^{(\ell)}_o \Big) \\[2pt]
        y^{(\ell)}_o(\mathbf r)
            &\leftarrow \gamma_o(\text{cell})\, y^{(\ell)}_o(\mathbf r)
               + \beta_o(\text{cell})
               \qquad\text{(FiLM, unless \tt cell\_conditioning: false)} \\[2pt]
        v^{(\ell)}_o(\mathbf r)
            &= v^{(\ell-1)}_o(\mathbf r)
               + \sigma\big( y^{(\ell)}_o(\mathbf r) \big)
               \qquad\text{(residual)}

    **Read-out** — the only place :math:`P` appears:

    .. math::

        u_p(\mathbf r) &= \sigma\Big(
            \sum_{o=1}^{C} P^{(1)}_{po}\, v^{(L)}_o(\mathbf r)
            + b^{(1)}_p \Big),
            \qquad p = 1 \dots P \\
        \text{out}_q(\mathbf r) &=
            \sum_{p=1}^{P} P^{(2)}_{qp}\, u_p(\mathbf r) + b^{(2)}_q,
            \qquad q = 1 \dots C_{\rm out}

    The retained band is capped by the grid, not fixed by it:

    .. math::

        m_1 = \min(\texttt{modes}, N_x/2), \quad
        m_2 = \min(\texttt{modes}, N_y/2), \quad
        m_3 = \min(\texttt{modes}, N_z/2 + 1)

    There are **four** corners :math:`c` and not eight because :math:`v` is
    real, so :math:`\hat v(-\mathbf k) = \hat v(\mathbf k)^{*}` and only
    :math:`k_3 \ge 0` is stored: :math:`2\,(\pm k_1) \times 2\,(\pm k_2)
    \times 1`. That 4 is the leading axis of :math:`R`, whose shape is
    ``(4, C, C, m1, m2, m3)``.

    The parameter count is where the three settings separate:

    .. code-block:: text

        spectral   L * 4 * C^2 * m1*m2*m3   complex   <- dominates: C^2 and m^3
        pointwise  L * (C^2 + C)
        lift       C * (C_in + 3) + C
        read-out   C*P + P + P*C_out + C_out          <- linear in P, and once

    ``width`` (:math:`C`)
        How many **channels** carry the field through those layers. The input
        is lifted from 1-2 physical channels to :math:`C` at the start and
        projected back at the end; everything between works at width :math:`C`.

        This is the operator's *representational* capacity — how many distinct
        features of the density it can carry at once — and it is the setting
        that costs the most, because the spectral weights go as :math:`C^2`.
        Doubling it roughly quadruples both parameters and time.

    ``modes`` (:math:`m`)
        How many **Fourier coefficients per axis** the spectral multiplier
        acts on, counted from the lowest. The rest of the spectrum passes
        through the layer untouched by :math:`R` (it is still carried by
        :math:`W`).

        This is the operator's *spatial* reach: mode :math:`m` on a cell of
        length :math:`L` is a wavelength :math:`L/m`, so :math:`m` fixes the
        finest feature the learned convolution can resolve and, equivalently,
        the longest range it can correlate. It is a **capacity limit, not a
        requirement**: on a grid too coarse to supply :math:`m` modes, fewer
        are used automatically and the same weights still apply — which is
        what lets one model serve materials on different grids.

        The parameter count is :math:`4 C^2 m^3` complex numbers per layer, so
        ``modes`` is by far the most expensive knob: 8 → 12 is a 3.4× increase.

    ``projection_channels`` (:math:`P`)
        The hidden width of the **final two-layer head** that maps the
        :math:`C` channels back down to the physical output. It appears once,
        at the end, and never inside a Fourier layer.

        Purely pointwise, so it costs :math:`O(P)` and no spectral weights at
        all. It exists because collapsing :math:`C` channels to one with a
        single linear map is a bottleneck; a wider head is cheap and usually
        worth more than the same budget spent on ``width``.

    A useful way to hold them apart: ``width`` is how much information travels
    *through* the operator, ``modes`` is how far it travels *in space*, and
    ``projection_channels`` is only how it is *read out* at the end.

    Typical ranges are ``width`` 16-64, ``modes`` 8-16, ``projection_channels``
    64-128; ``modes`` above about 16 is rarely worth its cost, since the high
    coefficients of a smooth field carry little of its norm.
    """

    width: int = 16
    layers: int = 3
    modes: int = 8
    band: float = None
    projection_channels: int = 64
    activation: str = "silu"
    cell_conditioning: object = 32
    equivariant: EquivariantConfig = block(EquivariantConfig)
    precision: str = "float32"

    # What the backbone constructor calls these.
    @property
    def n_layers(self):
        return self.layers

    @property
    def mode_selection(self):
        return "fixed" if self.band is None else "physical"

    @property
    def g_max(self):
        return self.band

    @property
    def equivariant_enabled(self):
        return self.equivariant is not None

    def equivariant_kwargs(self):
        """``n_radial`` and ``g_basis`` for the radial layer; ``{}`` when off."""
        if self.equivariant is None:
            return {}
        return {"n_radial": self.equivariant.n_radial,
                "g_basis": self.equivariant.g_basis,
                "spherical_cutoff": True}

    def fno_kwargs(self):
        """Keyword arguments for :class:`~poraque.ml.fno.FNO3d`."""
        conditioning = self.cell_conditioning
        return {
            "width": self.width, "modes": self.modes,
            "n_layers": self.layers,
            "projection_channels": self.projection_channels,
            "activation": self.activation,
            "use_coordinates": self.equivariant is None,
            "cell_conditioning": conditioning is not False,
            "embedding_dim": 32 if conditioning is False else conditioning,
            "mode_selection": self.mode_selection, "g_max": self.band,
            "equivariant": self.equivariant is not None,
            **self.equivariant_kwargs(),
        }

    def validate(self, path):
        for key in ("width", "layers", "modes", "projection_channels"):
            _check(_is_count(getattr(self, key)),
                   f"{path}.{key} is {getattr(self, key)!r}; expected a "
                   f"positive integer.")
        _check(self.band is None or (_is_number(self.band) and self.band > 0),
               f"{path}.band is {self.band!r}; expected null or a positive "
               f"band in 1/Ang.")
        _choice(self.activation, ACTIVATION_NAMES, f"{path}.activation")
        _check(self.cell_conditioning is False
               or _is_count(self.cell_conditioning),
               f"{path}.cell_conditioning is {self.cell_conditioning!r}; "
               f"expected the embedding width or false.")
        _choice(self.precision, ("float32", "float64"), f"{path}.precision")
        if self.equivariant is not None:
            _check(_is_count(self.equivariant.n_radial),
                   f"{path}.equivariant.n_radial must be a positive integer.")
            basis = self.equivariant.g_basis
            _check(basis == "auto" or (_is_number(basis) and basis > 0),
                   f"{path}.equivariant.g_basis is {basis!r}; expected a "
                   f"positive radius in 1/Ang, or 'auto'.")


@dataclass
class StructureInputConfig:
    """
    Channels describing the structure, appended after the potential.

    ``superposition`` adds ρ_sup, the isolated atoms superposed; ``elements``
    adds one superposition per element (``auto``: every element of the
    cache, or a list, or ``[]``). Built from ``data.atoms``.
    """

    superposition: bool = True
    elements: object = "auto"


@dataclass
class OccupancyHeadConfig:
    """
    ``ext2chg``'s per-atom head: each atom's PAW augmentation occupancies.

    ``weight`` multiplies the occupancy term against the field objective, in
    training and in the choice of best epoch alike; ``magnetization_weight``
    weights the magnetization's record set within it. ``readout_band`` (1/Å)
    is what the head reads around each atom, ``auto`` 95 % of the coarsest
    grid's Nyquist; ``neighbor_min`` (Å) the first of the ten radial centers
    the neighbor expansion uses (0.6 also covers short CHON bonds).
    """

    weight: float = 1.0
    magnetization_weight: float = 1.0
    readout_band: object = "auto"
    neighbor_min: float = 2.2


@dataclass
class Ext2chgConfig:
    """
    The density operator, V_ext → pseudo-density + PAW occupancies.

    ``target: delta`` learns ρ − ρ_sup (the atoms superposed) and adds the
    superposition back on output; ``full`` learns ρ itself.
    ``occupancies: false`` trains the grid alone.
    """

    target: str = "delta"
    structure_input: StructureInputConfig = block(StructureInputConfig)
    model: ModelConfig = section(ModelConfig, width=64, layers=4, modes=16)
    occupancies: OccupancyHeadConfig = block(OccupancyHeadConfig, on=True)

    @property
    def task_name(self):
        """The internal task: ``ext2chg``, or ``ext2pseudo`` for the grid."""
        return "ext2pseudo" if self.occupancies is None else "ext2chg"

    @property
    def delta_density(self):
        return self.target == "delta"

    def structure_input_settings(self, elements=None):
        """``{superposition, elements}`` with ``auto`` resolved, or ``None``."""
        if self.structure_input is None:
            return None
        chosen = self.structure_input.elements
        if chosen == "auto":
            if elements is None:
                raise ValueError("ext2chg.structure_input.elements: auto "
                                 "needs the training data's elements to "
                                 "resolve.")
            chosen = list(elements)
        elif chosen is None:
            chosen = []
        return {"superposition": self.structure_input.superposition,
                "elements": [str(element) for element in chosen]}

    def paw_settings(self, readout_cache_mib=1024):
        """The occupancy head's settings, as the operator builder wants them."""
        head = self.occupancies or OccupancyHeadConfig()
        return {"width": self.model.width, "modes": self.model.modes,
                "n_layers": self.model.layers,
                "readout_g_max": head.readout_band,
                "occupancy_weight": head.weight,
                "neighbor_min": head.neighbor_min,
                "magnetization_weight": head.magnetization_weight,
                "readout_cache_mib": readout_cache_mib}

    def validate(self, path="ext2chg"):
        _choice(self.target, ("delta", "full"), f"{path}.target")
        self.model.validate(f"{path}.model")
        if self.structure_input is not None:
            _check(self.target == "delta",
                   f"{path}.structure_input is built from the isolated atoms, "
                   f"which only target: delta loads.")
            _check(isinstance(self.structure_input.superposition, bool),
                   f"{path}.structure_input.superposition must be true or "
                   f"false.")
            chosen = self.structure_input.elements
            _check(chosen in ("auto", None) or (
                isinstance(chosen, (list, tuple))
                and all(isinstance(item, str) for item in chosen)),
                f"{path}.structure_input.elements is {chosen!r}; expected "
                f"'auto' or a list of element symbols.")
            _check(self.structure_input.superposition or chosen not in
                   (None, [], ()),
                   f"{path}.structure_input has superposition false and no "
                   f"elements: that is no channel at all.")
        head = self.occupancies
        if head is not None:
            band = head.readout_band
            _check(band == "auto" or (_is_number(band) and band > 0),
                   f"{path}.occupancies.readout_band is {band!r}; expected "
                   f"'auto' or a positive band in 1/Ang.")
            for key in ("weight", "magnetization_weight"):
                value = getattr(head, key)
                _check(_is_number(value) and value >= 0,
                       f"{path}.occupancies.{key} is {value!r}; expected a "
                       f"non-negative number.")
            _check(_is_number(head.neighbor_min)
                   and 0 <= head.neighbor_min < 5.8,
                   f"{path}.occupancies.neighbor_min is "
                   f"{head.neighbor_min!r}; expected a length from 0 "
                   f"(inclusive) to 5.8 Ang (exclusive).")


@dataclass
class OccupancyInputConfig:
    """
    ``chg2tau``'s per-atom input: each atom's occupancies written as fields.

    ``channels`` is how many field channels the records are written into, by
    one learned map per element and angular momentum.
    """

    channels: int = 8


@dataclass
class Chg2tauConfig:
    """
    The kinetic operator, pseudo-density + PAW occupancies → τ.

    ``head: pauli`` predicts τ = τ_vW[ρ] + s·softplus(f), which makes
    τ ≥ τ_vW structural, with the scale ``s`` fitted on the training split
    and learned; ``plain`` predicts τ directly. ``occupancies: false`` trains
    on the grid alone.
    """

    head: str = "plain"
    model: ModelConfig = section(ModelConfig)
    occupancies: OccupancyInputConfig = block(OccupancyInputConfig, on=True)

    @property
    def task_name(self):
        """The internal task: ``chg2tau``, or ``pseudo2tau`` for the grid."""
        return "pseudo2tau" if self.occupancies is None else "chg2tau"

    @property
    def pauli_residual(self):
        return self.head == "pauli"

    @property
    def occupancy_channels(self):
        return (self.occupancies or OccupancyInputConfig()).channels

    def validate(self, path="chg2tau"):
        _choice(self.head, ("plain", "pauli"), f"{path}.head")
        self.model.validate(f"{path}.model")
        if self.occupancies is not None:
            _check(_is_count(self.occupancies.channels),
                   f"{path}.occupancies.channels must be a positive integer.")
            _check(self.head == "plain",
                   f"{path}.head: pauli is implemented for the grid alone; "
                   f"set occupancies: false, or head: plain.")


# ---------------------------------------------------------------------- #
# training, fine_tuning, runtime, report
# ---------------------------------------------------------------------- #
@dataclass
class OptimizerConfig:
    """AdamW (or Adam, SGD) on a cosine schedule, gradient norm clipped at 1."""

    name: str = "adamw"
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4


@dataclass
class LossConfig:
    """
    The objective: an L² or H¹ norm, absolute or relative per sample.

    ``sobolev_weight`` weights the gradient term of the two H¹ norms.
    """

    norm: str = "relative_l2"
    sobolev_weight: float = 0.1


@dataclass
class SpinLossConfig:
    """
    How the two channels of a spin target are combined.

    ``joint`` divides both by the norm of the whole (ρ, m) target;
    ``per_channel`` divides each by its own, floored at
    ``magnetization_floor``·‖ρ‖, and weights m by ``magnetization_weight``.
    The two weights are read under ``per_channel`` only.
    """

    mode: str = "joint"
    magnetization_weight: float = 1.0
    magnetization_floor: float = 0.01


@dataclass
class ValidationConfig:
    """
    The held-out data and what is done with it.

    ``fraction`` of the structures is held out (0 trains on everything);
    ``every`` epochs it is evaluated; ``patience`` evaluations' worth of
    epochs without improvement stops the run and restores the best weights
    (0 never stops). ``kfold: K`` cross-validates over K folds instead.
    """

    fraction: float = 0.2
    every: int = 10
    patience: int = 300
    kfold: int = None


@dataclass
class PhysicsConfig:
    r"""
    Physics penalties on the predicted fields, weighted against the data.

    ``electron_count``: ∫ρ = N; ``positivity``: ρ, τ ≥ 0;
    ``von_weizsacker``: :math:`\tau \ge |\nabla\rho|^2/8\rho`. All zero
    switches the block off.
    """

    electron_count: float = 0.0
    positivity: float = 0.0
    von_weizsacker: float = 0.0


@dataclass
class OptimizationConfig:
    """How both operators are trained."""

    epochs: int = 500
    batch_size: int = 4
    samples_per_step: int = 0
    optimizer: OptimizerConfig = section(OptimizerConfig)
    loss: LossConfig = section(LossConfig)
    spin: SpinLossConfig = section(SpinLossConfig)
    validation: ValidationConfig = section(ValidationConfig)
    physics: PhysicsConfig = section(PhysicsConfig)

    #: No longer choices.
    SCHEDULER = "cosine"
    GRAD_CLIP = 1.0

    @property
    def kfold_enabled(self):
        return self.validation.kfold is not None

    def physics_weights(self):
        """The four weights :class:`~poraque.ml.losses.PhysicsInformedLoss` takes."""
        return {"electron_count_weight": float(self.physics.electron_count),
                "positivity_weight": float(self.physics.positivity),
                "von_weizsacker_weight": float(self.physics.von_weizsacker),
                "euler_lagrange_weight": 0.0}

    @property
    def physics_enabled(self):
        return any(self.physics_weights().values())

    def validate(self, path="training"):
        for key in ("epochs", "batch_size"):
            _check(_is_count(getattr(self, key)),
                   f"{path}.{key} must be a positive integer.")
        _check(_is_count(self.samples_per_step, 0),
               f"{path}.samples_per_step must be a non-negative integer.")
        _choice(self.optimizer.name, ("adamw", "adam", "sgd"),
                f"{path}.optimizer.name")
        _check(_is_number(self.optimizer.learning_rate)
               and self.optimizer.learning_rate > 0,
               f"{path}.optimizer.learning_rate must be positive.")
        _choice(self.loss.norm, ("absolute_l2", "relative_l2", "absolute_h1",
                                 "relative_h1"), f"{path}.loss.norm")
        _choice(self.spin.mode, ("joint", "per_channel"), f"{path}.spin.mode")
        _check(_is_number(self.validation.fraction)
               and 0 <= self.validation.fraction < 1,
               f"{path}.validation.fraction must be in [0, 1).")
        _check(_is_count(self.validation.every),
               f"{path}.validation.every must be a positive integer.")
        _check(_is_count(self.validation.patience, 0),
               f"{path}.validation.patience must be a non-negative integer.")
        _check(self.validation.kfold is None
               or _is_count(self.validation.kfold, 2),
               f"{path}.validation.kfold is {self.validation.kfold!r}; "
               f"expected null or a number of folds of at least 2.")
        for key in ("electron_count", "positivity", "von_weizsacker"):
            value = getattr(self.physics, key)
            _check(_is_number(value) and value >= 0,
                   f"{path}.physics.{key} must be a non-negative weight.")


@dataclass
class LoRAConfig:
    """Rank-``rank`` adapters on the dense lifting and projection layers."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


@dataclass
class FineTuningConfig:
    """
    Start from a trained checkpoint instead of a fresh initialization.

    Normalizations come from ``checkpoint`` and are never refitted.
    ``freeze_lifting`` holds the input embedding fixed; ``lora`` freezes every
    weight and learns low-rank adapters instead.
    """

    checkpoint: str = "models/poraque_models/poraque_models" + BUNDLE_SUFFIX
    learning_rate: float = 1e-5
    freeze_lifting: bool = False
    lora: LoRAConfig = block(LoRAConfig)

    # What the fine-tuning code calls these.
    @property
    def pretrained_checkpoint(self):
        return self.checkpoint

    @property
    def use_lora(self):
        return self.lora is not None


@dataclass
class RuntimeConfig:
    """
    The hardware, which changes how fast a run is and never what it computes.

    ``device``: auto (CUDA, then MPS, then CPU) or a backend;
    ``strict_device`` refuses a fallback. ``distributed: auto`` forms a DDP
    group when Slurm or torchrun describes one. ``workers``: processes that
    build the cache (``auto``, a count, or ``0``); ``loader_workers``: DataLoader processes.
    ``tf32``: TensorFloat-32 on Ampere and later. ``readout_cache_mib``:
    device memory for ``ext2chg``'s readout kernels.
    """

    device: str = "auto"
    strict_device: bool = False
    distributed: str = "auto"
    workers: object = "auto"
    loader_workers: int = 0
    tf32: bool = True
    readout_cache_mib: int = 1024

    #: No longer choices.
    PIN_MEMORY = "auto"
    DISTRIBUTED_TIMEOUT = 30.0

    def validate(self, path="runtime"):
        _choice(self.distributed, ("auto", "off"), f"{path}.distributed")
        _check(_is_count(self.loader_workers, 0),
               f"{path}.loader_workers must be a non-negative integer.")
        _check(_is_number(self.readout_cache_mib)
               and self.readout_cache_mib >= 0,
               f"{path}.readout_cache_mib must be a non-negative number.")


@dataclass
class ReportConfig:
    """
    Figures and the PDF, under ``<run.output>/<run.name>/``.

    ``raw_data`` writes the numbers behind every figure beside it.
    """

    figures: bool = True
    pdf: bool = True
    format: str = "png"
    dpi: int = 200
    raw_data: bool = False


# ---------------------------------------------------------------------- #
# The run
# ---------------------------------------------------------------------- #
@dataclass
class TrainingConfig:
    """
    Complete definition of a training run.

    Examples
    --------
    >>> config = TrainingConfig.from_yaml("configs/train_simple.yaml")  # doctest: +SKIP
    >>> config.task_names()                                             # doctest: +SKIP
    ['ext2chg', 'chg2tau']
    """

    run: RunConfig = section(RunConfig)
    data: DataConfig = section(DataConfig)
    ext2chg: Ext2chgConfig = block(Ext2chgConfig)
    chg2tau: Chg2tauConfig = block(Chg2tauConfig)
    training: OptimizationConfig = section(OptimizationConfig)
    fine_tuning: FineTuningConfig = block(FineTuningConfig)
    runtime: RuntimeConfig = section(RuntimeConfig)
    report: ReportConfig = section(ReportConfig)

    #: The order the sections print in.
    SECTIONS = ("run", "data", "ext2chg", "chg2tau", "training",
                "fine_tuning", "runtime", "report")

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, mapping):
        """
        Build and validate a configuration from a plain mapping.

        Raises
        ------
        ValueError
            On an unknown or retired section or key, naming the replacement
            where there is one, and on any value out of range.
        """
        mapping = dict(mapping or {})
        retired = sorted(set(mapping) & set(RETIRED_SECTIONS))
        if retired:
            raise ValueError(
                "Removed configuration section(s): "
                + "; ".join(f"'{name}' -- {RETIRED_SECTIONS[name]}"
                            for name in retired) + ".")
        unknown = sorted(set(mapping) - set(cls.SECTIONS))
        if unknown:
            raise ValueError(f"Unknown configuration section(s): {unknown}. "
                             f"Expected any of {list(cls.SECTIONS)}.")
        known = {entry.name: entry for entry in fields(cls)}
        config = cls(**{name: _value(known[name], value, name)
                        for name, value in mapping.items()})
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path):
        """Read a configuration from a YAML file."""
        _require_yaml()
        with open(path, "r") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def to_dict(self):
        """Every setting, as a plain mapping :meth:`from_dict` reads back."""
        return _dump(self)

    def to_yaml(self, path=None, minimal=False):
        """
        The configuration as YAML, written to ``path`` when given.

        ``minimal`` keeps only what differs from the defaults.
        """
        _require_yaml()
        payload = self.non_default_dict() if minimal else self.to_dict()
        text = yaml.safe_dump(payload, sort_keys=False,
                              default_flow_style=False)
        if path is not None:
            with open(path, "w") as handle:
                handle.write(text)
        return text

    def non_default_dict(self):
        """The settings that differ from a default configuration's."""
        def trim(current, reference):
            if not isinstance(current, dict) or not isinstance(reference,
                                                               dict):
                return current
            out = {}
            for key, value in current.items():
                if key not in reference:
                    out[key] = value
                elif isinstance(value, dict) and isinstance(reference[key],
                                                            dict):
                    nested = trim(value, reference[key])
                    if nested:
                        out[key] = nested
                elif value != reference[key]:
                    out[key] = value
            return out

        reference = type(self)()
        # A task section that is on is compared against its own defaults,
        # not against `false`.
        for name in ("ext2chg", "chg2tau"):
            if getattr(self, name) is not None:
                setattr(reference, name, _value(
                    next(e for e in fields(self) if e.name == name), {}, name))
        current = self.to_dict()
        trimmed = trim(current, reference.to_dict())
        for name in ("ext2chg", "chg2tau"):
            if current[name] is not False and name not in trimmed:
                trimmed[name] = None
        return trimmed

    def validate(self):
        """Check every section; called by :meth:`from_dict` and the overrides."""
        self.run.validate()
        self.data.validate()
        self.training.validate()
        self.runtime.validate()
        for name in ("ext2chg", "chg2tau"):
            operator = getattr(self, name)
            if operator is not None:
                operator.validate(name)
        if self.training.kfold_enabled:
            _check(self.fine_tuning is None,
                   "training.validation.kfold trains every fold from "
                   "scratch, so it cannot be combined with fine_tuning.")
        return self

    def apply_overrides(self, overrides):
        """
        Apply ``{"dotted.path": value}`` overrides, skipping ``None`` values.

        A path walks sections and blocks (``training.optimizer.learning_rate``,
        ``ext2chg.model.width``). Setting a block to ``true``, ``false`` or a
        mapping switches it; a path into a block that is off is refused,
        rather than switching it on behind the configuration's back.

        Raises
        ------
        ValueError
            On an unknown path, a path into a block that is off, or a value
            the result does not validate with.
        """
        for key, value in (overrides or {}).items():
            if value is None:
                continue
            parts = key.split(".")
            target = self
            for depth, part in enumerate(parts[:-1]):
                known = {entry.name: entry for entry in fields(target)} \
                    if hasattr(target, "__dataclass_fields__") else {}
                if part not in known:
                    raise ValueError(f"Unknown override {key!r}.")
                child = getattr(target, part)
                if child is None:
                    raise ValueError(
                        f"Override {key!r} reaches into "
                        f"{'.'.join(parts[:depth + 1])}, which is off. "
                        f"Switch it on in the configuration first.")
                target = child
            known = {entry.name: entry for entry in fields(target)}
            last = parts[-1]
            if last not in known:
                raise ValueError(f"Unknown override {key!r}.")
            setattr(target, last, _value(known[last], value, key))
        return self.validate()

    # ------------------------------------------------------------------ #
    # The tasks
    # ------------------------------------------------------------------ #
    def task_names(self):
        """
        The internal tasks this run trains, density operator first.

        ``ext2chg`` or ``ext2pseudo`` for the ``ext2chg`` section (by its
        ``occupancies``), ``chg2tau`` or ``pseudo2tau`` for ``chg2tau``.

        Raises
        ------
        ValueError
            When neither section is present: there is nothing to train.
        """
        names = [operator.task_name for operator in (self.ext2chg,
                                                     self.chg2tau)
                 if operator is not None]
        if not names:
            raise ValueError("Nothing to train: the configuration has neither "
                             "an `ext2chg:` nor a `chg2tau:` section.")
        return names

    def operator(self, task_name):
        """The section describing internal task ``task_name``."""
        section_name = {"ext2chg": "ext2chg", "ext2pseudo": "ext2chg",
                        "chg2tau": "chg2tau", "pseudo2tau": "chg2tau"}.get(
            str(task_name))
        if section_name is None:
            raise ValueError(f"Unknown task {task_name!r}.")
        operator = getattr(self, section_name)
        if operator is None or operator.task_name != task_name:
            raise ValueError(f"This configuration does not train "
                             f"{task_name!r}.")
        return operator

    def model(self, task_name):
        """The :class:`ModelConfig` of internal task ``task_name``."""
        return self.operator(task_name).model

    def model_kwargs(self, task_name):
        """Keyword arguments for the :class:`~poraque.ml.fno.FNO3d` of a task."""
        return self.model(task_name).fno_kwargs()

    @property
    def delta_density(self):
        """Whether the density target is the residual over the atoms."""
        return self.ext2chg is None or self.ext2chg.delta_density

    # ------------------------------------------------------------------ #
    # Where things go
    # ------------------------------------------------------------------ #
    def run_name(self):
        return str(self.run.name or "poraque_models")

    def run_dir(self):
        """``<run.output>/<run.name>``, or ``None`` when nothing is written."""
        if not self.run.output:
            return None
        import os

        return os.path.join(self.run.output, self.run_name())

    def _subdir(self, name, wanted=True):
        root = self.run_dir()
        if root is None or not wanted:
            return None
        import os

        return os.path.join(root, name)

    def log_dir(self):
        return self._subdir("log")

    def plot_dir(self):
        return self._subdir("plots", self.report.figures)

    def report_dir(self):
        return self._subdir("report", self.report.pdf)

    def checkpoint_path(self):
        """``<run dir>/<name>.poraque`` (``_finetuned`` for a fine-tune)."""
        root = self.run_dir()
        if root is None:
            return None
        import os

        stem = self.run_name()
        if self.fine_tuning is not None:
            stem += "_finetuned"
        return os.path.join(root, stem + BUNDLE_SUFFIX)

    def log_path(self):
        return self._log_artifact(".log")

    def json_path(self):
        return self._log_artifact(".json")

    def _log_artifact(self, suffix):
        directory = self.log_dir()
        if directory is None:
            return None
        import os

        return os.path.join(directory, f"{self.run_name()}{suffix}")

    # ------------------------------------------------------------------ #
    # The run header
    # ------------------------------------------------------------------ #
    _LITERALS = {None: "null", True: "true", False: "false"}

    @classmethod
    def _describe_value(cls, value):
        if isinstance(value, bool) or value is None:
            return cls._LITERALS[value]
        if isinstance(value, (list, tuple)):
            return "[]" if not value else ", ".join(str(item) for item in value)
        return str(value)

    @classmethod
    def _describe_section(cls, mapping, indent):
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
        """Every setting, one per line, nested blocks indented."""
        lines = []
        for name, values in self.to_dict().items():
            if isinstance(values, dict):
                lines.append(f"{name}:")
                lines.extend(self._describe_section(values, indent=2))
            else:
                lines.append(f"{name}: {self._describe_value(values)}")
        return "\n".join(lines)
