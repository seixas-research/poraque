# -*- coding: utf-8 -*-
# file: training.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Training loop and the end-to-end field-to-field operator.

:class:`FieldOperator` is the object a user actually holds: it owns the network
*and* the normalizations, so it can be handed a
:class:`~poraque.fields.ExternalPotential` and return a
:class:`~poraque.fields.ChargeDensity` on the very same grid — closing the loop
with :mod:`poraque.fields`.
"""

import os
import warnings

import numpy as np
import torch

from ..fields import ChargeDensity, ExternalPotential, KineticEnergyDensity
from .data import make_dataloader
from .device import describe_device, resolve_device
from .fno import FNO3d
from .losses import PhysicsInformedLoss, relative_error
from .tasks import resolve_task
from .transforms import FieldTransform, Identity

#: Output field class produced by each task.
_OUTPUT_CLASSES = {
    "EXTCAR": ExternalPotential,
    "CHGCAR": ChargeDensity,
    "TAUCAR": KineticEnergyDensity,
}


def clip_gradients(parameters, max_norm):
    r"""
    Global gradient-norm clipping that tolerates **complex** parameters.

    :func:`torch.nn.utils.clip_grad_norm_` computes the norm with
    ``linalg.vector_norm``, which Apple's MPS backend does not implement for
    complex dtypes — and the FNO's spectral weights are complex, so the stock
    helper raises there.

    Viewing a complex tensor as its stacked ``(real, imag)`` representation is
    **exact** rather than an approximation: for :math:`z = a + ib`,

    .. math:: \sum_k |z_k|^2 = \sum_k (a_k^2 + b_k^2),

    so the Frobenius norm is unchanged. The clip is therefore identical on
    every backend.

    Parameters
    ----------
    parameters : iterable of torch.nn.Parameter
        Parameters whose ``.grad`` should be clipped in place.
    max_norm : float
        Maximum global 2-norm; non-positive disables clipping.

    Returns
    -------
    float
        The total gradient norm *before* clipping.
    """
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or max_norm is None or max_norm <= 0:
        return 0.0

    squares = [
        (torch.view_as_real(g) if g.is_complex() else g).pow(2).sum()
        for g in gradients
    ]
    total = torch.sqrt(torch.stack(squares).sum())

    coefficient = max_norm / (total + 1e-6)
    if float(coefficient) < 1.0:
        for gradient in gradients:
            gradient.mul_(coefficient.to(gradient.device))
    return float(total)


class FieldOperator:
    """
    A trained (or trainable) map between two 3D scalar fields.

    Parameters
    ----------
    task : str or TaskSpec
        ``"ext2chg"`` or ``"chg2tau"``.
    model : torch.nn.Module, optional
        Network; defaults to a :class:`~poraque.ml.fno.FNO3d` built from
        ``**model_kwargs``.
    input_transform, target_transform : FieldTransform, optional
        Normalizations; usually taken from
        :meth:`~poraque.ml.data.FieldPairDataset.fit_transforms`.
    device : str or torch.device, optional
        ``"auto"`` (default) selects CUDA, then Apple MPS, then CPU; an
        explicit backend that is unavailable warns and falls back to CPU. See
        :func:`~poraque.ml.device.resolve_device`.
    pauli_residual : bool, optional
        Wrap the backbone in a
        :class:`~poraque.ml.heads.PauliResidualOperator`, so the model
        predicts :math:`\\tau = \\tau_{\\rm vW}[\\rho] + s\\,
        \\mathrm{softplus}(f_\\theta)` and the Hoffmann-Ostenhof bound holds by
        construction. Only meaningful for ``chg2tau``; requested for any other
        task it raises, since :math:`\\tau_{\\rm vW}` is a functional of the
        density and the density is not the input elsewhere.
    pauli_scale : float, optional
        Initial Pauli-term magnitude in eV/Å³; fit it with
        :func:`~poraque.ml.heads.fit_pauli_scale`.
    learn_pauli_scale : bool, optional
        Optimize the scale together with the backbone.
    training_resolution : int, optional
        Longest grid axis of the data this operator was trained on. Carried
        into the checkpoint so a consumer can size an evaluation grid the model
        has actually seen: an FNO is resolution-*flexible* but not
        resolution-*indifferent*, and nothing downstream can detect that it is
        being extrapolated.
    init_seed : int, optional
        Seed for the **weight initialisation only**. The global RNG state is
        saved and restored around the draw, so everything downstream — batch
        order, any stochastic layer — is left on the stream it would otherwise
        have had.

        That isolation is what makes a *query-by-committee* ensemble
        interpretable: members share a seed for the data pipeline and differ
        only in ``init_seed``, so their disagreement measures the spread of
        optimisation outcomes rather than a confounded mixture of that and a
        reshuffled dataset. ``None`` leaves initialisation on the ambient
        stream.
    **model_kwargs
        Forwarded to :class:`~poraque.ml.fno.FNO3d`.
    """

    def __init__(self, task, model=None, input_transform=None,
                 target_transform=None, device=None, pauli_residual=False,
                 pauli_scale=1.0, learn_pauli_scale=True,
                 training_resolution=None, init_seed=None, **model_kwargs):
        self.task = resolve_task(task)
        self.device = resolve_device(device)
        self.input_transform = input_transform or Identity()
        self.target_transform = target_transform or Identity()
        self.training_resolution = (None if training_resolution is None
                                    else int(training_resolution))
        self.init_seed = None if init_seed is None else int(init_seed)

        self.pauli_residual = bool(pauli_residual)
        self.pauli_scale = float(pauli_scale)
        self.learn_pauli_scale = bool(learn_pauli_scale)

        # Channel counts are part of the architecture and are recorded in
        # state(); two of them means a spin-polarised (rho, m) field.
        self.in_channels = int(model_kwargs.get("in_channels", 1))
        self.out_channels = int(model_kwargs.get("out_channels", 1))
        self.use_coordinates = bool(model_kwargs.get("use_coordinates", True))

        if model is not None:
            backbone = model
            self.in_channels = int(getattr(model, "in_channels",
                                           self.in_channels))
            self.out_channels = int(getattr(model, "out_channels",
                                            self.out_channels))
            self.use_coordinates = bool(getattr(model, "use_coordinates",
                                                self.use_coordinates))
        elif self.init_seed is None:
            backbone = FNO3d(**model_kwargs)
        else:
            # Seed the draw, then hand the global stream back untouched. A bare
            # manual_seed would also re-align every later consumer of the RNG,
            # so two committee members would differ in their batch order as
            # well as their weights -- and the disagreement would no longer
            # isolate the effect it is meant to measure.
            state = torch.random.get_rng_state()
            try:
                torch.manual_seed(self.init_seed)
                backbone = FNO3d(**model_kwargs)
            finally:
                torch.random.set_rng_state(state)

        if self.pauli_residual:
            if self.task.name != "chg2tau":
                raise ValueError(
                    f"pauli_residual is only defined for the chg2tau task "
                    f"(tau = tau_vW[rho] + ...), not {self.task.name!r}."
                )
            from .heads import PauliResidualOperator

            backbone = PauliResidualOperator(
                backbone, self.input_transform, self.target_transform,
                scale=self.pauli_scale, learn_scale=self.learn_pauli_scale,
            )

        self.model = backbone.to(self.device)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, field):
        """
        Apply the operator to a :class:`~poraque.fields.ScalarField`.

        Parameters
        ----------
        field : ScalarField
            Input field; its grid is reused for the output, so the prediction
            is defined on exactly the same mesh as the input.

        Returns
        -------
        ScalarField or SpinDensity
            An instance of the task's target class, in physical units. A
            two-channel operator returns a
            :class:`~poraque.fields.SpinDensity` carrying
            :math:`(\\rho, m)`; a one-channel operator returns the plain field.
        """
        self.model.eval()
        values = torch.as_tensor(np.ascontiguousarray(field.data),
                                 dtype=torch.float32, device=self.device)
        cell = torch.as_tensor(field.grid.cell, dtype=torch.float32,
                               device=self.device).unsqueeze(0)

        # A field is either (Nx, Ny, Nz) or, when spin-polarised, already
        # (C, Nx, Ny, Nz). Only the batch axis has to be added in the second
        # case, so the channel axis is inserted from the field's own rank
        # rather than assumed.
        normalized = self.input_transform(values)
        if normalized.ndim == 3:
            normalized = normalized.unsqueeze(0)
        normalized = normalized.unsqueeze(0)

        prediction = self.target_transform.inverse(self.model(normalized, cell))

        # .float() before .numpy(): accelerators may hand back a dtype numpy
        # cannot consume directly, and .cpu() alone does not convert it.
        channels = prediction[0].detach().to("cpu", torch.float32).numpy()
        metadata = {"predicted_by": type(self.model).__name__,
                    "task": self.task.name,
                    "device": str(self.device)}

        if channels.shape[0] == 1:
            return _OUTPUT_CLASSES[self.task.target_field](
                channels[0], field.grid, field.structure, metadata=metadata)

        if channels.shape[0] == 2 and self.task.target_field == "CHGCAR":
            from ..fields import SpinDensity

            metadata["ispin"] = 2
            return SpinDensity.from_channels(channels, field.grid,
                                             field.structure,
                                             metadata=metadata)

        raise ValueError(
            f"An operator with {channels.shape[0]} output channels has no "
            f"field representation for target {self.task.target_field!r}. "
            f"One channel is a scalar field; two on CHGCAR is a spin-polarised "
            f"(rho, m) density."
        )

    @property
    def device_description(self):
        """Human-readable description of the active device."""
        return describe_device(self.device)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def state(self):
        """
        Everything needed to rebuild this operator, as a plain ``dict``.

        The head flags are included because they change the *architecture*, not
        just the weights: restoring a Pauli-residual model into a bare backbone
        would silently load mismatched tensors.

        Returns
        -------
        dict
            Suitable for :meth:`from_state`, and the per-task value stored
            inside a bundle by :func:`save_bundle`.
        """
        return {
            "task": self.task.name,
            "model_state": self.model.state_dict(),
            "model_class": type(self.model).__name__,
            # Recorded rather than inferred: the lifting layer cannot tell
            # in_channels apart from the coordinate channels, so a spin model
            # reloaded by inference alone would silently become a one-channel
            # model with coordinates. See infer_backbone_kwargs.
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_coordinates": self.use_coordinates,
            "input_transform": self.input_transform.state_dict(),
            "target_transform": self.target_transform.state_dict(),
            "pauli_residual": self.pauli_residual,
            "pauli_scale": self.pauli_scale,
            "learn_pauli_scale": self.learn_pauli_scale,
            "training_resolution": self.training_resolution,
            "init_seed": self.init_seed,
        }

    @classmethod
    def from_state(cls, state, model=None, device=None, **model_kwargs):
        """
        Rebuild an operator from a :meth:`state` payload.

        The backbone's shape is **inferred from the stored tensors** unless
        ``model_kwargs`` overrides it — see :func:`infer_backbone_kwargs`. A
        remembered hyper-parameter that disagreed with the weights would load
        mismatched tensors or fail obscurely; the tensors cannot disagree with
        themselves.

        Parameters
        ----------
        state : dict
            As produced by :meth:`state`.
        model : torch.nn.Module, optional
            Pre-built **backbone** with matching architecture.
        device : str or torch.device, optional
        **model_kwargs
            Override individual inferred hyper-parameters.

        Returns
        -------
        FieldOperator
        """
        inferred = infer_backbone_kwargs(state["model_state"])
        # Explicitly recorded channel counts win over the inference, which
        # cannot separate in_channels from the coordinate channels.
        for key in ("in_channels", "out_channels", "use_coordinates"):
            if state.get(key) is not None:
                inferred[key] = state[key]
        inferred.update(model_kwargs)
        if model is not None:
            inferred = model_kwargs

        operator = cls(
            state["task"], model=model, device=device,
            input_transform=FieldTransform.from_state_dict(state["input_transform"]),
            target_transform=FieldTransform.from_state_dict(state["target_transform"]),
            pauli_residual=state.get("pauli_residual", False),
            pauli_scale=state.get("pauli_scale", 1.0),
            learn_pauli_scale=state.get("learn_pauli_scale", True),
            training_resolution=state.get("training_resolution"),
            init_seed=state.get("init_seed"),
            **inferred,
        )
        operator.model.load_state_dict(state["model_state"])
        return operator

    def save(self, path):
        """
        Save this single operator to ``path``.

        Used for per-fold diagnostic checkpoints. The deployable artefact is a
        *bundle* holding every task at once — see :func:`save_bundle`.
        """
        torch.save(self.state(), path)
        return str(path)

    @classmethod
    def load(cls, path, model=None, device=None, **model_kwargs):
        """
        Restore an operator saved by :meth:`save`.

        Parameters
        ----------
        path : str
            Checkpoint file.
        model : torch.nn.Module, optional
            Pre-built **backbone** with matching architecture.
        device : str or torch.device, optional
        **model_kwargs
            Override individual inferred hyper-parameters.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_state(state, model=model, device=device, **model_kwargs)


#: Identifies a bundle payload, so a stray tensor file fails with a clear
#: message instead of a ``KeyError`` three frames deep.
BUNDLE_FORMAT = "poraque-bundle-1"

#: Conventional filename for the unified checkpoint.
BUNDLE_FILENAME = "poraque_models.pfno"

#: Extensions Poraquê used to write bundles under. Retained only so a stale
#: file can be *found and named*, never to change how one is read: the format
#: is unchanged and the extension was never inspected when loading.
LEGACY_BUNDLE_SUFFIXES = (".pth", ".pt")


def resolve_bundle_path(path, log=None):
    """
    Fall back to a legacy extension when the ``.pfno`` file is absent.

    Renaming the default output from ``.pth`` to ``.pfno`` would otherwise make
    an existing trained model invisible to every default path, which reads as
    "no model" rather than "renamed". This looks beside the requested file for
    the same stem under an old extension and says what it did.

    Parameters
    ----------
    path : str
        Requested bundle.
    log : callable, optional
        Sink for the notice. Silent when omitted.

    Returns
    -------
    str
        ``path`` when it exists, the legacy file when only that does, and
        ``path`` unchanged when neither does — so the caller still reports the
        name the user actually asked for.
    """
    import os

    if os.path.exists(path):
        return path

    stem = os.path.splitext(path)[0]
    for suffix in LEGACY_BUNDLE_SUFFIXES:
        legacy = stem + suffix
        if os.path.exists(legacy):
            if log is not None:
                log(f"  NOTE: {path} does not exist, but {legacy} does — "
                    f"using it.")
                log(f"        Poraque now writes {os.path.basename(stem)}"
                    f".pfno; rename the file to silence this.")
            return legacy
    return path


#: Filename for a fine-tuned bundle. Deliberately distinct from
#: :data:`BUNDLE_FILENAME`: a fine-tune is a specialisation of the base model,
#: usually to a narrower set of materials, and writing it over the general
#: model would silently replace something broad with something narrow.
FINETUNED_BUNDLE_FILENAME = "poraque_finetuned.pfno"


def infer_backbone_kwargs(model_state):
    """
    Recover an :class:`~poraque.ml.fno.FNO3d`'s shape from its tensors.

    Storing the hyper-parameters separately would allow the record and the
    weights to disagree; the weights cannot disagree with themselves. The
    Pauli-residual head wraps the backbone under a ``backbone.`` prefix, which
    is detected here rather than having to be remembered.

    Parameters
    ----------
    model_state : dict
        A ``state_dict`` as stored by :meth:`FieldOperator.state`.

    Returns
    -------
    dict
        ``width``, ``modes``, ``n_layers``, ``projection_channels``,
        ``out_channels`` and ``use_coordinates``, where each can be determined.

    Notes
    -----
    ``in_channels`` and ``use_coordinates`` are **not** separable from the
    tensors alone: the lifting layer sees ``in_channels + 3*use_coordinates``
    channels, and 4 could be either one field with coordinates or four fields
    without. A single-channel model is assumed here, which is what every
    checkpoint written before spin support contains.
    :meth:`FieldOperator.from_state` prefers the values recorded in the state
    payload and only falls back to this.
    """
    prefix = "backbone." if any(k.startswith("backbone.") for k in model_state) else ""
    spectral = [v for k, v in model_state.items()
                if k.startswith(f"{prefix}blocks.")
                and k.endswith(".spectral.weight")]
    projection = [v for k, v in model_state.items()
                  if k.startswith(f"{prefix}project.") and k.endswith(".weight")]
    lift = model_state.get(f"{prefix}lift.weight")

    kwargs = {}
    if spectral:
        # (4, in, out, m1, m2, m3) -- four corner blocks of the rfftn.
        kwargs["width"] = int(spectral[0].shape[1])
        kwargs["modes"] = int(spectral[0].shape[3])
        kwargs["n_layers"] = len(spectral)
    if projection:
        # First conv widens to projection_channels, last narrows to the output.
        kwargs["projection_channels"] = int(projection[0].shape[0])
        kwargs["out_channels"] = int(projection[-1].shape[0])
    if lift is not None:
        # Assumes one field channel; see the note above.
        kwargs["use_coordinates"] = bool(int(lift.shape[1]) > 1)
    return kwargs


def save_bundle(path, operators, metadata=None):
    """
    Save several operators into one unified checkpoint.

    The whole pipeline is a chain, so its artefact is a chain: one file holding
    every stage means the two halves cannot drift apart, be copied
    individually, or be mixed across training runs.

    Parameters
    ----------
    path : str
        Destination, conventionally ``models/poraque_models.pfno``.
    operators : dict
        ``{task_name: FieldOperator}``.
    metadata : dict, optional
        Extra provenance stored alongside, e.g. the dataset size.

    Returns
    -------
    str
        ``path``.

    Examples
    --------
    >>> save_bundle("models/poraque_models.pfno",
    ...             {"ext2chg": first, "chg2tau": second})   # doctest: +SKIP
    """
    from ..version import __version__

    payload = {
        "format": BUNDLE_FORMAT,
        "poraque_version": __version__,
        "tasks": sorted(operators),
        "metadata": dict(metadata or {}),
    }
    for name, operator in operators.items():
        payload[name] = operator.state()

    directory = os.path.dirname(str(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)
    return str(path)


def read_bundle(path):
    """
    Read a bundle's raw payload, validating that it *is* one.

    Parameters
    ----------
    path : str

    Returns
    -------
    dict

    Raises
    ------
    ValueError
        If the file is not a Poraquê bundle. A single-operator checkpoint is
        named explicitly in the message, since that is the likely mistake.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
        detail = ""
        if isinstance(payload, dict) and "model_state" in payload:
            detail = (f" It looks like a single-operator checkpoint for task "
                      f"{payload.get('task')!r}; load it with "
                      f"FieldOperator.load instead.")
        raise ValueError(
            f"{path} is not a Poraque model bundle (expected "
            f"format={BUNDLE_FORMAT!r}).{detail}"
        )
    return payload


def bundle_tasks(path):
    """Task names stored in the bundle at ``path``."""
    return list(read_bundle(path).get("tasks", []))


def load_bundle(path, task, model=None, device=None, **model_kwargs):
    """
    Load one task's operator out of a unified checkpoint.

    Parameters
    ----------
    path : str
        Bundle file.
    task : str
        ``"ext2chg"`` or ``"chg2tau"``.
    model : torch.nn.Module, optional
    device : str or torch.device, optional
    **model_kwargs
        Override individual inferred hyper-parameters.

    Returns
    -------
    FieldOperator

    Raises
    ------
    KeyError
        If the bundle holds no such task, listing what it does hold.
    """
    payload = read_bundle(path)
    if task not in payload:
        raise KeyError(
            f"{path} holds no {task!r} model; it contains "
            f"{sorted(payload.get('tasks', []))}."
        )
    return FieldOperator.from_state(payload[task], model=model, device=device,
                                    **model_kwargs)


#: Submodule prefixes of :class:`~poraque.ml.fno.FNO3d` that make up the input
#: lifting path — everything before the first Fourier block.
LIFTING_PREFIXES = ("lift.", "cell_encoder.")


def freeze_lifting_layers(model, freeze=True):
    r"""
    Hold the input lifting path fixed, leaving the rest trainable.

    The lifting map embeds the input field into the network's channel space
    before any operator acts on it, so it is the most general part of the
    model and the part least in need of specialising to a new material family.
    Freezing it also removes parameters a small fine-tuning set could
    otherwise overfit.

    The cell encoder goes with it: it embeds the lattice, which is a property
    of the input rather than of the mapping being adapted.

    The projection head is deliberately left trainable — it decodes to
    physical units, which is precisely what differs between families.

    Parameters
    ----------
    model : torch.nn.Module
        Backbone. A model without these submodules is left untouched.
    freeze : bool, optional
        Set ``False`` to release them again.

    Returns
    -------
    dict
        ``{"frozen": int, "trainable": int}`` parameter counts, so a caller can
        report what actually happened rather than what was requested. Counted
        the way :meth:`~poraque.ml.fno.FNO3d.n_parameters` counts — a complex
        weight is two real numbers — so the two figures sum to the total the
        rest of the log quotes instead of appearing to differ by half a model.
    """
    for name, parameter in model.named_parameters():
        if name.startswith(LIFTING_PREFIXES):
            parameter.requires_grad = not freeze

    def size(parameter):
        return parameter.numel() * (2 if parameter.is_complex() else 1)

    return {
        "frozen": sum(size(p) for p in model.parameters()
                      if not p.requires_grad),
        "trainable": sum(size(p) for p in model.parameters()
                         if p.requires_grad),
    }


#: Weights on :class:`~poraque.ml.losses.PhysicsInformedLoss` that, when any is
#: positive, make the objective more than plain data fidelity.
PHYSICS_WEIGHTS = ("positivity_weight", "electron_count_weight",
                   "von_weizsacker_weight", "euler_lagrange_weight")


def physics_terms_active(criterion):
    """
    Whether ``criterion`` adds anything to the data term.

    Read from the configured weights rather than from the first batch's
    returned components, because the console header is written before any
    batch has run and the header and the rows must agree.

    Parameters
    ----------
    criterion : nn.Module
        Typically a :class:`~poraque.ml.losses.PhysicsInformedLoss`. Anything
        without the weight attributes counts as data-only.

    Returns
    -------
    bool
    """
    return any(float(getattr(criterion, name, 0.0) or 0.0) > 0.0
               for name in PHYSICS_WEIGHTS)


def train(operator, dataset, epochs=100, batch_size=1, learning_rate=1e-3,
          weight_decay=1e-4, validation=None, loss=None, scheduler="cosine",
          grad_clip=1.0, eval_every=1, early_stopping=0, checkpoint=None,
          seed=0, verbose=True, log=None):
    """
    Train a :class:`FieldOperator`.

    Parameters
    ----------
    operator : FieldOperator
        Operator to train, updated in place.
    dataset : FieldPairDataset
        Training data.
    epochs : int, optional
        Number of passes.
    batch_size : int, optional
        Maximum batch size within one shape bucket.
    learning_rate, weight_decay : float, optional
        AdamW hyper-parameters.
    validation : FieldPairDataset, optional
        Held-out materials, evaluated every ``eval_every`` epochs.
    loss : nn.Module, optional
        Objective; defaults to a supervised
        :class:`~poraque.ml.losses.PhysicsInformedLoss` for the task.
    scheduler : {"cosine", None}, optional
        Learning-rate schedule.
    grad_clip : float, optional
        Gradient-norm clipping; ``0`` disables. Spectral layers can produce
        large gradients early in training, so this is on by default.
    eval_every : int, optional
        Epoch interval for evaluating and reporting. Validation is computed
        *only* on these epochs, so this is a real saving on a large validation
        set, not merely a quieter log. The final epoch is always reported, so
        a run never ends without a current number.
    early_stopping : int, optional
        Stop after this many epochs without an improvement in the validation
        error, and restore the best weights seen. ``0`` (default) disables it.

        Requires ``validation``: without it the only measurable quantity is the
        training loss, which falls monotonically by construction and therefore
        can never signal that training should stop. Asking for early stopping
        with no validation set warns rather than silently doing nothing.

        Because the best weights are restored, the operator returned is the
        best one *measured*, not merely the last one reached — stopping partway
        down a degrading curve would otherwise hand back the degraded model.
    checkpoint : str, optional
        Path to save the best-validation model to. Only epochs on which
        validation was computed can improve on the best, which is the intended
        behaviour: a checkpoint is written against a measured score, never an
        assumed one.
    seed : int, optional
        RNG seed.
    verbose : bool, optional
        Report progress.
    log : callable, optional
        Receives each progress line. Defaults to :func:`print`; pass the
        caller's logger to get training progress into the run log rather than
        only onto the terminal.

    Returns
    -------
    dict
        ``train_loss`` (one entry per epoch) and, when validating,
        ``val_error`` together with ``val_epoch`` — the 1-based epochs those
        errors were measured on. The two validation lists are the same length
        and shorter than ``train_loss`` whenever ``eval_every > 1``, so
        anything plotting them must use ``val_epoch`` for the x-axis rather
        than assuming one point per epoch.

        When validating, also ``best_epoch`` and ``best_error``, and
        ``stopped_early`` recording whether patience ran out.
    """
    torch.manual_seed(seed)
    criterion = loss or PhysicsInformedLoss(task=operator.task.name)
    # Only unfrozen parameters reach the optimiser. A frozen one would receive
    # no gradient and so never move, but AdamW's decoupled weight decay is
    # applied regardless of the gradient -- handing it frozen weights would
    # shrink them towards zero every step, quietly undoing the pre-training the
    # freeze was meant to preserve.
    trainable = [p for p in operator.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate,
                                  weight_decay=weight_decay)
    lr_schedule = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if scheduler == "cosine" else None
    )

    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=True, seed=seed)
    validation_loader = (
        make_dataloader(validation, batch_size=batch_size, shuffle=False, seed=seed)
        if validation is not None else None
    )

    history = {"train_loss": [], "val_error": [], "val_epoch": [],
               "data_loss": [], "physics_loss": []}
    physics_active = physics_terms_active(criterion)
    best_error = float("inf")
    best_epoch, best_state, stopped_early = 0, None, False
    emit = log if log is not None else print
    eval_every = max(1, int(eval_every))
    early_stopping = max(0, int(early_stopping))

    if early_stopping and validation_loader is None:
        warnings.warn(
            "early_stopping was requested without a validation split, so it "
            "cannot act: the training loss falls monotonically by construction "
            "and never signals that training should stop. Set holdout or "
            "valid_fraction, or set early_stopping=0 to silence this.",
            RuntimeWarning, stacklevel=2,
        )
        early_stopping = 0

    # Column header, emitted next to the row formatter below so the two cannot
    # drift apart. A bare column of numbers is not self-describing: the two
    # quantities are measured differently -- one is the *objective*, whatever
    # that has been configured to be, the other a plain relative L2 in physical
    # units -- and reading the second as if it were the first is an easy and
    # expensive mistake.
    validating = validation_loader is not None
    # With every physics weight at zero -- the shipped default -- `total` and
    # `data` are the same number and the physics column is a column of zeros.
    # Three columns saying that is less readable than one, so the breakdown
    # appears only when a constraint is actually switched on. It is recorded in
    # `history` either way.
    loss_column = "total loss" if physics_active else "train loss"
    header = f"    {'epoch':>11s}  {loss_column:>13s}"
    if physics_active:
        header += f"  {'data loss':>13s}  {'physics loss':>13s}"
    if validating:
        header += f"  {'val rel L2':>13s}"
    if verbose:
        if physics_active:
            legend = ("    total loss = data + physics, mean per batch; "
                      "physics is the weighted contribution")
        else:
            legend = f"    train loss: mean {type(criterion).__name__} per batch"
        if validating:
            legend += "   |   val rel L2: held-out error, physical units"
        else:
            legend += "   |   no validation split: this is a TRAINING FIT"
        emit(legend)
        if validating and (checkpoint or early_stopping):
            emit("    * marks an epoch that improved on the best score so far")
        if early_stopping:
            emit(f"    early stopping: after {early_stopping} epochs without "
                 f"improvement; the best weights are restored")
        emit(header)
        emit("    " + "-" * (len(header) - 4))

    for epoch in range(int(epochs)):
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)

        operator.model.train()
        running, batches = 0.0, 0
        running_data, running_physics = 0.0, 0.0
        components = {}
        for batch in loader:
            inputs = batch["input"].to(operator.device)
            targets = batch["target"].to(operator.device)
            cell = batch["cell"].to(operator.device)

            optimizer.zero_grad(set_to_none=True)
            prediction = operator.model(inputs, cell)

            # The reference in physical units serves two terms: it is rho for
            # the von Weizsacker bound on chg2tau -- where the *input* is the
            # density -- and its integral is the electron count the
            # charge-conservation term on ext2chg is measured against.
            physical_target = batch["target_physical"].to(operator.device)
            terms = criterion(
                prediction, targets, cell=cell,
                physical_prediction=operator.target_transform.inverse(prediction),
                physical_input=operator.input_transform.inverse(inputs),
                physical_target=physical_target,
            )
            terms["total"].backward()

            if grad_clip:
                clip_gradients(operator.model.parameters(), grad_clip)
            optimizer.step()

            # The composite loss reports its parts; only `total` is
            # differentiated. `data` is the fidelity term as the objective saw
            # it, and the physics contribution is taken as total - data rather
            # than by summing the components, because the components are
            # reported *unweighted* -- summing them would give a number that
            # is not what the optimiser stepped on.
            total_value = float(terms["total"].detach())
            data_value = float(terms.get("data", terms["total"]))
            running += total_value
            running_data += data_value
            running_physics += total_value - data_value
            for name, value in terms.items():
                if name in ("total", "data"):
                    continue
                components[name] = components.get(name, 0.0) + float(value)
            batches += 1

        if lr_schedule is not None:
            lr_schedule.step()

        divisor = max(batches, 1)
        mean_loss = running / divisor
        history["train_loss"].append(mean_loss)
        history["data_loss"].append(running_data / divisor)
        history["physics_loss"].append(running_physics / divisor)
        # Per-term magnitudes, unweighted: what each constraint is worth before
        # its weight scales it, which is what you need to choose that weight.
        for name, accumulated in components.items():
            history.setdefault(f"physics_{name}", []).append(accumulated / divisor)

        # The last epoch always reports, so a run never finishes without a
        # current number just because `epochs` is not a multiple of the
        # interval.
        reporting = ((epoch + 1) % eval_every == 0) or (epoch == epochs - 1)
        if not reporting:
            continue

        # Columns line up under `header` above; widths are shared with it.
        message = f"    {f'{epoch + 1}/{epochs}':>11s}  {mean_loss:>13.5f}"
        if physics_active:
            message += (f"  {history['data_loss'][-1]:>13.5f}"
                        f"  {history['physics_loss'][-1]:>13.5f}")
        exhausted = False
        if validating:
            error = evaluate(operator, validation_loader)
            history["val_error"].append(error)
            history["val_epoch"].append(epoch + 1)
            message += f"  {error:>13.5f}"

            if error < best_error:
                best_error, best_epoch = error, epoch + 1
                if early_stopping:
                    # Kept in memory so the best model can be returned even
                    # when no checkpoint path was given.
                    best_state = {k: v.detach().clone()
                                  for k, v in operator.model.state_dict().items()}
                if checkpoint:
                    operator.save(checkpoint)
                message += "  *"
            elif early_stopping and (epoch + 1) - best_epoch >= early_stopping:
                exhausted = True

        if verbose:
            emit(message)

        if exhausted:
            stopped_early = True
            if verbose:
                emit(f"    stopped early at epoch {epoch + 1}: no improvement "
                     f"in {early_stopping} epochs "
                     f"(best {best_error:.5f} at epoch {best_epoch})")
            break

    if early_stopping and best_state is not None:
        operator.model.load_state_dict(best_state)
        if verbose and not stopped_early and best_epoch != len(history["train_loss"]):
            emit(f"    restored the best weights, from epoch {best_epoch} "
                 f"(val rel L2 {best_error:.5f})")

    if validating:
        history["best_epoch"] = best_epoch
        history["best_error"] = best_error
        history["stopped_early"] = stopped_early

    return history


@torch.no_grad()
def evaluate(operator, loader):
    """
    Mean relative :math:`L^2` error over a loader, in *physical* units.

    Evaluating in physical units matters: a small error in a compressed
    (asinh) representation can hide a large error in the density itself.

    Parameters
    ----------
    operator : FieldOperator
    loader : torch.utils.data.DataLoader

    Returns
    -------
    float
    """
    operator.model.eval()
    errors = []
    for batch in loader:
        inputs = batch["input"].to(operator.device)
        cell = batch["cell"].to(operator.device)
        physical_target = batch["target_physical"].to(operator.device)

        prediction = operator.target_transform.inverse(operator.model(inputs, cell))
        errors.append(relative_error(prediction, physical_target).cpu())

    return float(torch.cat(errors).mean()) if errors else float("nan")
