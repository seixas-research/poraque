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
import time
import warnings

import numpy as np
import torch

from ..fields import ChargeDensity, ExternalPotential, KineticEnergyDensity
from .config import BUNDLE_SUFFIX
from .data import make_dataloader
from .device import describe_device, resolve_device, synchronize
from .distributed import DistributedContext, all_reduce_mean, barrier
from .fno import FNO3d
from .losses import PhysicsInformedLoss, data_error
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


def _resolve_operator_baseline(baseline):
    """
    Accept a library, a path, a serialised payload from a checkpoint, or ``None``.

    The dict form is what :meth:`FieldOperator.state` writes, so a checkpoint
    round-trips without the caller having to know the library ever existed.
    """
    if baseline is None:
        return None

    from ..fields.atomic import AtomicReference, AtomicReferenceLibrary

    if isinstance(baseline, AtomicReferenceLibrary):
        return baseline if len(baseline) else None
    if isinstance(baseline, dict):
        entries = baseline.get("entries") or {}
        if not entries:
            return None
        return AtomicReferenceLibrary(
            {key: AtomicReference.from_dict(value)
             for key, value in entries.items()})

    library = AtomicReferenceLibrary.load(str(baseline))
    return library if len(library) else None


class FieldOperator:
    r"""
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
    strict_device : bool, optional
        Raise instead of falling back to the CPU when ``device`` cannot be
        honoured. Off by default, which is right for a workstation and wrong
        for a queue — a batch job that quietly moves to the CPU spends its GPU
        allocation not using a GPU. Forwarded straight to
        :func:`~poraque.ml.device.resolve_device`; a run sets it from
        ``training.strict_device``.
    pauli_residual : bool, optional
        Wrap the backbone in a
        :class:`~poraque.ml.heads.PauliResidualOperator`, so the model
        predicts :math:`\tau = \tau_{\rm vW}[\rho] + s\,
        \mathrm{softplus}(f_\theta)` and the Hoffmann-Ostenhof bound holds by
        construction. Only meaningful for ``chg2tau``; requested for any other
        task it raises, since :math:`\tau_{\rm vW}` is a functional of the
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
    baseline : AtomicReferenceLibrary, str, dict or None, optional
        Isolated-atom database. Its presence puts the operator in
        **delta-density mode**: the network's output is
        :math:`\delta\rho = \rho - \rho_{\rm sup}`, and :meth:`predict` adds
        the superposition back before returning, so a caller sees an absolute
        density either way and nothing downstream has to know which mode the
        model was trained in.

        The library **travels inside the checkpoint**, not beside it. A
        δ-density model whose baseline changed after training is not a model of
        anything: the residual it learned was defined against one particular
        superposition. Storing the table with the weights makes that
        impossible to get wrong.
    **model_kwargs
        Forwarded to :class:`~poraque.ml.fno.FNO3d`.
    """

    def __init__(self, task, model=None, input_transform=None,
                 target_transform=None, device=None, pauli_residual=False,
                 pauli_scale=1.0, learn_pauli_scale=True,
                 training_resolution=None, init_seed=None, baseline=None,
                 strict_device=False, **model_kwargs):
        self.task = resolve_task(task)
        self.device = resolve_device(device, strict=strict_device)
        self.input_transform = input_transform or Identity()
        self.target_transform = target_transform or Identity()
        self.baseline = _resolve_operator_baseline(baseline)
        #: LoRA settings when this operator has been adapted, else ``None``.
        #: Set by the fine-tuning path; read by :meth:`state` so the record can
        #: say where the frozen base lives.
        self.lora = None
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
    def compute_dtype(self):
        """
        The real dtype the model computes in, read from its own weights.

        Returns
        -------
        torch.dtype
            ``torch.float32`` or ``torch.float64``. Falls back to
            ``torch.float32`` for a model with no floating parameters at all,
            which only a stub has.
        """
        for parameter in self.model.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return torch.float32

    def baseline_for(self, field):
        r"""
        The atomic superposition on ``field``'s grid, or ``None``.

        Parameters
        ----------
        field : ScalarField
            The input field, which supplies both the grid and the structure.

        Returns
        -------
        numpy.ndarray or None
            ``None`` in absolute mode, for a task whose target is not a
            density, or when the structure has a species the library does not
            cover --- the last of which raises rather than returning a partial
            superposition, because a baseline missing whole atoms is wrong in a
            way that looks plausible.
        """
        if self.baseline is None or self.task.target_field != "CHGCAR":
            return None

        from ..fields.atomic import atomic_superposition

        return atomic_superposition(field.structure, field.grid,
                                    self.baseline).data

    def set_precision(self, precision):
        """
        Convert the operator to another precision.

        Parameters
        ----------
        precision : str or torch.dtype
            ``"float32"`` or ``"float64"``. See
            :func:`~poraque.ml.fno.set_precision` for why neither
            ``model.double()`` nor ``model.to(torch.float64)`` is a substitute:
            the first leaves the complex spectral weights behind, the second
            deletes their imaginary part.

        Returns
        -------
        FieldOperator
            ``self``, so this can be chained onto a constructor.
        """
        from .fno import set_precision as convert

        convert(self.model, precision)
        return self

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
        # The model's own precision, not a literal float32: an operator
        # converted with `set_precision(model, "float64")` would otherwise be
        # handed single-precision input and fail on the first spectral layer.
        # The dtype is read from the weights, so the two cannot disagree.
        compute = self.compute_dtype()
        values = torch.as_tensor(np.ascontiguousarray(field.data),
                                 dtype=compute, device=self.device)
        cell = torch.as_tensor(field.grid.cell, dtype=compute,
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

        # To CPU in the compute dtype before .numpy(): accelerators may hand
        # back a dtype numpy cannot consume directly, and .cpu() alone does not
        # convert it. Narrowing to float32 here would throw away exactly the
        # precision a float64 run was asked for.
        channels = prediction[0].detach().to("cpu", compute).numpy()
        metadata = {"predicted_by": type(self.model).__name__,
                    "task": self.task.name,
                    "device": str(self.device)}

        # Delta-density mode: the network predicted rho - rho_sup, so the
        # superposition goes back on *here*, before the caller can apply
        # positivity or an electron-count normalization. Both of those are
        # statements about the absolute density and neither is true of a
        # signed residual. Clipping delta-rho at zero deletes the bonding
        # charge, which is negative wherever charge moved away from the free
        # atoms -- exactly the signal the mode exists to model; and rescaling
        # it to an electron count divides by an integral that is approximately
        # zero. Both alternatives return something that still looks like a
        # density.
        baseline = self.baseline_for(field)
        if baseline is not None:
            # Channel 0 only: a spin-polarised prediction is (rho, m) and the
            # magnetisation has no free-atom superposition to restore.
            channels = np.array(channels, copy=True)
            channels[0] = channels[0] + baseline
            metadata["delta_density"] = True
            metadata["baseline"] = "atomic_superposition"

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
        # Architecture that no tensor shape encodes. Without this record, a
        # model trained with mode_selection="physical" reloads as "fixed" and
        # its never-trained masked modes go live -- silently.
        backbone = getattr(self.model, "backbone", self.model)
        architecture = {
            key: getattr(backbone, key)
            for key in ("modes", "mode_selection", "g_max", "activation",
                        "projection_activation",
                        "cell_conditioning", "embedding_dim",
                        "kan_grid_size", "kan_spline_order", "kan_grid_range",
                        "kan_degree", "kan_rational_num_degree",
                        "kan_rational_den_degree", "kan_use_base",
                        "equivariant", "n_radial", "g_basis",
                        "spherical_cutoff")
            if hasattr(backbone, key)
        }
        # A LoRA fine-tune stores its adapter and *not* the base it adapts.
        # That is the whole economy of the method: the frozen tensors are
        # already on disk in the checkpoint being adapted, and writing them
        # again per fine-tune is precisely the cost LoRA exists to avoid --
        # 12 MB against 8 kB on this project's own model. What the record has
        # to carry instead is where the base lives, or the file names weights
        # it cannot reconstruct.
        from .lora import is_adapted, lora_state_dict

        adapted = is_adapted(self.model)
        return {
            "task": self.task.name,
            "model_state": ({} if adapted else self.model.state_dict()),
            "lora": (None if not adapted else {
                **dict(self.lora or {}),
                "state": lora_state_dict(self.model),
            }),
            "model_class": type(self.model).__name__,
            # Recorded rather than inferred: the lifting layer cannot tell
            # in_channels apart from the coordinate channels, so a spin model
            # reloaded by inference alone would silently become a one-channel
            # model with coordinates. See infer_backbone_kwargs.
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "use_coordinates": self.use_coordinates,
            "architecture": architecture,
            "input_transform": self.input_transform.state_dict(),
            "target_transform": self.target_transform.state_dict(),
            "pauli_residual": self.pauli_residual,
            "pauli_scale": self.pauli_scale,
            "learn_pauli_scale": self.learn_pauli_scale,
            "training_resolution": self.training_resolution,
            "init_seed": self.init_seed,
            # The baseline is part of what the weights *mean*, not a runtime
            # convenience: the residual was defined against this particular
            # superposition. Stored as the serialised library so a checkpoint
            # is self-contained; `None` in absolute-density mode.
            "baseline": (None if self.baseline is None
                         else {"version": 1,
                               "fingerprint": self.baseline.fingerprint,
                               "entries": {k: v.to_dict() for k, v
                                           in self.baseline.entries.items()}}),
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
        lora = state.get("lora")
        if lora:
            return cls._from_lora_state(state, lora, device=device,
                                        **model_kwargs)

        inferred = infer_backbone_kwargs(state["model_state"])
        architecture = dict(state.get("architecture") or {})
        # The read-out's nonlinearity followed a hard-coded GELU until
        # 2026-08-28 and now follows `activation`. A checkpoint that does not
        # record which it used is by definition one written before the change,
        # so "gelu" is not a default here -- it is the answer. Restoring it as
        # `activation` instead would silently change what every silu-trained
        # model computes, which is the exact failure the architecture record
        # exists to prevent.
        architecture.setdefault("projection_activation", "gelu")
        # Same rule, same reason: a checkpoint with no `equivariant` entry
        # predates the flag, so False is the answer and not a default. Here it
        # is also checkable -- `infer_backbone_kwargs` reads the variant off
        # the rank of the spectral weight -- which is why the two must agree
        # rather than one quietly overriding the other.
        architecture.setdefault("equivariant", False)
        # Recorded architecture wins over the inference: mode_selection,
        # g_max, activation, cell_conditioning and embedding_dim live in no
        # tensor shape, so inference alone would silently reset them to the
        # constructor defaults. Checkpoints written before the record existed
        # simply have no entry here and keep the inferred/default values.
        inferred.update(architecture)
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
            baseline=state.get("baseline"),
            **inferred,
        )
        operator.model.load_state_dict(state["model_state"])
        return operator

    @classmethod
    def _from_lora_state(cls, state, lora, device=None, **model_kwargs):
        """
        Rebuild an adapted operator: the recorded base, then the adapter.

        The base is loaded from the checkpoint the fine-tune named. That
        indirection is the price of a small file, and it is stated in the
        error rather than left to be discovered: a LoRA checkpoint whose base
        has moved cannot reconstruct weights it never stored.
        """
        import os

        from .lora import apply_lora, load_lora_state_dict

        base_path = lora.get("base_checkpoint")
        if base_path:
            # The base may have been renamed since the fine-tune recorded it
            # — the extension changed under the whole project on 2026-09-02 —
            # and a LoRA checkpoint that cannot find its base holds no weights
            # at all.
            base_path = resolve_bundle_path(str(base_path))
        if not base_path or not os.path.exists(str(base_path)):
            raise FileNotFoundError(
                f"This is a LoRA checkpoint: it holds a "
                f"{len(lora.get('state') or {})}-tensor adapter and not the "
                f"weights it adapts, which live in "
                f"{base_path!r}. That file is missing, so the model cannot be "
                f"rebuilt. Point fine_tuning.pretrained_checkpoint at it "
                f"again, or re-run the fine-tune with fine_tuning.use_lora "
                f"disabled to get a self-contained bundle.")

        base = load_bundle(str(base_path), state["task"], device=device,
                           **model_kwargs)
        # The transforms, the baseline and the task come from *this* record:
        # a fine-tune may have been trained against a different normalisation
        # than the base was, and taking them from the base would decode the
        # adapted model's outputs with the wrong scale.
        base.input_transform = FieldTransform.from_state_dict(
            state["input_transform"])
        base.target_transform = FieldTransform.from_state_dict(
            state["target_transform"])
        if state.get("baseline") is not None:
            base.baseline = _resolve_operator_baseline(state["baseline"])

        settings = apply_lora(base.model, rank=int(lora.get("rank", 8)),
                              alpha=float(lora.get("alpha", 16.0)),
                              dropout=float(lora.get("dropout", 0.0)))
        load_lora_state_dict(base.model, lora.get("state") or {})
        base.lora = {key: value for key, value in lora.items()
                     if key != "state"}
        base.lora.update(adapters=settings["adapters"])
        base.model.to(base.device)
        return base

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

#: Conventional filename for the unified checkpoint. The extension itself is
#: :data:`poraque.ml.config.BUNDLE_SUFFIX`, re-exported here so that everything
#: about naming a bundle is reachable from one module.
BUNDLE_FILENAME = "poraque_models" + BUNDLE_SUFFIX

#: Extensions Poraquê used to write bundles under, newest first. Retained only
#: so a stale file can be *found and named*, never to change how one is read:
#: the format is unchanged and the extension was never inspected when loading.
#: ``.pfno`` was the extension until 2026-09-02 and is what every model trained
#: before then is called, so it is the one most likely to be hit.
LEGACY_BUNDLE_SUFFIXES = (".pfno", ".pth", ".pt")


def resolve_bundle_path(path, log=None):
    """
    Find the bundle the caller means, whichever extension it is under.

    Renaming the default output — ``.pth`` to ``.pfno``, then ``.pfno`` to
    ``.poraque`` — would otherwise make an existing trained model invisible to
    every default path, which reads as "no model" rather than "renamed". This
    looks beside the requested file for the same stem under any of the names
    Poraquê has used, and says what it did.

    The search runs **both ways**, which the single-direction version did not.
    A config that asks for ``.poraque`` and finds a ``.pfno`` is the obvious
    case; the other one is a checkpoint that *records* the path it was trained
    against — a LoRA adapter names its base — written before the rename and
    read after the base was renamed. Only the extension is guessed at; a
    different stem is a different model and is never substituted.

    Parameters
    ----------
    path : str
        Requested bundle.
    log : callable, optional
        Sink for the notice. Silent when omitted.

    Returns
    -------
    str
        ``path`` when it exists, the file under another of Poraquê's own
        extensions when only that does, and ``path`` unchanged when neither
        does — so the caller still reports the name the user actually asked
        for.
    """
    if os.path.exists(path):
        return path

    stem, requested = os.path.splitext(path)
    alternatives = [suffix for suffix
                    in (BUNDLE_SUFFIX,) + LEGACY_BUNDLE_SUFFIXES
                    if suffix != requested]
    for suffix in alternatives:
        candidate = stem + suffix
        if os.path.exists(candidate):
            if log is not None:
                log(f"  NOTE: {path} does not exist, but {candidate} does — "
                    f"using it.")
                log(f"        Poraque now writes {os.path.basename(stem)}"
                    f"{BUNDLE_SUFFIX}; rename the file to silence this.")
            return candidate
    return path


#: Filename for a fine-tuned bundle. Deliberately distinct from
#: :data:`BUNDLE_FILENAME`: a fine-tune is a specialisation of the base model,
#: usually to a narrower set of materials, and writing it over the general
#: model would silently replace something broad with something narrow.
FINETUNED_BUNDLE_FILENAME = "poraque_finetuned" + BUNDLE_SUFFIX


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
        An equivariant backbone reports ``equivariant`` and ``n_radial``
        instead of ``modes``: its kernel is radial, so no mode index survives
        in a tensor shape and ``modes`` comes from the architecture record.

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
        kwargs["n_layers"] = len(spectral)
        if spectral[0].dim() == 3:
            # (in, out, n_radial) -- RadialSpectralConv3d. The kernel is a
            # function of |G| and carries no per-mode index, so `modes` is not
            # in the tensor at all and has to come from the architecture
            # record. `equivariant` is: a rank-3 spectral weight cannot be
            # anything else, which is the one architectural fact here that
            # does not depend on a record having been written.
            kwargs["width"] = int(spectral[0].shape[1])
            kwargs["n_radial"] = int(spectral[0].shape[2])
            kwargs["equivariant"] = True
        else:
            # (4, in, out, m1, m2, m3) -- four corner blocks of the rfftn.
            kwargs["width"] = int(spectral[0].shape[1])
            kwargs["modes"] = int(spectral[0].shape[3])
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
        Destination, conventionally ``models/poraque_models.poraque``.
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
    >>> save_bundle("models/poraque_models.poraque",
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


#: Optimisers selectable through ``training.optimizer``.
#:
#: ``adamw`` is the default and was the only one for most of this project's
#: life. The distinction from ``adam`` is not cosmetic: both keep the same
#: per-parameter adaptive step, but Adam folds weight decay into the gradient,
#: where the adaptive denominator then rescales it, so a parameter with small
#: historical gradients is decayed *harder* than one with large ones. AdamW
#: applies the decay directly to the weight, decoupled from the gradient and
#: from that denominator, which is what makes ``weight_decay`` mean the same
#: thing for every parameter.
#:
#: With ``weight_decay = 0`` the two are numerically identical, which is worth
#: knowing before reading anything into a comparison at that setting.
#:
#: ``sgd`` carries momentum 0.9 and is here as the non-adaptive control: it is
#: what the adaptive methods have to beat, and on a spectral operator whose
#: parameter scales differ by orders of magnitude between the pointwise and
#: the Fourier weights, it usually does not.
OPTIMIZERS = ("adamw", "adam", "sgd")


def build_optimizer(parameters, name="adamw", learning_rate=1e-3,
                    weight_decay=1e-4, momentum=0.9):
    """
    Construct one of :data:`OPTIMIZERS`.

    Parameters
    ----------
    parameters : iterable
        Tensors to optimise -- the *trainable* ones; see the note at the call
        site about frozen weights and decoupled decay.
    name : str, optional
        One of :data:`OPTIMIZERS`.
    learning_rate, weight_decay : float, optional
    momentum : float, optional
        ``sgd`` only; ignored by the adaptive methods, which derive their own
        first moment from ``betas``.

    Returns
    -------
    torch.optim.Optimizer

    Raises
    ------
    ValueError
        For an unknown name, listing what is available -- a typo here would
        otherwise fall through to a default and train something other than
        what the config describes.
    """
    key = str(name).strip().lower()
    if key == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate,
                                 weight_decay=weight_decay)
    if key == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate,
                                weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate,
                               momentum=momentum, weight_decay=weight_decay)
    raise ValueError(
        f"Unknown optimizer {name!r}; expected one of {list(OPTIMIZERS)}.")


def _distributed_forward(operator, context, emit, verbose):
    """
    Wrap the model in ``DistributedDataParallel``, or return it unchanged.

    The wrapper is used to *call* the model and is never assigned onto the
    operator: DDP prefixes every ``state_dict`` key with ``module.``, so an
    operator holding one would write checkpoints that load nowhere else. The parameters are the same tensors either way, which is
    what lets the optimiser, the gradient clipping and the best-weight snapshot
    all keep addressing ``operator.model`` directly.

    Parameters
    ----------
    operator : FieldOperator
        Already on ``cuda:<local_rank>``; DDP requires it, and putting every
        rank on ``cuda:0`` is the classic way to get a fourfold slowdown
        reported as a scaling failure.
    context : DistributedContext
    emit : callable
    verbose : bool

    Returns
    -------
    torch.nn.Module
    """
    if not context or not context.initialized:
        return operator.model

    from torch.nn.parallel import DistributedDataParallel

    wrapped = DistributedDataParallel(
        operator.model, device_ids=[context.local_rank],
        output_device=context.local_rank,
        # The Fourier layers are complex-valued and every parameter takes a
        # gradient every step, so there is no unused branch for DDP to hunt
        # for -- and the search costs a full graph traversal per iteration.
        find_unused_parameters=False,
    )
    if verbose:
        emit(f"    distributed: {context.describe()}")
    return wrapped


def train(operator, dataset, epochs=100, batch_size=1, learning_rate=1e-3,
          weight_decay=1e-4, validation=None, loss=None, scheduler="cosine",
          grad_clip=1.0, eval_every=1, early_stopping=0, checkpoint=None,
          seed=0, verbose=True, log=None, optimizer="adamw",
          num_workers=0, pin_memory="auto", distributed=None):
    """
    Train a :class:`FieldOperator`.

    **One GPU, unless a group is passed.** With ``distributed`` left at
    ``None`` this is the single-device loop it has always been. Given an
    initialised :class:`~poraque.ml.distributed.DistributedContext` it becomes
    data-parallel over NCCL: the model is wrapped in
    :class:`~torch.nn.parallel.DistributedDataParallel`, the shape-bucketed
    batches are partitioned across ranks by
    :class:`~poraque.ml.data.DistributedShapeBucketSampler`, and rank 0 alone
    prints and writes.

    Three things about that are worth knowing before reading a scaling number.
    The **effective batch size is multiplied by the world size**, so a
    four-rank run at ``batch_size`` 32 is stepping on 128 samples and is not
    the same optimiser as the one-rank run it is being compared against.
    **Validation is not distributed**: every rank evaluates the whole held-out
    set, redundantly, so that ``best_error`` and the early-stopping decision
    are identical on every rank by construction rather than by a reduction that
    could be forgotten — ranks that disagreed about whether to stop would leave
    the ones still training waiting on a collective nobody joins. And the
    **training loss is all-reduced** at the end of each epoch, so the logged
    number is over the whole dataset and not over this rank's quarter of it.

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
        Step size and decay, passed to whichever optimiser is selected.
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
    num_workers : int, optional
        DataLoader worker processes. Leave at ``0`` unless the dataset's
        in-memory cache had to be turned off: measured, the two are
        alternatives and adding workers to a cached dataset makes it *slower*.
        See :func:`~poraque.ml.data.make_dataloader`.
    pin_memory : {"auto", True, False}, optional
        Page-locked staging for the host-to-device copy. ``"auto"`` resolves to
        ``True`` on CUDA and ``False`` elsewhere, which is the whole decision.
    distributed : DistributedContext, optional
        The process group this rank belongs to, already initialised by
        :func:`~poraque.ml.distributed.initialize`. ``None`` and a disabled
        context are the same thing, so a caller passes it unconditionally and
        the single-device path is what runs when there is no group.

    Returns
    -------
    dict
        ``train_loss`` (one entry per epoch) — the **total** objective, data
        fidelity plus every weighted physics term, which is what the optimiser
        stepped on — and, when validating,
        ``val_error`` together with ``val_epoch`` — the 1-based epochs those
        errors were measured on. The two validation lists are the same length
        and shorter than ``train_loss`` whenever ``eval_every > 1``, so
        anything plotting them must use ``val_epoch`` for the x-axis rather
        than assuming one point per epoch.

        When validating, also ``best_epoch`` and ``best_error``,
        ``stopped_early`` recording whether patience ran out, and
        ``val_metric`` naming the norm ``val_error`` is measured in —
        ``"rel L2"``, or ``"rel H1"`` when the objective carries a Sobolev
        gradient term. The final per-structure evaluation reports plain
        relative :math:`L^2` whatever the objective was, so the two are
        comparable across runs.

        Always ``seconds_per_epoch`` — wall time per epoch, measured after
        :func:`~poraque.ml.device.synchronize`, so it is compute and not
        queueing — and, on CUDA, ``peak_vram_bytes`` and
        ``peak_vram_reserved_bytes``. Those three are what a ``batch_size``
        study needs and what previously could only be had by sampling
        ``nvidia-smi`` from outside the process.
    """
    torch.manual_seed(seed)
    context = distributed if distributed is not None else DistributedContext()
    # Rank 0 alone writes and prints. Four ranks appending to one log truncate
    # each other's lines, and four calling `operator.save` on one path race on
    # the same inode -- which does not raise, it leaves a checkpoint that loads
    # and holds a mixture. Applied here, once, rather than guarded at each of
    # the dozen sites below.
    if not context.is_main:
        verbose = False
        checkpoint = None
    criterion = loss or PhysicsInformedLoss(task=operator.task.name)
    # Only unfrozen parameters reach the optimiser. A frozen one would receive
    # no gradient and so never move, but AdamW's decoupled weight decay is
    # applied regardless of the gradient -- handing it frozen weights would
    # shrink them towards zero every step, quietly undoing the pre-training the
    # freeze was meant to preserve.
    trainable = [p for p in operator.model.parameters() if p.requires_grad]
    optimizer = build_optimizer(trainable, name=optimizer,
                                learning_rate=learning_rate,
                                weight_decay=weight_decay)
    lr_schedule = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if scheduler == "cosine" else None
    )

    # Pinning is a CUDA transfer optimisation and nothing else: it does not
    # help on MPS and some PyTorch versions warn when asked for it there. This
    # is the layer that knows the device, so this is where "auto" is answered.
    if isinstance(pin_memory, str):
        pin_memory = (operator.device.type == "cuda"
                      if pin_memory.strip().lower() == "auto" else
                      pin_memory.strip().lower() in ("1", "true", "yes"))
    pin_memory = bool(pin_memory)

    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=True,
                             seed=seed, num_workers=num_workers,
                             pin_memory=pin_memory, distributed=context)
    # Deliberately **not** distributed. Every rank evaluates the whole held-out
    # set, which is redundant work -- forward-only, on a fifth of the data --
    # bought in exchange for `best_error` being identical on every rank without
    # a reduction anyone could forget to add. Early stopping is decided from
    # that number, and ranks that disagreed about whether to break would leave
    # the ones still training in a collective that never completes.
    validation_loader = (
        make_dataloader(validation, batch_size=batch_size, shuffle=False,
                        seed=seed, num_workers=num_workers,
                        pin_memory=pin_memory)
        if validation is not None else None
    )

    # Called instead of `operator.model` inside the loop, never assigned to it:
    # DDP renames every `state_dict` key to `module.<key>`, and storing the
    # wrapper would silently make every checkpoint this run writes unloadable
    # by every other code path.
    emit = log if log is not None else print
    forward = _distributed_forward(operator, context, emit, verbose)

    history = {"train_loss": [], "val_error": [], "val_epoch": [],
               "seconds_per_epoch": []}
    if operator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(operator.device)
    best_error = float("inf")
    best_epoch, best_state, stopped_early = 0, None, False
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
    # The validation column names the norm it is actually measured in. An H1
    # objective constrains the gradient as well as the values and an absolute
    # one does not normalise per sample, so watching any of the four on a plain
    # relative L2 reports a quantity the optimiser is not minimising -- and
    # early stopping then selects against the wrong one. Both are read off the
    # criterion rather than passed in, so the label cannot disagree with the
    # objective it describes.
    data_loss_name = str(getattr(criterion, "loss", "relative_l2"))
    sobolev_weight = float(getattr(criterion, "sobolev_weight", 0.0) or 0.0)
    # Read off the criterion, never re-derived from a config this function does
    # not see -- the same discipline `sobolev_weight` above is read with, and
    # for the same reason: a loop that decided this for itself could decide it
    # differently from the objective it is stepping on.
    #
    # The default is **True**, which is the opposite of what saves the work,
    # and deliberately so: an injected `loss=` that is not a
    # `PhysicsInformedLoss` has always been handed the physical fields, and
    # withholding them from it on the strength of a missing attribute would
    # change what somebody else's objective computes without saying so. The
    # saving comes from `PhysicsInformedLoss` itself answering False, which it
    # does whenever no weight is set -- the default configuration.
    physics_informed = bool(getattr(criterion, "physics_informed", True))
    val_metric = str(getattr(criterion, "metric_label", "rel L2"))
    # One number for the objective, whatever it is made of. A physics-informed
    # run's `train loss` is the total the optimiser actually stepped on --
    # data plus every weighted constraint -- and that total is the only
    # quantity a reader can compare against another run's.
    header = f"    {'epoch':>11s}  {'train loss':>13s}"
    if validating:
        header += f"  {f'val {val_metric}':>13s}"
    if verbose:
        legend = f"    train loss: mean {type(criterion).__name__} per batch"
        if validating:
            legend += (f"   |   val {val_metric}: held-out error, "
                       f"physical units")
            if sobolev_weight > 0.0:
                norm = "relative" if val_metric.startswith("rel") else "absolute"
                base = val_metric.replace("H1", "L2")
                legend += (f"\n    val {val_metric} = {base} + "
                           f"{sobolev_weight:g} x the {norm} L2 of the "
                           f"gradient, matching the objective; the final "
                           f"per-structure table below reports plain rel L2")
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
        epoch_start = time.perf_counter()
        # Accumulated **on the device**. `float(loss)` per batch is a
        # synchronisation per batch: the CPU stops until the queue drains, so a
        # run pays the latency once per step to record a number it does not
        # read until the epoch ends. One `.item()` below is enough.
        running = torch.zeros((), device=operator.device)
        batches = 0
        for batch in loader:
            # non_blocking only does anything with pin_memory, and is harmless
            # without it -- an unpinned copy is synchronous whatever is asked.
            inputs = batch["input"].to(operator.device, non_blocking=pin_memory)
            targets = batch["target"].to(operator.device, non_blocking=pin_memory)
            cell = batch["cell"].to(operator.device, non_blocking=pin_memory)

            optimizer.zero_grad(set_to_none=True)
            prediction = forward(inputs, cell)

            # Everything in this block exists for the constraints and for
            # nothing else, so with them off it is work whose result is
            # discarded: a host-to-device copy of a whole reference field, two
            # inverse transforms, and in delta-density mode a second copy and
            # two full-tensor adds -- per batch, with the data term needing
            # none of it. Measured on MPS at batch 8, 32^3, width 16: 303.0 ms
            # against 295.7, so 2.4 % of a step whose cost is overwhelmingly
            # the operator itself.
            #
            # Note what is *not* saved, because it would be easy to claim: the
            # backward pass. `total` never depended on `physical_prediction`
            # when every weight was zero, so autograd never traversed the
            # inverse transform. What it did do was hold that graph's
            # intermediates alive until the graph was freed, which is memory
            # rather than arithmetic.
            physical = {}
            if physics_informed:
                # The reference in physical units serves two terms: it is rho
                # for the von Weizsacker bound on chg2tau -- where the *input*
                # is the density -- and its integral is the electron count the
                # charge-conservation term on ext2chg is measured against.
                physical_target = batch["target_physical"].to(
                    operator.device, non_blocking=pin_memory)
                physical_prediction = operator.target_transform.inverse(
                    prediction)

                # Delta-density mode: the data term above compares residuals,
                # which is the whole point, but every *physics* term is a
                # statement about the absolute density -- positivity, the
                # electron count, the Euler-Lagrange residual. Adding the
                # baseline back here is what makes them true again, and it
                # means no loss term had to change.
                baseline = batch.get("baseline")
                if baseline is not None:
                    baseline = baseline.to(operator.device,
                                           non_blocking=pin_memory)
                    physical_prediction = physical_prediction + baseline
                    physical_target = physical_target + baseline

                physical = {
                    "physical_prediction": physical_prediction,
                    "physical_input": operator.input_transform.inverse(inputs),
                    "physical_target": physical_target,
                }

            terms = criterion(prediction, targets, cell=cell, **physical)
            terms["total"].backward()

            if grad_clip:
                clip_gradients(operator.model.parameters(), grad_clip)
            optimizer.step()

            # The composite loss reports its parts, but only `total` is
            # differentiated and only `total` is recorded: it is the objective
            # the optimiser stepped on, and the parts are reported *unweighted*
            # so they do not sum to it.
            running += terms["total"].detach()
            batches += 1

        if lr_schedule is not None:
            lr_schedule.step()

        divisor = max(batches, 1)
        # Reduced before it leaves the device, so the collective costs one
        # small all-reduce rather than a host round trip per rank. Every rank
        # holds the same number afterwards, which is what keeps the early-
        # stopping branch below in agreement across the group. A no-op without
        # one, so the single-device path is unchanged.
        running = all_reduce_mean(running / divisor, context)
        # The one synchronisation per epoch that the device-side accumulator
        # above exists to reduce the count of.
        mean_loss = float(running)
        history["train_loss"].append(mean_loss)
        # After the sync `.item()` forced, so this is compute rather than
        # queueing -- an unsynchronised clock on an asynchronous backend
        # measures how fast work was *submitted*.
        synchronize(operator.device)
        history["seconds_per_epoch"].append(time.perf_counter() - epoch_start)

        # The last epoch always reports, so a run never finishes without a
        # current number just because `epochs` is not a multiple of the
        # interval.
        reporting = ((epoch + 1) % eval_every == 0) or (epoch == epochs - 1)
        if not reporting:
            continue

        # Columns line up under `header` above; widths are shared with it.
        message = f"    {f'{epoch + 1}/{epochs}':>11s}  {mean_loss:>13.5f}"
        exhausted = False
        if validating:
            error = evaluate(operator, validation_loader,
                             loss=data_loss_name,
                             sobolev_weight=sobolev_weight)
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
                 f"(val {val_metric} {best_error:.5f})")

    if validating:
        history["best_epoch"] = best_epoch
        history["best_error"] = best_error
        history["stopped_early"] = stopped_early
        # Which norm `val_error` is in, so a figure or a report cannot label a
        # stored series by assuming it.
        history["val_metric"] = val_metric

    # Reported in the run's own JSON rather than sampled from outside it with
    # `nvidia-smi`, which is how every number in the CUDA work list had to be
    # obtained and cannot be attributed to one task of a multi-task run.
    if operator.device.type == "cuda":
        history["peak_vram_bytes"] = int(
            torch.cuda.max_memory_allocated(operator.device))
        history["peak_vram_reserved_bytes"] = int(
            torch.cuda.max_memory_reserved(operator.device))

    # Rank 0 writes the checkpoint just above; the others must not run ahead
    # into the next task and start reading a file that is still being written.
    barrier(context)
    return history


@torch.no_grad()
def evaluate(operator, loader, loss="relative_l2", sobolev_weight=0.0):
    r"""
    Mean held-out error over a loader, in *physical* units.

    Evaluating in physical units matters: a small error in a compressed
    (asinh) representation can hide a large error in the density itself.

    Parameters
    ----------
    operator : FieldOperator
    loader : torch.utils.data.DataLoader
    loss : str, optional
        Which of :data:`~poraque.ml.losses.DATA_LOSSES` to measure. Pass the
        run's own, so the number watched is the number optimised — the
        checkpoint and early stopping are decided on it.
    sobolev_weight : float, optional
        Gradient weight, for the two :math:`H^1` forms.

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

        # Delta-density mode: the reported error is on the **absolute**
        # density, not on the residual. A relative L2 on delta_rho has a
        # denominator some twenty times smaller, so quoting it would make every
        # delta-mode run look catastrophically worse than an absolute-mode one
        # while measuring the same physical error. The two numbers have to be
        # comparable or the ablation this mode exists for cannot be run.
        baseline = batch.get("baseline")
        if baseline is not None:
            baseline = baseline.to(operator.device)
            prediction = prediction + baseline
            physical_target = physical_target + baseline

        errors.append(data_error(prediction, physical_target, cell,
                                 loss=loss,
                                 sobolev_weight=sobolev_weight).cpu())

    return float(torch.cat(errors).mean()) if errors else float("nan")
