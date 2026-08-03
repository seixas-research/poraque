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
    **model_kwargs
        Forwarded to :class:`~poraque.ml.fno.FNO3d`.
    """

    def __init__(self, task, model=None, input_transform=None,
                 target_transform=None, device=None, pauli_residual=False,
                 pauli_scale=1.0, learn_pauli_scale=True, **model_kwargs):
        self.task = resolve_task(task)
        self.device = resolve_device(device)
        self.input_transform = input_transform or Identity()
        self.target_transform = target_transform or Identity()

        self.pauli_residual = bool(pauli_residual)
        self.pauli_scale = float(pauli_scale)
        self.learn_pauli_scale = bool(learn_pauli_scale)

        backbone = model if model is not None else FNO3d(**model_kwargs)
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
        ScalarField
            An instance of the task's target class, in physical units.
        """
        self.model.eval()
        values = torch.as_tensor(np.ascontiguousarray(field.data),
                                 dtype=torch.float32, device=self.device)
        cell = torch.as_tensor(field.grid.cell, dtype=torch.float32,
                               device=self.device).unsqueeze(0)

        normalized = self.input_transform(values).unsqueeze(0).unsqueeze(0)
        prediction = self.model(normalized, cell)
        physical = self.target_transform.inverse(prediction)[0, 0]

        # .float() before .numpy(): accelerators may hand back a dtype numpy
        # cannot consume directly, and .cpu() alone does not convert it.
        return _OUTPUT_CLASSES[self.task.target_field](
            physical.detach().to("cpu", torch.float32).numpy(),
            field.grid, field.structure,
            metadata={"predicted_by": type(self.model).__name__,
                      "task": self.task.name,
                      "device": str(self.device)},
        )

    @property
    def device_description(self):
        """Human-readable description of the active device."""
        return describe_device(self.device)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path):
        """
        Save weights, normalizations and the head configuration to ``path``.

        The head flags are stored because they change the *architecture*, not
        just the weights: restoring a Pauli-residual model into a bare backbone
        would silently load mismatched tensors.
        """
        torch.save(
            {
                "task": self.task.name,
                "model_state": self.model.state_dict(),
                "model_class": type(self.model).__name__,
                "input_transform": self.input_transform.state_dict(),
                "target_transform": self.target_transform.state_dict(),
                "pauli_residual": self.pauli_residual,
                "pauli_scale": self.pauli_scale,
                "learn_pauli_scale": self.learn_pauli_scale,
            },
            path,
        )
        return str(path)

    @classmethod
    def load(cls, path, model=None, device=None, **model_kwargs):
        """
        Restore an operator saved by :meth:`save`.

        The head configuration is taken from the checkpoint, so a
        Pauli-residual model reloads as one without the caller having to
        remember. ``**model_kwargs`` must still reproduce the backbone's
        hyper-parameters.

        Parameters
        ----------
        path : str
            Checkpoint file.
        model : torch.nn.Module, optional
            Pre-built **backbone** with matching architecture. When omitted an
            :class:`~poraque.ml.fno.FNO3d` is built from ``**model_kwargs``.
        device : str or torch.device, optional
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        operator = cls(
            state["task"], model=model, device=device,
            input_transform=FieldTransform.from_state_dict(state["input_transform"]),
            target_transform=FieldTransform.from_state_dict(state["target_transform"]),
            pauli_residual=state.get("pauli_residual", False),
            pauli_scale=state.get("pauli_scale", 1.0),
            learn_pauli_scale=state.get("learn_pauli_scale", True),
            **model_kwargs,
        )
        operator.model.load_state_dict(state["model_state"])
        return operator


def train(operator, dataset, epochs=100, batch_size=1, learning_rate=1e-3,
          weight_decay=1e-4, validation=None, loss=None, scheduler="cosine",
          grad_clip=1.0, log_every=1, checkpoint=None, seed=0, verbose=True):
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
        Held-out materials; evaluated each epoch.
    loss : nn.Module, optional
        Objective; defaults to a supervised
        :class:`~poraque.ml.losses.PhysicsInformedLoss` for the task.
    scheduler : {"cosine", None}, optional
        Learning-rate schedule.
    grad_clip : float, optional
        Gradient-norm clipping; ``0`` disables. Spectral layers can produce
        large gradients early in training, so this is on by default.
    log_every : int, optional
        Epoch interval for logging.
    checkpoint : str, optional
        Path to save the best-validation model to.
    seed : int, optional
        RNG seed.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    dict
        History with ``train_loss`` and, when validating, ``val_error``.
    """
    torch.manual_seed(seed)
    criterion = loss or PhysicsInformedLoss(task=operator.task.name)
    optimizer = torch.optim.AdamW(operator.model.parameters(), lr=learning_rate,
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

    history = {"train_loss": [], "val_error": []}
    best_error = float("inf")

    for epoch in range(int(epochs)):
        if hasattr(loader.batch_sampler, "set_epoch"):
            loader.batch_sampler.set_epoch(epoch)

        operator.model.train()
        running, batches = 0.0, 0
        for batch in loader:
            inputs = batch["input"].to(operator.device)
            targets = batch["target"].to(operator.device)
            cell = batch["cell"].to(operator.device)

            optimizer.zero_grad(set_to_none=True)
            prediction = operator.model(inputs, cell)

            terms = criterion(
                prediction, targets, cell=cell,
                physical_prediction=operator.target_transform.inverse(prediction),
                physical_input=batch["target_physical"].to(operator.device)
                if operator.task.name == "chg2tau" else
                operator.input_transform.inverse(inputs),
            )
            terms["total"].backward()

            if grad_clip:
                clip_gradients(operator.model.parameters(), grad_clip)
            optimizer.step()

            running += float(terms["total"].detach())
            batches += 1

        if lr_schedule is not None:
            lr_schedule.step()

        mean_loss = running / max(batches, 1)
        history["train_loss"].append(mean_loss)

        message = f"epoch {epoch + 1:4d}/{epochs}  train {mean_loss:.5f}"
        if validation_loader is not None:
            error = evaluate(operator, validation_loader)
            history["val_error"].append(error)
            message += f"  val_rel_L2 {error:.5f}"
            if checkpoint and error < best_error:
                best_error = error
                operator.save(checkpoint)
                message += "  *"

        if verbose and (epoch % log_every == 0 or epoch == epochs - 1):
            print(message)

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
