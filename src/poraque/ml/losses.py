# -*- coding: utf-8 -*-
# file: losses.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Data-fidelity losses for operator learning, plus the composite objective.

The default choice is the **relative** :math:`L^2` error rather than a plain
MSE. Fields from different materials differ in magnitude by orders of
magnitude; an absolute loss would let a handful of high-density systems
dominate the gradient and would silently make the reported error a function of
the dataset composition. Normalizing per sample makes every material count
equally and makes the loss value directly interpretable as a percentage.

:class:`SobolevLoss` adds a gradient term. Matching :math:`\nabla\rho` matters
because the von Weizsäcker kinetic energy — the dominant term in the
inhomogeneous regions the model must get right — depends on the gradient, not
on the density alone. A model with a small pointwise error but noisy
derivatives is useless downstream.
"""

import torch
import torch.nn as nn

from .physics import (
    electron_count_loss,
    euler_lagrange_loss,
    integrate,
    positivity_loss,
    spectral_gradient,
    von_weizsacker_bound_loss,
)


class RelativeL2Loss(nn.Module):
    r"""
    Per-sample relative :math:`L^2` error,
    :math:`\lVert \hat f - f\rVert_2 / \lVert f\rVert_2`.

    Parameters
    ----------
    epsilon : float, optional
        Floor on the target norm.
    reduction : {"mean", "sum", "none"}, optional
        Reduction over the batch.
    """

    def __init__(self, epsilon=1e-8, reduction="mean"):
        super().__init__()
        self.epsilon = float(epsilon)
        self.reduction = reduction

    def forward(self, prediction, target):
        """Compare ``(B, C, Nx, Ny, Nz)`` tensors."""
        difference = (prediction - target).flatten(1).norm(dim=1)
        scale = target.flatten(1).norm(dim=1).clamp_min(self.epsilon)
        ratio = difference / scale
        if self.reduction == "sum":
            return ratio.sum()
        if self.reduction == "none":
            return ratio
        return ratio.mean()


class AbsoluteL2Loss(nn.Module):
    r"""
    Per-sample **absolute** :math:`L^2` error,
    :math:`\lVert \hat f - f\rVert_2`.

    The unnormalised counterpart of :class:`RelativeL2Loss`, and the honest
    choice for a dataset whose materials are genuinely comparable in magnitude:
    it weights every *voxel* equally rather than every *material* equally, so a
    system with a larger field contributes proportionally more gradient.

    On a heterogeneous set that is usually wrong — fields differ by orders of
    magnitude between materials, and a handful of dense systems would dominate
    — which is why the relative form is the default. It is offered because
    "which of these two the optimiser stepped on" is a real question about a
    run, and it should be answerable from the config rather than inferred.

    Parameters
    ----------
    reduction : {"mean", "sum", "none"}, optional
        Reduction over the batch.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        """Compare ``(B, C, Nx, Ny, Nz)`` tensors."""
        norms = (prediction - target).flatten(1).norm(dim=1)
        if self.reduction == "sum":
            return norms.sum()
        if self.reduction == "none":
            return norms
        return norms.mean()


class SobolevLoss(nn.Module):
    r"""
    :math:`H^1` loss: values *and* spectral gradients.

    .. math::

        \mathcal{L} = \mathcal{L}_{L^2}
                    + w_{\rm Sob}\,\mathcal{L}_{L^2}[\nabla]

    in whichever of the two norms ``relative`` selects — both halves in the
    same one, so the weight is a pure ratio between values and derivatives and
    does not also have to absorb a change of scale.

    Matching :math:`\nabla\rho` matters because the von Weizsäcker kinetic
    energy — the dominant term in the inhomogeneous regions the model must get
    right — depends on the gradient, not on the density alone. A model with a
    small pointwise error but noisy derivatives is useless downstream.

    Parameters
    ----------
    weight : float, optional
        Weight of the gradient term relative to the value term
        (``training.sobolev_weight``).
    relative : bool, optional
        Normalise each term by the target's own norm (the default).
    epsilon : float, optional
        Floor on the target norms, used only when ``relative``.
    """

    def __init__(self, weight=0.1, relative=True, epsilon=1e-8):
        super().__init__()
        self.weight = float(weight)
        self.relative = bool(relative)
        self.epsilon = float(epsilon)
        self.value_loss = (RelativeL2Loss(epsilon) if relative
                           else AbsoluteL2Loss())

    def forward(self, prediction, target, cell):
        r"""
        Compare fields and their gradients; ``cell`` is ``(B, 3, 3)`` in Å.

        Every channel contributes its own gradient. This is a *data* term, not
        a physical constraint: whatever the operator predicts is part of the
        target and should be as smooth as the target is, so a spin-polarised
        run constrains :math:`\nabla m` alongside :math:`\nabla\rho` rather
        than leaving the magnetisation free. That is the opposite of what
        :func:`~poraque.ml.physics.von_weizsacker_tau` wants, which is a
        functional of :math:`\rho` alone and takes channel 0.
        """
        loss = self.value_loss(prediction, target)
        if self.weight > 0.0:
            loss = loss + self.weight * _gradient_error(
                prediction, target, cell, relative=self.relative,
                epsilon=self.epsilon).mean()
        return loss


def _gradient_error(prediction, target, cell, relative=True, epsilon=1e-8):
    """Per-sample :math:`L^2` error of the spectral gradients."""
    difference = (spectral_gradient(prediction, cell)
                  - spectral_gradient(target, cell)).flatten(1).norm(dim=1)
    if not relative:
        return difference
    scale = spectral_gradient(target, cell).flatten(1).norm(dim=1)
    return difference / scale.clamp_min(epsilon)


#: The objectives ``training.loss`` accepts, and what each one is.
#:
#: Two axes, named rather than implied: **absolute or relative** (is each
#: sample normalised by its own target norm?) and **L2 or H1** (are spatial
#: gradients part of the objective?). The old spelling had one axis explicit
#: and the other hidden — ``"sobolev"`` said "gradients" and said nothing about
#: the norm, and there was no way to ask for an unnormalised error at all.
#:
#: ``sobolev_weight`` scales the gradient term of the two :math:`H^1` forms and
#: is ignored by the two :math:`L^2` ones.
DATA_LOSSES = {
    "absolute_l2": {"relative": False, "gradient": False, "label": "abs L2"},
    "relative_l2": {"relative": True, "gradient": False, "label": "rel L2"},
    "absolute_h1": {"relative": False, "gradient": True, "label": "abs H1"},
    "relative_h1": {"relative": True, "gradient": True, "label": "rel H1"},
}


def resolve_data_loss(name):
    """
    The settings behind one of :data:`DATA_LOSSES`, by name.

    Raises
    ------
    ValueError
        Naming the four that exist. ``"sobolev"`` — the spelling this replaced
        — is called out on its own, because it had no norm in its name and the
        two possible readings of it are both offered now.
    """
    key = str(name).lower()
    if key in DATA_LOSSES:
        return DATA_LOSSES[key]
    hint = ""
    if key == "sobolev":
        hint = (" 'sobolev' was renamed: it named the gradient term and not "
                "the norm. Use 'relative_h1' for what it did, or "
                "'absolute_h1' for the unnormalised form.")
    raise ValueError(
        f"Unknown loss {name!r}; expected one of {sorted(DATA_LOSSES)}.{hint}")


def build_data_loss(name="relative_l2", sobolev_weight=0.1):
    """
    The data-fidelity module for one of :data:`DATA_LOSSES`.

    Parameters
    ----------
    name : str
    sobolev_weight : float, optional
        Gradient weight, used by the :math:`H^1` forms only.

    Returns
    -------
    nn.Module
    """
    spec = resolve_data_loss(name)
    if spec["gradient"]:
        return SobolevLoss(sobolev_weight, relative=spec["relative"])
    return RelativeL2Loss() if spec["relative"] else AbsoluteL2Loss()


def _total_density(field):
    r"""
    The channel carrying :math:`\rho`, dropping the magnetisation if present.

    A spin-polarised density is stored as :math:`(\rho, m)` and only the first
    channel integrates to the electron count — :math:`\int m` is the net
    moment. Charge conservation is a statement about the first alone.
    """
    return field[:, :1] if field.dim() == 5 and field.shape[1] > 1 else field


class PhysicsInformedLoss(nn.Module):
    r"""
    Composite objective: data fidelity plus physical constraints.

    .. math::

        \mathcal{L} = \mathcal{L}_{\rm data}
                    + \lambda_{N}\mathcal{L}_{N}
                    + \lambda_{+}\mathcal{L}_{+}
                    + \lambda_{\rm vW}\mathcal{L}_{\rm vW}
                    + \lambda_{\rm EL}\mathcal{L}_{\rm EL}

    Every physics weight defaults to **zero**, so the class reduces exactly to
    the supervised baseline until a term is switched on deliberately. That is
    intentional: physics terms should be introduced one at a time against a
    measured baseline, since a badly scaled constraint degrades accuracy while
    looking principled. See ``docs/notes/pi_fno.md`` for the recommended schedule.

    Parameters
    ----------
    task : str, optional
        ``"ext2chg"`` or ``"chg2tau"``; selects which constraints are
        applicable.
    data_loss : nn.Module, optional
        Fidelity term; defaults to :class:`RelativeL2Loss`.
    loss : str, optional
        Which data-fidelity term to use: one of :data:`DATA_LOSSES`.
    sobolev_weight : float, optional
        Gradient weight for the two :math:`H^1` forms; ignored by the others.
    electron_count_weight : float, optional
        Weight of the **charge-conservation** term :math:`\int\rho = N`
        (``ext2chg`` only). :math:`N` is taken from ``n_electrons`` when the
        caller knows it and otherwise from the integral of the reference
        density, which every batch carries — so the constraint needs no extra
        labels and works on an archive that publishes densities and nothing
        else.
    positivity_weight : float, optional
        Weight of the non-negativity penalty.
    von_weizsacker_weight : float, optional
        Weight of :math:`\tau \ge \tau_{\rm vW}` (``chg2tau`` only).
    euler_lagrange_weight : float, optional
        Weight of the OF-DFT stationarity residual (``ext2chg`` only).
    euler_lagrange_lambda : float, optional
        von Weizsäcker fraction in the kinetic functional used by that residual.
    physics_informed : {"auto", True, False}, optional
        Whether the constraints run at all. ``"auto"`` (the default) answers
        from the weights: physics is informed iff at least one of them is
        positive, which is what every existing caller already meant. ``True``
        **raises** when no weight is set — a run that asks for physics-informed
        training and gets the supervised baseline is the failure this flag
        exists to make impossible. ``False`` zeroes every weight, so a
        configured constraint is inert *and known to be*, rather than quietly
        contributing.

        The resolved answer is published as :attr:`physics_informed`, and
        :func:`~poraque.ml.training.train` reads it from there to decide
        whether to decode a prediction into physical units at all. That is the
        one place the decision can live without two layers disagreeing: the
        loop cannot ask a config it does not see, and a weight it inferred for
        itself would drift from the one the objective was built with.
    """

    def __init__(self, task="ext2chg", data_loss=None, loss="relative_l2",
                 sobolev_weight=0.1,
                 electron_count_weight=0.0, positivity_weight=0.0,
                 von_weizsacker_weight=0.0, euler_lagrange_weight=0.0,
                 euler_lagrange_lambda=1.0 / 9.0,
                 physics_informed="auto"):
        super().__init__()
        self.task = str(task)
        # Recorded so the training loop can label its validation column with
        # the norm it is actually measuring, without re-deriving it from a
        # config it does not see. `loss` names the objective; `metric_label`
        # names it for a human.
        spec = resolve_data_loss(loss)
        self.loss = str(loss).lower()
        self.metric_label = spec["label"]
        self.sobolev_weight = (float(sobolev_weight) if spec["gradient"]
                               else 0.0)
        self.data_loss = data_loss or build_data_loss(self.loss,
                                                      self.sobolev_weight)
        self.electron_count_weight = float(electron_count_weight)
        self.positivity_weight = float(positivity_weight)
        self.von_weizsacker_weight = float(von_weizsacker_weight)
        self.euler_lagrange_weight = float(euler_lagrange_weight)
        self.euler_lagrange_lambda = float(euler_lagrange_lambda)
        self.physics_informed = self._resolve_physics_informed(
            physics_informed)
        if not self.physics_informed:
            # Zeroed rather than merely skipped, so `physics_informed` and the
            # weights cannot tell a reader two different stories -- the log
            # header, the PDF report and `state()` all print the weights.
            self.electron_count_weight = 0.0
            self.positivity_weight = 0.0
            self.von_weizsacker_weight = 0.0
            self.euler_lagrange_weight = 0.0

    #: The four constraint weights, in the order they are applied.
    PHYSICS_WEIGHT_NAMES = ("electron_count_weight", "positivity_weight",
                            "von_weizsacker_weight", "euler_lagrange_weight")

    def _resolve_physics_informed(self, requested):
        """
        Turn ``"auto"`` / ``True`` / ``False`` into one boolean.

        Raises
        ------
        ValueError
            When ``True`` was asked for and no weight is positive. Silently
            training the supervised baseline under a flag that says otherwise
            is the one outcome this must not have: the loss curve is ordinary,
            the report says "physics-informed", and nothing anywhere is.
        """
        active = [name for name in self.PHYSICS_WEIGHT_NAMES
                  if getattr(self, name) > 0.0]
        if requested == "auto" or requested is None:
            return bool(active)
        if not isinstance(requested, bool):
            # `bool("off")` is True, and a config saying `off` would switch the
            # constraints on. Three spellings, and nothing that looks like a
            # fourth is guessed at.
            raise ValueError(
                f"physics_informed={requested!r} is not one of 'auto', true "
                f"or false.")
        if requested and not active:
            raise ValueError(
                "physics_informed is true but every constraint weight is "
                "zero, so the objective would be the supervised baseline "
                "under a name that says it is not. Set one of "
                + ", ".join(self.PHYSICS_WEIGHT_NAMES)
                + " in training.physics_informed_setup, or set "
                  "training.physics_informed: false.")
        return bool(requested)

    def forward(self, prediction, target, cell=None, physical_prediction=None,
                physical_input=None, physical_target=None, n_electrons=None):
        r"""
        Evaluate the composite loss.

        Parameters
        ----------
        prediction, target : torch.Tensor
            Fields in **normalized** units; the data term acts here.
        cell : torch.Tensor, optional
            ``(B, 3, 3)`` lattice vectors, Å. Required by every physics term.
        physical_prediction : torch.Tensor, optional
            Prediction decoded to physical units. Physics terms act on this —
            constraints are statements about physics, not about whatever
            normalization the training happens to use.
        physical_input : torch.Tensor, optional
            The network input in physical units: :math:`v_{\rm ext}` for
            ``ext2chg``, :math:`\rho` for ``chg2tau``.
        physical_target : torch.Tensor, optional
            The reference field in physical units. Supplies the electron count
            :math:`N = \int\rho\,d^3r` for the charge-conservation term when
            ``n_electrons`` is not given separately.
        n_electrons : torch.Tensor, optional
            Valence electron count per structure. Takes precedence over
            ``physical_target``: a count read from the pseudopotentials is
            exact, whereas one integrated from a downsampled reference carries
            that grid's error.

        Returns
        -------
        dict
            ``total`` plus each active component, for logging.
        """
        if isinstance(self.data_loss, SobolevLoss):
            if cell is None:
                raise ValueError("SobolevLoss requires `cell`.")
            total = self.data_loss(prediction, target, cell)
        else:
            total = self.data_loss(prediction, target)
        terms = {"data": total.detach()}

        # Two ways to reach the data term alone, and they mean different
        # things. `physics_informed` false is a decision -- the caller has
        # skipped decoding the prediction and there is nothing to evaluate a
        # constraint on. `physical_prediction is None` is the older contract,
        # kept because plenty of callers pass only what the data term needs.
        if not self.physics_informed or physical_prediction is None:
            return {"total": total, **terms}

        if self.positivity_weight > 0.0 and self.task in ("ext2chg", "chg2tau"):
            # Only rho is sign-constrained: the magnetisation channel of a
            # spin-polarised prediction is legitimately negative.
            value = positivity_loss(_total_density(physical_prediction))
            total = total + self.positivity_weight * value
            terms["positivity"] = value.detach()

        if self.task == "ext2chg":
            if self.electron_count_weight > 0.0:
                # The reference density is in every batch, so its integral is
                # always available as the target electron count. Requiring the
                # count to be supplied separately is what previously left this
                # term configured-but-inert on any dataset that ships densities
                # and no valence table -- which is every public archive.
                count = n_electrons
                predicted = _total_density(physical_prediction)
                if count is None and physical_target is not None:
                    count = integrate(_total_density(physical_target),
                                      cell).detach()
                if count is not None:
                    value = electron_count_loss(predicted, cell, count)
                    total = total + self.electron_count_weight * value
                    terms["electron_count"] = value.detach()

            if self.euler_lagrange_weight > 0.0 and physical_input is not None:
                value = euler_lagrange_loss(physical_prediction, physical_input,
                                            cell, lam=self.euler_lagrange_lambda)
                total = total + self.euler_lagrange_weight * value
                terms["euler_lagrange"] = value.detach()

        elif self.task == "chg2tau":
            if self.von_weizsacker_weight > 0.0 and physical_input is not None:
                value = von_weizsacker_bound_loss(physical_prediction,
                                                  physical_input, cell)
                total = total + self.von_weizsacker_weight * value
                terms["von_weizsacker"] = value.detach()

        return {"total": total, **terms}


@torch.no_grad()
def relative_error(prediction, target):
    """
    Per-sample relative :math:`L^2` error, for reporting.

    Returns
    -------
    torch.Tensor
        ``(B,)`` errors.
    """
    return RelativeL2Loss(reduction="none")(prediction, target)


def data_error(prediction, target, cell=None, loss="relative_l2",
               sobolev_weight=0.1, epsilon=1e-8):
    r"""
    Per-sample held-out error **in the norm the run is optimising**.

    The same combination :func:`build_data_loss` minimises, evaluated per
    sample rather than reduced — that identity is the point. A run should be
    *watched* on the quantity it is minimising, so the validation curve and the
    training curve describe the same thing and early stopping selects the model
    the objective prefers. Reporting a relative :math:`L^2` for an
    :math:`H^1` run means the checkpoint is chosen against a functional the
    optimiser never saw.

    Parameters
    ----------
    prediction, target : torch.Tensor
        ``(B, C, Nx, Ny, Nz)`` fields, in physical units for reporting.
    cell : torch.Tensor, optional
        ``(B, 3, 3)`` lattice vectors in Å. Required by the :math:`H^1` forms,
        whose gradient is spectral.
    loss : str, optional
        One of :data:`DATA_LOSSES`.
    sobolev_weight : float, optional
        Gradient weight, for the :math:`H^1` forms.
    epsilon : float, optional
        Floor on the target norms.

    Returns
    -------
    torch.Tensor
        ``(B,)`` errors.
    """
    spec = resolve_data_loss(loss)
    if spec["relative"]:
        value = RelativeL2Loss(epsilon, reduction="none")(prediction, target)
    else:
        value = AbsoluteL2Loss(reduction="none")(prediction, target)

    if not spec["gradient"] or sobolev_weight <= 0.0:
        return value
    if cell is None:
        raise ValueError(f"loss={loss!r} needs the cell: its gradient is "
                         f"spectral and cannot be taken without one.")
    return value + sobolev_weight * _gradient_error(
        prediction, target, cell, relative=spec["relative"], epsilon=epsilon)


