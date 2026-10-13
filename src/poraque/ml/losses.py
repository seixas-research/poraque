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

    The unnormalized counterpart of :class:`RelativeL2Loss`, and the honest
    choice for a dataset whose materials are genuinely comparable in magnitude:
    it weights every *voxel* equally rather than every *material* equally, so a
    system with a larger field contributes proportionally more gradient.

    On a heterogeneous set that is usually wrong — fields differ by orders of
    magnitude between materials, and a handful of dense systems would dominate
    — which is why the relative form is the default. It is offered because
    "which of these two the optimizer stepped on" is a real question about a
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
        Normalize each term by the target's own norm (the default).
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
        target and should be as smooth as the target is, so a spin-polarized
        run constrains :math:`\nabla m` alongside :math:`\nabla\rho` rather
        than leaving the magnetization free. That is the opposite of what
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


def _channel_norms(field):
    r"""
    ``(B, C)``: the :math:`L^2` norm of **each channel** of a field.

    A field with no channel axis is one channel, so the answer is always
    two-dimensional and the callers below never branch on the layout.
    """
    if field.dim() >= 5:
        return field.flatten(2).norm(dim=2)
    return field.flatten(1).norm(dim=1).unsqueeze(1)


def _gradient_channel_norms(field):
    r"""
    ``(B, C)``: the :math:`L^2` norm of each channel of a *gradient*.

    Separate from :func:`_channel_norms` because
    :func:`~poraque.ml.physics.spectral_gradient` does not keep a width-1
    channel axis: it returns ``(B, 3, Nx, Ny, Nz)`` for a single channel and
    ``(B, C, 3, Nx, Ny, Nz)`` only for :math:`C > 1`. Reducing the first of
    those as if axis 1 were the channel axis reads the three *spatial
    components* as three channels, which silently tripled the value term of a
    single-channel :math:`H^1` objective and handed the magnetization weight to
    two of the three derivatives.
    """
    if field.dim() == 6:
        return field.flatten(2).norm(dim=2)
    return field.flatten(1).norm(dim=1).unsqueeze(1)


#: How the two channels of a spin-polarized target enter the data term.
#:
#: ``joint`` is one norm over both channels at once --- every channel's error
#: divided by the whole :math:`(\rho, m)` target's norm and combined in
#: quadrature, which is exactly :math:`\lVert\delta(\rho,m)\rVert /
#: \lVert(\rho,m)\rVert`. ``per_channel`` divides each channel by **its own**
#: norm and sums the two with ``magnetization_weight``.
#:
#: ``joint`` is the default **because it was measured to be better**, which is
#: the opposite of what the arithmetic suggests. See :func:`channel_errors`.
SPIN_LOSS_MODES = ("joint", "per_channel")


def resolve_spin_loss(mode):
    """One of :data:`SPIN_LOSS_MODES`, or raise naming both."""
    key = str(mode).lower()
    if key not in SPIN_LOSS_MODES:
        raise ValueError(f"Unknown training.spin_loss {mode!r}; expected one "
                         f"of {list(SPIN_LOSS_MODES)}.")
    return key


def _channel_scales(norms, magnetization_floor, epsilon):
    r"""
    ``(B, C)`` denominator for each channel: **its own** norm, floored.

    Channel 0 is normalized by :math:`\lVert\rho\rVert`, which for a
    single-channel field is the whole target and reproduces the plain relative
    error exactly. Every further channel is normalized by

    .. math::

        \sqrt{\lVert m\rVert^2 + (\eta\lVert\rho\rVert)^2},

    its own norm softened by a floor :math:`\eta` (``magnetization_floor``)
    expressed as a fraction of the density's. The floor is not a safety hack,
    it is the physics of the quantity: :math:`m \equiv 0` *exactly* on 126 of
    the 432 CHON cells and sits in the numerical noise
    (:math:`\lVert m\rVert/\lVert\rho\rVert \le 10^{-4}`) on 239 more, so an
    unfloored per-channel relative error would divide by zero on a quarter of
    the set and chase rounding noise on half of it. Above the floor the term is
    the honest relative error of :math:`m`; below it, the term measures a
    spurious predicted moment against a fixed fraction of the density, which
    is what drives it to zero.

    The smooth form rather than ``max`` so the denominator is differentiable
    in the target --- it never is in practice, but a kink here would be a
    discontinuity in the reported error as a cell crosses the floor.
    """
    if norms.shape[1] == 1:
        return norms.clamp_min(epsilon)
    charge = norms[:, :1]
    floor = float(magnetization_floor) * charge
    rest = torch.sqrt(norms[:, 1:] ** 2 + floor ** 2)
    return torch.cat([charge, rest], dim=1).clamp_min(epsilon)


def channel_error_parts(prediction, target, cell=None, loss="relative_l2",
                        sobolev_weight=0.1, spin_loss="joint",
                        magnetization_floor=1e-2, epsilon=1e-8):
    r"""
    ``(value, gradient)``, each ``(B, C)`` or ``gradient`` ``None``.

    Kept apart because the :math:`H^1` total is **not** the per-channel sum of
    ``value + w\,gradient``: :class:`SobolevLoss` combines the channels of
    each part and only then adds them, so reproducing it bit for bit requires
    the parts separately. The reported columns add them per channel, which is
    the honest per-channel reading of the same quantities.
    """
    spec = resolve_data_loss(loss)
    mode = resolve_spin_loss(spin_loss)

    def scaled(difference, reference):
        """One part, divided by whichever denominator the mode calls for."""
        if not spec["relative"]:
            return difference
        if mode == "per_channel":
            scales = _channel_scales(reference, magnetization_floor, epsilon)
        else:
            # The joint norm of the reference, shared by every channel:
            # sqrt(sum_c ||ref_c||^2), which is the whole target's norm.
            scales = reference.norm(dim=1, keepdim=True).clamp_min(epsilon)
        return difference / scales

    value = scaled(_channel_norms(prediction - target),
                   _channel_norms(target))
    if not spec["gradient"] or sobolev_weight <= 0.0:
        return value, None
    if cell is None:
        raise ValueError(f"loss={loss!r} needs the cell: its gradient is "
                         f"spectral and cannot be taken without one.")
    reference = spectral_gradient(target, cell)
    gradient = scaled(
        _gradient_channel_norms(spectral_gradient(prediction, cell)
                                - reference),
        _gradient_channel_norms(reference))
    return value, gradient


def combine_channel_parts(value, gradient, sobolev_weight=0.1,
                          magnetization_weight=1.0, spin_loss="joint"):
    r"""
    The data term from :func:`channel_error_parts`: ``(B,)``.

    ``joint`` combines each part over the channels in quadrature and then adds
    them, which is identically :math:`\lVert\delta f\rVert/\lVert
    f\rVert + w\lVert\nabla\delta f\rVert/\lVert\nabla f\rVert` over the
    whole field. ``per_channel`` weights each channel's own total and sums.
    """
    mode = resolve_spin_loss(spin_loss)

    def over_channels(part):
        if part.shape[1] == 1:
            return part[:, 0]
        if mode == "joint":
            # `norm`, not `pow(2).sum().sqrt()`: the two agree in value but
            # not in gradient at the origin, where the explicit form is 0/0
            # and gives NaN. A prediction that is exactly right is not a
            # pathological input -- `tests/test_physics_informed.py` scores
            # one deliberately, to check that a physics term's gradient
            # survives a zero data term.
            return part.norm(dim=1)
        weights = torch.cat([
            part.new_ones(1),
            part.new_full((part.shape[1] - 1,), float(magnetization_weight)),
        ])
        return part @ weights

    total = over_channels(value)
    if gradient is not None:
        total = total + sobolev_weight * over_channels(gradient)
    return total


def channel_errors(prediction, target, cell=None, loss="relative_l2",
                   sobolev_weight=0.1, spin_loss="joint",
                   magnetization_floor=1e-2, epsilon=1e-8):
    r"""
    Per-sample error of each channel separately: ``(B, C)``.

    The error of a spin-polarized field is really two numbers --- one for
    :math:`\rho` and one for :math:`m` --- and the relative forms normalize
    **each by its own scale**, so each column is that channel's own relative
    error and neither can be made negligible by the other being large.

    That is the whole point, and it is worth being concrete about what the
    alternative did. A single norm over both channels, or two numbers sharing
    the whole target's norm as denominator, weights the magnetization by its
    share of :math:`\lVert(\rho, m)\rVert`. Measured on the 432-cell CHON set
    (``experiments/chon_ext2chg/results/final/magnetization_diagnosis.json``)
    that share is 5 % in the mean and :math:`3\times10^{-8}` in the *median*:
    a handful of strongly magnetic cells carry all of it and the typical cell
    carries none. Trained against that objective the operator did the only
    sensible thing and predicted :math:`m \approx 0` --- 1.1 % of the
    reference norm on the 46 cells with
    :math:`\lVert m\rVert/\lVert\rho\rVert > 10^{-2}`, while over-predicting
    by 55x on the noise-level ones, for a relative :math:`L^2` of 0.99 on
    every split. The magnetization was not badly learned; it was not in the
    objective.

    See :func:`_channel_scales` for the denominators and why the
    magnetization's carries a floor.

    Parameters
    ----------
    prediction, target : torch.Tensor
        ``(B, C, Nx, Ny, Nz)`` fields.
    cell : torch.Tensor, optional
        ``(B, 3, 3)`` lattice vectors in Å; required by the :math:`H^1` forms.
    loss : str, optional
        One of :data:`DATA_LOSSES`.
    sobolev_weight : float, optional
        Gradient weight, for the :math:`H^1` forms.
    magnetization_floor : float, optional
        :math:`\eta`, as a fraction of :math:`\lVert\rho\rVert`
        (``training.magnetization_floor``).
    epsilon : float, optional
        Floor on the denominators, used only by the relative forms.

    Returns
    -------
    torch.Tensor
        ``(B, C)``. Summing it with :func:`combine_channel_errors` gives the
        objective; its columns are the ``charge`` and ``mag`` numbers the
        training log prints, which are now the objective's own terms rather
        than a diagnostic computed beside it.
    """
    value, gradient = channel_error_parts(
        prediction, target, cell, loss=loss, sobolev_weight=sobolev_weight,
        spin_loss=spin_loss, magnetization_floor=magnetization_floor,
        epsilon=epsilon)
    if gradient is None:
        return value
    return value + sobolev_weight * gradient


def combine_channel_errors(errors, magnetization_weight=1.0,
                           spin_loss="joint"):
    r"""
    Weighted **sum** over channels of a ``(B, C)`` error: ``(B,)``.

    .. math::

        \mathcal{L} = \mathcal{L}[\rho] + w_m\,\mathcal{L}[m]

    A sum and not a quadrature, which is the whole change: in quadrature each
    term's gradient is scaled by its own share of the total, so the smaller
    channel is suppressed exactly when it needs attention. In a sum each
    channel's gradient is whatever its weight says, independent of how the
    other one is doing.

    Channel 0 carries weight 1 and every further channel carries
    ``magnetization_weight``, so the weight is a pure ratio between the
    density term and the magnetization term and the overall scale stays where
    it belongs, in the learning rate. A single-channel field is returned
    untouched, which makes an unpolarized run bit-identical to what it was
    before the channels were separated.

    Parameters
    ----------
    errors : torch.Tensor
        ``(B, C)`` as :func:`channel_errors` returns.
    magnetization_weight : float, optional
        Weight of every channel past the first (``training.
        magnetization_weight``). Zero trains the density alone and says so.

    Returns
    -------
    torch.Tensor
        ``(B,)``.
    """
    if errors.shape[1] == 1:
        return errors[:, 0]
    if resolve_spin_loss(spin_loss) == "joint":
        # Quadrature, which reconstitutes the single norm over both channels
        # exactly: ||d(rho, m)|| = sqrt(||d rho||^2 + ||d m||^2). The weight
        # does not apply --- there is one term, and the columns are its parts.
        return errors.pow(2).sum(dim=1).sqrt()
    weights = torch.cat([
        errors.new_ones(1),
        errors.new_full((errors.shape[1] - 1,), float(magnetization_weight)),
    ])
    return errors @ weights


class ChannelWeightedLoss(nn.Module):
    r"""
    The data-fidelity objective, with the channels weighted separately.

    One of :data:`DATA_LOSSES` evaluated per channel by
    :func:`channel_errors` and summed by :func:`combine_channel_errors`. For a
    single-channel field it *is* the corresponding plain loss, to the last bit;
    for a spin-polarized one it is the sum
    :math:`\mathcal{L}[\rho] + w_m \mathcal{L}[m]` in place of the single norm
    over both channels.

    Parameters
    ----------
    loss : str, optional
        One of :data:`DATA_LOSSES`.
    sobolev_weight : float, optional
        Gradient weight, for the :math:`H^1` forms.
    magnetization_weight : float, optional
        Weight of the magnetization term relative to the density term's 1.
    epsilon : float, optional
        Floor on the target norms.
    reduction : {"mean", "sum", "none"}, optional
        Reduction over the batch.
    """

    def __init__(self, loss="relative_l2", sobolev_weight=0.1,
                 magnetization_weight=1.0, spin_loss="joint",
                 magnetization_floor=1e-2, epsilon=1e-8, reduction="mean"):
        super().__init__()
        spec = resolve_data_loss(loss)
        self.loss = str(loss).lower()
        self.metric_label = spec["label"]
        self.sobolev_weight = (float(sobolev_weight) if spec["gradient"]
                               else 0.0)
        self.magnetization_weight = float(magnetization_weight)
        self.spin_loss = resolve_spin_loss(spin_loss)
        self.magnetization_floor = float(magnetization_floor)
        self.epsilon = float(epsilon)
        self.reduction = reduction

    def channel_errors(self, prediction, target, cell=None):
        """``(B, C)`` per-channel error, before the weights are applied."""
        return channel_errors(prediction, target, cell, loss=self.loss,
                              sobolev_weight=self.sobolev_weight,
                              spin_loss=self.spin_loss,
                              magnetization_floor=self.magnetization_floor,
                              epsilon=self.epsilon)

    def forward(self, prediction, target, cell=None):
        """Compare ``(B, C, Nx, Ny, Nz)`` tensors; ``cell`` is ``(B, 3, 3)``."""
        value, gradient = channel_error_parts(
            prediction, target, cell, loss=self.loss,
            sobolev_weight=self.sobolev_weight, spin_loss=self.spin_loss,
            magnetization_floor=self.magnetization_floor,
            epsilon=self.epsilon)
        combined = combine_channel_parts(
            value, gradient, self.sobolev_weight,
            self.magnetization_weight, self.spin_loss)
        if self.reduction == "sum":
            return combined.sum()
        if self.reduction == "none":
            return combined
        return combined.mean()


#: The objectives ``training.loss`` accepts, and what each one is.
#:
#: Two axes, named rather than implied: **absolute or relative** (is each
#: sample normalized by its own target norm?) and **L2 or H1** (are spatial
#: gradients part of the objective?). The old spelling had one axis explicit
#: and the other hidden — ``"sobolev"`` said "gradients" and said nothing about
#: the norm, and there was no way to ask for an unnormalized error at all.
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
                "'absolute_h1' for the unnormalized form.")
    raise ValueError(
        f"Unknown loss {name!r}; expected one of {sorted(DATA_LOSSES)}.{hint}")


def build_data_loss(name="relative_l2", sobolev_weight=0.1,
                    magnetization_weight=1.0, spin_loss="joint",
                    magnetization_floor=1e-2):
    """
    The data-fidelity module for one of :data:`DATA_LOSSES`.

    Always a :class:`ChannelWeightedLoss`, which reduces to the plain
    :class:`RelativeL2Loss`, :class:`AbsoluteL2Loss` or :class:`SobolevLoss`
    bit for bit on a single-channel field and splits the two channels of a
    spin-polarized one into separately weighted terms. The three plain classes
    remain, because the per-structure report measures a plain relative
    :math:`L^2` whatever the objective was.

    Parameters
    ----------
    name : str
    sobolev_weight : float, optional
        Gradient weight, used by the :math:`H^1` forms only.
    magnetization_weight : float, optional
        Weight of the magnetization term relative to the density term's 1;
        ignored by a single-channel target.

    Returns
    -------
    nn.Module
    """
    return ChannelWeightedLoss(name, sobolev_weight=sobolev_weight,
                               magnetization_weight=magnetization_weight,
                               spin_loss=spin_loss,
                               magnetization_floor=magnetization_floor)


def _total_density(field):
    r"""
    The channel carrying :math:`\rho`, dropping the magnetization if present.

    A spin-polarized density is stored as :math:`(\rho, m)` and only the first
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
        ``"ext2pseudo"``, ``"ext2chg"``, ``"pseudo2tau"`` or ``"chg2tau"``; selects which constraints are
        applicable.
    data_loss : nn.Module, optional
        Fidelity term; defaults to :class:`RelativeL2Loss`.
    loss : str, optional
        Which data-fidelity term to use: one of :data:`DATA_LOSSES`.
    sobolev_weight : float, optional
        Gradient weight for the two :math:`H^1` forms; ignored by the others.
    electron_count_weight : float, optional
        Weight of the **charge-conservation** term :math:`\int\rho = N`
        (``ext2pseudo`` and ``ext2chg``). :math:`N` is taken from ``n_electrons`` when the
        caller knows it and otherwise from the integral of the reference
        density, which every batch carries — so the constraint needs no extra
        labels and works on an archive that publishes densities and nothing
        else.
    positivity_weight : float, optional
        Weight of the non-negativity penalty.
    von_weizsacker_weight : float, optional
        Weight of :math:`\tau \ge \tau_{\rm vW}` (``pseudo2tau`` only).
    euler_lagrange_weight : float, optional
        Weight of the OF-DFT stationarity residual (``ext2pseudo`` only).
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

    def __init__(self, task="ext2pseudo", data_loss=None, loss="relative_l2",
                 sobolev_weight=0.1, magnetization_weight=1.0,
                 spin_loss="joint", magnetization_floor=1e-2,
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
        self.magnetization_weight = float(magnetization_weight)
        self.spin_loss = resolve_spin_loss(spin_loss)
        self.magnetization_floor = float(magnetization_floor)
        self.data_loss = data_loss or build_data_loss(
            self.loss, self.sobolev_weight, self.magnetization_weight,
            self.spin_loss, self.magnetization_floor)
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
                + " (in a configuration, a weight under training.physics), "
                  "or pass physics_informed='auto'.")
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
            ``ext2pseudo``, :math:`\rho` for ``pseudo2tau``.
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
        if isinstance(self.data_loss, (SobolevLoss, ChannelWeightedLoss)):
            if cell is None and self.sobolev_weight > 0.0:
                raise ValueError(
                    f"{type(self.data_loss).__name__} requires `cell`: the "
                    f"gradient term of {self.loss!r} is spectral.")
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

        # tau >= 0 holds for the pseudo kinetic energy density of chg2tau as
        # for pseudo2tau. The von Weizsacker bound below is left to pseudo2tau: a
        # PAW pseudo-density carries the compensation charge, which no pseudo
        # orbital's kinetic energy is bounded by.
        if self.positivity_weight > 0.0 and self.task in ("ext2pseudo", "pseudo2tau",
                                                           "chg2tau"):
            # Only rho is sign-constrained: the magnetization channel of a
            # spin-polarized prediction is legitimately negative.
            value = positivity_loss(_total_density(physical_prediction))
            total = total + self.positivity_weight * value
            terms["positivity"] = value.detach()

        if self.task in ("ext2pseudo", "ext2chg"):
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

        if self.task == "ext2pseudo":
            if self.euler_lagrange_weight > 0.0 and physical_input is not None:
                value = euler_lagrange_loss(physical_prediction, physical_input,
                                            cell, lam=self.euler_lagrange_lambda)
                total = total + self.euler_lagrange_weight * value
                terms["euler_lagrange"] = value.detach()

        elif self.task == "pseudo2tau":
            if self.von_weizsacker_weight > 0.0 and physical_input is not None:
                value = von_weizsacker_bound_loss(physical_prediction,
                                                  physical_input, cell)
                total = total + self.von_weizsacker_weight * value
                terms["von_weizsacker"] = value.detach()

        return {"total": total, **terms}


    def channel_errors(self, prediction, target, cell=None):
        r"""
        ``(B, C)`` per-channel data error, the terms the objective is made of.

        What the training log prints as ``charge`` and ``mag``. They used to be
        computed beside the objective, by evaluating it with the prediction
        substituted into one channel and the reference left in the other; now
        they *are* its two terms, so the columns and the number stepped on
        cannot drift apart and the substitution's extra forward evaluation is
        gone.
        """
        if isinstance(self.data_loss, ChannelWeightedLoss):
            return self.data_loss.channel_errors(prediction, target, cell)
        # A caller-supplied data term of its own: fall back to the objective's
        # named settings, which is what the diagnostic always used.
        return channel_errors(prediction, target, cell, loss=self.loss,
                              sobolev_weight=self.sobolev_weight,
                              spin_loss=self.spin_loss,
                              magnetization_floor=self.magnetization_floor)


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
               sobolev_weight=0.1, magnetization_weight=1.0,
               spin_loss="joint", magnetization_floor=1e-2, epsilon=1e-8):
    r"""
    Per-sample held-out error **in the norm the run is optimizing**.

    The same combination :func:`build_data_loss` minimizes, evaluated per
    sample rather than reduced — that identity is the point. A run should be
    *watched* on the quantity it is minimizing, so the validation curve and the
    training curve describe the same thing and early stopping selects the model
    the objective prefers. Reporting a relative :math:`L^2` for an
    :math:`H^1` run means the checkpoint is chosen against a functional the
    optimizer never saw.

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
    magnetization_weight : float, optional
        Weight of the magnetization term relative to the density term's 1, for
        a spin-polarized field. Pass the run's own, for the same reason the
        norm is passed: the checkpoint is chosen on this number, so a weight
        here that differs from the objective's would select against a
        functional the optimizer never saw.
    epsilon : float, optional
        Floor on the target norms.

    Returns
    -------
    torch.Tensor
        ``(B,)`` errors.
    """
    value, gradient = channel_error_parts(
        prediction, target, cell, loss=loss, sobolev_weight=sobolev_weight,
        spin_loss=spin_loss, magnetization_floor=magnetization_floor,
        epsilon=epsilon)
    return combine_channel_parts(value, gradient, sobolev_weight,
                                 magnetization_weight, spin_loss)

