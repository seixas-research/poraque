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


class SobolevLoss(nn.Module):
    r"""
    Relative :math:`H^1` loss: values *and* spectral gradients.

    Parameters
    ----------
    weight : float, optional
        Weight of the gradient term relative to the value term.
    epsilon : float, optional
        Floor on the target norms.
    """

    def __init__(self, weight=0.1, epsilon=1e-8):
        super().__init__()
        self.weight = float(weight)
        self.value_loss = RelativeL2Loss(epsilon)

    def forward(self, prediction, target, cell):
        """Compare fields and their gradients; ``cell`` is ``(B, 3, 3)`` in Å."""
        loss = self.value_loss(prediction, target)
        if self.weight > 0.0:
            predicted_gradient = spectral_gradient(prediction, cell)
            target_gradient = spectral_gradient(target, cell)
            difference = (predicted_gradient - target_gradient).flatten(1).norm(dim=1)
            scale = target_gradient.flatten(1).norm(dim=1).clamp_min(1e-8)
            loss = loss + self.weight * (difference / scale).mean()
        return loss


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
    sobolev_weight : float, optional
        If positive, use :class:`SobolevLoss` with this gradient weight.
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
    """

    def __init__(self, task="ext2chg", data_loss=None, sobolev_weight=0.0,
                 electron_count_weight=0.0, positivity_weight=0.0,
                 von_weizsacker_weight=0.0, euler_lagrange_weight=0.0,
                 euler_lagrange_lambda=1.0 / 9.0):
        super().__init__()
        self.task = str(task)
        self.sobolev_weight = float(sobolev_weight)
        self.data_loss = data_loss or (
            SobolevLoss(sobolev_weight) if sobolev_weight > 0 else RelativeL2Loss()
        )
        self.electron_count_weight = float(electron_count_weight)
        self.positivity_weight = float(positivity_weight)
        self.von_weizsacker_weight = float(von_weizsacker_weight)
        self.euler_lagrange_weight = float(euler_lagrange_weight)
        self.euler_lagrange_lambda = float(euler_lagrange_lambda)

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

        if physical_prediction is None:
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
