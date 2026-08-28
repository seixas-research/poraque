# -*- coding: utf-8 -*-
# file: heads.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Output heads that enforce physical constraints **by construction**.

A soft penalty trades accuracy against constraint satisfaction and achieves
neither exactly. Where a constraint can be built into the parameterization it
should be: it then holds for every weight configuration, at initialization and
after any optimizer step, with no loss weight to tune and no inference cost.

:class:`PauliResidualOperator` implements the highest-value case for the
``CHGCAR -> TAUCAR`` task.

The Pauli decomposition
-----------------------
The kinetic energy density splits exactly into the von Weizsäcker term and the
Pauli term,

.. math::

    \tau(\mathbf r) \;=\; \underbrace{\frac{|\nabla\rho|^2}{8\rho}}_{\tau_{\rm vW}[\rho]}
    \;+\; \underbrace{\tau_P(\mathbf r)}_{\ge\,0},

where :math:`\tau_{\rm vW}` is a **closed-form functional of the input
density** and :math:`\tau_P \ge 0` by the Hoffmann-Ostenhof inequality. Writing
the network output as

.. math:: \tau_\theta \;=\; \tau_{\rm vW}[\rho] \;+\; s\,\mathrm{softplus}(f_\theta(\rho))

buys two things at once:

1. **The bound becomes structural.** ``softplus`` is non-negative everywhere,
   so :math:`\tau_\theta \ge \tau_{\rm vW}` identically — a theorem enforced by
   algebra rather than by
   :func:`~poraque.ml.physics.von_weizsacker_bound_loss`. The inequality is
   non-strict, exactly as Hoffmann-Ostenhof states it: for a strongly negative
   backbone output ``softplus`` underflows to zero and the head returns
   :math:`\tau = \tau_{\rm vW}` *exactly*. That is the correct one-orbital
   (nodeless) limit, reachable rather than merely approachable — a soft
   penalty can only ever approach it from above.
2. **The network stops re-deriving known physics.** On the Pt dataset
   :math:`\tau_{\rm vW}` accounts for ~31 % of :math:`\tau`; that fraction is
   now supplied analytically and exactly (spectral gradients on the plane-wave
   grid), leaving only the genuinely unknown Pauli term to learn. The residual
   :math:`\tau_P` is also the physically meaningful object — it is what
   distinguishes a fermionic system from a single-orbital one, and what every
   orbital-free KEDF is really trying to approximate.

Where the bound can fail
------------------------
The inequality is exact for *all-electron* densities. VASP's ``CHGCAR`` and
``TAUCAR`` are **pseudo** quantities, so it is not guaranteed a priori.
Measured on the present dataset it holds at every grid point of one structure
and at all but one point of the other — and that single point is where spectral
downsampling rang :math:`\tau` slightly negative, i.e. an artefact of the
resampling rather than of the physics. Verify with
:func:`pauli_bound_violation` before enabling the head on a new dataset; if a
pseudopotential family does violate it materially, use the soft penalty
instead.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .physics import von_weizsacker_tau
from .transforms import Identity


class PauliResidualOperator(nn.Module):
    r"""
    Wrap a backbone so it predicts only the Pauli term of :math:`\tau`.

    The module consumes and returns fields in the **normalized**
    representation, exactly like a bare backbone, so it is a drop-in
    replacement inside :class:`~poraque.ml.training.FieldOperator`. Internally
    it decodes its input to a physical density, evaluates
    :math:`\tau_{\rm vW}` exactly, adds the positive learned residual, and
    re-encodes.

    Parameters
    ----------
    backbone : torch.nn.Module
        Operator producing one unconstrained channel, e.g.
        :class:`~poraque.ml.fno.FNO3d`. It is called as
        ``backbone(x, cell)``.
    input_transform : FieldTransform
        Normalization applied to the density, used here to invert it. Must be
        the same object the dataset uses, or :math:`\tau_{\rm vW}` is computed
        from the wrong :math:`\rho`.
    target_transform : FieldTransform
        Normalization of :math:`\tau`; applied to the reconstructed field so
        the loss stays well-conditioned.
    scale : float, optional
        Magnitude of the Pauli term, in eV/Å³. Fit it with
        :func:`fit_pauli_scale` so the ``softplus`` operates near unit
        argument.
    learn_scale : bool, optional
        Optimize ``log(scale)`` alongside the backbone. Positivity is
        preserved regardless, since ``exp`` and ``softplus`` are both positive.

    Notes
    -----
    :math:`\tau_{\rm vW}[\rho]` is a fixed function of the *input*, so it
    contributes no gradient to the backbone parameters — it is a constant
    offset per sample, not a second trainable branch.
    """

    def __init__(self, backbone, input_transform=None, target_transform=None,
                 scale=1.0, learn_scale=True):
        super().__init__()
        self.backbone = backbone
        self.input_transform = input_transform or Identity()
        self.target_transform = target_transform or Identity()

        scale = float(scale)
        if scale <= 0.0:
            raise ValueError(f"scale must be positive, got {scale!r}.")
        log_scale = torch.tensor(float(np.log(scale)))
        if learn_scale:
            self.log_scale = nn.Parameter(log_scale)
        else:
            self.register_buffer("log_scale", log_scale)

    @property
    def scale(self):
        """Current Pauli-term scale in eV/Å³."""
        return torch.exp(self.log_scale)

    def decode_density(self, x):
        """Physical density (e/Å³) from the normalized input channel."""
        return self.input_transform.inverse(x)

    def pauli_term(self, x, cell):
        r"""
        The learned :math:`\tau_P \ge 0`, in eV/Å³.

        Exposed separately because it — not :math:`\tau` — is the object of
        physical interest and the one to inspect when diagnosing a model.
        """
        return F.softplus(self.backbone(x, cell)) * self.scale

    def von_weizsacker_term(self, x, cell):
        r""":math:`\tau_{\rm vW}[\rho]` for the current input, in eV/Å³."""
        return von_weizsacker_tau(self.decode_density(x), cell)

    def forward(self, x, cell=None):
        """
        Predict :math:`\\tau` in the normalized representation.

        Parameters
        ----------
        x : torch.Tensor
            ``(B, 1, Nx, Ny, Nz)`` normalized density.
        cell : torch.Tensor
            ``(B, 3, 3)`` lattice vectors in Å. Required: the von Weizsäcker
            term needs the metric to take a gradient.

        Returns
        -------
        torch.Tensor
            ``(B, 1, Nx, Ny, Nz)`` normalized :math:`\\tau`.
        """
        if cell is None:
            raise ValueError(
                "PauliResidualOperator requires `cell`: tau_vW is a gradient "
                "functional and needs the lattice metric."
            )
        physical = self.von_weizsacker_term(x, cell) + self.pauli_term(x, cell)
        return self.target_transform(physical)

    def n_parameters(self):
        """Total trainable parameters, including the backbone."""
        return sum(
            p.numel() * (2 if p.is_complex() else 1)
            for p in self.parameters() if p.requires_grad
        )

    def extra_repr(self):
        return f"scale={float(self.scale):.4g} eV/Ang^3"


# ---------------------------------------------------------------------- #
# Fitting and diagnostics
# ---------------------------------------------------------------------- #
def fit_pauli_scale(dataset, max_materials=8, quantile=0.5):
    r"""
    Estimate the typical magnitude of :math:`\tau_P` from training data.

    Initializing ``scale`` near the true magnitude keeps the ``softplus``
    argument close to unity, where its gradient is well behaved. A badly
    scaled head is slow to train even though it is always feasible.

    Parameters
    ----------
    dataset : FieldPairDataset
        A ``chg2tau`` dataset. Only the *training* split should be passed —
        fitting on held-out material leaks information.
    max_materials : int, optional
        Number of materials to sample.
    quantile : float, optional
        Quantile of :math:`\tau_P` to use; the median is robust to the sharp
        core peaks that would dominate a mean.

    Returns
    -------
    float
        Scale in eV/Å³, always positive.
    """
    from ..fields import von_weizsacker_tau as vw_numpy

    values = []
    for index in range(min(len(dataset), max_materials)):
        density, tau = dataset.load_fields(index)
        pauli = tau.data - vw_numpy(density.data, density.grid)
        values.append(np.quantile(pauli[np.isfinite(pauli)], quantile))

    scale = float(np.mean(values)) if values else 1.0
    return max(scale, 1e-6)


def pauli_bound_violation(dataset, max_materials=None):
    r"""
    Measure how often the reference data violates :math:`\tau \ge \tau_{\rm vW}`.

    Run this before enabling :class:`PauliResidualOperator` on a new dataset.
    A hard head cannot represent points below the bound, so a materially
    violated dataset must use the soft penalty
    (:func:`~poraque.ml.physics.von_weizsacker_bound_loss`) instead.

    Parameters
    ----------
    dataset : FieldPairDataset
        A ``chg2tau`` dataset.
    max_materials : int, optional
        Limit the number of materials inspected.

    Returns
    -------
    list of dict
        Per material: ``material``, ``points``, ``violations``,
        ``fraction``, ``worst_deficit`` (eV/Å³) and ``vw_fraction``, the share
        of :math:`\tau` the von Weizsäcker term already supplies.
    """
    from ..fields import von_weizsacker_tau as vw_numpy

    limit = len(dataset) if max_materials is None else min(len(dataset),
                                                           max_materials)
    report = []
    for index in range(limit):
        density, tau = dataset.load_fields(index)
        vw = vw_numpy(density.data, density.grid)
        pauli = tau.data - vw
        violated = pauli < 0

        report.append({
            "material": dataset.materials[index].identifier,
            "points": int(pauli.size),
            "violations": int(violated.sum()),
            "fraction": float(violated.mean()),
            "worst_deficit": float(pauli.min()),
            "vw_fraction": float(vw.mean() / tau.data.mean())
            if tau.data.mean() else float("nan"),
        })
    return report
