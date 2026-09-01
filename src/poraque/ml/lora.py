# -*- coding: utf-8 -*-
# file: lora.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Low-Rank Adaptation of a trained operator's dense layers.

Fine-tuning normally updates every weight, which needs optimiser state for
every weight and produces a checkpoint the size of the model. **LoRA** instead
freezes the trained weights and learns a low-rank correction beside them:

.. math::

    W' = W + \frac{\alpha}{r}\, B A ,
    \qquad A \in \mathbb{R}^{r \times d_{\rm in}},
    \quad B \in \mathbb{R}^{d_{\rm out} \times r} ,

with :math:`r \ll \min(d_{\rm in}, d_{\rm out})`. ``B`` is initialised to zero
and ``A`` to a small random matrix, so :math:`BA = 0` at step 0 and the adapted
model *is* the base model until training moves it — a fine-tune that starts
anywhere else has already discarded some of what it was adapting.

Where it is applied, and why only there
---------------------------------------
The **lifting and projection convolutions**, which are :math:`1\times1\times1`
and therefore ordinary dense maps over channels. They are the two ends of the
network: lifting embeds the input field into channel space, projection decodes
it back to physical units, and between them sit the spectral weights.

The spectral weights are deliberately left alone. They are complex tensors of
shape ``(width, width, m1, m2, m3)`` and hold ~99.8 % of the parameters, so
they look like the obvious target — but a rank-:math:`r` factorisation of a
5-index complex kernel is not one decomposition, it is a choice of which axes
to pair, and each choice is a different and unvalidated model. A LoRA that
adapted them would also stop being cheap, which is the point. Adapting the ends
and freezing the middle says something specific and defensible: *keep the
learned operator, re-fit how fields enter and leave it.*

.. warning::

   That is a real restriction, not a formality. A family whose *operator*
   differs from the base model's — not merely its input and output scales — is
   not something a LoRA on the two ends can reach, and the honest fix is a full
   fine-tune (``use_lora: false``). LoRA buys memory, not generality.
"""

import math

import torch
import torch.nn as nn

#: Submodules of :class:`~poraque.ml.fno.FNO3d` whose dense layers are adapted.
#: ``lift`` is a single ``Conv3d``; ``project`` is a ``Sequential`` whose two
#: ``Conv3d`` layers are both wrapped.
LORA_TARGETS = ("lift", "project")


class LoRAConv3d(nn.Module):
    r"""
    A frozen :math:`1\times1\times1` convolution plus a trainable low-rank term.

    Wraps the layer rather than rewriting it, so the base weight keeps its name
    inside the module tree and a checkpoint written before the adapter existed
    still loads into the thing being adapted.

    Parameters
    ----------
    base : torch.nn.Conv3d
        The trained layer. Its parameters are frozen in place.
    rank : int
        :math:`r`. Must be positive.
    alpha : float, optional
        Scaling numerator; the update is multiplied by ``alpha / rank``, which
        is what makes the learning rate roughly independent of the rank.
    dropout : float, optional
        Dropout on the adapter's input, off by default.
    """

    def __init__(self, base, rank, alpha=16.0, dropout=0.0):
        super().__init__()
        if not isinstance(base, nn.Conv3d):
            raise TypeError(f"LoRA wraps a Conv3d, got {type(base).__name__}.")
        if tuple(base.kernel_size) != (1, 1, 1):
            raise ValueError(
                f"LoRA is applied to the 1x1x1 (dense) convolutions, and this "
                f"one has kernel_size={tuple(base.kernel_size)}. A larger "
                f"kernel is a spatial operator, not a channel map, and "
                f"factorising it means something else.")
        if int(rank) <= 0:
            raise ValueError(f"lora_rank must be positive, got {rank!r}.")

        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Stored as plain matrices and applied with `conv3d`, so the adapter's
        # state dict holds two 2-D tensors per layer and nothing shaped like a
        # convolution kernel.
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base.in_channels))
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_channels, self.rank))
        # Kaiming on A, zeros on B: the product is exactly zero at step 0, so
        # the wrapped model reproduces the base model bit for bit before any
        # optimiser step. A non-zero start would silently perturb the weights
        # the fine-tune exists to preserve.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_channels(self):
        return self.base.in_channels

    @property
    def out_channels(self):
        return self.base.out_channels

    def forward(self, x):
        """``base(x) + (alpha / r) * B A x``, the second term channel-wise."""
        update = self.dropout(x)
        update = torch.nn.functional.conv3d(
            update, self.lora_A.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        update = torch.nn.functional.conv3d(
            update, self.lora_B.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        return self.base(x) + self.scaling * update

    def merged_weight(self):
        r"""
        The single equivalent convolution weight, :math:`W + \frac{\alpha}{r}BA`.

        For exporting an adapted model as an ordinary one. Not used in
        training: merging and then continuing would put the adapter's history
        into the frozen tensor.
        """
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.base.weight + delta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    def extra_repr(self):
        return (f"rank={self.rank}, alpha={self.alpha:g}, "
                f"scaling={self.scaling:g}")


def apply_lora(model, rank=8, alpha=16.0, dropout=0.0, targets=LORA_TARGETS):
    r"""
    Wrap a model's dense layers in adapters and freeze everything else.

    Every parameter of ``model`` is frozen first and the adapters are added
    afterwards, so "frozen except the adapters" is enforced rather than
    assumed: a layer this function does not know about cannot stay trainable by
    omission.

    Parameters
    ----------
    model : torch.nn.Module
        The backbone (:class:`~poraque.ml.fno.FNO3d`) or a head wrapping one.
    rank, alpha, dropout : see :class:`LoRAConv3d`
    targets : sequence of str, optional
        Attribute names to descend into.

    Returns
    -------
    dict
        ``{"adapters": int, "trainable": int, "frozen": int}`` — how many
        layers were wrapped and how many real parameters ended up on each side.
        Reported rather than assumed, because "it froze the base" is exactly
        the claim a caller should not have to take on trust.

    Raises
    ------
    ValueError
        If no layer was adapted. A LoRA fine-tune that silently wrapped nothing
        would train zero parameters and report a flat loss curve, which reads
        as a bad dataset rather than a misconfiguration.
    """
    backbone = getattr(model, "backbone", model)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    adapters = 0
    for name in targets:
        module = getattr(backbone, name, None)
        if module is None:
            continue
        if isinstance(module, nn.Conv3d):
            setattr(backbone, name, LoRAConv3d(module, rank, alpha, dropout))
            adapters += 1
        elif isinstance(module, nn.Sequential):
            for index, child in enumerate(module):
                if isinstance(child, nn.Conv3d):
                    module[index] = LoRAConv3d(child, rank, alpha, dropout)
                    adapters += 1

    if not adapters:
        raise ValueError(
            f"LoRA adapted no layer of {type(backbone).__name__}: none of "
            f"{list(targets)} is a 1x1x1 Conv3d or a Sequential holding one. "
            f"With nothing trainable the run would report a flat loss curve "
            f"and look like a data problem.")

    counts = parameter_counts(model)
    return {"adapters": adapters, **counts}


def parameter_counts(model):
    """
    ``{"trainable": int, "frozen": int}`` real-valued parameter counts.

    Counted the way :meth:`~poraque.ml.fno.FNO3d.n_parameters` counts — a
    complex weight is two real numbers — so the two sum to the total the rest
    of the log quotes rather than appearing to differ by half a model.
    """
    trainable = frozen = 0
    for parameter in model.parameters():
        size = parameter.numel() * (2 if parameter.is_complex() else 1)
        if parameter.requires_grad:
            trainable += size
        else:
            frozen += size
    return {"trainable": trainable, "frozen": frozen}


def lora_state_dict(model):
    """
    Only the adapter tensors, keyed as they are in the full state dict.

    This is what makes a LoRA checkpoint small: the frozen base is already on
    disk in the model being adapted, and storing it again per fine-tune is the
    cost LoRA exists to avoid.

    Returns
    -------
    dict
    """
    return {name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
            if ".lora_A" in name or ".lora_B" in name}


def load_lora_state_dict(model, state):
    """
    Load adapter tensors into an already-adapted model.

    Strict about *what it was given*, not about what the model has: a key in
    ``state`` with no home is a mismatch worth raising on, while an adapter the
    checkpoint does not mention keeps its zero-initialised value and so leaves
    that layer at the base weights.

    Raises
    ------
    KeyError
        Naming the keys the model has no parameter for.
    """
    own = dict(model.state_dict())
    unknown = sorted(set(state) - set(own))
    if unknown:
        raise KeyError(
            f"LoRA state has {len(unknown)} key(s) this model has no place "
            f"for: {unknown[:4]}{' ...' if len(unknown) > 4 else ''}. The "
            f"checkpoint was adapted with different targets or a different "
            f"rank.")
    model.load_state_dict(state, strict=False)
    return len(state)


def is_adapted(model):
    """Whether any :class:`LoRAConv3d` is present."""
    return any(isinstance(module, LoRAConv3d) for module in model.modules())
