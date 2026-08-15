# -*- coding: utf-8 -*-
# file: transforms.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Invertible normalizations for 3D physical fields.

Getting this layer right matters more than usual here, because the three fields
have wildly different statistics:

``EXTCAR``
    Roughly symmetric about zero (the :math:`\mathbf{G}=0` component is removed
    by construction) but with deep, narrow wells at the ionic sites. A plain
    standardization is appropriate; the tails are the physics.

``CHGCAR`` and ``TAUCAR``
    Strictly positive and spanning several orders of magnitude between
    interstitial voids and core regions. Standardizing them lets a handful of
    near-core points dominate the loss, and nothing constrains the prediction
    to stay positive. An :class:`Asinh` (or :class:`Log`) compression fixes
    both: it is smooth at the origin, behaves logarithmically in the tail, and
    is exactly invertible so predictions can be pushed back to physical units.

Every transform is invertible and differentiable, so a physics loss can be
evaluated in *physical* units on the decoded prediction while the network still
trains on well-conditioned targets.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch


class FieldTransform(ABC):
    """Base class for invertible, differentiable field normalizations."""

    @abstractmethod
    def forward(self, x):
        """Map physical values to normalized values."""

    @abstractmethod
    def inverse(self, y):
        """Map normalized values back to physical units."""

    def __call__(self, x):
        return self.forward(x)

    def state_dict(self):
        """
        Serializable description of the transform.

        Only public attributes are stored: they are exactly the constructor
        arguments, so :meth:`from_state_dict` can rebuild the object. Derived
        quantities (``_norm`` and friends) are recomputed on construction.
        """
        state = {"type": type(self).__name__}
        state.update({k: _to_python(v) for k, v in vars(self).items()
                      if not k.startswith("_")})
        return state

    @staticmethod
    def from_state_dict(state):
        """Rebuild a transform from :meth:`state_dict`."""
        state = dict(state)
        registry = {cls.__name__: cls for cls in
                    (Identity, Standardize, Asinh, Log, Channelwise)}
        cls = registry[state.pop("type")]
        if cls is Channelwise:
            return cls([FieldTransform.from_state_dict(entry)
                        for entry in state["transforms"]])
        return cls(**state)

    def __repr__(self):
        fields = ", ".join(f"{k}={_to_python(v)!r}" for k, v in vars(self).items()
                           if not k.startswith("_"))
        return f"{type(self).__name__}({fields})"


class Identity(FieldTransform):
    """No-op transform."""

    def forward(self, x):
        return x

    def inverse(self, y):
        return y


class Standardize(FieldTransform):
    r"""
    Affine normalization :math:`y = (x - \mu)/\sigma`.

    Parameters
    ----------
    mean, std : float
        Statistics to remove.
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = float(mean)
        self.std = float(std) if abs(float(std)) > 1e-30 else 1.0

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, y):
        return y * self.std + self.mean

    @classmethod
    def fit(cls, values):
        """Fit to a sample of values."""
        values = np.asarray(values, dtype=float)
        return cls(float(values.mean()), float(values.std()))


class Asinh(FieldTransform):
    r"""
    Compression :math:`y = \mathrm{asinh}(x/s) / \mathrm{asinh}(1/s)`.

    Linear for :math:`|x| \ll s`, logarithmic beyond, defined for both signs
    and smooth at the origin — the transform of choice for densities that span
    many decades yet must remain differentiable at zero.

    Parameters
    ----------
    scale : float
        Cross-over value ``s`` in physical units.
    """

    def __init__(self, scale=1.0):
        self.scale = float(scale) if abs(float(scale)) > 1e-30 else 1.0
        self._norm = float(np.arcsinh(1.0 / self.scale))

    def forward(self, x):
        backend = torch if torch.is_tensor(x) else np
        return backend.arcsinh(x / self.scale) / self._norm

    def inverse(self, y):
        backend = torch if torch.is_tensor(y) else np
        return backend.sinh(y * self._norm) * self.scale

    @classmethod
    def fit(cls, values, quantile=0.5):
        """
        Choose ``scale`` as a quantile of ``|values|``.

        The median puts half the grid points in the linear regime and half in
        the logarithmic one, which balances the loss between bulk and core.
        """
        values = np.abs(np.asarray(values, dtype=float))
        scale = float(np.quantile(values[np.isfinite(values)], quantile))
        return cls(max(scale, 1e-12))


class Log(FieldTransform):
    r"""
    Strictly positive compression :math:`y = \log(x + \epsilon)`.

    :meth:`inverse` returns ``exp(y) - epsilon``, which **guarantees a
    prediction above** ``-epsilon``: positivity of the density is enforced by
    the parameterization instead of by a penalty.

    Parameters
    ----------
    epsilon : float
        Floor added before taking the logarithm.
    """

    def __init__(self, epsilon=1e-6):
        self.epsilon = float(epsilon)

    def forward(self, x):
        backend = torch if torch.is_tensor(x) else np
        return backend.log(backend.clip(x, self.epsilon, None) + self.epsilon)

    def inverse(self, y):
        backend = torch if torch.is_tensor(y) else np
        return backend.exp(y) - self.epsilon


class Channelwise(FieldTransform):
    r"""
    One transform per channel, applied along the channel axis.

    Needed as soon as a field has more than one channel whose components live
    on different scales. The spin-polarised density is the motivating case:
    :math:`\rho` integrates to hundreds of electrons while :math:`m` integrates
    to a few :math:`\mu_B`, so a single :class:`Asinh` scale fitted on both at
    once lands in between and normalizes neither. The magnetisation would
    arrive at the network an order of magnitude below unity, contribute almost
    nothing to the loss, and be learned as approximately zero — which is a
    plausible-looking answer for a weakly magnetic system and wrong for
    everything else.

    Parameters
    ----------
    transforms : sequence of FieldTransform
        One per channel, in channel order.

    Notes
    -----
    The channel axis is taken as ``-4``, which is the same axis for an
    unbatched ``(C, X, Y, Z)`` sample and a batched ``(B, C, X, Y, Z)`` one.
    Counting from the right avoids having to know which of the two is being
    handed over — the dataset produces the first and the model consumes the
    second.
    """

    def __init__(self, transforms):
        self.transforms = list(transforms)
        if not self.transforms:
            raise ValueError("Channelwise needs at least one transform.")

    def _apply(self, values, method):
        backend = torch if torch.is_tensor(values) else np
        if values.shape[-4] != len(self.transforms):
            raise ValueError(
                f"Channelwise has {len(self.transforms)} transforms but the "
                f"field has {values.shape[-4]} channels."
            )
        parts = [getattr(transform, method)(values[..., index, :, :, :])
                 for index, transform in enumerate(self.transforms)]
        return backend.stack(parts, axis=-4) if backend is np else \
            backend.stack(parts, dim=-4)

    def forward(self, x):
        return self._apply(x, "forward")

    def inverse(self, y):
        return self._apply(y, "inverse")

    def state_dict(self):
        return {"type": "Channelwise",
                "transforms": [t.state_dict() for t in self.transforms]}

    def __repr__(self):
        return f"Channelwise({self.transforms!r})"


def _to_python(value):
    """Convert numpy scalars to plain Python for serialization."""
    return value.item() if isinstance(value, np.generic) else value


#: Sensible defaults per field type; see the module docstring for the reasoning.
DEFAULT_TRANSFORMS = {
    "EXTCAR": lambda values: Standardize.fit(values),
    "CHGCAR": lambda values: Asinh.fit(values),
    "TAUCAR": lambda values: Asinh.fit(values),
}
