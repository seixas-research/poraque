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
                    (Identity, Standardize, Asinh, Log, SymmetricLog)}
        cls = registry[state.pop("type")]
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


class SymmetricLog(FieldTransform):
    r"""
    Sign-preserving log, :math:`y = \mathrm{sgn}(x)\log(1 + |x|/s)`.

    Useful for signed fields with heavy tails, e.g. the external potential when
    deep pseudopotential wells would otherwise dominate a standardized loss.
    """

    def __init__(self, scale=1.0):
        self.scale = float(scale) if abs(float(scale)) > 1e-30 else 1.0

    def forward(self, x):
        backend = torch if torch.is_tensor(x) else np
        return backend.sign(x) * backend.log1p(backend.abs(x) / self.scale)

    def inverse(self, y):
        backend = torch if torch.is_tensor(y) else np
        return backend.sign(y) * (backend.exp(backend.abs(y)) - 1.0) * self.scale


def _to_python(value):
    """Convert numpy scalars to plain Python for serialization."""
    return value.item() if isinstance(value, np.generic) else value


#: Sensible defaults per field type; see the module docstring for the reasoning.
DEFAULT_TRANSFORMS = {
    "EXTCAR": lambda values: Standardize.fit(values),
    "CHGCAR": lambda values: Asinh.fit(values),
    "TAUCAR": lambda values: Asinh.fit(values),
}
