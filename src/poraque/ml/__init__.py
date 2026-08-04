# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Neural-operator learning between the 3D scalar fields of :mod:`poraque.fields`.

Two mappings are supported (see :mod:`poraque.ml.tasks`):

``ext2chg``
    :math:`V_{\rm ext} \mapsto \rho` — the Hohenberg-Kohn map.
``chg2tau``
    :math:`\rho \mapsto \tau` — the kinetic energy density functional.

The architecture is a Fourier Neural Operator, chosen because its convolution
is periodic by construction (matching a crystal unit cell) and because its
learned weights live in Fourier-mode space, which lets **one model serve
materials whose grids differ in shape** — the central constraint here, since
``ENCUT`` and cell size fix ``NGXF, NGYF, NGZF`` per material.

Quick start::

    from poraque.ml import FieldPairDataset, FieldOperator, train

    data = FieldPairDataset("dataset_root", task="ext2chg")
    data.fit_transforms()
    train_set, val_set = data.split(0.8)

    operator = FieldOperator("ext2chg", width=32, modes=16, n_layers=4)
    train(operator, train_set, validation=val_set, epochs=200, batch_size=4)

    from poraque.fields import ExternalPotential
    rho = operator.predict(ExternalPotential.read("some/EXTCAR"))
    rho.write("some/CHGCAR_predicted")

Physics-informed training is available through
:class:`~poraque.ml.losses.PhysicsInformedLoss` and the differentiable DFT
operators in :mod:`poraque.ml.physics`; the accompanying technical plan is in
``docs/notes/pi_fno.md``.

PyTorch is an optional dependency (``pip install poraque[ml]``); it is imported
lazily, so ``import poraque.ml`` fails with a clear message rather than at
package import time.
"""

_LAZY = {
    # tasks
    "TaskSpec": "poraque.ml.tasks",
    "TASKS": "poraque.ml.tasks",
    "resolve_task": "poraque.ml.tasks",
    "EXT_TO_CHG": "poraque.ml.tasks",
    "CHG_TO_TAU": "poraque.ml.tasks",
    # data
    "FieldPairDataset": "poraque.ml.data",
    "ShapeBucketSampler": "poraque.ml.data",
    "collate_fields": "poraque.ml.data",
    "discover_materials": "poraque.ml.data",
    "make_dataloader": "poraque.ml.data",
    # transforms
    "Asinh": "poraque.ml.transforms",
    "FieldTransform": "poraque.ml.transforms",
    "Identity": "poraque.ml.transforms",
    "Log": "poraque.ml.transforms",
    "Standardize": "poraque.ml.transforms",
    "SymmetricLog": "poraque.ml.transforms",
    # model
    "CellEncoder": "poraque.ml.fno",
    "FNO3d": "poraque.ml.fno",
    "FNOBlock": "poraque.ml.fno",
    "SpectralConv3d": "poraque.ml.fno",
    # constraint-enforcing output heads
    "PauliResidualOperator": "poraque.ml.heads",
    "fit_pauli_scale": "poraque.ml.heads",
    "pauli_bound_violation": "poraque.ml.heads",
    # losses
    # differentiable physics
    "functional_derivative": "poraque.ml.physics",
    "kinetic_potential": "poraque.ml.physics",
    "operator_kinetic_potential": "poraque.ml.physics",
    "euler_lagrange_residual": "poraque.ml.physics",
    "PhysicsInformedLoss": "poraque.ml.losses",
    "RelativeL2Loss": "poraque.ml.losses",
    "SobolevLoss": "poraque.ml.losses",
    # training
    "FieldOperator": "poraque.ml.training",
    "evaluate": "poraque.ml.training",
    "train": "poraque.ml.training",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    """Resolve public names on first use, so PyTorch is imported only if needed."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    try:
        module = import_module(module_name)
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            f"poraque.ml.{name} requires PyTorch. Install it with "
            f"`pip install poraque[ml]` or `pip install torch`."
        ) from error

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
