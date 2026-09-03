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
    train_set, val_set = data.split(0.8)
    # Fit normalizations on the training split only -- fitting before the
    # split would leak validation statistics into the transforms.
    val_set.input_transform, val_set.target_transform = \
        train_set.fit_transforms()

    operator = FieldOperator("ext2chg", width=32, modes=16, n_layers=4)
    train(operator, train_set, validation=val_set, epochs=200, batch_size=4)

    from poraque.fields import ExternalPotential
    rho = operator.predict(ExternalPotential.read("some/EXTCAR"))
    rho.write("some/CHGCAR_predicted")

Physics-informed training is available through
:class:`~poraque.ml.losses.PhysicsInformedLoss` and the differentiable DFT
operators in :mod:`poraque.ml.physics`; the accompanying technical plan is in
``docs/notes/pi_fno.md``.

PyTorch is imported lazily, so ``import poraque.ml`` costs nothing until a
name that needs it is first touched.
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
    "DistributedShapeBucketSampler": "poraque.ml.data",
    "collate_fields": "poraque.ml.data",
    "discover_materials": "poraque.ml.data",
    "make_dataloader": "poraque.ml.data",
    # multi-GPU
    "DistributedContext": "poraque.ml.distributed",
    # transforms
    "Asinh": "poraque.ml.transforms",
    "FieldTransform": "poraque.ml.transforms",
    "Identity": "poraque.ml.transforms",
    "Log": "poraque.ml.transforms",
    "Standardize": "poraque.ml.transforms",
    # model
    "CellEncoder": "poraque.ml.fno",
    "FNO3d": "poraque.ml.fno",
    "FNOBlock": "poraque.ml.fno",
    "SpectralConv3d": "poraque.ml.fno",
    "RadialSpectralConv3d": "poraque.ml.fno",
    # activations, incl. KAN-style learnable ones
    "ACTIVATIONS": "poraque.ml.kan",
    "KAN_ACTIVATIONS": "poraque.ml.kan",
    "BSplineKANActivation": "poraque.ml.kan",
    "ChebyKANActivation": "poraque.ml.kan",
    "RBFKANActivation": "poraque.ml.kan",
    "RationalKANActivation": "poraque.ml.kan",
    "build_activation": "poraque.ml.kan",
    "symbolic_expression": "poraque.ml.kan",
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
    "exact_kinetic_potential": "poraque.ml.physics",
    "exact_pauli_potential": "poraque.ml.physics",
    "xc_potential": "poraque.ml.physics",
    "xc_energy_density": "poraque.ml.physics",
    "XC_FUNCTIONALS": "poraque.ml.physics",
    "lda_exchange_potential": "poraque.ml.physics",
    "pw92_correlation_potential": "poraque.ml.physics",
    "AbsoluteL2Loss": "poraque.ml.losses",
    "DATA_LOSSES": "poraque.ml.losses",
    "PhysicsInformedLoss": "poraque.ml.losses",
    "RelativeL2Loss": "poraque.ml.losses",
    "SobolevLoss": "poraque.ml.losses",
    "build_data_loss": "poraque.ml.losses",
    "data_error": "poraque.ml.losses",
    # training
    "FieldOperator": "poraque.ml.training",
    "evaluate": "poraque.ml.training",
    "train": "poraque.ml.training",
    # fine-tuning
    "LIFTING_PREFIXES": "poraque.ml.training",
    "freeze_lifting_layers": "poraque.ml.training",
    "LORA_TARGETS": "poraque.ml.lora",
    "LoRAConv3d": "poraque.ml.lora",
    "apply_lora": "poraque.ml.lora",
    "is_adapted": "poraque.ml.lora",
    "load_lora_state_dict": "poraque.ml.lora",
    "lora_state_dict": "poraque.ml.lora",
    "BUNDLE_SUFFIX": "poraque.ml.training",
    "FINETUNED_BUNDLE_FILENAME": "poraque.ml.training",
    "LEGACY_BUNDLE_SUFFIXES": "poraque.ml.training",
    "resolve_bundle_path": "poraque.ml.training",
    # symbolic distillation
    "physics_constraints": "poraque.ml.symbolic",
    "physics_probes": "poraque.ml.symbolic",
    "FeatureTable": "poraque.ml.symbolic",
    "SymbolicDistiller": "poraque.ml.symbolic",
    "SymbolicResult": "poraque.ml.symbolic",
    "build_features": "poraque.ml.symbolic",
    "distill_dataset": "poraque.ml.symbolic",
    "expression_to_latex": "poraque.ml.symbolic",
    "native_engine": "poraque.ml.gp",
    "ConstrainedObjective": "poraque.ml.gp",
    "TreeFactory": "poraque.ml.gp",
    "sample_rows": "poraque.ml.symbolic",
    # query by committee
    "Committee": "poraque.ml.committee",
    "committee_spread": "poraque.ml.committee",
    "committee_integrals": "poraque.ml.committee",
    "disagreement_error_correlation": "poraque.ml.committee",
    "jensen_shannon_spread": "poraque.ml.committee",
    "jensen_shannon_divergence": "poraque.ml.committee",
    "probability_density": "poraque.ml.committee",
    # active learning
    "discover_pool": "poraque.ml.active_learning",
    "jsd_statistics": "poraque.ml.active_learning",
    "promote": "poraque.ml.active_learning",
    "run_round": "poraque.ml.active_learning",
    "score_candidate": "poraque.ml.active_learning",
    "score_pool": "poraque.ml.active_learning",
    "select_top_k": "poraque.ml.active_learning",
    # C backend for CPU inference
    "backend_available": "poraque.ml.backend",
    "backend_describe": "poraque.ml.backend",
    # unified checkpoint
    "BUNDLE_FILENAME": "poraque.ml.training",
    "BUNDLE_FORMAT": "poraque.ml.training",
    "bundle_tasks": "poraque.ml.training",
    "infer_backbone_kwargs": "poraque.ml.training",
    "load_bundle": "poraque.ml.training",
    "read_bundle": "poraque.ml.training",
    "save_bundle": "poraque.ml.training",
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
            f"`pip install torch`."
        ) from error

    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
