# -*- coding: utf-8 -*-
# file: tasks.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The regression tasks the neural operator is trained on.

Two maps are in scope, and they are the two links of the orbital-free DFT
chain rather than an arbitrary pair:

``ext2chg`` — :math:`V_{\rm ext} \mapsto \rho`
    The Hohenberg-Kohn map. Its existence and uniqueness is a *theorem*: for a
    given electron count the ground-state density is a functional of the
    external potential alone. Learning it is therefore learning a well-posed
    object, not fitting a correlation.

``chg2tau`` — :math:`\rho \mapsto \tau`
    The kinetic energy density functional, the missing ingredient of practical
    OF-DFT. Unlike the first map this one is *semi-local in character* — the
    exact :math:`\tau` is bounded below by the von Weizsäcker form and
    approaches Thomas-Fermi in the slowly-varying limit — which gives the
    physics losses of :mod:`poraque.ml.physics` firm anchors.

The two share one architecture and one dataset layout; only the endpoints of
the map differ.

A third task sits beside the chain rather than in it:

``ext2paw`` — :math:`V_{\rm ext} \mapsto (\tilde\rho, \rho^a_{ij})`
    The pseudo-density *and* the PAW augmentation occupancies of every atom,
    which together are what an all-electron density is reconstructed from.
    The occupancies are on-site quantities with no representation on the
    plane-wave grid --- they are why no grid model has so far predicted them
    --- so this task's target is a field plus a per-atom array, and its
    operator (``model.paw``) is not the field-to-field one: an FNO for the
    grid, and a readout at each atom over one band of :math:`|\mathbf G|`
    feeding an equivariant head per element --- :mod:`poraque.ml.paw`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """
    Description of one field-to-field regression task.

    Attributes
    ----------
    name : str
        Short identifier.
    input_field, target_field : str
        File names of the source and target fields.
    description : str
        Human-readable summary.
    input_unit, target_unit : str
        Physical units of the two fields.
    site_target : str or None
        A per-atom target carried beside the field target, read from the same
        file. ``"augmentation"`` is the PAW occupancies of ``CHGCAR``; ``None``
        for a pure field-to-field task.
    """

    name: str
    input_field: str
    target_field: str
    description: str
    input_unit: str = ""
    target_unit: str = ""
    site_target: str = None

    @property
    def required_files(self):
        """File names a material directory must contain for this task."""
        return (self.input_field, self.target_field)

    def __str__(self):
        return f"{self.name}: {self.input_field} -> {self.target_field}"


#: External potential to charge density (the Hohenberg-Kohn map).
EXT_TO_CHG = TaskSpec(
    name="ext2chg",
    input_field="EXTCAR",
    target_field="CHGCAR",
    description="Local external potential -> valence charge density.",
    input_unit="eV",
    target_unit="e/Ang^3",
)

#: Charge density to kinetic energy density (the KEDF).
CHG_TO_TAU = TaskSpec(
    name="chg2tau",
    input_field="CHGCAR",
    target_field="TAUCAR",
    description="Valence charge density -> kinetic energy density.",
    input_unit="e/Ang^3",
    target_unit="eV/Ang^3",
)

#: External potential to pseudo-density plus PAW augmentation occupancies.
EXT_TO_PAW = TaskSpec(
    name="ext2paw",
    input_field="EXTCAR",
    target_field="CHGCAR",
    description="Local external potential -> pseudo-density and PAW "
                "augmentation occupancies.",
    input_unit="eV",
    target_unit="e/Ang^3",
    site_target="augmentation",
)

#: Registry of every task a name can resolve to.
TASKS = {task.name: task for task in (EXT_TO_CHG, CHG_TO_TAU, EXT_TO_PAW)}

#: The orbital-free chain, in order: what ``task.type: all`` trains, what a
#: bundle must hold to reach a total energy, and what a dataset is asked it can
#: serve. ``ext2paw`` is not a link of it --- it replaces the first link's
#: target rather than feeding the second --- so it joins an ``all`` run only
#: when ``model.paw.enable`` asks.
CHAIN = ("ext2chg", "chg2tau")


def resolve_task(task):
    """
    Coerce a task name or :class:`TaskSpec` to a :class:`TaskSpec`.

    Parameters
    ----------
    task : str or TaskSpec
        Task name (``"ext2chg"``, ``"chg2tau"``, ``"ext2paw"``) or an explicit
        spec.

    Returns
    -------
    TaskSpec

    Raises
    ------
    KeyError
        If the name is unknown.
    """
    if isinstance(task, TaskSpec):
        return task
    try:
        return TASKS[str(task)]
    except KeyError:
        raise KeyError(
            f"Unknown task {task!r}; available: {sorted(TASKS)}."
        ) from None
