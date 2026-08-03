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
    """

    name: str
    input_field: str
    target_field: str
    description: str
    input_unit: str = ""
    target_unit: str = ""

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

#: Registry of the available tasks.
TASKS = {task.name: task for task in (EXT_TO_CHG, CHG_TO_TAU)}


def resolve_task(task):
    """
    Coerce a task name or :class:`TaskSpec` to a :class:`TaskSpec`.

    Parameters
    ----------
    task : str or TaskSpec
        Task name (``"ext2chg"``, ``"chg2tau"``) or an explicit spec.

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
