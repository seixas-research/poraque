# -*- coding: utf-8 -*-
# file: tasks.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The regression tasks the neural operator is trained on.

Four tasks, which are two readings of the same two-link chain. What separates
them is not the architecture but *what a density is taken to be*.

A VASP ``CHGCAR`` is not a grid. It is a grid **and** a per-atom set of PAW
augmentation occupancies, and the all-electron density is reconstructed from
the two together. The grid alone is the pseudo-density: smooth, representable
in a plane-wave basis, and wrong near a nucleus by construction. The task
names follow the files for that reason. The pair named for ``CHGCAR`` carries
everything the file carries; the pair named ``pseudo`` keeps the grid and
drops the rest.

**The complete pair**, and the route this project takes to a total energy:

``ext2chg`` --- :math:`V_{\rm ext} \mapsto (\tilde\rho, \rho^a_{ij})`
    The pseudo-density *and* the PAW augmentation occupancies of every atom,
    which together are what an all-electron density is reconstructed from.
    The occupancies are on-site quantities with no representation on the
    plane-wave grid --- they are why no grid model has so far predicted them
    --- so this task's target is a field plus a per-atom array, and its
    operator (:class:`~poraque.ml.paw.PAWOperatorModel`) is not the field-to-field one: an FNO for the
    grid, and a readout at each atom over one band of :math:`|\mathbf G|`
    feeding an equivariant head per element --- :mod:`poraque.ml.paw`.

``chg2tau`` --- :math:`(\tilde\rho, \rho^a_{ij}) \mapsto \tau`
    The kinetic energy density from the pseudo-density **and** the
    augmentation occupancies, where the on-site part the grid cannot hold
    enters as input. The occupancies are written into extra input channels by
    an equivariant :class:`~poraque.ml.paw.SiteWriter`; the field operator is
    the ordinary FNO. What ``ext2chg`` predicts is exactly what ``chg2tau``
    reads, which is what makes the two a chain and not a pair of unrelated
    fits.

**The grid-only pair**, the same chain in its pure field-to-field form:

``ext2pseudo`` --- :math:`V_{\rm ext} \mapsto \tilde\rho`
    The Hohenberg-Kohn map, restricted to what lives on the grid. Its
    existence and uniqueness is a *theorem*: for a given electron count the
    ground-state density is a functional of the external potential alone.
    Learning it is therefore learning a well-posed object, not fitting a
    correlation --- with the caveat the name now makes explicit, that the
    object on the grid is the pseudo-density and not the density itself.

``pseudo2tau`` --- :math:`\tilde\rho \mapsto \tau`
    The kinetic energy density functional, the missing ingredient of practical
    OF-DFT. Unlike the first map this one is *semi-local in character* --- the
    exact :math:`\tau` is bounded below by the von Weizsäcker form and
    approaches Thomas-Fermi in the slowly-varying limit --- which gives the
    physics losses of :mod:`poraque.ml.physics` firm anchors.

:data:`CHAIN` is the grid-only pair. A configuration names only the complete
pair -- its ``ext2chg`` and ``chg2tau`` sections -- and reaches the grid-only
tasks by switching a section's ``occupancies`` off: the two grid-only links
share one architecture and one dataset layout and need nothing a density file
does not already hold, whereas the complete pair needs the occupancy records
and the projector layouts the pseudopotentials declare.
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
    site_input : str or None
        A per-atom input carried beside the input field, read from the same
        file; ``"augmentation"`` as for ``site_target``.
    """

    name: str
    input_field: str
    target_field: str
    description: str
    input_unit: str = ""
    target_unit: str = ""
    site_target: str = None
    site_input: str = None

    @property
    def site_field(self):
        """The file the per-atom data is read from, or ``None`` without any."""
        if self.site_target is not None:
            return self.target_field
        if self.site_input is not None:
            return self.input_field
        return None

    @property
    def required_files(self):
        """File names a material directory must contain for this task."""
        return (self.input_field, self.target_field)

    def __str__(self):
        return f"{self.name}: {self.input_field} -> {self.target_field}"


#: External potential to the pseudo-density grid (the Hohenberg-Kohn map,
#: restricted to what the grid holds).
EXT_TO_PSEUDO = TaskSpec(
    name="ext2pseudo",
    input_field="EXTCAR",
    target_field="CHGCAR",
    description="Local external potential -> pseudo-density grid.",
    input_unit="eV",
    target_unit="e/Ang^3",
)

#: Pseudo-density to kinetic energy density (the KEDF, grid only).
PSEUDO_TO_TAU = TaskSpec(
    name="pseudo2tau",
    input_field="CHGCAR",
    target_field="TAUCAR",
    description="Pseudo-density grid -> kinetic energy density.",
    input_unit="e/Ang^3",
    target_unit="eV/Ang^3",
)

#: External potential to a whole CHGCAR: pseudo-density grid plus every
#: atom's PAW augmentation occupancies.
EXT_TO_CHG = TaskSpec(
    name="ext2chg",
    input_field="EXTCAR",
    target_field="CHGCAR",
    description="Local external potential -> pseudo-density grid and "
                "PAW augmentation occupancies.",
    input_unit="eV",
    target_unit="e/Ang^3",
    site_target="augmentation",
)

#: A whole CHGCAR --- grid plus PAW occupancies --- to kinetic energy density.
CHG_TO_TAU = TaskSpec(
    name="chg2tau",
    input_field="CHGCAR",
    target_field="TAUCAR",
    description="Pseudo-density and PAW augmentation occupancies -> kinetic "
                "energy density.",
    input_unit="e/Ang^3",
    target_unit="eV/Ang^3",
    site_input="augmentation",
)

#: Registry of every task a name can resolve to.
TASKS = {task.name: task
         for task in (EXT_TO_PSEUDO, PSEUDO_TO_TAU, EXT_TO_CHG, CHG_TO_TAU)}

#: The grid-only chain, in order: what a bundle must hold to reach a total
#: energy from fields alone, and what inference runs by default. The complete
#: pair --- ``ext2chg`` and ``chg2tau`` --- is not in it: each replaces a
#: link's endpoint with a whole CHGCAR, and each needs the per-atom records.
CHAIN = ("ext2pseudo", "pseudo2tau")


#: What each pre-rename task name became, for the error messages and for the
#: checkpoint translation in :func:`~poraque.ml.training.read_bundle`. The two
#: names absent from it --- ``ext2chg`` and ``chg2tau`` --- are the swap: they
#: still resolve, to a *different* task than they used to name, which is why
#: nothing can be decided from an old name alone.
RENAMED_TASKS = {
    "ext2paw": "ext2chg",
    "paw2tau": "chg2tau",
}


def resolve_task(task):
    """
    Coerce a task name or :class:`TaskSpec` to a :class:`TaskSpec`.

    Parameters
    ----------
    task : str or TaskSpec
        Task name --- one of :data:`TASKS` --- or an explicit spec.

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
    name = str(task)
    try:
        return TASKS[name]
    except KeyError:
        hint = RENAMED_TASKS.get(name, "")
        if hint:
            hint = (f" The tasks were renamed so that a name says whether the "
                    f"per-atom PAW records travel with the grid: {name!r} is "
                    f"now {hint!r}. Mind that the two CHGCAR names were "
                    f"*swapped* rather than retired --- a configuration or a "
                    f"checkpoint written before the rename means by "
                    f"'ext2chg' what is now called 'ext2pseudo'.")
        raise KeyError(
            f"Unknown task {name!r}; available: {sorted(TASKS)}.{hint}"
        ) from None
