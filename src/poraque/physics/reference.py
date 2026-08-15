# -*- coding: utf-8 -*-
# file: reference.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Isolated-atom reference energies, for cohesive energies.

A total energy is only meaningful against a reference state. Poraquê's own
totals sit :math:`\approx 10^3` eV per atom away from VASP's because the PAW
one-centre terms and VASP's ``EATOM`` are absent, and a DFT total energy is in
any case referenced to whatever the pseudopotential generator chose. Neither
number means anything on its own.

The **cohesive energy** removes the arbitrary part:

.. math::

    \Delta E = E_{\rm total} - E_{\rm ref},
    \qquad
    E_{\rm ref} = \sum_i E_{\rm iso}(Z_i),

with :math:`E_{\rm iso}` the energy of one isolated atom of that species,
computed with the *same* pseudopotential, functional and code. What is left is
the energy released on assembling the solid from free atoms — the bonding, and
nothing else.

What this does and does not buy
-------------------------------
It is worth being precise, because the shift is often over-sold.

**It does nothing for forces.** :math:`E_{\rm ref}` depends on the composition
and not on where the atoms are, so :math:`\nabla_{\mathbf R} E_{\rm ref} = 0`
and :math:`\nabla_{\mathbf R}\Delta E = \nabla_{\mathbf R}E_{\rm total}`
identically. Referencing cannot change a force.

**It does nothing for differences at fixed composition.** Two structures with
the same formula have the same :math:`E_{\rm ref}`, which cancels exactly in
:math:`\Delta E_1 - \Delta E_2`. An energy-volume curve, a polymorph ranking or
a rattled-structure comparison is numerically unchanged.

**It is essential across compositions.** A binding energy
(:math:`N_2` against two nitrogen atoms), a cohesive energy per atom, a
formation energy — these compare systems whose atom counts differ, the
per-atom offset does *not* cancel, and without a reference state the numbers
are not merely inaccurate but undefined. That is the case this module exists
for.

Layout
------
One directory per element, each holding an ordinary single-point calculation of
one isolated atom in a large box::

    data/vasp/ref/
        Au/     POSCAR POTCAR OSZICAR OUTCAR ...
        N/      ...
        C/      ...

The directory name is the element. Reading is by the same convention the rest
of the package uses — the energy comes from ``OUTCAR`` when present and
``OSZICAR`` otherwise — and new codes are added by registering a reader with
:func:`register_energy_reader` rather than by editing this module.

    from poraque.physics import ReferenceEnergies

    references = ReferenceEnergies.from_directory("data/vasp/ref")
    references["Au"]                       # -0.0786 eV
    references.total_for(structure)        # sum over the atoms present
"""

import os
import re
import warnings

# ===================================================================== #
# Per-code energy extraction
# ===================================================================== #
def _vasp_energy(directory):
    r"""
    Final total energy of a VASP run, in eV.

    ``OUTCAR`` is preferred because it states ``free energy TOTEN``
    unambiguously; ``OSZICAR``'s ``F=`` is the same quantity and is used when
    only it survives, which is common for archived runs.

    Returns
    -------
    float or None
        ``None`` when the directory holds neither file, so the caller can
        report *which* element failed rather than raising from inside a loop.
    """
    outcar = os.path.join(directory, "OUTCAR")
    if os.path.isfile(outcar):
        with open(outcar, errors="replace") as handle:
            matches = re.findall(r"free\s+energy\s+TOTEN\s*=\s*([-\d.eE+]+)",
                                 handle.read())
        if matches:
            return float(matches[-1])

    oszicar = os.path.join(directory, "OSZICAR")
    if os.path.isfile(oszicar):
        with open(oszicar, errors="replace") as handle:
            matches = re.findall(r"F=\s*([-.\dE+]+)", handle.read())
        if matches:
            return float(matches[-1])
    return None


#: ``code -> callable(directory) -> float or None``. Mirrors the reader
#: registry of :mod:`poraque.fields.io`: adding a code is a registration, not
#: an edit here.
ENERGY_READERS = {"vasp": _vasp_energy}


def register_energy_reader(code, reader):
    """
    Register a total-energy reader for a DFT code.

    Parameters
    ----------
    code : str
        Code name, matching :mod:`poraque.fields.io`.
    reader : callable
        ``reader(directory) -> float or None``, returning eV.
    """
    ENERGY_READERS[str(code).lower()] = reader


def read_total_energy(directory, code="auto"):
    """
    Total energy of a calculation directory, in eV.

    Parameters
    ----------
    directory : str
    code : str, optional
        ``"auto"`` tries every registered reader, which is right for a
        reference directory that may have come from anywhere.

    Returns
    -------
    float or None
    """
    if code != "auto":
        try:
            reader = ENERGY_READERS[str(code).lower()]
        except KeyError:
            raise ValueError(
                f"No energy reader for code {code!r}; known codes are "
                f"{sorted(ENERGY_READERS)}."
            ) from None
        return reader(directory)

    for reader in ENERGY_READERS.values():
        energy = reader(directory)
        if energy is not None:
            return energy
    return None


# ===================================================================== #
# The mapping
# ===================================================================== #
class ReferenceEnergies:
    r"""
    ``{element: E_iso}`` in eV, and the sum over a structure.

    Parameters
    ----------
    energies : dict
        ``{element: energy}`` in eV. Keys are matched on the *bare* element
        name, so a ``Au_pv`` POTCAR and a ``Au`` reference directory agree.
    source : str, optional
        Where the values came from, for :meth:`__repr__` and for provenance in
        an energy decomposition.

    Examples
    --------
    >>> references = ReferenceEnergies({"Au": -0.0786})
    >>> references["Au"]
    -0.0786
    """

    def __init__(self, energies=None, source=None):
        self.energies = {_bare(symbol): float(value)
                         for symbol, value in (energies or {}).items()}
        self.source = source

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_directory(cls, root, code="auto", strict=False, method="poraque",
                       functional="pbe"):
        r"""
        Read every ``<root>/<Element>/`` subdirectory.

        Parameters
        ----------
        root : str
            Directory of per-element reference calculations.
        code : str, optional
            DFT code, or ``"auto"``. Only consulted for ``method="code"``.
        strict : bool, optional
            Raise when a subdirectory yields no energy, instead of warning and
            skipping it. Off by default: a half-populated reference set is
            useful for the elements it does cover, and the structures it
            cannot cover report the omission by name through
            :meth:`missing_for`.
        method : {"poraque", "code"}, optional
            **Which energy to use as** :math:`E_{\rm iso}`, and the most
            consequential choice in this module.

            ``"poraque"`` (default) evaluates the isolated atom through
            Poraquê's own energy expression, from the reference directory's
            ``CHGCAR``, ``TAUCAR`` and ``POTCAR``. ``"code"`` reads the DFT
            code's total energy out of ``OUTCAR``/``OSZICAR``.

            The default is not a convenience. A cohesive energy is only
            meaningful when the two energies being subtracted carry the *same*
            systematic error, and Poraquê's totals sit :math:`\approx 10^3` eV
            per atom away from VASP's because the PAW one-centre terms are
            absent. Subtracting VASP's atomic energy from Poraquê's total
            leaves that offset entirely intact; subtracting Poraquê's own
            atomic energy cancels it, because the same terms are missing from
            both sides. Measured on gold, this is the difference between a
            cohesive energy of :math:`-1157` eV/atom and one of
            :math:`-1.9` eV/atom.

            Use ``"code"`` when the reference calculations are what you want
            to compare *against* — for instance to quote VASP's own cohesive
            energy beside Poraquê's.
        functional : str, optional
            Exchange-correlation approximation for ``method="poraque"``. Must
            match what the structures will be evaluated with, or the two sides
            of the subtraction use different functionals.

        Returns
        -------
        ReferenceEnergies

        Raises
        ------
        FileNotFoundError
            If ``root`` does not exist. A silently empty mapping here would
            surface much later as "no cohesive energy available", with nothing
            pointing at the typo that caused it.
        ValueError
            If ``method`` is not one of the two accepted values.
        """
        if method not in ("poraque", "code"):
            raise ValueError(
                f"method={method!r} is not known; expected 'poraque' (evaluate "
                f"the isolated atom with Poraquê's own energy expression, so "
                f"the missing PAW terms cancel) or 'code' (read the DFT code's "
                f"total energy)."
            )
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"No reference-energy directory at {root!r}. It should hold "
                f"one subdirectory per element, each an isolated-atom "
                f"calculation: {root}/Au/OSZICAR, {root}/N/OSZICAR, ..."
            )

        energies = {}
        for entry in sorted(os.listdir(root)):
            directory = os.path.join(root, entry)
            if not os.path.isdir(directory) or entry.startswith("."):
                continue

            if method == "code":
                energy = read_total_energy(directory, code=code)
                complaint = "no readable total energy"
            else:
                energy, complaint = _poraque_atom_energy(directory, functional)

            if energy is None:
                message = (f"{complaint} in {directory}; it is skipped, so "
                           f"structures containing {entry!r} will have no "
                           f"cohesive energy.")
                if strict:
                    raise ValueError(message)
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                continue

            _warn_if_not_a_single_atom(directory, entry)
            energies[_bare(entry)] = energy

        return cls(energies, source=f"{root} [{method}]")

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def __getitem__(self, symbol):
        return self.energies[_bare(symbol)]

    def __contains__(self, symbol):
        return _bare(symbol) in self.energies

    def __len__(self):
        return len(self.energies)

    def __iter__(self):
        return iter(self.energies)

    def get(self, symbol, default=None):
        """``E_iso`` for ``symbol``, or ``default``."""
        return self.energies.get(_bare(symbol), default)

    def items(self):
        """``(element, energy)`` pairs."""
        return self.energies.items()

    # ------------------------------------------------------------------ #
    # Structure sums
    # ------------------------------------------------------------------ #
    def missing_for(self, structure):
        """
        Elements of ``structure`` with no reference energy, sorted.

        Returns
        -------
        list of str
        """
        absent = {_bare(symbol) for symbol in structure.symbols
                  if _bare(symbol) not in self.energies}
        return sorted(absent)

    def covers(self, structure):
        """Whether every species in ``structure`` has a reference energy."""
        return not self.missing_for(structure)

    def total_for(self, structure):
        r"""
        :math:`E_{\rm ref} = \sum_i E_{\rm iso}(Z_i)` for ``structure``, in eV.

        Parameters
        ----------
        structure : Structure

        Returns
        -------
        float

        Raises
        ------
        KeyError
            When a species has no reference. Returning a partial sum would be
            an energy silently missing whole atoms, which is worse than no
            energy at all — the number would look plausible and be wrong by
            electron-volts per missing atom.
        """
        absent = self.missing_for(structure)
        if absent:
            raise KeyError(
                f"No isolated-atom reference for {absent}. Add "
                f"{'/'.join(absent)} to the reference directory"
                + (f" ({self.source})" if self.source else "")
                + ", or ask for the total energy rather than the cohesive one."
            )

        total = 0.0
        for symbol, atom_slice in structure.species_slices():
            count = atom_slice.stop - atom_slice.start
            total += count * self.energies[_bare(symbol)]
        return float(total)

    def __repr__(self):
        listed = ", ".join(f"{symbol}={value:.4f}"
                           for symbol, value in sorted(self.energies.items()))
        origin = f", source={self.source!r}" if self.source else ""
        return f"ReferenceEnergies({listed}{origin})"


# ===================================================================== #
# Helpers
# ===================================================================== #
def _bare(symbol):
    """``Au_pv`` -> ``Au``; matches :func:`poraque.fields.element_of`."""
    return str(symbol).split("_")[0].split(".")[0].strip().rstrip("0123456789")


def _poraque_atom_energy(directory, functional):
    r"""
    Evaluate an isolated atom through Poraquê's own energy expression.

    Uses the reference calculation's ``CHGCAR`` and ``TAUCAR`` directly rather
    than predicting them: the point of :math:`E_{\rm iso}` is to be a fixed
    property of the species, and running it through the operators would make
    every cohesive energy depend on the model's error on a free atom — a
    geometry unlike anything in the training set.

    Returns
    -------
    tuple of (float or None, str)
        The energy in eV and, when it is ``None``, what was missing.
    """
    from ..fields import ChargeDensity, ExternalPotential, KineticEnergyDensity
    from ..fields.vasp.potcar import Potcar
    from .energy import EnergyCalculator

    required = {name: os.path.join(directory, name)
                for name in ("CHGCAR", "TAUCAR", "POTCAR")}
    absent = [name for name, path in required.items() if not os.path.isfile(path)]
    if absent:
        return None, f"no {', '.join(absent)}"

    try:
        density = ChargeDensity.read(required["CHGCAR"])
        tau = KineticEnergyDensity.read(required["TAUCAR"], grid=density.grid)
        potcar = Potcar.from_file(required["POTCAR"], parse_tables=True)
        potential = ExternalPotential.from_potcar_tables(
            density.structure, density.grid, potcar)
    except (OSError, ValueError) as error:                # noqa: BLE001
        return None, f"unreadable reference fields ({error})"

    calculator = EnergyCalculator(
        grid=density.grid,
        structure=density.structure,
        charges={entry.element: entry.zval for entry in potcar},
        pscore={entry.element: entry.pscore for entry in potcar
                if entry.pscore is not None},
        functional=functional,
    )
    return float(calculator.compute(density, tau, potential).total), ""


def _warn_if_not_a_single_atom(directory, element):
    """
    Check that a reference directory really holds one isolated atom.

    Both failure modes are silent and change every cohesive energy that uses
    the value: a reference computed on a dimer halves nothing and doubles the
    subtraction, and one computed in a small box is not an isolated atom at
    all. Neither raises, because the caller may know better than this heuristic
    — but neither should pass unremarked.
    """
    poscar = os.path.join(directory, "POSCAR")
    if not os.path.isfile(poscar):
        return

    try:
        from ..fields.vasp.poscar import Poscar

        structure = Poscar.from_file(poscar)
    except Exception:                                   # noqa: BLE001
        return

    if structure.natoms != 1:
        warnings.warn(
            f"{directory} holds {structure.natoms} atoms, but an isolated-atom "
            f"reference must hold exactly one; E_iso({element}) will be too "
            f"low by a factor of about {structure.natoms}.",
            RuntimeWarning, stacklevel=3,
        )
        return

    import numpy as np

    lengths = np.linalg.norm(np.asarray(structure.cell, dtype=float), axis=1)
    if lengths.min() < 8.0:
        warnings.warn(
            f"{directory} puts the isolated {element} atom in a box only "
            f"{lengths.min():.1f} A across. The atom interacts with its own "
            f"periodic images, so E_iso is not an isolated-atom energy and the "
            f"cohesive energy inherits the error.",
            RuntimeWarning, stacklevel=3,
        )
