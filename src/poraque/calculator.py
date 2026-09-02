# -*- coding: utf-8 -*-
# file: calculator.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
ASE calculator interface.

:class:`Poraque` attaches the whole geometry-to-energy pipeline to an
:class:`ase.Atoms` object, so it can be driven by anything in ASE that speaks
the calculator protocol::

    from ase.build import bulk
    from poraque.calculator import Poraque

    atoms = bulk("Pt", "fcc", a=3.92, cubic=True)
    atoms.calc = Poraque("models/poraque_models.poraque", potcar_dir="POTCARs")

    rho = atoms.calc.get_charge_density(atoms)   # ChargeDensity, e/Ang^3
    charges = atoms.get_charges()                # net charge per atom, +e

Each call runs

.. code-block:: text

    Atoms -> Structure -> FieldGrid
          -> V_ext   (analytic, from the POTCAR tables)
          -> rho     (ext2chg operator)
          -> tau     (chg2tau operator)
          -> E       (poraque.physics.EnergyCalculator)

and the three intermediate fields stay on the calculator afterwards
(:attr:`Poraque.fields`), so a prediction can be written out or inspected
rather than only reduced to a number.

Differences from a conventional MLIP
------------------------------------
It behaves like MACE or NequIP at the interface, but not underneath. Two
consequences are worth stating plainly before anyone runs a relaxation:

**Forces are Hellmann-Feynman, and incomplete for PAW.** A machine-learned
interatomic potential differentiates a scalar energy with respect to the
positions it was built from. Here the positions enter :math:`V_{\rm ext}`
analytically, so the electron-ion and ion-ion terms are differentiated in
closed form (:mod:`poraque.physics.forces`) rather than by finite differences
on a fixed grid. That construction is exact for a *local* pseudopotential and
verified to :math:`10^{-7}` eV/Å against finite differences of the term it
differentiates — but a PAW calculation adds projector and one-centre forces
that no grid-based field carries. On the shipped platinum structures the result
gets the magnitude and not the direction. Read
:meth:`Poraque.compute_forces` before using them for anything; geometry
optimisation and molecular dynamics are *not* yet supportable.

**Absolute energies are not DFT energies.** The fields are pseudo-valence
quantities, so the PAW one-centre terms are missing. See
:class:`~poraque.physics.energy.EnergyComponents`.
"""

import os
import warnings

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
    _ASE_ERROR = None
except ImportError as error:                              # pragma: no cover
    Calculator, all_changes = object, None
    _ASE_ERROR = error

from .fields import ExternalPotential, FieldGrid, field_integral
from .fields.structure import Structure
from .ml import BUNDLE_FILENAME, resolve_bundle_path
from .physics import EnergyCalculator
from .physics.energy import total_density

#: Fallback grid resolution when neither the caller nor the checkpoints say.
DEFAULT_RESOLUTION = 32


class Poraque(Calculator):
    r"""
    Predict the total energy of an :class:`ase.Atoms` object.

    Parameters
    ----------
    models : str, optional
        The unified checkpoint written by ``poraque-train``, holding both
        operators. Defaults to ``models/poraque_models.poraque``.
    ext2chg, chg2tau : FieldOperator, optional
        Already-loaded operators, overriding the corresponding entry of
        ``models``. Useful for evaluating a model that is still in memory.
    potcar : str, optional
        A single concatenated ``POTCAR`` covering the species present.
        **Strongly preferred** when the composition is fixed: it supplies the
        tabulated local potential the operators were trained on, and the
        ``PSCORE`` values needed for the :math:`\mathbf G = 0` energy term.
    potcar_dir : str, optional
        A ``POTCAR`` *library* — one subdirectory per pseudopotential, as VASP
        ships them (``<potcar_dir>/Pt/POTCAR``, optionally ``.gz`` or ``.Z``).
        The right choice for a calculator that must serve arbitrary
        compositions: the entries for whatever elements an
        :class:`ase.Atoms` happens to contain are assembled on demand and
        cached per composition. A flat layout (``POTCAR.Pt``, ``Pt.POTCAR``)
        is also recognised.
    charges : dict, optional
        ``{element: Z_val}``, used only when no ``POTCAR`` is available.
        Selects the Gaussian pseudo-ion model — see the warning below.
    resolution : int, optional
        Longest grid axis. Defaults to the resolution recorded in the
        ``ext2chg`` checkpoint, else :data:`DEFAULT_RESOLUTION`.
    functional : str, optional
        Exchange-correlation approximation, one of
        :data:`~poraque.physics.energy.XC_FUNCTIONALS`: ``"pbe"`` (default),
        ``"lda"``, ``"pbe-x"``, ``"lda-x"`` or ``"none"``. PBE is the default
        because the reference calculations are PBE (``PAW_PBE`` potentials,
        ``LEXCH = PE``); an LDA :math:`E_{\rm xc}` on a PBE density answers a
        different question. Change it only to match a differently generated
        dataset.
    device : str, optional
        ``"auto"`` (default), ``"cuda"``, ``"mps"`` or ``"cpu"``.
    normalize_density : bool, optional
        Rescale the predicted density to the valence electron count the
        pseudopotentials fix, before any energy is integrated from it.
        **On by default, and turning it off is only for diagnosis.** The
        shipped operators predict a density whose electron count drifts by up
        to ~2 %, and because the electrostatic terms are of order
        :math:`10^{4}` eV that drift alone moves a total energy by tens of eV
        — far more than the differences the energy exists to resolve, and by a
        different amount for every structure, so it does not cancel. See
        :meth:`~poraque.fields.ChargeDensity.normalized`.
    references : ReferenceEnergies, str or dict, optional
        Isolated-atom energies, enabling :meth:`get_cohesive_energy`. Accepts
        a built mapping, a ``{element: energy}`` dict, or a directory holding
        one subdirectory per element (``data/vasp/ref``). Without it the
        calculator still returns total energies and forces; only the cohesive
        energy is unavailable.

        This does **not** change :meth:`get_potential_energy`, which keeps
        returning the total, nor the forces, which cannot depend on it —
        :math:`E_{\rm ref}` is a function of composition alone, so its gradient
        with respect to any atomic position is exactly zero.
    **kwargs
        Passed to :class:`ase.calculators.calculator.Calculator`.

    Attributes
    ----------
    fields : dict
        ``{"external", "density", "tau"}`` from the most recent evaluation,
        each a :class:`~poraque.fields.base.ScalarField`.
    components : EnergyComponents
        Full energy decomposition of the most recent evaluation.
    charge_analysis : PartialCharges or None
        Full population analysis from the most recent :meth:`get_charges`,
        which returns only the net charges.

    Warnings
    --------
    Without ``potcar``, the external potential falls back to the Gaussian
    pseudo-ion model, which reproduces a reference potential to a relative
    :math:`L^2` of about ``0.13`` — against ``2 \times 10^{-5}`` for the
    tabulated one. The operators were trained on tabulated potentials, so the
    Gaussian model feeds them an input **outside their training distribution**
    and the prediction is not trustworthy. The calculator warns once; it does
    not refuse, because the fallback is genuinely useful for smoke tests.

    Examples
    --------
    >>> from ase.build import bulk                                # doctest: +SKIP
    >>> atoms = bulk("Pt", "fcc", a=3.92, cubic=True)             # doctest: +SKIP
    >>> atoms.calc = Poraque("models/poraque_models.poraque",
    ...                      potcar_dir="POTCARs")                # doctest: +SKIP

    The fields are the prediction; the energy is one thing integrated from
    them:

    >>> rho = atoms.calc.get_charge_density(atoms)                # doctest: +SKIP
    >>> rho.electron_count()                                      # doctest: +SKIP
    44.0
    >>> rho.write("CHGCAR_pred")                                  # doctest: +SKIP
    >>> atoms.get_charges()                                       # doctest: +SKIP
    array([-0.001,  0.001, -0.000,  0.000])
    >>> print(atoms.calc.charge_analysis)                         # doctest: +SKIP
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    #: Relative electron-count drift above which the energy is warned about.
    #: Set where it is because the electrostatic terms are of order 1e4 eV, so
    #: 1e-3 already corresponds to a ~10 eV shift in the total.
    _DRIFT_WARNING = 1e-3

    def __init__(self, models=None, ext2chg=None, chg2tau=None, potcar=None,
                 potcar_dir=None, charges=None, resolution=None,
                 functional="pbe", device="auto", normalize_density=True,
                 references=None, **kwargs):
        if _ASE_ERROR is not None:                        # pragma: no cover
            raise ImportError(
                "The Poraque ASE calculator requires ASE: pip install ase"
            ) from _ASE_ERROR
        Calculator.__init__(self, **kwargs)

        self.device = device
        self.functional = functional
        self.normalize_density = bool(normalize_density)
        self.references = _resolve_references(references)
        # The directory is kept beside the parsed energies because Hirshfeld
        # partitioning needs the isolated-atom *densities*, which live in the
        # same tree but are not what ReferenceEnergies reads.
        self.reference_dir = (str(references)
                              if isinstance(references, (str, os.PathLike))
                              else None)
        self.charges = dict(charges) if charges else None
        self.potcar_dir = str(potcar_dir) if potcar_dir else None
        self.fields = {}
        self._fields_key = None
        self.components = None
        self.raw_electron_drift = None
        self.charge_analysis = None
        self._warned_gaussian = False
        self._potcar_cache = {}

        if models is None and (ext2chg is None or chg2tau is None):
            models = resolve_bundle_path(
                os.path.join("models", BUNDLE_FILENAME))
        self.models = str(models) if models is not None else None

        self.ext2chg = self._resolve_operator(
            ext2chg if ext2chg is not None else self.models, "ext2chg")
        self.chg2tau = self._resolve_operator(
            chg2tau if chg2tau is not None else self.models, "chg2tau")

        self.potcar = self._read_potcar(potcar) if potcar else None

        if self.potcar_dir is not None and not os.path.isdir(self.potcar_dir):
            raise ValueError(
                f"potcar_dir {self.potcar_dir!r} is not a directory."
            )
        if self.potcar is None and self.potcar_dir is None and not self.charges:
            raise ValueError(
                "Poraque needs a POTCAR (potcar=... for a fixed composition, "
                "potcar_dir=... for a library) or an explicit "
                "charges={'Pt': 11.0} mapping: the external potential cannot "
                "be built without the pseudo-ion valence charges."
            )

        self.resolution = int(
            resolution if resolution is not None
            else (self.ext2chg.training_resolution or DEFAULT_RESOLUTION)
        )

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _resolve_operator(self, source, task):
        """Accept a live operator, or load one from the unified checkpoint."""
        from .ml import FieldOperator, bundle_tasks, load_bundle

        if isinstance(source, FieldOperator):
            if source.task.name != task:
                raise ValueError(
                    f"Expected a {task!r} operator, got {source.task.name!r}. "
                    f"Chaining the wrong model produces plausible-looking "
                    f"garbage rather than an error."
                )
            return source

        try:
            return load_bundle(source, task, device=self.device)
        except KeyError:
            # A single-task bundle is a normal artefact now: the vast public
            # archives of charge densities train `ext2chg` and nothing else,
            # because no archive publishes a kinetic energy density. Say what
            # such a model *can* do rather than only that a key is missing --
            # the calculator needs both halves of the chain and this is where
            # a reader finds out why.
            held = bundle_tasks(source)
            raise KeyError(
                f"{source} holds no {task!r} model; it contains {held}. The "
                f"ASE calculator runs the whole chain V_ext -> rho -> tau -> E, "
                f"so it needs both. A bundle with only 'ext2chg' predicts "
                f"charge densities -- use FieldOperator.predict, or "
                f"poraque.ml.load_bundle(path, 'ext2chg') -- but cannot give a "
                f"total energy, which is an integral over tau. Train chg2tau "
                f"on data that carries a TAUCAR and save both into one bundle."
            ) from None

    @staticmethod
    def _read_potcar(path):
        """Read a POTCAR with its local-potential tables."""
        from .fields.vasp.potcar import Potcar

        return Poraque._validate_potcar(Potcar.from_file(path,
                                                         parse_tables=True))

    @staticmethod
    def _validate_potcar(potcar):
        """
        Return ``potcar`` if every entry carries a usable table, else ``None``.

        A truncated table is not a fatal error — the Gaussian fallback still
        produces a number — but it silently changes which model generated the
        input, so it warns rather than degrading quietly.
        """
        missing = [entry.element for entry in potcar
                   if not entry.has_local_table]
        if missing:
            warnings.warn(
                f"POTCAR entries {missing} carry no usable local-potential "
                f"table; the Gaussian pseudo-ion model will be used for the "
                f"whole cell instead. Predictions will be unreliable.",
                RuntimeWarning, stacklevel=3,
            )
            return None
        return potcar

    def _potcar_for(self, structure):
        """
        The ``POTCAR`` covering ``structure``, or ``None`` if unavailable.

        An explicit ``potcar=`` wins when it covers every element present; a
        ``potcar_dir=`` library is otherwise searched and the result cached by
        composition, so a scan over many geometries of one material parses the
        tables once.

        Parameters
        ----------
        structure : Structure

        Returns
        -------
        Potcar or None
        """
        elements = tuple(dict.fromkeys(structure.elements))

        if self.potcar is not None:
            covered = {entry.element for entry in self.potcar}
            absent = [e for e in elements if e not in covered]
            if not absent:
                return self.potcar
            if self.potcar_dir is None:
                raise ValueError(
                    f"The supplied POTCAR covers {sorted(covered)} but the "
                    f"structure contains {absent}. Pass potcar_dir= to look "
                    f"the missing species up in a POTCAR library."
                )

        if self.potcar_dir is None:
            return None

        if elements not in self._potcar_cache:
            self._potcar_cache[elements] = self._assemble_potcar(elements)
        return self._potcar_cache[elements]

    def _assemble_potcar(self, elements):
        """Build a :class:`Potcar` for ``elements`` from the library."""
        from .fields.vasp.potcar import Potcar

        return self._validate_potcar(
            Potcar.from_library(self.potcar_dir, elements))

    # ------------------------------------------------------------------ #
    # The pipeline
    # ------------------------------------------------------------------ #
    def build_external_potential(self, atoms):
        r"""
        Analytic :math:`V_{\rm ext}` for ``atoms``, on a fresh grid.

        Parameters
        ----------
        atoms : ase.Atoms
            Must be periodic in all three directions: every field in Poraquê
            is a periodic plane-wave quantity, and a slab or molecule in a box
            with ``pbc=False`` would be silently treated as periodic anyway.

        Returns
        -------
        ExternalPotential
        """
        if not all(atoms.get_pbc()):
            raise ValueError(
                "Poraque works on fully periodic cells (pbc=True in all three "
                "directions). For a molecule or slab, place it in a box with "
                "enough vacuum and set pbc=True explicitly."
            )
        if atoms.cell.rank != 3:
            raise ValueError("Atoms has no full 3D cell.")

        structure = Structure.from_ase(atoms)
        grid = FieldGrid(_grid_shape(structure.cell, self.resolution),
                         structure.cell)

        potcar = self._potcar_for(structure)
        if potcar is not None:
            return ExternalPotential.from_potcar_tables(structure, grid,
                                                        potcar)

        if not self.charges:
            raise ValueError(
                f"No POTCAR found for {sorted(set(structure.elements))} and no "
                f"charges= mapping was given, so the external potential cannot "
                f"be built."
            )

        if not self._warned_gaussian:
            warnings.warn(
                "No POTCAR: falling back to the Gaussian pseudo-ion model for "
                "V_ext. It differs from the tabulated potential by a relative "
                "L2 of ~0.13, which is far outside what the operators were "
                "trained on. Treat the resulting energy as a smoke test, not "
                "a prediction.",
                RuntimeWarning, stacklevel=3,
            )
            self._warned_gaussian = True
        return ExternalPotential.compute(structure, grid, self.charges)

    def valence_electrons(self, structure):
        r"""
        :math:`\sum_s N_s Z^{\rm val}_s` for ``structure``, or ``None``.

        Read from the ``POTCAR`` when one covers the composition, else from an
        explicit ``charges=`` mapping.

        Parameters
        ----------
        structure : Structure

        Returns
        -------
        float or None
        """
        charges = None
        potcar = self._potcar_for(structure)
        if potcar is not None:
            charges = {entry.element: entry.zval for entry in potcar}
        elif self.charges:
            charges = self.charges
        if not charges:
            return None

        total = 0.0
        for symbol, atom_slice in structure.species_slices():
            element = symbol.split("_")[0].split(".")[0]
            if element not in charges:
                return None
            count = atom_slice.stop - atom_slice.start
            total += count * float(charges[element])
        return total

    def predict_fields(self, atoms):
        r"""
        Run the full chain and return the three fields.

        The predicted density is rescaled to the valence electron count the
        pseudopotentials fix before :math:`\tau` is predicted from it — see
        :meth:`~poraque.fields.ChargeDensity.normalized` for why this is a
        correction rather than a cosmetic touch-up. The count is exactly known
        from the ``POTCAR``, so nothing is being assumed here that the
        calculation does not already state. When no valence charges are
        available the raw prediction is used and
        :attr:`~poraque.physics.energy.EnergyComponents.electron_drift`
        reports ``None``.

        Parameters
        ----------
        atoms : ase.Atoms

        Returns
        -------
        dict
            ``{"external", "density", "tau"}``. ``density`` is the normalized
            field; the raw one is kept under ``"density_raw"``.
        """
        potential = self.build_external_potential(atoms)
        raw = self.ext2chg.predict(potential)

        density = raw
        nominal = self.valence_electrons(potential.structure)
        if nominal is not None and self.normalize_density:
            try:
                density = raw.normalized(nominal)
            except ValueError:
                # An untrained or badly broken operator can predict a field
                # that integrates to zero, which no rescaling can repair. The
                # normalization is a correction, not a precondition, so a
                # failure here degrades to the raw prediction rather than
                # taking down a pipeline that would otherwise return a number.
                # The drift check below reports it either way.
                warnings.warn(
                    "The predicted density integrates to zero, so it could "
                    "not be normalized to the valence electron count. The raw "
                    "prediction is being used and the energy is meaningless.",
                    RuntimeWarning, stacklevel=3,
                )

        tau = self.chg2tau.predict(density)
        return {"external": potential, "density": density, "tau": tau,
                "density_raw": raw}

    def energy_components(self, atoms):
        """
        Full energy decomposition for ``atoms``.

        Also records :attr:`raw_electron_drift`, the relative electron-count
        error of the density *before* normalization. That is the honest
        measure of how well the ``ext2chg`` operator did on this structure;
        after normalization the count is exact by construction and no longer
        tells you anything.

        Returns
        -------
        EnergyComponents
        """
        fields = self.predict_fields(atoms)
        potential = fields["external"]

        # Same POTCAR the potential was built from, so PSCORE and the valence
        # charges cannot come from different pseudopotentials.
        potcar = self._potcar_for(potential.structure)
        pscore = None
        if potcar is not None:
            pscore = {entry.element: entry.pscore for entry in potcar
                      if entry.pscore is not None}

        calculator = EnergyCalculator(
            grid=potential.grid,
            structure=potential.structure,
            charges=potential.metadata.get("charges") or self.charges,
            pscore=pscore,
            functional=self.functional,
            references=self.references,
        )
        components = calculator.compute(fields["density"], fields["tau"],
                                        potential)

        nominal = calculator.nominal_electrons
        # `electron_count` on a spin pair, `integrate` on a plain field: the
        # two-channel class deliberately offers no `integrate`, since the
        # integral of (rho, m) is not one number. One helper decides that for
        # the whole codebase -- this was the second of three copies, and the
        # third was the one that did not exist, in `poraque-inference`.
        raw_count = field_integral(fields["density_raw"])
        self.raw_electron_drift = (
            None if not nominal else (raw_count - nominal) / nominal)

        # The total is a cancellation of terms of order 1e4 eV down to order
        # 1 eV, so the fields need a relative accuracy of ~1e-5 for the result
        # to mean anything. The electron count is the only component of that
        # accuracy which can be checked without a reference calculation, so it
        # is checked: a density that misses the count it is *known* to have has
        # certainly not got the rest right either.
        if (self.raw_electron_drift is not None
                and abs(self.raw_electron_drift) > self._DRIFT_WARNING):
            warnings.warn(
                f"The predicted density integrates to {raw_count:.3f} "
                f"electrons against a nominal {nominal:.3f} "
                f"({self.raw_electron_drift:+.2%}). It has been rescaled to "
                f"the nominal count, but a drift this large means the shape is "
                f"also wrong, and the energy terms it is integrated into are "
                f"of order 1e4 eV. Treat this total energy as indicative only "
                f"— see Poraque.get_potential_energy's accuracy note.",
                RuntimeWarning, stacklevel=3,
            )

        self.fields = fields
        self._fields_key = self._atoms_key(atoms)
        return components

    # ------------------------------------------------------------------ #
    # ASE protocol
    # ------------------------------------------------------------------ #
    def calculate(self, atoms=None, properties=("energy",),
                  system_changes=all_changes):
        """
        Evaluate the requested properties. Called by ASE, not directly.

        Parameters
        ----------
        atoms : ase.Atoms, optional
        properties : sequence of str, optional
        system_changes : list of str, optional
        """
        Calculator.calculate(self, atoms, properties, system_changes)

        components = self.energy_components(self.atoms)
        self.components = components

        # ASE's "energy" stays the TOTAL energy, deliberately. Everything in
        # ASE that consumes it -- optimizers, equations of state, phonon codes
        # -- differences it against another calculation of the same object, and
        # E_ref cancels in every one of those. Returning a cohesive energy here
        # would silently redefine a quantity the ecosystem already agrees on,
        # for no gain: E_total = dE + E_ref reconstructs the same number, and
        # the cohesive energy is available by name.
        energy = components.total
        self.results["energy"] = energy
        # No electronic smearing enters this pipeline, so the free energy and
        # the total energy are the same number. ASE optimizers request
        # 'free_energy' by name, and omitting it makes them fail on a
        # calculator that can in fact answer.
        self.results["free_energy"] = energy
        self.results["n_electrons"] = components.n_electrons
        self.results["raw_electron_drift"] = self.raw_electron_drift
        self.results["energy_components"] = components.as_dict()
        self.results["reference_energy"] = components.reference
        self.results["cohesive_energy"] = components.cohesive

        # Computed only on request: it costs one extra pass over the grid per
        # atom, which an energy-only scan should not pay for.
        if "forces" in properties:
            self.results["forces"] = self.compute_forces(self.atoms)

    # ------------------------------------------------------------------ #
    # Field accessors
    # ------------------------------------------------------------------ #
    @staticmethod
    def _atoms_key(atoms):
        """A fingerprint of what the fields were predicted *for*."""
        if atoms is None:
            return None
        return (np.asarray(atoms.cell).tobytes(),
                np.asarray(atoms.positions).tobytes(),
                np.asarray(atoms.numbers).tobytes())

    def _field(self, atoms, key):
        """
        One field of the current evaluation, predicting first if needed.

        Reuses :attr:`fields` only when the cached fields were predicted for
        *these* atoms — a cache keyed on nothing served structure A's density
        to a caller asking about structure B, silently.
        """
        atoms = atoms if atoms is not None else self.atoms
        wanted = self._atoms_key(atoms)
        if not self.fields or wanted != self._fields_key:
            self.fields = self.predict_fields(atoms)
            self._fields_key = wanted
        return self.fields[key]

    def get_external_potential(self, atoms=None):
        r"""
        :math:`V_{\rm ext}` in eV, computed analytically from the ``POTCAR``.

        Returns
        -------
        ExternalPotential
        """
        return self._field(atoms, "external")

    def get_charge_density(self, atoms=None):
        r"""
        Predicted valence density :math:`\rho` in e/Å³.

        Normalized to the valence electron count unless
        ``normalize_density=False``; the raw prediction is
        :meth:`get_raw_charge_density`.

        Returns
        -------
        ChargeDensity or SpinDensity
        """
        return self._field(atoms, "density")

    def get_raw_charge_density(self, atoms=None):
        """The ``ext2chg`` output before charge normalization."""
        return self._field(atoms, "density_raw")

    def get_kinetic_energy_density(self, atoms=None):
        r"""
        Predicted :math:`\tau` in eV/Å³.

        Returns
        -------
        KineticEnergyDensity
        """
        return self._field(atoms, "tau")

    def get_hartree_potential(self, atoms=None, with_external=False):
        r"""
        Hartree potential :math:`v_{\rm H}` in eV, on the shared grid.

        **Solved, not predicted.** Poisson's equation relates
        :math:`v_{\rm H}` to :math:`\rho` exactly, and on a periodic
        plane-wave grid the reciprocal-space solution
        :math:`v_{\rm H}(\mathbf G) = 4\pi e^2\rho(\mathbf G)/G^2` is exact for
        any band-limited density at the cost of two FFTs. So this field
        inherits the density's error and adds none of its own: nothing is
        learned here, and nothing needs to be.

        Parameters
        ----------
        atoms : ase.Atoms, optional
        with_external : bool, optional
            Return :math:`v_{\rm H} + V_{\rm ext}`, the total local potential,
            which is what a plain VASP ``LOCPOT`` holds. Off by default,
            because the Hartree term alone is the one this method is named for
            and the one that is derived here.

        Returns
        -------
        HartreePotential
            A field like any other: it carries the grid and the structure, and
            :meth:`~poraque.fields.base.ScalarField.write` serializes it in
            ``LOCPOT`` format.

        Examples
        --------
        >>> potential = atoms.calc.get_hartree_potential()      # doctest: +SKIP
        >>> potential.write("LOCPOT")                           # doctest: +SKIP
        """
        from .fields import HartreePotential

        density = self.get_charge_density(atoms)
        hartree = HartreePotential.from_density(density)
        if with_external:
            return hartree.total_with(self.get_external_potential(atoms))
        return hartree

    def valence_charges(self, structure):
        """
        ``{element: Z_val}`` for ``structure``, from the ``POTCAR`` or
        ``charges=``.

        Returns
        -------
        dict or None
        """
        potcar = self._potcar_for(structure)
        if potcar is not None:
            return {entry.element: entry.zval for entry in potcar}
        return dict(self.charges) if self.charges else None

    def get_charges(self, atoms=None, method="bader", **kwargs):
        r"""
        Partial charges from the predicted density.

        Compatible with the standard ASE interface: ``atoms.get_charges()``
        calls this with the atoms alone and gets the default partitioning.

        Parameters
        ----------
        atoms : ase.Atoms, optional
        method : {"bader", "hirshfeld", "voronoi"}, optional
            Which partitioning. See :mod:`poraque.analysis.charges` for what
            each one means and when it misleads.
        **kwargs
            Forwarded to the partitioner — ``backend=`` for Bader,
            ``references=`` for Hirshfeld. The Hirshfeld reference defaults to
            the directory passed as ``references=`` to the constructor, so
            free-atom densities and free-atom energies are read from one place.

        Returns
        -------
        numpy.ndarray
            ``(natoms,)`` net charges in units of ``+e``, positive for
            electron-deficient. The full decomposition — populations, the
            valence subtracted, and which promolecule or Bader backend was
            used — is left on :attr:`charge_analysis`.

        Notes
        -----
        These are partitions of the **pseudo** valence density: the PAW core is
        absent, so the charges are systematically compressed toward zero
        relative to an all-electron analysis, and they inherit whatever error
        the predicted density carries. Use them to compare across a series, not
        as absolute numbers.

        Examples
        --------
        >>> atoms.get_charges()                              # doctest: +SKIP
        >>> atoms.calc.get_charges(method="hirshfeld")       # doctest: +SKIP
        >>> print(atoms.calc.charge_analysis)                # doctest: +SKIP
        """
        from .analysis import partial_charges

        atoms = atoms if atoms is not None else self.atoms
        density = self.get_charge_density(atoms)
        structure = self.fields["external"].structure

        if method == "hirshfeld" and "references" not in kwargs:
            kwargs["references"] = self.reference_dir

        analysis = partial_charges(
            density, structure=structure, grid=density.grid, method=method,
            valence=self.valence_charges(structure), **kwargs)

        self.charge_analysis = analysis
        return analysis.charges

    def verify_charge(self, atoms=None, tolerance=1e-3, warn=True):
        r"""
        Check that the predicted density holds the right number of electrons.

        Parameters
        ----------
        atoms : ase.Atoms, optional
        tolerance : float, optional
            Relative tolerance.
        warn : bool, optional

        Returns
        -------
        ChargeCheck

        Notes
        -----
        With ``normalize_density=True`` — the default — this passes by
        construction, because the density has already been rescaled to the
        nominal count. It is the *raw* prediction that is worth checking, and
        :attr:`raw_electron_drift` reports that after any energy evaluation.
        Run this with ``normalize_density=False`` to measure the operator
        rather than the repair.
        """
        from .analysis import verify_total_charge

        atoms = atoms if atoms is not None else self.atoms
        density = self.get_charge_density(atoms)
        structure = self.fields["external"].structure

        expected = self.valence_electrons(structure)
        if expected is None:
            raise ValueError(
                "No valence charges available, so there is no expected "
                "electron count to check against. Supply a POTCAR or "
                "charges={'Pt': 11.0}."
            )
        return verify_total_charge(density, density.grid.cell, expected,
                                   tolerance=tolerance, warn=warn)

    def get_cohesive_energy(self, atoms=None, per_atom=False):
        r"""
        :math:`\Delta E = E_{\rm total} - \sum_i E_{\rm iso}(Z_i)`, in eV.

        The energy released on assembling ``atoms`` from isolated atoms. Unlike
        :meth:`get_potential_energy` this is referenced to a defined state, so
        it can be compared against another code, against experiment, or across
        compositions — none of which the raw total supports.

        Parameters
        ----------
        atoms : ase.Atoms, optional
        per_atom : bool, optional
            Divide by the atom count.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            When no reference energies cover the composition. Falling back to
            the total would return a number about :math:`10^3` eV per atom
            away from a cohesive energy, in the same units, with nothing to
            mark it as the wrong quantity.
        """
        atoms = atoms if atoms is not None else self.atoms
        components = self.energy_components(atoms)

        if components.cohesive is None:
            structure = self.fields["external"].structure
            absent = (sorted(set(structure.symbols)) if self.references is None
                      else self.references.missing_for(structure))
            raise ValueError(
                f"No isolated-atom reference energy for {absent}. Pass "
                f"references='data/vasp/ref' (a directory with one "
                f"subdirectory per element) or a ReferenceEnergies instance to "
                f"Poraque(...)."
            )
        return (components.cohesive_per_atom if per_atom
                else components.cohesive)

    def compute_forces(self, atoms):
        r"""
        Hellmann-Feynman forces on every atom, in eV/Å.

        Uses the fields of the most recent evaluation, so it is the caller's
        job to have run :meth:`energy_components` for ``atoms`` first;
        :meth:`calculate` does that.

        .. warning::

           **These forces are not accurate for PAW datasets with strong
           non-locality**, which includes every transition metal. The
           Hellmann-Feynman construction is complete only for a *local*
           pseudopotential; a PAW calculation adds a projector (non-local)
           force and a one-centre force, and neither is recoverable from
           :math:`\rho`, :math:`\tau` and :math:`V_{\rm loc}` on a grid.
           Measured against VASP on the shipped platinum structures — using VASP's
           *own* density, so with no model error at all — this reproduces the
           magnitude but not the direction: mean absolute error ~0.7 eV/Å
           against forces of ~1.4 eV/Å.

           The cancellation is the reason it is delicate. The electron-ion and
           ion-ion terms are each ~100 eV/Å and cancel to ~0.5 eV/Å, a residual
           of half a percent, so a relative error :math:`\epsilon` in
           :math:`\rho` arrives in the force amplified roughly 200-fold.

        Parameters
        ----------
        atoms : ase.Atoms

        Returns
        -------
        numpy.ndarray
            ``(natoms, 3)`` in eV/Å.

        Raises
        ------
        ValueError
            Without a ``POTCAR``. The Gaussian pseudo-ion fallback has no
            tabulated form factor to differentiate, and the electron-ion term
            is not optional — it cancels almost all of the Ewald force.
        """
        from .physics import hellmann_feynman_forces

        current = bool(self.fields) and self._atoms_key(atoms) == self._fields_key
        fields = self.fields if current else self.predict_fields(atoms)
        potential = fields["external"]
        structure = potential.structure

        potcar = self._potcar_for(structure)
        if potcar is None:
            raise ValueError(
                "Forces need the tabulated local potential from a POTCAR. The "
                "Gaussian pseudo-ion fallback cannot supply the form factor "
                "the Hellmann-Feynman term differentiates, and the ion-ion "
                "force alone is wrong by two orders of magnitude."
            )

        # The Hellmann-Feynman term is an integral of rho against dV_ext/dR,
        # so it takes the total density; a spin pair's `.data` is the (rho, m)
        # stack and would enter it as an extra leading axis.
        return hellmann_feynman_forces(
            np.asarray(total_density(fields["density"])), structure,
            potential.grid, potcar=potcar)

    def get_stress(self, atoms=None):
        """
        Not implemented.

        Raises
        ------
        NotImplementedError
            Always. The stress needs the energy's response to a strain, which
            deforms the cell *and* the grid the fields live on.
        """
        raise NotImplementedError(
            "Poraque does not compute the stress tensor yet."
        )

    def __repr__(self):
        if self.potcar is not None:
            source = "potcar"
        elif self.potcar_dir is not None:
            source = f"potcar_dir={self.potcar_dir!r}"
        else:
            source = "gaussian"
        return (f"Poraque(resolution={self.resolution}, v_ext={source}, "
                f"functional={self.functional!r}, "
                f"device={self.ext2chg.device_description})")


# ===================================================================== #
# Helpers
# ===================================================================== #
def _resolve_references(references):
    """
    Accept a :class:`ReferenceEnergies`, a directory path, a dict, or ``None``.

    A path is the common case and the one worth being forgiving about; the
    other forms exist so a caller that has already built the mapping — a test,
    or a sweep over many structures — does not re-read the directory per
    calculator.
    """
    if references is None:
        return None

    from .physics import ReferenceEnergies

    if isinstance(references, ReferenceEnergies):
        return references
    if isinstance(references, dict):
        return ReferenceEnergies(references)
    return ReferenceEnergies.from_directory(str(references))





def _grid_shape(cell, resolution):
    """
    FFT-friendly grid whose longest axis is ``resolution`` points.

    The cell's aspect ratio is preserved, so an elongated cell is not forced
    onto a cubic mesh with wildly different spacing along each axis.
    """
    from .fields.grid import fft_friendly_size

    lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
    scale = resolution / lengths.max()
    return tuple(fft_friendly_size(max(4, int(round(length * scale))))
                 for length in lengths)
