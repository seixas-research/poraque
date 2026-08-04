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

    atoms = bulk("Au", "fcc", a=4.08, cubic=True)
    atoms.calc = Poraque("models/poraque_models.pfno", potcar="POTCAR")
    energy = atoms.get_potential_energy()

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
that no grid-based field carries. On the shipped gold structures the result
gets the magnitude and not the direction. Read
:meth:`Poraque.compute_forces` before using them for anything; geometry
optimisation and molecular dynamics are *not* yet supportable.

**Absolute energies are not DFT energies.** The fields are pseudo-valence
quantities, so the PAW one-centre terms are missing. See
:class:`~poraque.physics.energy.EnergyComponents`.
"""

import gzip
import os
import warnings

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
    _ASE_ERROR = None
except ImportError as error:                              # pragma: no cover
    Calculator, all_changes = object, None
    _ASE_ERROR = error

from .fields import ExternalPotential, FieldGrid
from .fields.structure import Structure
from .ml import BUNDLE_FILENAME, resolve_bundle_path
from .physics import EnergyCalculator

#: Fallback grid resolution when neither the caller nor the checkpoints say.
DEFAULT_RESOLUTION = 32


class Poraque(Calculator):
    r"""
    Predict the total energy of an :class:`ase.Atoms` object.

    Parameters
    ----------
    models : str, optional
        The unified checkpoint written by ``poraque-train``, holding both
        operators. Defaults to ``models/poraque_models.pfno``.
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
        ships them (``<potcar_dir>/Au/POTCAR``, optionally ``.gz`` or ``.Z``).
        The right choice for a calculator that must serve arbitrary
        compositions: the entries for whatever elements an
        :class:`ase.Atoms` happens to contain are assembled on demand and
        cached per composition. A flat layout (``POTCAR.Au``, ``Au.POTCAR``)
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
    **kwargs
        Passed to :class:`ase.calculators.calculator.Calculator`.

    Attributes
    ----------
    implemented_properties : list of str
        ``["energy", "free_energy", "forces"]``. ``free_energy`` is the same
        number as ``energy``: there is no electronic entropy in this pipeline,
        and ASE optimisers ask for it by name. See :meth:`compute_forces` for
        what ``forces`` is and is not.
    fields : dict
        ``{"external", "density", "tau"}`` from the most recent evaluation,
        each a :class:`~poraque.fields.base.ScalarField`.
    components : EnergyComponents
        Full energy decomposition of the most recent evaluation.

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
    >>> atoms = bulk("Au", "fcc", a=4.08, cubic=True)             # doctest: +SKIP
    >>> atoms.calc = Poraque("models/poraque_models.pfno",
    ...                      potcar="POTCAR")                     # doctest: +SKIP
    >>> atoms.get_potential_energy()                              # doctest: +SKIP
    -123.456
    >>> print(atoms.calc.components)                              # doctest: +SKIP
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    #: Relative electron-count drift above which the energy is warned about.
    #: Set where it is because the electrostatic terms are of order 1e4 eV, so
    #: 1e-3 already corresponds to a ~10 eV shift in the total.
    _DRIFT_WARNING = 1e-3

    def __init__(self, models=None, ext2chg=None, chg2tau=None, potcar=None,
                 potcar_dir=None, charges=None, resolution=None,
                 functional="pbe", device="auto", normalize_density=True,
                 **kwargs):
        if _ASE_ERROR is not None:                        # pragma: no cover
            raise ImportError(
                "The Poraque ASE calculator requires ASE: pip install ase"
            ) from _ASE_ERROR
        Calculator.__init__(self, **kwargs)

        self.device = device
        self.functional = functional
        self.normalize_density = bool(normalize_density)
        self.charges = dict(charges) if charges else None
        self.potcar_dir = str(potcar_dir) if potcar_dir else None
        self.fields = {}
        self.components = None
        self.raw_electron_drift = None
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
                "charges={'Au': 11.0} mapping: the external potential cannot "
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
        from .ml import FieldOperator, load_bundle

        if isinstance(source, FieldOperator):
            if source.task.name != task:
                raise ValueError(
                    f"Expected a {task!r} operator, got {source.task.name!r}. "
                    f"Chaining the wrong model produces plausible-looking "
                    f"garbage rather than an error."
                )
            return source

        return load_bundle(source, task, device=self.device)

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

        entries = []
        for element in elements:
            path = _find_potcar(self.potcar_dir, element)
            single = Potcar.from_string(_read_maybe_compressed(path),
                                        parse_tables=True)
            if not single:
                raise ValueError(f"{path} contains no POTCAR dataset.")
            if len(single) > 1:
                raise ValueError(
                    f"{path} holds {len(single)} datasets; a library entry "
                    f"must contain exactly one."
                )
            found = single[0].element
            if found != element:
                raise ValueError(
                    f"{path} is a POTCAR for {found!r}, not {element!r}."
                )
            entries.append(single[0])

        return self._validate_potcar(Potcar(entries))

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
        )
        components = calculator.compute(fields["density"], fields["tau"],
                                        potential)

        nominal = calculator.nominal_electrons
        raw_count = float(fields["density_raw"].integrate())
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

        # Computed only on request: it costs one extra pass over the grid per
        # atom, which an energy-only scan should not pay for.
        if "forces" in properties:
            self.results["forces"] = self.compute_forces(self.atoms)

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
           Measured against VASP on the shipped gold structures — using VASP's
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

        fields = self.fields or self.predict_fields(atoms)
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

        return hellmann_feynman_forces(
            fields["density"].data, structure, potential.grid, potcar=potcar)

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
#: Filenames a library entry may use, in preference order.
_POTCAR_NAMES = ("POTCAR", "POTCAR.gz", "POTCAR.Z")


def _find_potcar(directory, element):
    r"""
    Locate the ``POTCAR`` for ``element`` inside a library directory.

    Recognised layouts, in preference order:

    1. ``<dir>/<element>/POTCAR`` --- what VASP ships;
    2. ``<dir>/<element>_<variant>/POTCAR`` --- ``Au_pv``, ``Fe_sv``, ...;
    3. ``<dir>/POTCAR.<element>`` or ``<dir>/<element>.POTCAR`` --- flat.

    Each accepts a ``.gz`` or ``.Z`` suffix.

    Parameters
    ----------
    directory : str
        Library root.
    element : str
        Bare chemical symbol.

    Returns
    -------
    str
        Path to the file.

    Raises
    ------
    FileNotFoundError
        When nothing matches, listing what the directory does contain.
    ValueError
        When only *variant* directories match and there is more than one. The
        choice between ``Fe`` and ``Fe_pv`` changes ``ZVAL`` and therefore
        every energy, so it is the user's to make, not a coin flip.
    """
    exact = os.path.join(directory, element)
    if os.path.isdir(exact):
        for name in _POTCAR_NAMES:
            candidate = os.path.join(exact, name)
            if os.path.isfile(candidate):
                return candidate

    for stem in (f"POTCAR.{element}", f"{element}.POTCAR"):
        for suffix in ("", ".gz", ".Z"):
            candidate = os.path.join(directory, stem + suffix)
            if os.path.isfile(candidate):
                return candidate

    variants = sorted(
        entry for entry in os.listdir(directory)
        if entry.startswith(f"{element}_")
        and os.path.isdir(os.path.join(directory, entry))
        and any(os.path.isfile(os.path.join(directory, entry, name))
                for name in _POTCAR_NAMES)
    )
    if len(variants) == 1:
        chosen = os.path.join(directory, variants[0])
        for name in _POTCAR_NAMES:
            candidate = os.path.join(chosen, name)
            if os.path.isfile(candidate):
                warnings.warn(
                    f"No plain {element!r} POTCAR in {directory}; using the "
                    f"only variant present, {variants[0]!r}.",
                    RuntimeWarning, stacklevel=4,
                )
                return candidate
    if len(variants) > 1:
        raise ValueError(
            f"No plain {element!r} POTCAR in {directory}, and several "
            f"variants exist: {variants}. They differ in ZVAL and therefore "
            f"in every energy, so name the one you want by passing an "
            f"explicit potcar= file."
        )

    available = sorted(entry for entry in os.listdir(directory)
                       if not entry.startswith("."))[:20]
    raise FileNotFoundError(
        f"No POTCAR for {element!r} under {directory}. Expected "
        f"{element}/POTCAR, POTCAR.{element} or {element}.POTCAR. "
        f"The directory contains: {available}"
    )


def _read_maybe_compressed(path):
    """Read a POTCAR, transparently handling ``.gz``/``.Z`` compression."""
    if path.endswith((".gz", ".Z")):
        with gzip.open(path, "rt", errors="replace") as handle:
            return handle.read()
    with open(path, "r", errors="replace") as handle:
        return handle.read()


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
