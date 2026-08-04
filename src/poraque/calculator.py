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
    atoms.calc = Poraque(ext2chg="models/ext2chg.pt",
                         chg2tau="models/chg2tau.pt",
                         potcar="POTCAR")
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

**No forces.** A machine-learned interatomic potential differentiates a scalar
energy with respect to the positions it was built from. Here the positions
enter through :math:`V_{\rm ext}` on a *fixed grid*, and the energy also
depends on them through the Ewald sum and the grid discretisation. The
derivative is well defined but is not wired up, so :meth:`get_forces` raises.
Geometry optimisation and molecular dynamics are therefore out of reach; single
points, energy-volume curves and ranking of fixed geometries are not.

**Absolute energies are not DFT energies.** The fields are pseudo-valence
quantities, so the PAW one-centre terms are missing. See
:class:`~poraque.physics.energy.EnergyComponents`.
"""

import warnings

import numpy as np

try:
    from ase.calculators.calculator import Calculator, all_changes
    _ASE_ERROR = None
except ImportError as error:                              # pragma: no cover
    Calculator, all_changes = object, None
    _ASE_ERROR = error

from .fields import ChargeDensity, ExternalPotential, FieldGrid
from .fields.structure import Structure
from .physics import EnergyCalculator

#: Fallback grid resolution when neither the caller nor the checkpoints say.
DEFAULT_RESOLUTION = 32


class Poraque(Calculator):
    r"""
    Predict the total energy of an :class:`ase.Atoms` object.

    Parameters
    ----------
    ext2chg : str or FieldOperator
        Checkpoint path or a loaded operator for :math:`V_{\rm ext}\to\rho`.
    chg2tau : str or FieldOperator
        Checkpoint path or a loaded operator for :math:`\rho\to\tau`.
    potcar : str, optional
        ``POTCAR`` for the species present. **Strongly preferred**: it supplies
        the tabulated local potential the operators were trained on, and the
        ``PSCORE`` values needed for the :math:`\mathbf G = 0` energy term.
    charges : dict, optional
        ``{element: Z_val}``, used only when ``potcar`` is absent. Selects the
        Gaussian pseudo-ion model — see the warning below.
    resolution : int, optional
        Longest grid axis. Defaults to the resolution recorded in the
        ``ext2chg`` checkpoint, else :data:`DEFAULT_RESOLUTION`.
    functional : str, optional
        Exchange-correlation approximation for
        :func:`~poraque.physics.energy.xc_energy`.
    device : str, optional
        ``"auto"`` (default), ``"cuda"``, ``"mps"`` or ``"cpu"``.
    **kwargs
        Passed to :class:`ase.calculators.calculator.Calculator`.

    Attributes
    ----------
    implemented_properties : list of str
        ``["energy", "free_energy"]``. ``free_energy`` is the same number:
        there is no electronic entropy in this pipeline, and ASE optimisers ask
        for it by name.
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
    >>> atoms.calc = Poraque("models/ext2chg.pt", "models/chg2tau.pt",
    ...                      potcar="POTCAR")                     # doctest: +SKIP
    >>> atoms.get_potential_energy()                              # doctest: +SKIP
    -123.456
    >>> print(atoms.calc.components)                              # doctest: +SKIP
    """

    implemented_properties = ["energy", "free_energy"]

    def __init__(self, ext2chg, chg2tau, potcar=None, charges=None,
                 resolution=None, functional="lda", device="auto", **kwargs):
        if _ASE_ERROR is not None:                        # pragma: no cover
            raise ImportError(
                "The Poraque ASE calculator requires ASE: pip install ase"
            ) from _ASE_ERROR
        Calculator.__init__(self, **kwargs)

        self.device = device
        self.functional = functional
        self.charges = dict(charges) if charges else None
        self.fields = {}
        self.components = None
        self._warned_gaussian = False

        self.ext2chg = self._resolve_operator(ext2chg, "ext2chg")
        self.chg2tau = self._resolve_operator(chg2tau, "chg2tau")

        self.potcar = self._read_potcar(potcar) if potcar else None
        if self.potcar is None and not self.charges:
            raise ValueError(
                "Poraque needs either a POTCAR (preferred) or an explicit "
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
        """Accept a live operator or load one from a checkpoint."""
        from .ml import FieldOperator

        if isinstance(source, FieldOperator):
            if source.task.name != task:
                raise ValueError(
                    f"Expected a {task!r} operator, got {source.task.name!r}. "
                    f"Chaining the wrong model produces plausible-looking "
                    f"garbage rather than an error."
                )
            return source

        import torch

        state = torch.load(source, map_location="cpu", weights_only=False)
        if state.get("task") != task:
            raise ValueError(
                f"{source}: checkpoint is for task {state.get('task')!r}, "
                f"but {task!r} is required at this stage of the pipeline."
            )
        return FieldOperator.load(source, device=self.device,
                                  **_backbone_kwargs(state["model_state"]))

    @staticmethod
    def _read_potcar(path):
        """Read a POTCAR with its local-potential tables."""
        from .fields.vasp.potcar import Potcar

        potcar = Potcar.from_file(path, parse_tables=True)
        missing = [entry.element for entry in potcar if not entry.has_local_table]
        if missing:
            warnings.warn(
                f"POTCAR entries {missing} carry no usable local-potential "
                f"table; the Gaussian pseudo-ion model will be used for the "
                f"whole cell instead. Predictions will be unreliable.",
                RuntimeWarning, stacklevel=3,
            )
            return None
        return potcar

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

        if self.potcar is not None:
            return ExternalPotential.from_potcar_tables(structure, grid,
                                                        self.potcar)

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

    def predict_fields(self, atoms):
        r"""
        Run the full chain and return the three fields.

        Parameters
        ----------
        atoms : ase.Atoms

        Returns
        -------
        dict
            ``{"external", "density", "tau"}``.
        """
        potential = self.build_external_potential(atoms)
        density = self.ext2chg.predict(potential)
        tau = self.chg2tau.predict(density)
        return {"external": potential, "density": density, "tau": tau}

    def energy_components(self, atoms):
        """
        Full energy decomposition for ``atoms``.

        Returns
        -------
        EnergyComponents
        """
        fields = self.predict_fields(atoms)
        potential = fields["external"]

        pscore = None
        if self.potcar is not None:
            pscore = {entry.element: entry.pscore for entry in self.potcar
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
        self.results["energy_components"] = components.as_dict()

    def get_forces(self, atoms=None):
        r"""
        Not implemented.

        Raises
        ------
        NotImplementedError
            Always. Two independent pieces are missing: the derivative of
            :math:`V_{\rm ext}` with respect to the ionic positions (analytic,
            a Hellmann-Feynman term), and back-propagation of
            :math:`\partial E/\partial\rho` through both operators. Neither is
            wired up, and a finite-difference stand-in on a fixed grid would
            be dominated by the grid's own discontinuity as atoms move between
            voxels.
        """
        raise NotImplementedError(
            "Poraque does not compute forces yet, so geometry optimisation and "
            "molecular dynamics are unavailable. Single-point energies and "
            "energy-volume scans work."
        )

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
        source = "potcar" if self.potcar is not None else "gaussian"
        return (f"Poraque(resolution={self.resolution}, v_ext={source}, "
                f"functional={self.functional!r}, "
                f"device={self.ext2chg.device_description})")


# ===================================================================== #
# Helpers
# ===================================================================== #
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


def _backbone_kwargs(weights):
    """
    Recover the FNO hyper-parameters from a checkpoint's tensor shapes.

    Storing them separately would let the two disagree; the shapes cannot.
    """
    prefix = "backbone." if any(k.startswith("backbone.") for k in weights) else ""
    spectral = [v for k, v in weights.items()
                if k.startswith(f"{prefix}blocks.") and k.endswith(".spectral.weight")]
    projection = [v for k, v in weights.items()
                  if k.startswith(f"{prefix}project.") and k.endswith(".weight")]
    lift = weights.get(f"{prefix}lift.weight")

    kwargs = {}
    if spectral:
        kwargs["width"] = int(spectral[0].shape[1])       # (4, in, out, m, m, m)
        kwargs["modes"] = int(spectral[0].shape[3])
        kwargs["n_layers"] = len(spectral)
    if projection:
        kwargs["projection_channels"] = int(projection[0].shape[0])
    if lift is not None:
        kwargs["use_coordinates"] = bool(int(lift.shape[1]) > 1)
    return kwargs
