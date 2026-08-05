# -*- coding: utf-8 -*-
# file: charges.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Charge conservation and atomic population analysis.

Three partitionings of a predicted density into per-atom charges, plus the
check that has to pass before any of them is worth reading.

.. math::

    q_A = Z^{\rm val}_A - \int w_A(\mathbf r)\,\rho(\mathbf r)\,d^3r ,
    \qquad \sum_A w_A(\mathbf r) = 1 ,

so the three methods differ only in the weight :math:`w_A`, and all three
conserve charge exactly by construction: the partitions are exhaustive, so the
populations sum to :math:`\int\rho` whatever the weights are.

======================  ==================================================
Method                  :math:`w_A(\mathbf r)`
======================  ==================================================
``"voronoi"``           1 for the nearest atom, 0 otherwise
``"hirshfeld"``         :math:`\rho^{\rm at}_A / \sum_B \rho^{\rm at}_B`
``"bader"``             1 inside :math:`A`'s zero-flux basin, 0 outside
======================  ==================================================

Which to use
------------
They answer different questions and disagree by design.

**Voronoi** is purely geometric — it never looks at the density except to
integrate it — so it is fast, has no parameters, and is a poor chemical
partition for atoms of unequal size: a large anion beside a small cation is cut
in half regardless of where the charge actually sits. Use it as a sanity check
and a baseline.

**Hirshfeld** weights by the free atoms, so it is chemically sensible and
smooth, but it is *defined by its reference*: the answer depends on the
promolecule, and a poor set of isolated-atom densities gives poor charges
quietly. Hirshfeld charges are also known to be small in magnitude compared to
other schemes.

**Bader** (QTAIM) is the only one with a definition intrinsic to the density —
the basins are bounded by surfaces of zero flux in :math:`\nabla\rho` — which
is what makes it the usual default despite being the most expensive.

.. warning::

   All three integrate the **pseudo** valence density. The PAW core is absent,
   so a Bader volume here is not the all-electron Bader volume, and the charges
   are systematically compressed toward zero relative to an all-electron
   analysis. This matters most for Bader, whose basin boundaries are set by the
   density's topology near the nuclei — exactly where the pseudisation is
   largest. Treat the numbers as comparative across a series, not absolute.

   For a proper Bader analysis VASP's own guidance is to sum ``AECCAR0`` and
   ``AECCAR2`` and pass that; Poraquê does not predict those.
"""

import itertools
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field

import numpy as np

#: Partitioning schemes accepted by :func:`partial_charges`.
PARTITION_METHODS = ("voronoi", "hirshfeld", "bader")

#: Fallback decay length for the promolecule, as a fraction of the covalent
#: radius. An exponential density ``exp(-r/a)`` has ``<r> = 3a``, so this puts
#: the mean radius of the model free atom at its covalent radius.
_DECAY_FRACTION = 1.0 / 3.0


# ===================================================================== #
# Results
# ===================================================================== #
@dataclass
class ChargeCheck:
    r"""
    Outcome of :func:`verify_total_charge`.

    Attributes
    ----------
    integrated : float
        :math:`\int\rho\,d^3r` on the grid, in electrons.
    expected : float
        The count the pseudopotentials fix, :math:`\sum_s N_s Z^{\rm val}_s`.
    absolute_error, relative_error : float
        ``integrated - expected`` and that divided by ``expected``.
    ok : bool
        Whether ``|relative_error|`` is within the requested tolerance.
    tolerance : float
        The tolerance it was judged against.
    """

    integrated: float
    expected: float
    absolute_error: float
    relative_error: float
    ok: bool
    tolerance: float

    def __str__(self):
        verdict = "OK" if self.ok else "FAILED"
        return (f"charge check {verdict}: {self.integrated:.6f} electrons "
                f"against an expected {self.expected:.6f} "
                f"({self.relative_error:+.3e} relative, "
                f"tolerance {self.tolerance:g})")


@dataclass
class PartialCharges:
    """
    Per-atom populations and net charges.

    Attributes
    ----------
    symbols : list of str
        One chemical symbol per atom, in the structure's order.
    populations : numpy.ndarray
        Electrons assigned to each atom.
    valence : numpy.ndarray
        Reference valence charge subtracted from each population.
    method : str
        Which partitioning produced this.
    details : dict
        Method-specific extras — the Bader backend used, the Hirshfeld
        reference source, the number of basin maxima found.
    """

    symbols: list
    populations: np.ndarray
    valence: np.ndarray
    method: str
    details: dict = field(default_factory=dict)

    @property
    def charges(self):
        r"""
        Net charge per atom, :math:`q_A = Z^{\rm val}_A - N_A`, in units of
        ``+e``.

        Positive is electron-deficient, the usual chemical convention.
        """
        return np.asarray(self.valence, dtype=float) - np.asarray(
            self.populations, dtype=float)

    @property
    def total_charge(self):
        """Sum of :attr:`charges`; zero for a neutral cell and a partition."""
        return float(np.sum(self.charges))

    @property
    def total_population(self):
        """Sum of :attr:`populations`, which must equal the integrated density."""
        return float(np.sum(self.populations))

    def as_dict(self):
        """Plain ``dict``, for a JSON summary."""
        return {
            "method": self.method,
            "symbols": list(self.symbols),
            "populations": np.asarray(self.populations).tolist(),
            "valence": np.asarray(self.valence).tolist(),
            "charges": self.charges.tolist(),
            "total_charge": self.total_charge,
            "details": dict(self.details),
        }

    def __str__(self):
        lines = [f"  {self.method} charges (e):",
                 f"    {'#':>4s} {'atom':<6s} {'population':>12s} "
                 f"{'valence':>9s} {'charge':>9s}"]
        charges = self.charges
        for index, symbol in enumerate(self.symbols):
            lines.append(f"    {index:4d} {symbol:<6s} "
                         f"{self.populations[index]:12.4f} "
                         f"{self.valence[index]:9.2f} {charges[index]:9.4f}")
        lines.append(f"    {'':>4s} {'sum':<6s} "
                     f"{self.total_population:12.4f} "
                     f"{np.sum(self.valence):9.2f} {self.total_charge:9.4f}")
        return "\n".join(lines)


# ===================================================================== #
# Task 1: charge conservation
# ===================================================================== #
def verify_total_charge(density_grid, lattice_vectors, expected_electrons,
                        tolerance=1e-3, warn=True):
    r"""
    Check that a density integrates to the electron count it should.

    .. math::

        \int\rho\,d^3r \;\approx\; \sum_{ijk}\rho_{ijk}\,\mathrm{d}V ,
        \qquad
        \mathrm{d}V = \frac{\Omega}{N_x N_y N_z} ,

    which is exact for a band-limited field on a uniform mesh — the trapezoid
    and the rectangle rules coincide under periodic boundary conditions, so
    there is no quadrature error to worry about here, only the field itself.

    Parameters
    ----------
    density_grid : array_like
        :math:`\rho` in e/Å³, shape ``(Nx, Ny, Nz)``. A
        :class:`~poraque.fields.ChargeDensity` or
        :class:`~poraque.fields.SpinDensity` is also accepted, in which case
        its total channel is used.
    lattice_vectors : array_like or FieldGrid
        ``(3, 3)`` cell in Å, rows being :math:`\mathbf a_1,\mathbf a_2,
        \mathbf a_3`; or a grid carrying one.
    expected_electrons : float
        :math:`\sum_s N_s Z^{\rm val}_s`, from the pseudopotentials.
    tolerance : float, optional
        Relative tolerance for :attr:`ChargeCheck.ok`. The default of
        :math:`10^{-3}` is not arbitrary: the electrostatic terms are of order
        :math:`10^{4}` eV, so a relative drift of :math:`10^{-3}` already moves
        a total energy by roughly 10 eV.
    warn : bool, optional
        Emit a :class:`RuntimeWarning` when the check fails.

    Returns
    -------
    ChargeCheck

    Examples
    --------
    >>> check = verify_total_charge(rho, cell, 297.0)      # doctest: +SKIP
    >>> check.ok                                           # doctest: +SKIP
    True
    """
    values = _density_values(density_grid)
    cell = _cell_of(lattice_vectors, values.shape)

    volume = float(abs(np.linalg.det(cell)))
    voxel = volume / values.size
    integrated = float(np.sum(values) * voxel)

    expected = float(expected_electrons)
    absolute = integrated - expected
    relative = absolute / expected if expected else float("inf")
    ok = bool(abs(relative) <= tolerance)

    check = ChargeCheck(integrated=integrated, expected=expected,
                        absolute_error=absolute, relative_error=relative,
                        ok=ok, tolerance=float(tolerance))
    if warn and not ok:
        warnings.warn(
            f"Charge is not conserved: the density integrates to "
            f"{integrated:.6f} electrons against an expected {expected:.6f} "
            f"({relative:+.3%}). Every electrostatic energy term is at least "
            f"linear in rho, so this propagates directly into the energy. "
            f"Normalize the density (ChargeDensity.normalized, or "
            f"normalize_density=True on the calculator) before trusting any "
            f"quantity derived from it.",
            RuntimeWarning, stacklevel=2,
        )
    return check


# ===================================================================== #
# Geometry shared by the partitionings
# ===================================================================== #
def _density_values(density):
    """Total density as a plain array, from a field or an array."""
    values = getattr(density, "total", None)          # SpinDensity
    if values is None:
        values = getattr(density, "data", density)    # ScalarField or array
    return np.asarray(values, dtype=float)


def _cell_of(source, shape=None):
    """Lattice vectors from a grid, a structure, or a bare array."""
    cell = getattr(source, "cell", source)
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3):
        raise ValueError(
            f"Expected (3, 3) lattice vectors, got shape {cell.shape}."
        )
    return cell


def _is_orthogonal(cell, tolerance=1e-8):
    """Whether the lattice vectors are mutually perpendicular."""
    gram = cell @ cell.T
    return bool(np.all(np.abs(gram - np.diag(np.diag(gram))) < tolerance
                       * max(1.0, np.abs(gram).max())))


def _distance_to_atom(scaled, atom_scaled, cell, orthogonal):
    r"""
    Minimum-image distance from every grid point to one atom, in Å.

    Parameters
    ----------
    scaled : numpy.ndarray
        ``(Nx, Ny, Nz, 3)`` fractional coordinates of the grid.
    atom_scaled : array_like
        ``(3,)`` fractional position of the atom.
    cell : numpy.ndarray
        ``(3, 3)`` lattice vectors.
    orthogonal : bool
        When true, wrapping the fractional difference into
        :math:`[-\tfrac12, \tfrac12)` already gives the minimum image and the
        27-image search is skipped. That shortcut is *only* valid for a
        rectangular cell: in a skewed one the nearest image can be a diagonal
        neighbour, and taking the wrap alone would overestimate the distance.

    Returns
    -------
    numpy.ndarray
        ``(Nx, Ny, Nz)`` distances in Å.
    """
    delta = scaled - np.asarray(atom_scaled, dtype=float)
    delta -= np.round(delta)

    if orthogonal:
        return np.linalg.norm(delta @ cell, axis=-1)

    best = None
    for shift in itertools.product((-1.0, 0.0, 1.0), repeat=3):
        distance = np.linalg.norm((delta + np.asarray(shift)) @ cell, axis=-1)
        best = distance if best is None else np.minimum(best, distance)
    return best


def _nearest_atom(scaled, positions_scaled, cell):
    """
    Index of the closest atom for every grid point, and that distance.

    Returns
    -------
    tuple of numpy.ndarray
        ``(owner, distance)``, both of the grid's shape.
    """
    orthogonal = _is_orthogonal(cell)
    owner = np.zeros(scaled.shape[:3], dtype=np.int32)
    best = None
    for index, atom in enumerate(positions_scaled):
        distance = _distance_to_atom(scaled, atom, cell, orthogonal)
        if best is None:
            best, owner[...] = distance, index
        else:
            closer = distance < best
            best = np.where(closer, distance, best)
            owner[closer] = index
    return owner, best


def _resolve_inputs(density, structure, grid):
    """Pull ``(values, structure, grid)`` out of a field or explicit arguments."""
    values = _density_values(density)
    grid = grid if grid is not None else getattr(density, "grid", None)
    structure = (structure if structure is not None
                 else getattr(density, "structure", None))
    if grid is None or structure is None:
        raise ValueError(
            "A bare density array needs an explicit structure= and grid=; "
            "pass a ChargeDensity and both come with it."
        )
    if values.shape != tuple(grid.shape):
        raise ValueError(
            f"density has shape {values.shape} but the grid is "
            f"{tuple(grid.shape)}."
        )
    return values, structure, grid


def _valence_array(structure, valence):
    """
    ``(natoms,)`` reference valence charges.

    Parameters
    ----------
    valence : dict, array_like or None
        ``{element: Z_val}``, one value per atom, or ``None`` for zeros —
        which makes :attr:`PartialCharges.charges` the negative population
        rather than a net charge, and is flagged in the result's details.
    """
    natoms = structure.natoms
    if valence is None:
        return np.zeros(natoms, dtype=float)

    if isinstance(valence, dict):
        lookup = {str(k).split("_")[0].split(".")[0]: float(v)
                  for k, v in valence.items()}
        out = np.empty(natoms, dtype=float)
        for symbol, atom_slice in structure.species_slices():
            element = symbol.split("_")[0].split(".")[0]
            if element not in lookup:
                raise KeyError(
                    f"No valence charge for {element!r}; pass valence="
                    f"{{'{element}': Z}} or read it from a POTCAR."
                )
            out[atom_slice] = lookup[element]
        return out

    out = np.asarray(valence, dtype=float).ravel()
    if out.size != natoms:
        raise ValueError(f"{out.size} valence charges for {natoms} atoms.")
    return out


def _atom_symbols(structure):
    """One symbol per atom, expanded from the species blocks."""
    symbols = []
    for symbol, atom_slice in structure.species_slices():
        symbols.extend([symbol] * (atom_slice.stop - atom_slice.start))
    return symbols


def _populations_from_weights(values, weights, voxel):
    """Integrate ``rho`` against a stack of weights, giving one number each."""
    return np.array([float(np.sum(values * weight) * voxel)
                     for weight in weights], dtype=float)


# ===================================================================== #
# Task 2: Voronoi
# ===================================================================== #
def voronoi_charges(density, structure=None, grid=None, valence=None):
    r"""
    Assign each voxel wholly to its nearest atom.

    The weight is a hard indicator, :math:`w_A(\mathbf r) = 1` where :math:`A`
    is the closest atom under the minimum-image convention and zero elsewhere.
    Distances respect periodic boundary conditions, and in a non-orthogonal
    cell the 27 nearest images are searched rather than trusting a wrap of the
    fractional coordinates — in a skewed lattice the nearest image can be a
    diagonal neighbour.

    Parameters
    ----------
    density : ChargeDensity, SpinDensity or array_like
        :math:`\rho` in e/Å³.
    structure : Structure, optional
    grid : FieldGrid, optional
    valence : dict or array_like, optional
        ``{element: Z_val}`` to subtract; without it the charges are minus the
        populations.

    Returns
    -------
    PartialCharges

    Notes
    -----
    Purely geometric: the density enters only through the integral, never
    through where the boundary is put. For atoms of unequal size that is a
    poor chemical partition — the boundary sits at the midpoint whatever the
    density does there — and it is the reason this is offered as a baseline
    rather than as a default.

    Voxels equidistant from several atoms are **shared equally** rather than
    awarded to the lowest index. The number of such voxels is reported as
    ``details["shared_voxels"]``; it is normally zero, and large on a
    symmetric cell sampled by a commensurate grid, which is exactly the case
    where an index-order tie-break would bias the result.
    """
    values, structure, grid = _resolve_inputs(density, structure, grid)
    cell = _cell_of(grid)
    orthogonal = _is_orthogonal(cell)
    scaled = grid.scaled_coordinates()
    positions = structure.scaled_positions
    natoms = structure.natoms

    # Equidistant voxels are shared, not given to the lowest index. On a real
    # density exact ties are rare, but on a symmetric structure sampled by a
    # commensurate grid whole *planes* of voxels are equidistant -- and
    # breaking those ties by index would hand one atom a systematic excess
    # that grows with the symmetry of the cell, which is precisely when the
    # answer is expected to be symmetric.
    tolerance = 1e-9 * float(np.linalg.norm(cell, axis=1).max())

    nearest = None
    for index in range(natoms):
        distance = _distance_to_atom(scaled, positions[index], cell, orthogonal)
        nearest = distance if nearest is None else np.minimum(nearest, distance)

    shares = np.zeros(grid.shape, dtype=float)
    for index in range(natoms):
        distance = _distance_to_atom(scaled, positions[index], cell, orthogonal)
        shares += (distance <= nearest + tolerance)

    voxel = grid.volume / values.size
    weighted = values / shares
    populations = np.empty(natoms, dtype=float)
    ties = 0
    for index in range(natoms):
        distance = _distance_to_atom(scaled, positions[index], cell, orthogonal)
        owned = distance <= nearest + tolerance
        populations[index] = float(np.sum(weighted[owned]) * voxel)
        ties += int(np.count_nonzero(owned))

    return PartialCharges(
        symbols=_atom_symbols(structure),
        populations=populations,
        valence=_valence_array(structure, valence),
        method="voronoi",
        details={"partition": "nearest atom, minimum image",
                 "orthogonal_cell": orthogonal,
                 "shared_voxels": int(np.count_nonzero(shares > 1))},
    )


# ===================================================================== #
# Task 3: Hirshfeld
# ===================================================================== #
def atomic_radial_profile(density, structure=None, grid=None, atom_index=0,
                          n_bins=200, max_radius=None):
    r"""
    Spherically average a density about one atom.

    Turns an isolated-atom calculation into the radial profile
    :math:`\rho^{\rm at}(r)` a promolecule is built from. Binning rather than
    interpolating is deliberate: the grid points are not on a radial mesh, and
    averaging what falls in each shell is the unbiased estimator of the
    spherical average.

    Parameters
    ----------
    density : ChargeDensity or array_like
    structure : Structure, optional
    grid : FieldGrid, optional
    atom_index : int, optional
        Which atom to centre on; an isolated-atom reference has only one.
    n_bins : int, optional
        Radial bins.
    max_radius : float, optional
        Outer radius in Å. Defaults to half the shortest cell dimension, past
        which the average is contaminated by periodic images.

    Returns
    -------
    tuple of numpy.ndarray
        ``(radii, density)`` in Å and e/Å³, suitable for
        :func:`numpy.interp`.
    """
    values, structure, grid = _resolve_inputs(density, structure, grid)
    cell = _cell_of(grid)

    distance = _distance_to_atom(grid.scaled_coordinates(),
                                 structure.scaled_positions[atom_index],
                                 cell, _is_orthogonal(cell))

    if max_radius is None:
        max_radius = 0.5 * float(np.linalg.norm(cell, axis=1).min())

    edges = np.linspace(0.0, max_radius, n_bins + 1)
    counts, _ = np.histogram(distance, bins=edges)
    totals, _ = np.histogram(distance, bins=edges, weights=values)

    occupied = counts > 0
    radii = 0.5 * (edges[:-1] + edges[1:])
    profile = np.zeros(n_bins, dtype=float)
    profile[occupied] = totals[occupied] / counts[occupied]
    return radii[occupied], profile[occupied]


def _exponential_promolecule(element, valence):
    r"""
    Fallback free-atom density: :math:`\rho(r) = \frac{N}{8\pi a^3}e^{-r/a}`.

    Normalized so :math:`\int\rho\,4\pi r^2 dr = N`, with the decay length set
    from the covalent radius through :math:`\langle r\rangle = 3a`. This is a
    crude stand-in for a real free-atom density and is used only when no
    reference calculation is available.

    Returns
    -------
    callable
        ``f(r) -> density`` in e/Å³.
    """
    try:
        from ase.data import atomic_numbers, covalent_radii

        radius = float(covalent_radii[atomic_numbers[element]])
    except (ImportError, KeyError):
        radius = 1.3
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.3

    decay = max(_DECAY_FRACTION * radius, 0.05)
    amplitude = float(valence) / (8.0 * np.pi * decay ** 3)

    def profile(r):
        return amplitude * np.exp(-np.asarray(r, dtype=float) / decay)

    return profile


def _reference_profiles(elements, valence_of, references):
    """
    ``{element: callable(r)}`` for the promolecule.

    A real isolated-atom calculation is used where one exists; the exponential
    model fills in the rest. Which was used for each element is reported in the
    result's details, because a promolecule silently built from a crude model
    gives crude charges with no other symptom.
    """
    profiles, sources = {}, {}

    for element in elements:
        directory = (os.path.join(str(references), element)
                     if references else None)
        chgcar = (os.path.join(directory, "CHGCAR") if directory else None)

        if chgcar and os.path.isfile(chgcar):
            try:
                from ..fields import ChargeDensity

                reference = ChargeDensity.read(chgcar)
                radii, values = atomic_radial_profile(reference)
                # Outside the tabulated range the free atom is essentially
                # empty; clamping to the last bin rather than extrapolating
                # keeps the weights finite and positive far from every atom.
                profiles[element] = (
                    lambda r, radii=radii, values=values:
                    np.interp(np.asarray(r, dtype=float), radii, values,
                              left=values[0], right=0.0))
                sources[element] = "isolated-atom CHGCAR"
                continue
            except (OSError, ValueError) as error:       # noqa: BLE001
                warnings.warn(
                    f"Could not read the isolated-atom reference for "
                    f"{element!r} ({error}); falling back to the exponential "
                    f"promolecule.",
                    RuntimeWarning, stacklevel=3,
                )

        profiles[element] = _exponential_promolecule(element,
                                                     valence_of(element))
        sources[element] = "exponential model"

    return profiles, sources


def hirshfeld_charges(density, structure=None, grid=None, valence=None,
                      references=None, epsilon=1e-30):
    r"""
    Partition by the free-atom weight
    :math:`w_A = \rho^{\rm at}_A / \sum_B \rho^{\rm at}_B`.

    Parameters
    ----------
    density : ChargeDensity, SpinDensity or array_like
    structure : Structure, optional
    grid : FieldGrid, optional
    valence : dict or array_like, optional
        ``{element: Z_val}``. Also sets the norm of the fallback free-atom
        densities, so supplying it improves the promolecule as well as the
        charges.
    references : str, optional
        Directory of isolated-atom calculations, one subdirectory per element
        (``data/vasp/ref``). Each ``CHGCAR`` found there is spherically
        averaged into :math:`\rho^{\rm at}(r)`. **Strongly preferred**: without
        it the promolecule is an exponential model, and Hirshfeld charges are
        defined by their reference.
    epsilon : float, optional
        Floor on the promolecule denominator, guarding the far interstitial
        where every free atom has decayed to nothing.

    Returns
    -------
    PartialCharges

    Notes
    -----
    Where the promolecule underflows, the weight is shared equally between all
    atoms rather than left undefined. That region carries essentially no charge
    — it is where every free-atom density has decayed — so the choice does not
    move the answer, but it does keep :math:`\sum_A w_A = 1` everywhere, which
    is what makes the populations sum to the integrated density exactly.
    """
    values, structure, grid = _resolve_inputs(density, structure, grid)
    cell = _cell_of(grid)
    orthogonal = _is_orthogonal(cell)
    scaled = grid.scaled_coordinates()

    valence_array = _valence_array(structure, valence)
    symbols = _atom_symbols(structure)
    elements = sorted({s.split("_")[0].split(".")[0] for s in symbols})

    def valence_of(element):
        for index, symbol in enumerate(symbols):
            if symbol.split("_")[0].split(".")[0] == element:
                return valence_array[index] or 1.0
        return 1.0

    profiles, sources = _reference_profiles(elements, valence_of, references)

    # Free-atom density of every atom on the grid, then the promolecule.
    natoms = structure.natoms
    proatom = np.empty((natoms,) + tuple(grid.shape), dtype=float)
    for index, symbol in enumerate(symbols):
        element = symbol.split("_")[0].split(".")[0]
        distance = _distance_to_atom(scaled, structure.scaled_positions[index],
                                     cell, orthogonal)
        proatom[index] = profiles[element](distance)

    promolecule = proatom.sum(axis=0)
    empty = promolecule < epsilon
    # Equal shares where the promolecule has underflowed, so the weights still
    # sum to one and no charge is lost.
    weights = np.where(empty, 1.0 / natoms,
                       proatom / np.where(empty, 1.0, promolecule))

    voxel = grid.volume / values.size
    populations = _populations_from_weights(values, weights, voxel)

    return PartialCharges(
        symbols=symbols,
        populations=populations,
        valence=valence_array,
        method="hirshfeld",
        details={"promolecule": sources,
                 "reference_directory": str(references) if references else None},
    )


# ===================================================================== #
# Task 4: Bader
# ===================================================================== #
def bader_charges(density, structure=None, grid=None, valence=None,
                  backend="auto", executable="bader"):
    r"""
    Partition into zero-flux (QTAIM) basins.

    Parameters
    ----------
    density : ChargeDensity, SpinDensity or array_like
    structure : Structure, optional
    grid : FieldGrid, optional
    valence : dict or array_like, optional
    backend : {"auto", "native", "external"}, optional
        ``"native"`` runs the vectorized on-grid steepest ascent below.
        ``"external"`` writes a ``CHGCAR`` and calls the Henkelman group's
        ``bader`` program. ``"auto"`` prefers the external one when it is on
        ``PATH`` — it is faster and is the reference implementation — and
        falls back to native otherwise.
    executable : str, optional
        Name or path of the external program.

    Returns
    -------
    PartialCharges

    Notes
    -----
    The native backend implements on-grid steepest ascent: every voxel points
    at whichever of its 26 neighbours gives the largest density increase *per
    unit distance*, and those pointers are followed to a local maximum. The
    distance weighting matters — without it the diagonal neighbours, which are
    :math:`\sqrt3` further away, win too often and the basins acquire a
    staircase bias along the cell diagonals.

    This is the grid-constrained approximation to the true zero-flux surface.
    It converges to the exact partition as the mesh is refined, and on a coarse
    mesh the boundaries are quantised to voxel faces; the Henkelman
    implementation additionally offers refinement schemes this does not.
    """
    values, structure, grid = _resolve_inputs(density, structure, grid)

    if backend not in ("auto", "native", "external"):
        raise ValueError(
            f"backend={backend!r} is not known; expected 'auto', 'native' or "
            f"'external'."
        )

    available = shutil.which(executable) is not None
    if backend == "external" and not available:
        raise FileNotFoundError(
            f"The {executable!r} program is not on PATH. Install the Henkelman "
            f"group's Bader analysis code, or use backend='native'."
        )

    use_external = available if backend == "auto" else (backend == "external")

    if use_external:
        populations, details = _bader_external(values, structure, grid,
                                               executable)
    else:
        populations, details = _bader_native(values, structure, grid)

    return PartialCharges(
        symbols=_atom_symbols(structure),
        populations=populations,
        valence=_valence_array(structure, valence),
        method="bader",
        details=details,
    )


def _neighbour_offsets():
    """The 26 neighbours of a voxel, as integer displacements."""
    return [offset for offset in itertools.product((-1, 0, 1), repeat=3)
            if offset != (0, 0, 0)]


def _bader_native(values, structure, grid):
    """
    On-grid steepest ascent, fully vectorized.

    Returns
    -------
    tuple of (numpy.ndarray, dict)
    """
    shape = tuple(grid.shape)
    cell = _cell_of(grid)

    flat_index = np.arange(values.size, dtype=np.int64).reshape(shape)
    parent = flat_index.copy()
    best_gain = np.zeros(shape, dtype=float)

    for offset in _neighbour_offsets():
        # Cartesian length of this step, so the gradient is per unit distance.
        step = np.array([offset[axis] / shape[axis] for axis in range(3)])
        length = float(np.linalg.norm(step @ cell))
        if length <= 0.0:
            continue

        shift = tuple(-component for component in offset)
        neighbour = np.roll(values, shift=shift, axis=(0, 1, 2))
        gain = (neighbour - values) / length

        steeper = gain > best_gain
        if steeper.any():
            best_gain = np.where(steeper, gain, best_gain)
            parent = np.where(steeper,
                              np.roll(flat_index, shift=shift, axis=(0, 1, 2)),
                              parent)

    # Follow the pointers to a fixed point. Doubling the pointer each round
    # gives a path of length L in log2(L) rounds, so this terminates quickly
    # even for long ascents across a large grid.
    parent = parent.ravel()
    for _ in range(64):
        successor = parent[parent]
        if np.array_equal(successor, parent):
            break
        parent = successor
    else:                                                # pragma: no cover
        warnings.warn(
            "The Bader ascent did not reach a fixed point in 64 rounds; the "
            "basins may be incomplete. This should not happen on a physical "
            "density — check for NaNs in the field.",
            RuntimeWarning, stacklevel=3,
        )

    maxima = np.unique(parent)

    # Each maximum belongs to the atom it is nearest to. A density can have
    # more maxima than atoms -- spurious ones in the interstitial from grid
    # noise, or genuine non-nuclear attractors in a metal -- so this is a
    # many-to-one assignment, not a pairing.
    maxima_scaled = np.stack(np.unravel_index(maxima, shape), axis=-1) / np.array(shape)
    owner_of_maximum, _ = _nearest_atom(maxima_scaled[:, None, None, :],
                                        structure.scaled_positions, cell)
    owner_of_maximum = owner_of_maximum.reshape(-1)

    lookup = np.zeros(parent.max() + 1, dtype=np.int32)
    lookup[maxima] = owner_of_maximum
    owner = lookup[parent]

    voxel = grid.volume / values.size
    populations = np.bincount(owner, weights=values.ravel(),
                              minlength=structure.natoms) * voxel

    return populations, {
        "backend": "native",
        "algorithm": "on-grid steepest ascent, 26 neighbours, "
                     "distance-weighted",
        "maxima": int(maxima.size),
        "atoms": int(structure.natoms),
    }


def _bader_external(values, structure, grid, executable):
    """
    Run the Henkelman ``bader`` program on a temporary ``CHGCAR``.

    Returns
    -------
    tuple of (numpy.ndarray, dict)
    """
    from ..fields import ChargeDensity

    workdir = tempfile.mkdtemp(prefix="poraque_bader_")
    try:
        chgcar = os.path.join(workdir, "CHGCAR")
        ChargeDensity(values, grid, structure).write(chgcar)

        try:
            completed = subprocess.run(
                [executable, "CHGCAR"], cwd=workdir, capture_output=True,
                text=True, timeout=3600, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(
                f"Running {executable!r} failed: {error}. Use "
                f"backend='native' to avoid the external dependency."
            ) from error

        acf = os.path.join(workdir, "ACF.dat")
        if completed.returncode != 0 or not os.path.isfile(acf):
            raise RuntimeError(
                f"{executable!r} exited with status {completed.returncode} and "
                f"wrote no ACF.dat.\n{completed.stderr.strip()[:400]}"
            )

        populations = _parse_acf(acf, structure.natoms)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return populations, {"backend": "external", "executable": executable}


def _parse_acf(path, natoms):
    """
    Read the ``CHARGE`` column of the Henkelman program's ``ACF.dat``.

    The file is a fixed table between two dashed rules, one row per atom:
    index, x, y, z, charge, minimum distance, atomic volume.
    """
    rows = []
    with open(path) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 5 or not fields[0].isdigit():
                continue
            try:
                rows.append(float(fields[4]))
            except ValueError:                           # pragma: no cover
                continue

    if len(rows) < natoms:
        raise RuntimeError(
            f"{path} lists {len(rows)} atoms but the structure has {natoms}."
        )
    return np.asarray(rows[:natoms], dtype=float)


# ===================================================================== #
# Unified entry point
# ===================================================================== #
def partial_charges(density, structure=None, grid=None, method="bader",
                    valence=None, **kwargs):
    r"""
    Partial charges by any of the supported partitionings.

    Parameters
    ----------
    density : ChargeDensity, SpinDensity or array_like
    structure : Structure, optional
    grid : FieldGrid, optional
    method : {"bader", "hirshfeld", "voronoi"}, optional
    valence : dict or array_like, optional
        ``{element: Z_val}`` to subtract from the populations.
    **kwargs
        Forwarded to the chosen partitioner — ``references=`` for Hirshfeld,
        ``backend=`` for Bader.

    Returns
    -------
    PartialCharges

    Raises
    ------
    ValueError
        On an unknown method, naming the ones that exist.
    """
    partitioners = {
        "voronoi": voronoi_charges,
        "hirshfeld": hirshfeld_charges,
        "bader": bader_charges,
    }
    key = str(method).lower()
    if key not in partitioners:
        raise ValueError(
            f"method={method!r} is not a known partitioning; expected one of "
            f"{list(PARTITION_METHODS)}."
        )
    return partitioners[key](density, structure=structure, grid=grid,
                             valence=valence, **kwargs)
