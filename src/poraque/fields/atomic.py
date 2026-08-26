# -*- coding: utf-8 -*-
# file: atomic.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Isolated atoms, and the superposition of them that most of a density already is.

Two things follow from having one converged calculation per element.

**A baseline for the operator.** The overwhelming majority of a crystal's
valence density is just its free atoms placed side by side. Predicting
:math:`\rho` therefore spends most of the network's capacity reproducing
something that was never in doubt — the four orders of magnitude of core peak
that the ``asinh`` transform exists to absorb. Predicting the *residual*

.. math::

    \delta\rho(\mathbf r) = \rho(\mathbf r) - \rho_{\rm sup}(\mathbf r)

leaves only the bonding charge: small, smooth, sign-changing, and the part that
actually depends on the chemistry. See ``DESIGN_PAW.md`` §3.1 for the trade-offs
this brings with it, of which the sign change is the one that matters.

**A source of one-centre terms.** The isolated atom's own PAW augmentation
record travels with it, which is the only thing available for an element the
training set has never seen. It is *not* the best available source when the
element **is** in the training set — measured on this project's gold data, the
free-atom record is 86.6 % RMS away from a bulk Au site while the training-set
average is 9.9 % away. ``DESIGN_PAW.md`` §3.2 has the numbers and the reasoning;
:mod:`poraque.fields.vasp.augmentation` remains the default source.

How an atom is stored
---------------------
As a **radial reciprocal-space form factor**

.. math::

    f_s(|\mathbf G|) = \int \rho^{\rm at}_s(\mathbf r)\,
                       e^{-i\mathbf G\cdot\mathbf r}\, d^3r,
    \qquad f_s(0) = Z^{\rm val}_s ,

rather than as a real-space array. Three reasons, and the last is decisive:

1. The superposition is a reciprocal-space sum anyway, and the structure factor
   factorises into three 1D phase vectors — the same construction
   :class:`~poraque.fields.ExternalPotential` already uses for pseudo-ions.
2. The table is **grid-independent**: one entry serves every cell and every FFT
   shape, which is the invariance the whole architecture is built around.
3. The electron count comes out **exact by construction**, since
   :math:`f_s(0) = Z^{\rm val}_s` makes
   :math:`\int\rho_{\rm sup} = \sum_a Z^{\rm val}_a` with no normalization step.
   That is what keeps the interaction with the electron-count constraint
   tractable at inference.

Is an atom radial enough for this? Measured on ``data/vasp/ref/Au`` — one gold
atom in a 10 Å cube on a 108³ grid — the recentred :math:`f(\mathbf G)` is real
to :math:`7\times10^{-16}` relative, and the worst within-bin scatter is
**0.48 %** of :math:`f(0)` (recorded per entry as ``radial_scatter``).

That per-bin figure is the pessimistic one. The number that matters is
end-to-end: superposing the stored table back onto the reference atom's own grid
reproduces its density to a relative :math:`L^2` of
:math:`3.0\times10^{-4}` — the anisotropy is scattered across directions and
largely averages out of the sum. That is the *total* cost of the radial
reduction plus the binning, and it is the error the baseline contributes to
every :math:`\delta\rho` target.

For scale, the residual it leaves behind on a real supercell is
:math:`\lVert\rho - \rho_{\rm sup}\rVert / \lVert\rho\rVert =` **0.036**
(``struct_000``) and **0.037** (``struct_015``). So the baseline removes about
96 % of the field while introducing an error two orders of magnitude smaller
than what remains to be learned.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field as dataclass_field

import numpy as np

#: Default filename of the database.
LIBRARY_FILENAME = "atomic_reference.json"

#: Schema version, so a database written by an older Poraquê is recognised
#: rather than silently misread.
SCHEMA_VERSION = 1

#: Radial bins used when reducing f(G) to a table. 512 over the available range
#: is far finer than the physics — f is smooth on the scale of 1/R_core — and
#: cheap, so the binning is never the limiting approximation.
DEFAULT_BINS = 512


def base_element(symbol):
    """
    ``"Au_pv"``, ``"Au.pbe"``, ``"Au1"`` all to ``"Au"``.

    One more copy of an idiom this tree already spells five ways; ``FUTURE.md``
    tracks unifying them. Kept local rather than importing one of the five,
    because picking a winner is a change to every call site of the other four.
    """
    text = str(symbol).split("_")[0].split(".")[0]
    return "".join(character for character in text
                   if character.isalpha()) or text


@dataclass
class AtomicReference:
    r"""
    One isolated atom: its form factor, its one-centre record, its provenance.

    Attributes
    ----------
    element : str
        Bare chemical symbol.
    valence_charge : float
        :math:`Z^{\rm val}`, equal to :math:`f(0)` by construction.
    g_grid, form_factor : list of float
        The radial table, ascending in :math:`|G|` (Å⁻¹). ``form_factor`` is
        dimensionless and :math:`f(0) = Z^{\rm val}`.
    g_max : float
        Range of the table. Beyond it the species contributes **zero**, not an
        extrapolation — the same truncation VASP applies to its own tabulated
        local potential beyond ``PSGMAX``, and for the same reason: an
        extrapolated form factor is a fabricated one.
    radial_scatter : float
        Worst within-bin relative scatter measured when the table was built —
        i.e. how non-spherical the reference atom actually was.
    augmentation : list of float or None
        The atom's own PAW augmentation record, if its ``CHGCAR`` carried one.
    potcar_title, potcar_sha256 : str or None
        Which pseudopotential this atom was computed with. Two variants of one
        element are different atoms and must not be merged.
    source, vasp_version, incar_sha256 : str or None
        Where it came from.
    cell_volume : float or None
    grid : list of int or None
    """

    element: str
    valence_charge: float
    g_grid: list = dataclass_field(default_factory=list)
    form_factor: list = dataclass_field(default_factory=list)
    g_max: float = 0.0
    radial_scatter: float = 0.0
    augmentation: list = None
    potcar_title: str = None
    potcar_sha256: str = None
    source: str = None
    vasp_version: str = None
    incar_sha256: str = None
    cell_volume: float = None
    grid: list = None

    @property
    def key(self):
        """
        ``element|potcar_title|hash16`` — the identity of one reference.

        Keyed on the pseudopotential as well as the element because ``Au`` and
        ``Au_pv`` have different valence counts and different pseudo-densities;
        merging them would produce a baseline that is wrong for both.
        """
        title = self.potcar_title or "unknown"
        digest = (self.potcar_sha256 or "0" * 16)[:16]
        return f"{self.element}|{title}|{digest}"

    def evaluate(self, g_magnitude):
        r"""
        The form factor on an arbitrary array of :math:`|G|`.

        Interpolated **linearly in** :math:`G^2`, not in :math:`|G|`, and zero
        beyond ``g_max``.

        The square matters, and it matters most exactly where the table is
        sparsest. A spherically symmetric density has a form factor that is an
        *even* function of :math:`\mathbf G`, so :math:`f` is analytic in
        :math:`G^2` and has **zero slope at the origin**:

        .. math::

            f(G) = Z^{\rm val} - \tfrac{1}{6}\langle r^2\rangle G^2
                   + \mathcal{O}(G^4).

        Meanwhile the reference cell's own reciprocal lattice has nothing
        between :math:`G = 0` and its first shell at :math:`2\pi/L` — a gap of
        0.63 Å⁻¹ for the shipped 10 Å gold atom — so any target grid with
        points inside that gap is served by interpolation alone. A chord drawn
        in :math:`|G|` cuts straight across the curvature and gets the slope at
        the origin wrong by construction. Measured against a Gaussian atom,
        whose :math:`f` is known in closed form, interpolating in :math:`|G|`
        was off by 3.3 % of :math:`f(0)` in that first gap; in :math:`G^2` the
        same table is off by 0.2 %.

        Linear rather than cubic, still: the table is far finer than :math:`f`
        varies everywhere else, and a spline can overshoot near the truncation.

        Parameters
        ----------
        g_magnitude : array_like
            :math:`|G|` in Å⁻¹.

        Returns
        -------
        numpy.ndarray
        """
        magnitude = np.asarray(g_magnitude, dtype=float)
        squares = np.square(np.asarray(self.g_grid, dtype=float))
        values = np.interp(magnitude ** 2, squares, self.form_factor,
                           left=self.form_factor[0], right=0.0)
        return np.where(magnitude <= self.g_max, values, 0.0)

    def to_dict(self):
        return {
            "element": self.element,
            "valence_charge": float(self.valence_charge),
            "g_grid": [float(v) for v in self.g_grid],
            "form_factor": [float(v) for v in self.form_factor],
            "g_max": float(self.g_max),
            "radial_scatter": float(self.radial_scatter),
            "augmentation": (None if self.augmentation is None
                             else [float(v) for v in self.augmentation]),
            "potcar_title": self.potcar_title,
            "potcar_sha256": self.potcar_sha256,
            "source": self.source,
            "vasp_version": self.vasp_version,
            "incar_sha256": self.incar_sha256,
            "cell_volume": (None if self.cell_volume is None
                            else float(self.cell_volume)),
            "grid": (None if self.grid is None
                     else [int(v) for v in self.grid]),
        }

    @classmethod
    def from_dict(cls, mapping):
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in mapping.items() if k in known})


class AtomicReferenceLibrary:
    """
    The isolated-atom database: entries, lookup, and a hash of the whole thing.

    The hash matters. A model trained in δ-density mode has the baseline built
    into its target, so a database that changes underneath it silently
    invalidates every weight. :attr:`fingerprint` is recorded in the checkpoint
    and compared on load.

    Parameters
    ----------
    entries : dict, optional
        ``{key: AtomicReference}``.
    """

    def __init__(self, entries=None):
        self.entries = dict(entries or {})

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.entries)

    def __contains__(self, element):
        return self.lookup(element) is not None

    def __repr__(self):
        return (f"AtomicReferenceLibrary({len(self.entries)} entries: "
                f"{sorted(self.elements())})")

    def elements(self):
        """Bare element symbols covered, deduplicated."""
        return sorted({entry.element for entry in self.entries.values()})

    def add(self, reference):
        """Insert or replace one entry."""
        self.entries[reference.key] = reference
        return reference

    def lookup(self, symbol, potcar_title=None):
        """
        Find the reference for a species.

        Parameters
        ----------
        symbol : str
            Chemical symbol, decorated (``"Au_pv"``) or not.
        potcar_title : str, optional
            Pin the pseudopotential variant. Without it, and with exactly one
            variant stored for the element, that one is returned — the normal
            case, and the one that keeps a caller who has no ``POTCAR`` working.
            With several stored and no title given the answer is **ambiguous**
            and ``None`` comes back rather than an arbitrary pick.

        Returns
        -------
        AtomicReference or None
        """
        element = base_element(symbol)
        candidates = [entry for entry in self.entries.values()
                      if entry.element == element]
        if not candidates:
            return None
        if potcar_title:
            titled = [entry for entry in candidates
                      if entry.potcar_title == potcar_title]
            if titled:
                return titled[0]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def missing_for(self, structure, titles=None):
        """Species of ``structure`` with no usable entry, sorted."""
        titles = titles or {}
        absent = set()
        for symbol in structure.symbols:
            element = base_element(symbol)
            if self.lookup(symbol, titles.get(element)) is None:
                absent.add(element)
        return sorted(absent)

    # ------------------------------------------------------------------ #
    @property
    def fingerprint(self):
        """
        SHA-256 over every entry's table and identity.

        This is what a δ-density checkpoint records, so a baseline that has
        changed since training is an error rather than a silent bias.
        """
        digest = hashlib.sha256()
        for key in sorted(self.entries):
            entry = self.entries[key]
            digest.update(key.encode())
            digest.update(np.asarray(entry.g_grid, dtype=float).tobytes())
            digest.update(np.asarray(entry.form_factor, dtype=float).tobytes())
        return digest.hexdigest()

    def save(self, path):
        """Write the database as JSON."""
        path = str(path)
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "version": SCHEMA_VERSION,
            "fingerprint": self.fingerprint,
            "entries": {key: entry.to_dict()
                        for key, entry in sorted(self.entries.items())},
        }
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)
        return path

    @classmethod
    def load(cls, path):
        """
        Read a database, or an empty one when the file is absent.

        Parameters
        ----------
        path : str
            The JSON file, or a directory containing
            :data:`LIBRARY_FILENAME`.

        Returns
        -------
        AtomicReferenceLibrary
        """
        path = str(path)
        if os.path.isdir(path):
            path = os.path.join(path, LIBRARY_FILENAME)
        if not os.path.exists(path):
            return cls()

        with open(path) as handle:
            payload = json.load(handle)

        version = int(payload.get("version", 0))
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"{path} declares schema version {version}; this Poraquê "
                f"understands up to {SCHEMA_VERSION}. Reading it would "
                f"silently drop whatever the newer version added.")

        return cls({key: AtomicReference.from_dict(value)
                    for key, value in (payload.get("entries") or {}).items()})


# ---------------------------------------------------------------------- #
# Building an entry from a calculation
# ---------------------------------------------------------------------- #
def form_factor_from_density(density, grid=None, structure=None, bins=None,
                             g_max=None):
    r"""
    Reduce one isolated atom's density to a radial form factor table.

    The steps, and why each is there:

    1. Take the Fourier **series** coefficients,
       :math:`c_{\mathbf G} = \Omega_0^{-1}\!\int\rho\,e^{-i\mathbf G\cdot
       \mathbf r}`, which is ``fftn(rho) / npoints``.
    2. **Recentre** on the atom by multiplying by
       :math:`e^{+i\mathbf G\cdot\boldsymbol\tau}`. A reference cell normally
       puts the atom at the middle, not the origin, and the uncorrected phase
       would be baked into the table and then applied to every atom of every
       structure.
    3. Multiply by :math:`\Omega_0` to get :math:`f(\mathbf G)` with
       :math:`f(0) = Z^{\rm val}`.
    4. Bin by :math:`|G|` and average. The scatter within each bin is the
       measurement of how non-spherical the atom was, and is returned.

    Parameters
    ----------
    density : ChargeDensity or array_like
        The isolated atom's density in e/Å³.
    grid : FieldGrid, optional
        Taken from ``density`` when it is a field.
    structure : Structure, optional
        Likewise. Must hold exactly one atom.
    bins : int, optional
        Radial bins; :data:`DEFAULT_BINS` by default.
    g_max : float, optional
        Table range in Å⁻¹. Defaults to the largest :math:`|G|` the reference
        grid resolves along its *shortest* reciprocal axis, which is the
        largest radius at which the mesh samples a full sphere.

    Returns
    -------
    dict
        ``g_grid``, ``form_factor``, ``g_max``, ``radial_scatter``,
        ``valence_charge``.

    Raises
    ------
    ValueError
        If the structure does not hold exactly one atom. Averaging several
        would silently produce the form factor of a molecule.
    """
    values = np.asarray(getattr(density, "data", density), dtype=float)
    grid = grid if grid is not None else density.grid
    structure = structure if structure is not None else density.structure

    positions = np.atleast_2d(structure.scaled_positions)
    if positions.shape[0] != 1:
        raise ValueError(
            f"An atomic reference needs exactly one atom; this structure has "
            f"{positions.shape[0]}. A cell with several atoms gives the form "
            f"factor of the group, not of an atom.")

    coefficients = np.fft.fftn(values) / grid.npoints
    m1, m2, m3 = grid.fft_frequencies()
    tau = positions[0]
    phase = (np.exp(2j * np.pi * m1 * tau[0])[:, None, None]
             * np.exp(2j * np.pi * m2 * tau[1])[None, :, None]
             * np.exp(2j * np.pi * m3 * tau[2])[None, None, :])
    f_g = np.real(coefficients * phase) * grid.volume

    magnitude = np.sqrt(grid.get_g2())
    if g_max is None:
        # The Nyquist radius of the *shortest* reciprocal axis: beyond it the
        # mesh samples only the corners of the Brillouin zone, so a radial
        # average there would be averaging over whichever directions happen to
        # survive rather than over a sphere.
        g_max = float(min(
            0.5 * n * np.linalg.norm(b)
            for n, b in zip(grid.shape, _reciprocal_vectors(grid))))

    bins = int(bins or DEFAULT_BINS)
    edges = np.linspace(0.0, float(g_max), bins + 1)
    flat_g, flat_f = magnitude.ravel(), f_g.ravel()
    inside = flat_g <= g_max
    flat_g, flat_f = flat_g[inside], flat_f[inside]

    index = np.clip(np.digitize(flat_g, edges) - 1, 0, bins - 1)
    counts = np.bincount(index, minlength=bins)
    sums = np.bincount(index, weights=flat_f, minlength=bins)
    squares = np.bincount(index, weights=flat_f ** 2, minlength=bins)
    g_sums = np.bincount(index, weights=flat_g, minlength=bins)

    occupied = counts > 0
    means = np.zeros(bins)
    means[occupied] = sums[occupied] / counts[occupied]

    # The bin's **mean |G|**, not its geometric centre. The reciprocal lattice
    # points inside a shell are not uniformly spread across it -- there are few
    # of them at small |G| and they cluster -- so pairing the mean of f with
    # the midpoint of the bin mismatches the two by the local curvature of f.
    # Measured on a Gaussian atom, whose f is known in closed form, that
    # mismatch alone cost an order of magnitude: 9e-3 relative L2 on a
    # round-trip against 6e-4 with the mean.
    centres = np.zeros(bins)
    centres[occupied] = g_sums[occupied] / counts[occupied]

    variance = np.zeros(bins)
    variance[occupied] = np.maximum(
        squares[occupied] / counts[occupied] - means[occupied] ** 2, 0.0)

    valence = float(f_g.flat[0])
    scatter = (float(np.sqrt(variance).max() / abs(valence))
               if valence else 0.0)

    # G = 0 shares its bin with a shell of small but non-zero |G|, so the bin's
    # mean is not f(0). It is prepended explicitly instead: the total charge of
    # every superposition is f(0), and interpolating it out of the leading bin
    # would move the electron count off the integer it is supposed to be.
    g_grid = np.concatenate(([0.0], centres[occupied]))
    table = np.concatenate(([valence], means[occupied]))
    keep = np.concatenate(([True], g_grid[1:] > 0.0))
    g_grid, table = g_grid[keep], table[keep]

    return {"g_grid": g_grid.tolist(), "form_factor": table.tolist(),
            "g_max": float(g_max), "radial_scatter": scatter,
            "valence_charge": valence}


def reference_from_calculation(directory, filename="CHGCAR", bins=None):
    """
    Build one :class:`AtomicReference` from an isolated-atom calculation.

    Parameters
    ----------
    directory : str
        A calculation directory holding a single-atom ``CHGCAR``.
    filename : str, optional
    bins : int, optional

    Returns
    -------
    AtomicReference

    Raises
    ------
    FileNotFoundError
        When the density is not there.
    ValueError
        When the cell holds more than one atom.
    """
    from .density import ChargeDensity
    from .grid import FieldGrid
    from .vasp.augmentation import parse_augmentation
    from .vasp.volumetric import read_augmentation

    path = (directory if os.path.isfile(directory)
            else os.path.join(directory, filename))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path}: no isolated-atom density here. An entry needs the "
            f"CHGCAR of a single atom in a box.")

    grid = FieldGrid.from_file(path)
    density = ChargeDensity.read(path, grid=grid)
    table = form_factor_from_density(density, bins=bins)

    root = directory if os.path.isdir(directory) else os.path.dirname(path)

    augmentation = None
    try:
        _, block = read_augmentation(path)
        records = parse_augmentation(block)
        if len(records) == 1:
            augmentation = records[0].tolist()
    except (OSError, ValueError):
        augmentation = None

    title, potcar_hash = _potcar_identity(root)
    version, incar_hash = _run_identity(root)

    return AtomicReference(
        element=base_element(density.structure.symbols[0]),
        valence_charge=table["valence_charge"],
        g_grid=table["g_grid"], form_factor=table["form_factor"],
        g_max=table["g_max"], radial_scatter=table["radial_scatter"],
        augmentation=augmentation,
        potcar_title=title, potcar_sha256=potcar_hash,
        source=str(directory), vasp_version=version, incar_sha256=incar_hash,
        cell_volume=float(grid.volume), grid=list(grid.shape),
    )


def build_library(directories, filename="CHGCAR", bins=None, library=None,
                  log=None):
    """
    Ingest several isolated-atom calculations into one database.

    Parameters
    ----------
    directories : iterable of str
        One per atom. A parent directory holding one subdirectory per element
        is expanded automatically, so ``data/vasp/ref`` works as written.
    filename : str, optional
    bins : int, optional
    library : AtomicReferenceLibrary, optional
        Extend this one rather than starting fresh.
    log : callable, optional

    Returns
    -------
    AtomicReferenceLibrary
    """
    emit = log or (lambda *_: None)
    library = library if library is not None else AtomicReferenceLibrary()

    for directory in _expand(directories, filename):
        try:
            reference = reference_from_calculation(directory, filename, bins)
        except (FileNotFoundError, ValueError) as error:
            emit(f"  skipped {directory}: {error}")
            continue
        library.add(reference)
        emit(f"  {reference.element:<4s} Zval {reference.valence_charge:6.3f}  "
             f"g_max {reference.g_max:5.2f} 1/Ang  "
             f"{len(reference.g_grid)} table points  "
             f"anisotropy {100 * reference.radial_scatter:.2f}%  "
             f"aug {'yes' if reference.augmentation else 'no'}  "
             f"[{reference.potcar_title or 'no POTCAR'}]")
    return library


# ---------------------------------------------------------------------- #
# The superposition
# ---------------------------------------------------------------------- #
def resolve_library(reference, cache=None, log=None):
    r"""
    Turn whatever ``data.atomic_reference`` names into a loaded database.

    Three spellings are accepted, because three are what people actually have:

    ``atomic_reference.json``
        A database built earlier by ``poraque-atoms``. Loaded as it stands.
    a directory of isolated-atom runs
        ``~/Simulations/vasp/metals/Pt`` holding ``1.atom/``, or the atom
        directory itself. Ingested on the spot, and — when ``cache`` is given —
        written there as :data:`LIBRARY_FILENAME` so the next run reads the
        JSON instead of re-reducing a 140³ density.
    a directory that already holds one
        The JSON inside it wins over re-ingesting, for the same reason.

    Accepting the raw calculation directory matters more than it looks. The
    isolated atoms are the reference for *both* the delta-density baseline and
    the PAW augmentation records, so requiring a separate build step before
    either works is a step that gets skipped, and skipping it silently changes
    what the model is trained on.

    Parameters
    ----------
    reference : str
        Path, in any of the three forms above.
    cache : str, optional
        Where to memoise an ingested database.
    log : callable, optional

    Returns
    -------
    AtomicReferenceLibrary

    Raises
    ------
    FileNotFoundError
        When the path does not exist.
    ValueError
        When it exists but yields no isolated atom — naming what was looked
        for, since the commonest cause is pointing at a *bulk* run.
    """
    emit = log or (lambda *_: None)
    # `~` and `$VAR` expanded here rather than left to the caller: this value
    # comes out of a YAML file, where writing `~/Simulations/...` is the
    # natural thing to type and the only alternative is hard-coding a home
    # directory into a committed config -- which this repo already does
    # elsewhere and regrets.
    reference = os.path.expanduser(os.path.expandvars(str(reference)))

    if not os.path.exists(reference):
        raise FileNotFoundError(
            f"{reference}: no such path. data.atomic_reference wants either an "
            f"atomic_reference.json built by `poraque-atoms`, or a directory "
            f"of isolated-atom calculations to ingest.")

    # A file, or a directory with a database already in it.
    direct = reference
    if os.path.isdir(reference):
        candidate = os.path.join(reference, LIBRARY_FILENAME)
        direct = candidate if os.path.exists(candidate) else None
    if direct and os.path.isfile(direct):
        library = AtomicReferenceLibrary.load(direct)
        if len(library):
            emit(f"      atomic reference: {len(library)} atom(s) "
                 f"{library.elements()} from {direct}")
            return library

    # A memoised ingest from an earlier run of this same cache.
    if cache:
        memo = os.path.join(cache, LIBRARY_FILENAME)
        if os.path.exists(memo):
            library = AtomicReferenceLibrary.load(memo)
            if len(library):
                emit(f"      atomic reference: {len(library)} atom(s) "
                     f"{library.elements()} (cached from {reference})")
                return library

    emit(f"      atomic reference: ingesting isolated atoms from {reference}")
    library = build_library(reference, log=emit)
    if not len(library):
        raise ValueError(
            f"{reference} yielded no isolated-atom reference. Each one must be "
            f"a directory holding the CHGCAR of a **single atom** in a box; a "
            f"bulk or slab run has more than one atom and is skipped. Looked "
            f"in {reference} and one level below it.")

    if cache:
        os.makedirs(cache, exist_ok=True)
        written = library.save(os.path.join(cache, LIBRARY_FILENAME))
        emit(f"      atomic reference: memoised to {written}")
    return library


def augmentation_reference(library):
    """
    The per-element PAW table, in the shape the model bundle already stores.

    :mod:`poraque.fields.vasp.augmentation`'s ``build_reference`` produces
    ``{element: {"values": [...], "atoms": n, "structures": n}}`` by averaging
    over *material* calculations. This produces the same shape from the
    **isolated atoms** instead, so everything downstream —
    ``records_for_structure``, the bundle metadata, the inference writer — is
    unchanged and only the provenance of the numbers differs.

    ``atoms`` and ``structures`` are 1 apiece and honestly so: a free atom is
    one atom in one calculation, and a reader comparing this against an
    averaged table should be able to see that immediately.

    Parameters
    ----------
    library : AtomicReferenceLibrary

    Returns
    -------
    dict
        ``{element: {...}}``, empty when no stored atom carried a record.
    """
    reference = {}
    for entry in library.entries.values():
        if not entry.augmentation:
            continue
        # Two POTCAR variants of one element would collide here. The first
        # wins and the second is dropped rather than averaged: they are
        # different pseudopotentials with different projector counts, and
        # mixing their occupancies would produce a record of neither.
        if entry.element in reference:
            continue
        reference[entry.element] = {
            "values": list(entry.augmentation),
            "atoms": 1,
            "structures": 1,
            "source": "isolated_atom",
            "potcar_title": entry.potcar_title,
        }
    return reference


def atomic_superposition(structure, grid, library, titles=None,
                         metadata=None):
    r"""
    Place the isolated atoms of ``structure`` onto ``grid``.

    .. math::

        \rho_{\rm sup}(\mathbf G) = \frac{1}{\Omega}\sum_s
            f_s(|\mathbf G|)\, S_s(\mathbf G),
        \qquad
        S_s(\mathbf G) = \sum_{a\in s} e^{-i\mathbf G\cdot\boldsymbol\tau_a}

    followed by one inverse FFT. Exactly periodic, exactly translation
    covariant, and :math:`O(N\log N)` — no real-space cutoff and no
    minimum-image approximation anywhere.

    The electron count is exact rather than approximate:
    :math:`\rho_{\rm sup}(\mathbf G = 0) = \Omega^{-1}\sum_a Z^{\rm val}_a`, so
    :math:`\int\rho_{\rm sup}\,d^3r = \sum_a Z^{\rm val}_a` to machine
    precision.

    Parameters
    ----------
    structure : Structure
        Geometry. Only species grouping and fractional coordinates are used, so
        the result does not depend on the cell's orientation.
    grid : FieldGrid
        Target mesh — any shape, not the reference atoms'.
    library : AtomicReferenceLibrary
        The isolated-atom database.
    titles : dict, optional
        ``{element: potcar_title}``, to pin a pseudopotential variant.
    metadata : dict, optional
        Extra provenance for the returned field.

    Returns
    -------
    ChargeDensity

    Raises
    ------
    KeyError
        When a species has no entry. A *partial* superposition is refused
        deliberately: it has the right units and a plausible shape and is wrong
        by whole atoms, which is the worst kind of answer to return.
    """
    from .density import ChargeDensity
    from .external import structure_factor

    titles = titles or {}
    missing = library.missing_for(structure, titles)
    if missing:
        raise KeyError(
            f"No isolated-atom reference for {missing}. The database covers "
            f"{library.elements()}. Ingest the missing atom(s) with "
            f"`poraque-atoms`, or train in absolute-density mode "
            f"(`data.delta_density: false`).")

    magnitude = np.sqrt(grid.get_g2())
    rho_g = np.zeros(grid.shape, dtype=complex)
    charges = {}

    for symbol, atom_slice in structure.species_slices():
        element = base_element(symbol)
        entry = library.lookup(symbol, titles.get(element))
        charges[element] = entry.valence_charge
        rho_g += (entry.evaluate(magnitude)
                  * structure_factor(grid,
                                      structure.scaled_positions[atom_slice]))

    rho_g /= grid.volume
    data = np.real(np.fft.ifftn(rho_g) * grid.npoints)

    payload = {"model": "atomic_superposition",
               "charges": charges,
               "library_fingerprint": library.fingerprint}
    payload.update(metadata or {})
    return ChargeDensity(data, grid, structure, metadata=payload)


def augmentation_from_atoms(structure, library, titles=None):
    """
    Per-atom augmentation records taken from the isolated-atom database.

    The **fallback** source, not the preferred one. Measured on this project's
    gold data, a free Au atom's record is 86.6 % RMS away from a bulk Au site
    while the training-set average is 9.9 % away, so
    :func:`~poraque.fields.vasp.augmentation.records_for_structure` is what a
    prediction should normally use. This exists for the case that one cannot
    cover: an element the training set has never seen.

    Parameters
    ----------
    structure : Structure
    library : AtomicReferenceLibrary
    titles : dict, optional

    Returns
    -------
    tuple of (list of str, list of str)
        The formatted lines, and the elements that had no record. As with
        :func:`~poraque.fields.vasp.augmentation.records_for_structure`, a
        partial block is never returned.
    """
    from .vasp.augmentation import format_augmentation, species_of_each_atom

    titles = titles or {}
    records, missing = [], set()
    for symbol in species_of_each_atom(structure):
        element = base_element(symbol)
        entry = library.lookup(symbol, titles.get(element))
        if entry is None or not entry.augmentation:
            missing.add(element)
            continue
        records.append(np.asarray(entry.augmentation, dtype=float))

    if missing:
        return [], sorted(missing)
    return format_augmentation(records), []


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _reciprocal_vectors(grid):
    """Rows of :math:`2\\pi(A^{-1})^{T}`, i.e. the reciprocal lattice."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(grid.cell, dtype=float)).T


def _expand(directories, filename):
    """Accept atom directories, or one parent holding several."""
    if isinstance(directories, (str, os.PathLike)):
        directories = [directories]

    expanded = []
    for entry in directories:
        entry = str(entry)
        if os.path.isfile(entry):
            expanded.append(entry)
            continue
        if os.path.exists(os.path.join(entry, filename)):
            expanded.append(entry)
            continue
        children = [os.path.join(entry, name) for name in sorted(os.listdir(entry))
                    if os.path.isdir(os.path.join(entry, name))]
        expanded.extend(child for child in children
                        if os.path.exists(os.path.join(child, filename)))
    return expanded


def _potcar_identity(directory):
    """``(title, sha256)`` of the run's POTCAR, or ``(None, None)``."""
    path = os.path.join(directory, "POTCAR")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "r", errors="replace") as handle:
            title = handle.readline().strip()
    except OSError:
        return None, None

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return title, digest.hexdigest()


def _run_identity(directory):
    """``(vasp_version, incar_sha256)``, reusing the tau gate's readers."""
    from ..data.validation import code_version, file_hash

    return (code_version(directory),
            file_hash(os.path.join(directory, "INCAR")))
