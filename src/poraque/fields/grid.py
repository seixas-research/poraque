# -*- coding: utf-8 -*-
# file: grid.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
The shared 3D field grid.

Every scalar field of a given material — the external potential (``EXTCAR``),
the charge density (``CHGCAR``) and the kinetic energy density (``TAUCAR``) —
lives on **one** :class:`FieldGrid` instance. Constructing the grid once and
passing it to each field is what guarantees that the three files of a material
are defined point-for-point on the same mesh, which is the precondition for the
field-to-field learning pipeline in :mod:`poraque.ml`.

Units follow the VASP convention: lengths in Ångström, wavevectors in Å⁻¹.
"""

import numpy as np

from .constants import HBAR2_OVER_2M_EV_ANGSTROM2


def fft_friendly_size(n, factors=(2, 3, 5, 7), force_even=True):
    """
    Smallest integer ``>= n`` that factorizes into ``factors`` (and is even).

    FFT libraries are dramatically faster on such sizes, and VASP rounds its
    grids the same way.

    Parameters
    ----------
    n : float
        Lower bound on the grid size.
    factors : tuple of int, optional
        Allowed prime factors.
    force_even : bool, optional
        Require an even result (VASP always uses even FFT grids).

    Returns
    -------
    int
    """
    candidate = max(2, int(np.ceil(n - 1e-9)))
    while True:
        if not (force_even and candidate % 2):
            residue = candidate
            for factor in factors:
                while residue % factor == 0:
                    residue //= factor
            if residue == 1:
                return candidate
        candidate += 1


class FieldGrid:
    """
    Uniform real-space mesh spanning a periodic unit cell.

    Parameters
    ----------
    shape : tuple of int
        ``(Nx, Ny, Nz)`` grid points along each lattice vector.
    cell : array_like
        ``(3, 3)`` lattice vectors in Ångström (rows are ``a1, a2, a3``).
    encut : float, optional
        Plane-wave cutoff (eV) the grid was derived from; recorded for
        provenance.
    prec : str, optional
        ``PREC`` setting the grid was derived from; recorded for provenance.

    Attributes
    ----------
    shape : tuple of int
    cell : numpy.ndarray
        ``(3, 3)`` lattice vectors, Å.
    reciprocal_cell : numpy.ndarray
        ``(3, 3)`` reciprocal lattice vectors ``2*pi*inv(cell).T``, Å⁻¹
        (rows are ``b1, b2, b3``).
    volume : float
        Cell volume, Å³.
    """

    def __init__(self, shape, cell, encut=None, prec=None):
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or any(n < 1 for n in self.shape):
            raise ValueError(f"Grid shape must be three positive integers, got {shape!r}.")

        self.cell = np.asarray(cell, dtype=float).reshape(3, 3)
        self.volume = float(abs(np.linalg.det(self.cell)))
        if self.volume <= 0.0:
            raise ValueError("Cell is singular (zero volume).")

        self.reciprocal_cell = 2.0 * np.pi * np.linalg.inv(self.cell).T
        self.encut = None if encut is None else float(encut)
        self.prec = prec

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_encut(cls, cell, encut, prec="normal", fine=True):
        """
        Size the grid from a plane-wave cutoff, VASP style.

        The cutoff fixes the radius of the plane-wave sphere,

        .. math:: G_{\\rm cut} = \\sqrt{E_{\\rm cut} / (\\hbar^2/2m_e)},

        and hence the largest Miller index along each lattice vector,
        ``n_i^max = floor(G_cut |a_i| / 2 pi)``. Representing the *wavefunctions*
        without aliasing needs ``2 n_i^max + 1`` points; the *density*, being
        their square, contains components up to ``2 G_cut`` and needs
        ``4 n_i^max + 1`` for a strictly wrap-around-free representation
        (``PREC = Accurate``). ``PREC = Normal`` relaxes this to the cheaper
        ``3 n_i^max + 1`` rule. Results are rounded up to even, FFT-friendly
        sizes.

        Parameters
        ----------
        cell : array_like
            ``(3, 3)`` lattice vectors, Å.
        encut : float
            Plane-wave cutoff, eV.
        prec : str, optional
            ``'normal'`` (3/2 rule) or ``'accurate'`` (wrap-around free).
        fine : bool, optional
            Return the fine/density grid (the one a ``CHGCAR`` uses, default)
            rather than the coarse wavefunction grid.

        Returns
        -------
        FieldGrid

        Notes
        -----
        This reproduces VASP's default ``NG(X,Y,Z)F`` to within one
        FFT-friendly step for typical cells, but VASP's exact rounding is
        version- and setting-dependent. When the field must line up with an
        existing VASP run, take the grid from that run instead — via explicit
        ``NGXF/NGYF/NGZF`` tags in the ``INCAR`` or via :meth:`from_file`.
        """
        cell = np.asarray(cell, dtype=float).reshape(3, 3)
        g_cut = np.sqrt(float(encut) / HBAR2_OVER_2M_EV_ANGSTROM2)
        lengths = np.linalg.norm(cell, axis=1)
        n_max = np.floor(g_cut * lengths / (2.0 * np.pi)).astype(int)

        key = str(prec).strip().lower()
        multiplier = 4.0 if key in ("accurate", "high") else 3.0
        if not fine:
            multiplier = 2.0

        shape = tuple(fft_friendly_size(multiplier * n + 1) for n in n_max)
        return cls(shape, cell, encut=encut, prec=prec)

    @classmethod
    def from_file(cls, path):
        """
        Adopt the grid of an existing VASP volumetric file.

        This is the safest way to build fields that must align with a real
        calculation: the shape is read from the file rather than re-derived.

        Parameters
        ----------
        path : str or pathlib.Path
            A ``CHGCAR``/``CHG``/``LOCPOT``/``TAUCAR``-format file.

        Returns
        -------
        FieldGrid
        """
        from .vasp.volumetric import read_volumetric

        structure, data, _ = read_volumetric(path)
        return cls(data.shape, structure.cell)

    @classmethod
    def from_parameters(cls, structure, parameters=None, pseudopotentials=None,
                        shape=None, encut=None, prec=None):
        """
        Build the grid from code-agnostic ingestion results.

        This is the general entry point, fed by any
        :class:`~poraque.fields.io.base.CalculationReader`. Precedence, first
        hit wins:

        1. an explicit ``shape`` argument;
        2. :attr:`CalculationParameters.grid_shape` (an explicit grid stated in
           the input file);
        3. a cutoff — the ``encut`` argument, then
           :attr:`CalculationParameters.cutoff`, then the largest
           ``recommended_cutoff`` among the pseudopotentials — combined with
           the precision setting.

        Parameters
        ----------
        structure : Structure
            Provides the cell.
        parameters : CalculationParameters, optional
            Cutoff (**eV**), precision and any explicit grid shape.
        pseudopotentials : dict, optional
            ``{element: PseudopotentialInfo}``, used only for the fallback
            cutoff.
        shape : tuple of int, optional
            Explicit grid shape; overrides everything else.
        encut : float, optional
            Explicit cutoff in eV.
        prec : str, optional
            Explicit precision setting.

        Returns
        -------
        FieldGrid
        """
        precision = prec or (parameters.precision if parameters else None) or "normal"

        if shape is not None:
            return cls(shape, structure.cell, encut=encut, prec=precision)

        if parameters is not None and parameters.grid_shape is not None:
            return cls(parameters.grid_shape, structure.cell,
                       encut=encut or parameters.cutoff, prec=precision)

        if encut is None and parameters is not None:
            encut = parameters.cutoff
        if encut is None and pseudopotentials:
            cutoffs = [info.recommended_cutoff for info in pseudopotentials.values()
                       if info.recommended_cutoff is not None]
            encut = max(cutoffs) if cutoffs else None
        if encut is None:
            raise ValueError(
                "Cannot size the grid: no explicit shape, no grid shape in the "
                "input file, no cutoff, and no recommended cutoff from the "
                "pseudopotentials."
            )

        return cls.from_encut(structure.cell, encut, prec=precision, fine=True)

    @classmethod
    def from_vasp_inputs(cls, poscar, incar=None, potcar=None, shape=None,
                         encut=None, prec=None):
        """
        Build the grid from a set of VASP inputs, honouring the usual precedence.

        The shape is resolved in this order, first hit wins:

        1. an explicit ``shape`` argument;
        2. explicit ``NGXF/NGYF/NGZF`` tags in the ``INCAR``;
        3. ``ENCUT`` (argument, then ``INCAR``, then the largest ``ENMAX`` in
           the ``POTCAR``) combined with ``PREC``.

        Parameters
        ----------
        poscar : Poscar
            Provides the cell.
        incar : Incar, optional
            Provides ``ENCUT``, ``PREC`` and any explicit grid tags.
        potcar : Potcar, optional
            Provides the fallback cutoff (``ENMAX``).
        shape : tuple of int, optional
            Explicit grid shape; overrides everything else.
        encut : float, optional
            Explicit cutoff in eV; overrides the ``INCAR``/``POTCAR`` values.
        prec : str, optional
            Explicit ``PREC``; overrides the ``INCAR`` value.

        Returns
        -------
        FieldGrid
        """
        prec = prec or (incar.prec if incar is not None else "normal")

        if shape is not None:
            return cls(shape, poscar.cell, encut=encut, prec=prec)

        if incar is not None and incar.fine_shape is not None:
            return cls(incar.fine_shape, poscar.cell,
                       encut=encut or incar.encut, prec=prec)

        if encut is None and incar is not None:
            encut = incar.encut
        if encut is None and potcar is not None:
            encut = potcar.enmax
        if encut is None:
            raise ValueError(
                "Cannot size the grid: no explicit shape, no NGXF/NGYF/NGZF in "
                "the INCAR, no ENCUT in the INCAR and no ENMAX in the POTCAR."
            )

        return cls.from_encut(poscar.cell, encut, prec=prec, fine=True)

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    @property
    def npoints(self):
        """Total number of grid points."""
        return int(np.prod(self.shape))

    @property
    def volume_element(self):
        """Volume per grid point, Å³."""
        return self.volume / self.npoints

    @property
    def lengths(self):
        """``(3,)`` lattice vector lengths, Å."""
        return np.linalg.norm(self.cell, axis=1)

    @property
    def spacing(self):
        """``(3,)`` nominal grid spacing along each lattice vector, Å."""
        return self.lengths / np.asarray(self.shape, dtype=float)

    def fft_frequencies(self):
        """
        Integer FFT frequencies along each axis.

        Returns
        -------
        tuple of numpy.ndarray
            ``(m1, m2, m3)`` with the usual ``0, 1, ..., -2, -1`` ordering.
        """
        return tuple(
            np.fft.fftfreq(n, d=1.0 / n).astype(int) for n in self.shape
        )

    def get_g_vectors(self):
        """
        Cartesian reciprocal-lattice vectors of the full FFT mesh.

        Valid for any Bravais lattice: the integer FFT frequencies are
        projected onto the reciprocal lattice vectors.

        Returns
        -------
        tuple of numpy.ndarray
            ``(Gx, Gy, Gz)``, each of shape :attr:`shape`, in Å⁻¹.
        """
        m1, m2, m3 = self.fft_frequencies()
        M1, M2, M3 = np.meshgrid(m1, m2, m3, indexing="ij")
        b1, b2, b3 = self.reciprocal_cell
        gx = M1 * b1[0] + M2 * b2[0] + M3 * b3[0]
        gy = M1 * b1[1] + M2 * b2[1] + M3 * b3[1]
        gz = M1 * b1[2] + M2 * b2[2] + M3 * b3[2]
        return gx, gy, gz

    def get_g2(self):
        """``|G|^2`` on the full FFT mesh, Å⁻²."""
        gx, gy, gz = self.get_g_vectors()
        return gx * gx + gy * gy + gz * gz

    def scaled_coordinates(self):
        """
        Fractional coordinates of every grid point.

        Returns
        -------
        numpy.ndarray
            ``(Nx, Ny, Nz, 3)`` array with values in ``[0, 1)``.
        """
        axes = [np.arange(n, dtype=float) / n for n in self.shape]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack(mesh, axis=-1)

    def cartesian_coordinates(self):
        """
        Cartesian coordinates of every grid point, Å.

        Returns
        -------
        numpy.ndarray
            ``(Nx, Ny, Nz, 3)`` array.
        """
        return self.scaled_coordinates() @ self.cell

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def integrate(self, field):
        """
        Integrate a scalar field over the cell.

        Parameters
        ----------
        field : array_like
            Array of shape :attr:`shape`.

        Returns
        -------
        float
        """
        return float(np.sum(field) * self.volume_element)

    def matches(self, other, tol=1e-6):
        """
        True when ``other`` describes the same mesh (shape *and* cell).

        Parameters
        ----------
        other : FieldGrid
            Grid to compare against.
        tol : float, optional
            Absolute tolerance on the lattice vectors, Å.

        Returns
        -------
        bool
        """
        if not isinstance(other, FieldGrid):
            # A plain False, not NotImplemented: that sentinel is truthy, so
            # returning it from an ordinary method made every mismatched-type
            # comparison silently pass.
            return False
        return (self.shape == other.shape
                and np.allclose(self.cell, other.cell, atol=tol))

    def __eq__(self, other):
        return self.matches(other) if isinstance(other, FieldGrid) else NotImplemented

    def __hash__(self):
        # Rounded to the same 1e-6 Å scale `matches` compares at, so grids
        # that compare equal land in the same bucket. (Exact bytes hashed
        # grids apart that __eq__ called equal.)
        return hash((self.shape, np.round(self.cell, 6).tobytes()))

    def __repr__(self):
        return (f"FieldGrid(shape={self.shape}, volume={self.volume:.3f} A^3"
                + (f", encut={self.encut:g} eV" if self.encut else "") + ")")
