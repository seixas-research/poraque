# -*- coding: utf-8 -*-
# file: base.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Common machinery for 3D scalar fields.

:class:`ScalarField` factors out everything that the external potential, the
charge density and the kinetic energy density share:

* they are ``(Nx, Ny, Nz)`` arrays attached to a :class:`~poraque.fields.FieldGrid`
  and a :class:`~poraque.fields.vasp.Poscar`;
* they are serialized in the **``CHGCAR`` format**, so ``EXTCAR``, ``CHGCAR``
  and ``TAUCAR`` are read and written by one code path;
* they compose only when defined on the same grid.

Subclasses supply a name, a filename, a unit, and a ``compute`` constructor.
"""

from abc import ABC

import numpy as np

from .grid import FieldGrid
from .vasp.volumetric import read_volumetric, write_volumetric


class ScalarField(ABC):
    """
    A real scalar field sampled on a :class:`FieldGrid`.

    Parameters
    ----------
    data : array_like
        ``(Nx, Ny, Nz)`` values, in :attr:`unit`.
    grid : FieldGrid
        The mesh the values live on.
    structure : Poscar
        The atomic structure; written into the file header.
    metadata : dict, optional
        Free-form provenance (model parameters, source file, ...).

    Attributes
    ----------
    name : str
        Human-readable field name, e.g. ``"external potential"``.
    default_filename : str
        Conventional file name, e.g. ``"EXTCAR"``.
    unit : str
        Physical unit of the stored values.
    volume_scaled : bool
        Whether the on-disk values are multiplied by the cell volume. VASP does
        this for the *charge density* in ``CHGCAR`` (it stores ``rho * Omega``)
        but not for potentials in ``LOCPOT``. Subclasses declare their own
        convention so that :meth:`read` and :meth:`write` stay symmetric.
    """

    name = "scalar field"
    default_filename = "FIELD"
    unit = ""
    volume_scaled = False

    def __init__(self, data, grid, structure, metadata=None):
        self.data = np.asarray(data, dtype=float)
        self.grid = grid
        self.structure = structure
        self.metadata = dict(metadata or {})

        if self.data.shape != grid.shape:
            raise ValueError(
                f"{type(self).__name__}: data shape {self.data.shape} does not "
                f"match grid shape {grid.shape}."
            )

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    def write(self, path=None, comment=None, columns=5, width=17, decimals=11,
              augmentation=None):
        """
        Write the field in ``CHGCAR`` format.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Destination; defaults to :attr:`default_filename` in the current
            directory.
        comment : str, optional
            Header comment line. Defaults to a self-describing string naming
            the field and its unit — VASP ignores this line, but it makes the
            file readable by humans and by the ML dataset loader.
        columns : int, optional
            Values per line (5 = ``CHGCAR`` style, 10 = ``CHG`` style).
        width, decimals : int, optional
            Fortran ``Ew.d`` field per value; the defaults reproduce VASP's
            ``(1X,E17.11)`` density block exactly.
        augmentation : sequence of str, optional
            PAW augmentation records to append, from
            :func:`~poraque.fields.vasp.volumetric.read_augmentation`. Needed
            when the file is to seed a VASP run with ``ICHARG=1``.

        Returns
        -------
        str
            The path written.
        """
        path = path if path is not None else self.default_filename
        if comment is None:
            formula = "".join(
                f"{s}{c}" for s, c in zip(self.structure.symbols, self.structure.counts)
            )
            comment = f"{formula}  {self.default_filename}: {self.name} [{self.unit}]"

        return write_volumetric(
            path,
            self.structure,
            self.to_file_values(),
            comment=comment,
            columns=columns,
            width=width,
            decimals=decimals,
            augmentation=augmentation,
        )

    @classmethod
    def read(cls, path, grid=None):
        """
        Read a field from a ``CHGCAR``-format file.

        Parameters
        ----------
        path : str or pathlib.Path
            File to read.
        grid : FieldGrid, optional
            Grid to attach. When given, its shape must match the file — this is
            how the three fields of one material are tied to a single shared
            grid object. When omitted, a grid is built from the file header.

        Returns
        -------
        ScalarField
            An instance of the calling subclass.
        """
        structure, raw, _ = read_volumetric(path)

        if grid is None:
            grid = FieldGrid(raw.shape, structure.cell)
        elif tuple(grid.shape) != raw.shape:
            raise ValueError(
                f"{path}: grid shape {raw.shape} does not match the supplied "
                f"shared grid {tuple(grid.shape)}. All fields of one material "
                f"must be defined on the same mesh."
            )

        return cls(
            cls.from_file_values(raw, grid),
            grid,
            structure,
            metadata={"source": str(path)},
        )

    def to_file_values(self):
        """Convert :attr:`data` to the on-disk convention."""
        return self.data * self.grid.volume if self.volume_scaled else self.data

    @classmethod
    def from_file_values(cls, raw, grid):
        """Convert on-disk values back to physical units."""
        return raw / grid.volume if cls.volume_scaled else raw

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def integrate(self):
        """Integral of the field over the cell (``unit * Å³``)."""
        return self.grid.integrate(self.data)

    def mean(self):
        """Volume-average of the field."""
        return float(np.mean(self.data))

    def statistics(self):
        """
        Summary statistics, handy for dataset normalization.

        Returns
        -------
        dict
            ``min``, ``max``, ``mean``, ``std`` and ``integral``.
        """
        return {
            "min": float(np.min(self.data)),
            "max": float(np.max(self.data)),
            "mean": float(np.mean(self.data)),
            "std": float(np.std(self.data)),
            "integral": self.integrate(),
        }

    def smooth(self, sigma, method="spectral"):
        r"""
        Gaussian blur of the field, respecting periodicity.

        Convolution with a normalized Gaussian of width :math:`\sigma`. Because
        the field lives on a torus the convolution must wrap; a filter that
        pads or reflects at the cell face would corrupt exactly the region
        where periodic images meet.

        Parameters
        ----------
        sigma : float
            Gaussian width in **Ångström**. Zero or ``None`` returns a copy.
        method : {"spectral", "ndimage"}, optional
            ``"spectral"`` multiplies by :math:`e^{-G^2\sigma^2/2}` in
            reciprocal space. This is the exact periodic Gaussian convolution
            for a band-limited field, it costs two FFTs, and it uses the true
            reciprocal metric — so it stays isotropic in Cartesian space even
            for a triclinic cell.
            ``"ndimage"`` uses :func:`scipy.ndimage.gaussian_filter` with
            ``mode="wrap"``, with the width converted from Ångström to voxels
            per axis.

        Returns
        -------
        ScalarField
            A new instance of the same class.

        Notes
        -----
        The two methods agree to rounding on an orthogonal cell. They do
        **not** agree on a skewed one: ``ndimage`` filters along grid axes with
        a per-axis voxel width, which is an anisotropic blur in Cartesian space
        when the lattice vectors are not orthogonal. ``"spectral"`` is the
        default for that reason.
        """
        if not sigma:
            return type(self)(self.data.copy(), self.grid, self.structure,
                              metadata=dict(self.metadata))

        sigma = float(sigma)
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}.")

        if method == "spectral":
            kernel = np.exp(-0.5 * self.grid.get_g2() * sigma * sigma)
            smoothed = np.real(np.fft.ifftn(np.fft.fftn(self.data) * kernel))
        elif method == "ndimage":
            from scipy.ndimage import gaussian_filter

            # gaussian_filter measures sigma in voxels, and the spacing differs
            # per axis, so the width must be converted per axis.
            voxels = sigma / self.grid.spacing
            smoothed = gaussian_filter(self.data, sigma=voxels, mode="wrap")
        else:
            raise ValueError(
                f"Unknown smoothing method {method!r}; use 'spectral' or 'ndimage'."
            )

        metadata = dict(self.metadata)
        metadata["gaussian_blur"] = sigma
        metadata["gaussian_blur_method"] = method
        return type(self)(smoothed, self.grid, self.structure, metadata=metadata)

    def same_grid_as(self, other):
        """True when ``other`` is defined on an identical mesh."""
        return self.grid.matches(other.grid)

    def _require_same_grid(self, other):
        if isinstance(other, ScalarField) and not self.same_grid_as(other):
            raise ValueError(
                f"Cannot combine fields on different grids: "
                f"{self.grid} vs {other.grid}."
            )

    # ------------------------------------------------------------------ #
    # Arithmetic / interop
    # ------------------------------------------------------------------ #
    def _binary(self, other, op):
        self._require_same_grid(other)
        values = other.data if isinstance(other, ScalarField) else other
        return type(self)(op(self.data, values), self.grid, self.structure,
                          metadata=dict(self.metadata))

    def __add__(self, other):
        return self._binary(other, np.add)

    def __sub__(self, other):
        return self._binary(other, np.subtract)

    def __mul__(self, other):
        return self._binary(other, np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary(other, np.divide)

    def __array__(self, dtype=None, copy=None):
        array = self.data if dtype is None else self.data.astype(dtype)
        return np.array(array, copy=False) if copy is False else array

    @property
    def shape(self):
        """Grid shape of the field."""
        return self.data.shape

    def __repr__(self):
        return (f"{type(self).__name__}(shape={self.data.shape}, "
                f"unit={self.unit!r}, range=[{self.data.min():.4g}, "
                f"{self.data.max():.4g}])")
