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

#: Precisions a field may be stored in, by name.
#:
#: ``float64``
#:     The default, and what every field used before this was selectable.
#:     Integrals over the cell — the electron count, the energy terms — are
#:     sums of :math:`N^3` terms, and at :math:`128^3` that is two million
#:     values whose accumulated rounding in single precision is no longer
#:     obviously negligible against the quantities being compared.
#: ``float32``
#:     Half the memory, which is the whole argument: a :math:`160^3` field is
#:     16 MB in double and 8 MB in single, and a committee of five models
#:     scoring a pool holds several at once. Adequate wherever the field is on
#:     its way into a network that will compute in single precision anyway.
#: ``float16``
#:     Storage only, for archiving or for a first pass over a very large pool.
#:     Roughly three decimal digits: too coarse for any integral, and offered
#:     so that a deliberate choice is available rather than an improvised
#:     ``astype`` somewhere downstream.
FIELD_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}

#: Dtype used when a field is built without an explicit one.
_DEFAULT_DTYPE = np.float64


def resolve_dtype(dtype=None):
    """
    Normalise a dtype argument to a numpy floating type.

    Parameters
    ----------
    dtype : str, numpy.dtype or None, optional
        A name from :data:`FIELD_DTYPES`, anything ``numpy.dtype`` accepts, or
        ``None`` for :func:`get_default_dtype`.

    Returns
    -------
    numpy.dtype

    Raises
    ------
    TypeError
        For a non-floating dtype. A field holds physical values; storing them
        as integers would silently truncate a density to zero rather than lose
        a little precision, so it is refused rather than allowed through.
    """
    if dtype is None:
        return np.dtype(_DEFAULT_DTYPE)
    if isinstance(dtype, str):
        dtype = FIELD_DTYPES.get(dtype.strip().lower(), dtype)
    resolved = np.dtype(dtype)
    if resolved.kind != "f":
        raise TypeError(
            f"A field must be stored in a floating type, not {resolved!r}. "
            f"Known names: {sorted(FIELD_DTYPES)}.")
    return resolved


def get_default_dtype():
    """The dtype new fields are built in when none is given."""
    return np.dtype(_DEFAULT_DTYPE)


def set_default_dtype(dtype):
    """
    Set the dtype new fields are built in.

    Process-wide, and deliberately so: the point is to load a whole dataset in
    one precision without threading an argument through every reader. It does
    **not** touch fields that already exist — use :meth:`ScalarField.astype`
    for those.

    Parameters
    ----------
    dtype : str or numpy.dtype
        See :func:`resolve_dtype`.

    Returns
    -------
    numpy.dtype
        The previous default, so a caller can restore it.

    Examples
    --------
    >>> from poraque.fields import set_default_dtype
    >>> previous = set_default_dtype("float32")   # doctest: +SKIP
    >>> set_default_dtype(previous)               # doctest: +SKIP
    """
    global _DEFAULT_DTYPE
    previous = np.dtype(_DEFAULT_DTYPE)
    _DEFAULT_DTYPE = resolve_dtype(dtype)
    return previous


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
    dtype : str or numpy.dtype, optional
        Precision the values are stored in; see :data:`FIELD_DTYPES`. Defaults
        to :func:`get_default_dtype`, i.e. ``float64`` unless the process has
        been told otherwise. The grid and the structure are **not** affected:
        geometry stays in double precision, where it costs nine numbers per
        material and where a rounding error moves an atom.

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
        but not for potentials in ``LOCPOT``, and not for the kinetic energy
        density in ``TAUCAR``. Subclasses declare their own convention so that
        :meth:`read` and :meth:`write` stay symmetric.
    reads_all_blocks : bool
        Whether :meth:`read` parses every grid block in the file rather than
        stopping after the first. Off by default because the blocks after the
        first are the expensive half of a ``CHGCAR``: the augmentation records
        and the magnetisation grid, neither of which a single-channel field
        wants. A subclass whose file spreads *one* physical quantity over
        several blocks turns it on and combines them in
        :meth:`combine_blocks`.
    """

    name = "scalar field"
    default_filename = "FIELD"
    unit = ""
    volume_scaled = False
    reads_all_blocks = False

    def __init__(self, data, grid, structure, metadata=None, dtype=None):
        self.data = np.asarray(data, dtype=resolve_dtype(dtype))
        self.grid = grid
        self.structure = structure
        self.metadata = dict(metadata or {})

        if self.data.shape != grid.shape:
            raise ValueError(
                f"{type(self).__name__}: data shape {self.data.shape} does not "
                f"match grid shape {grid.shape}."
            )

    @property
    def dtype(self):
        """Precision the values are stored in."""
        return self.data.dtype

    def astype(self, dtype):
        """
        The same field, stored in another precision.

        A new object: fields are shared between the three members of a
        material and converting one in place would change the others.

        Parameters
        ----------
        dtype : str or numpy.dtype
            See :func:`resolve_dtype`.

        Returns
        -------
        ScalarField
            Of the same subclass, on the same grid and structure.
        """
        # `dtype=` as well as the cast: without it the constructor applies the
        # process default and converts straight back, so `astype("float32")`
        # on a default build returned float64.
        resolved = resolve_dtype(dtype)
        return type(self)(self.data.astype(resolved), self.grid,
                          self.structure, metadata=dict(self.metadata),
                          dtype=resolved)

    def nbytes(self):
        """Memory the values occupy, in bytes. The reason ``dtype`` exists."""
        return int(self.data.nbytes)

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    def write(self, path=None, comment=None, columns=5, width=17, decimals=11,
              augmentation=None, compression=None, level=4):
        """
        Write the field in ``CHGCAR`` format, or into an HDF5 store.

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
        compression : str or None, optional
            HDF5 only: ``"none"``, ``"gzip"`` or ``"lzf"``. Ignored for a text
            path, where the format itself fixes the encoding.
        level : int, optional
            Gzip level, HDF5 only.

        Returns
        -------
        str
            The path written. For an HDF5 target this is the
            ``file.h5::DATASET`` address, which is what a reader needs and
            what the plain filename would not be.

        Notes
        -----
        An ``.h5``/``.hdf5`` path writes the field into an HDF5 store instead
        (:mod:`poraque.fields.hdf5`), carrying the same values under the same
        convention. ``augmentation`` has no HDF5 equivalent and is refused
        rather than dropped: PAW occupancies are what make a density readable
        by VASP, and losing them silently would produce a file that looks
        complete and is not.
        """
        path = path if path is not None else self.default_filename

        from .hdf5 import is_hdf5_path

        if is_hdf5_path(path):
            from .hdf5 import split_target, write_field

            if augmentation:
                raise ValueError(
                    "PAW augmentation records cannot be stored in an HDF5 "
                    "field store: they are text records VASP reads out of a "
                    "CHGCAR, and nothing reads them back from HDF5. Write the "
                    "density as a CHGCAR when it has to seed a VASP run.")
            _, dataset = split_target(path)
            return write_field(path, dataset or self.default_filename, self,
                               compression=compression, level=level)
        if comment is None:
            formula = "".join(
                f"{s}{c}" for s, c in zip(self.structure.symbols, self.structure.counts)
            )
            comment = f"{formula}  {self.default_filename}: {self.name} [{self.unit}]"

        # The volumetric writer serialises the header via Poscar.to_string,
        # which a bare Structure (e.g. from a cube-file reader) does not
        # have -- adapt it rather than crash on the first non-VASP source.
        structure = self.structure
        if not hasattr(structure, "to_string"):
            from .vasp.poscar import Poscar

            structure = Poscar.from_structure(structure)

        return write_volumetric(
            path,
            structure,
            self.to_file_values(),
            comment=comment,
            columns=columns,
            width=width,
            decimals=decimals,
            augmentation=augmentation,
        )

    @classmethod
    def read(cls, path, grid=None, dtype=None):
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
        dtype : str or numpy.dtype, optional
            Precision to store the values in; see :data:`FIELD_DTYPES`. The
            file is parsed in double precision regardless and narrowed at the
            end, so the volume division of a ``volume_scaled`` field is not
            done in the narrower type.

        Returns
        -------
        ScalarField
            An instance of the calling subclass.
        """
        structure, raw, extra = read_volumetric(path,
                                                read_all=cls.reads_all_blocks)
        raw = cls.combine_blocks(raw, extra)

        if grid is None:
            grid = FieldGrid(raw.shape, structure.cell)
        elif tuple(grid.shape) != raw.shape:
            raise ValueError(
                f"{path}: grid shape {raw.shape} does not match the supplied "
                f"shared grid {tuple(grid.shape)}. All fields of one material "
                f"must be defined on the same mesh."
            )
        elif not np.allclose(structure.cell, grid.cell, atol=1e-5):
            # Shape alone is not identity: a volume-scaled field read against
            # a grid with a different cell would be silently rescaled by the
            # volume ratio, corrupting the electron count and every integral.
            raise ValueError(
                f"{path}: the file's cell does not match the supplied shared "
                f"grid's cell (max difference "
                f"{np.abs(structure.cell - grid.cell).max():.2e} Å). All "
                f"fields of one material must share one mesh in one cell."
            )

        return cls(
            cls.from_file_values(raw, grid),
            grid,
            structure,
            metadata={"source": str(path)},
            dtype=dtype,
        )

    @classmethod
    def combine_blocks(cls, raw, extra):
        """
        Reduce the grid blocks of one file to the single field they describe.

        The default keeps the first block and discards the rest, which is right
        for every field VASP writes one block per channel of: a spin-polarised
        ``CHGCAR``'s second block is the *magnetisation*, a different quantity
        on the same mesh, and belongs to :class:`~poraque.fields.SpinDensity`
        rather than here.

        Parameters
        ----------
        raw : numpy.ndarray
            The first grid block, in file units.
        extra : sequence of numpy.ndarray
            The blocks after it. Empty unless :attr:`reads_all_blocks` is set.

        Returns
        -------
        numpy.ndarray
        """
        return raw

    def to_file_values(self):
        """
        Convert :attr:`data` to the on-disk convention.

        Widened to double first. The file format writes eleven significant
        digits, and a ``float32`` field multiplied by the cell volume in
        ``float32`` would print eleven digits of which only seven mean
        anything — a written file that silently claims more precision than it
        holds is worse than one that holds less.
        """
        values = self.data.astype(np.float64, copy=False)
        return values * self.grid.volume if self.volume_scaled else values

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
