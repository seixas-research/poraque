# -*- coding: utf-8 -*-
# file: external.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Local external (electron-ion) potential on the shared 3D grid.

Physics
-------
The valence electrons of a pseudopotential calculation see a local ionic
potential built from *pseudo-ions* of charge :math:`Z^{\rm val}_s` (the
``ZVAL`` of the ``POTCAR``) sitting at the atomic sites. In a periodic cell
this potential is evaluated in reciprocal space, where the Coulomb kernel is
diagonal:

.. math::

    V_{\rm ext}(\mathbf{G}) \;=\; -\,\frac{4\pi e^2}{\Omega}\,
    \sum_{s} \frac{Z^{\rm val}_{s}\, f_{s}(G)}{G^{2}}\,
    S_{s}(\mathbf{G}),
    \qquad
    S_{s}(\mathbf{G}) = \sum_{a \in s} e^{-i\mathbf{G}\cdot\boldsymbol{\tau}_a},

with :math:`V_{\rm ext}(\mathbf{G}=0) \equiv 0`. Dropping the
:math:`\mathbf{G}=0` component is the standard neutralizing-background
convention of periodic plane-wave codes: it is exactly the divergent term that
cancels against the Hartree and ion-ion :math:`\mathbf{G}=0` contributions in
the total energy, and it fixes the average of the potential to zero. The real
space field follows from a single inverse FFT,
:math:`V_{\rm ext}(\mathbf{r}) = \sum_{\mathbf{G}} V_{\rm ext}(\mathbf{G})\,
e^{i\mathbf{G}\cdot\mathbf{r}}`, so the whole construction is
:math:`\mathcal{O}(N \log N)` and *exactly* periodic — no real-space cutoff, no
minimum-image approximation, no Ewald parameter to tune.

The form factor :math:`f_s(G)` selects the pseudo-ion model:

``model="gaussian"`` (default)
    :math:`f_s(G) = e^{-G^2\sigma_s^2/2}`, i.e. a normalized Gaussian ion of
    width :math:`\sigma_s`, whose real-space potential is the regularized
    :math:`-Z^{\rm val}_s e^2\,\mathrm{erf}(r/\sqrt{2}\sigma_s)/r`. It is
    Coulombic beyond a few :math:`\sigma_s` and finite at the nucleus, which
    removes the Gibbs ringing a bare :math:`1/G^2` produces on a finite FFT
    mesh. :math:`\sigma_s` defaults to ``rcore_factor * RCORE_s`` with the
    pseudization radius taken from the ``POTCAR``, so the softening follows the
    length scale of the actual pseudopotential rather than the grid.

``model="coulomb"``
    :math:`f_s(G) = 1`: bare point pseudo-ions. Parameter-free and the exact
    long-range limit, but visibly aliased near the nuclei on any finite grid.

.. note::
   This is the *long-range, local* part of the ionic potential. It is not the
   full VASP ``PAW`` local pseudopotential: the short-range pseudization inside
   :math:`R_{\rm core}` is modelled by :math:`f_s(G)` rather than read from the
   ``POTCAR`` table, and non-local projectors are outside the scope of a local
   field descriptor by construction. See
   :mod:`poraque.fields.vasp.potcar` for why the tabulated ``local part`` is
   parsed but not consumed.

Sign and units
--------------
Values are the **potential energy of an electron** in eV — negative near the
ions — matching the convention of VASP's ``LOCPOT``. The file is written in
``CHGCAR`` format but, like ``LOCPOT`` and unlike ``CHGCAR``, the values are
*not* multiplied by the cell volume.
"""

import os

import numpy as np

from .base import ScalarField
from .constants import COULOMB_CONSTANT_EV_ANGSTROM
from .grid import FieldGrid
from .vasp.incar import Incar
from .vasp.poscar import Poscar
from .vasp.potcar import Potcar

#: Fallback Gaussian width (Å) when no ``RCORE`` is available from the POTCAR.
DEFAULT_SIGMA = 0.5
#: Smallest Gaussian width (Å) accepted, to keep the FFT representation sane.
MIN_SIGMA = 0.1


class ExternalPotential(ScalarField):
    """
    Local external potential of the pseudo-ions, written as ``EXTCAR``.

    Build it from a VASP input set with :meth:`from_vasp`, or directly from a
    structure and a grid with :meth:`compute`. Read an existing file with
    :meth:`~poraque.fields.base.ScalarField.read`.

    Examples
    --------
    >>> from poraque.fields import ExternalPotential
    >>> potential = ExternalPotential.from_vasp("path/to/run")   # doctest: +SKIP
    >>> potential.write("path/to/run/EXTCAR")                     # doctest: +SKIP
    """

    name = "local external potential"
    default_filename = "EXTCAR"
    unit = "eV"
    volume_scaled = False

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_vasp(cls, directory=".", poscar=None, incar=None, potcar=None,
                  grid=None, shape=None, encut=None, prec=None,
                  model="gaussian", sigma=None, rcore_factor=0.5,
                  zval=None):
        """
        Build the external potential from a VASP input set.

        Parameters
        ----------
        directory : str or pathlib.Path, optional
            Directory holding ``POSCAR``, ``INCAR`` and ``POTCAR``. Used only
            for the files not passed explicitly. ``CONTCAR`` is accepted as a
            fallback for ``POSCAR``.
        poscar, incar, potcar : str or object, optional
            Either a path to the corresponding file or an already-parsed
            :class:`~poraque.fields.vasp.Poscar` /
            :class:`~poraque.fields.vasp.Incar` /
            :class:`~poraque.fields.vasp.Potcar` object.
        grid : FieldGrid, optional
            Pre-built shared grid. **Pass this when the material's ``CHGCAR``
            and ``TAUCAR`` already exist**, so all three fields are guaranteed
            to share one mesh.
        shape : tuple of int, optional
            Explicit ``(NGXF, NGYF, NGZF)``; overrides the ``INCAR``/``ENCUT``
            derivation.
        encut : float, optional
            Cutoff in eV overriding the ``INCAR`` value.
        prec : str, optional
            ``PREC`` setting overriding the ``INCAR`` value.
        model : {"gaussian", "coulomb"}, optional
            Pseudo-ion model; see the module docstring.
        sigma : float or dict, optional
            Gaussian width in Å — a scalar applied to every species, or a
            ``{element: sigma}`` mapping. Defaults to ``rcore_factor * RCORE``
            per species from the ``POTCAR``.
        rcore_factor : float, optional
            Multiplier turning the ``POTCAR`` pseudization radius into a
            Gaussian width. The default ``0.5`` puts ``R_core`` at two standard
            deviations, where the model has recovered 95% of the bare Coulomb
            tail.
        zval : dict, optional
            ``{element: charge}`` overriding the ``POTCAR`` valence charges.
            Required when no ``POTCAR`` is available.

        Returns
        -------
        ExternalPotential
        """
        poscar = _resolve_poscar(poscar, directory)
        incar = _resolve_incar(incar, directory)
        potcar = _resolve_potcar(potcar, directory)

        if potcar is not None and not potcar.matches(poscar):
            raise ValueError(
                f"POTCAR species {potcar.elements} do not match POSCAR species "
                f"{[s.split('_')[0] for s in poscar.symbols]} (order matters)."
            )

        if grid is None:
            grid = FieldGrid.from_vasp_inputs(
                poscar, incar=incar, potcar=potcar,
                shape=shape, encut=encut, prec=prec,
            )

        charges = _resolve_charges(poscar, potcar, zval)
        widths = _resolve_widths(poscar, potcar, sigma, rcore_factor, model)

        return cls.compute(poscar, grid, charges, widths=widths, model=model,
                           metadata={
                               "encut": grid.encut,
                               "prec": grid.prec,
                               "model": model,
                               "rcore_factor": rcore_factor,
                           })

    @classmethod
    def compute(cls, structure, grid, charges, widths=None, model="gaussian",
                metadata=None):
        """
        Evaluate the potential on ``grid`` for ``structure``.

        Parameters
        ----------
        structure : Poscar
            Atomic structure; only fractional coordinates and species grouping
            are used, so the result is independent of the cell orientation.
        grid : FieldGrid
            Shared mesh.
        charges : dict
            ``{element: Z_val}`` pseudo-ion charges in units of ``+e``.
        widths : dict, optional
            ``{element: sigma}`` Gaussian widths in Å. Ignored for
            ``model="coulomb"``.
        model : {"gaussian", "coulomb"}, optional
            Pseudo-ion model.
        metadata : dict, optional
            Provenance to attach to the field.

        Returns
        -------
        ExternalPotential
        """
        if model not in ("gaussian", "coulomb"):
            raise ValueError(f"Unknown pseudo-ion model: {model!r}.")

        widths = widths or {}
        g2 = grid.get_g2()

        # 1 / G^2 with the G = 0 term excluded (neutralizing background).
        inverse_g2 = np.zeros_like(g2)
        nonzero = g2 > 1e-12
        inverse_g2[nonzero] = 1.0 / g2[nonzero]

        v_g = np.zeros(grid.shape, dtype=complex)

        for symbol, atom_slice in structure.species_slices():
            element = symbol.split("_")[0]
            charge = float(charges[element])
            if charge == 0.0:
                continue

            structure_factor = _structure_factor(
                grid, structure.scaled_positions[atom_slice]
            )

            form_factor = 1.0
            if model == "gaussian":
                sigma = float(widths.get(element, DEFAULT_SIGMA))
                form_factor = np.exp(-0.5 * g2 * sigma * sigma)

            v_g += charge * form_factor * structure_factor * inverse_g2

        v_g *= -4.0 * np.pi * COULOMB_CONSTANT_EV_ANGSTROM / grid.volume

        # V(r) = sum_G V(G) exp(i G.r);  numpy's ifftn carries a 1/N factor.
        data = np.real(np.fft.ifftn(v_g) * grid.npoints)

        payload = {"model": model, "charges": dict(charges)}
        if model == "gaussian":
            payload["widths"] = dict(widths)
        payload.update(metadata or {})

        return cls(data, grid, structure, metadata=payload)

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #
    def interaction_energy(self, density):
        r"""
        Electron-ion interaction energy :math:`\int V_{\rm ext}\,\rho\,d^3r`.

        Parameters
        ----------
        density : ChargeDensity or array_like
            Electron density on the same grid, in electrons/Å³.

        Returns
        -------
        float
            Energy in eV.

        Notes
        -----
        Sign convention: ``density`` is the (positive) electron number density,
        while :attr:`data` is already the potential energy *of an electron*, so
        the product integrates directly to the interaction energy.
        """
        values = np.asarray(density, dtype=float)
        if values.shape != self.data.shape:
            raise ValueError(
                f"Density shape {values.shape} does not match the potential "
                f"grid {self.data.shape}."
            )
        return self.grid.integrate(self.data * values)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _structure_factor(grid, scaled_positions):
    r"""
    Structure factor :math:`\sum_a e^{-i\mathbf{G}\cdot\boldsymbol{\tau}_a}`.

    Because :math:`\mathbf{G}\cdot\boldsymbol{\tau}_a = 2\pi\sum_j m_j s_{aj}`
    for FFT frequencies :math:`m_j` and fractional coordinates :math:`s_{aj}`,
    the phase factorizes into an outer product of three 1D vectors. That keeps
    the cost at ``O(N_atoms * (N1 + N2 + N3))`` complex exponentials instead of
    ``O(N_atoms * N1*N2*N3)``, and avoids allocating a full complex grid per
    atom.

    Parameters
    ----------
    grid : FieldGrid
        Mesh supplying the FFT frequencies.
    scaled_positions : array_like
        ``(n, 3)`` fractional coordinates of one species.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``grid.shape``.
    """
    m1, m2, m3 = grid.fft_frequencies()
    total = np.zeros(grid.shape, dtype=complex)

    for position in np.atleast_2d(scaled_positions):
        phase1 = np.exp(-2j * np.pi * m1 * position[0])
        phase2 = np.exp(-2j * np.pi * m2 * position[1])
        phase3 = np.exp(-2j * np.pi * m3 * position[2])
        total += phase1[:, None, None] * phase2[None, :, None] * phase3[None, None, :]

    return total


def _resolve_poscar(poscar, directory):
    """Return a :class:`Poscar`, reading ``POSCAR``/``CONTCAR`` if needed."""
    if isinstance(poscar, Poscar):
        return poscar
    if poscar is not None:
        return Poscar.from_file(poscar)
    for name in ("POSCAR", "CONTCAR"):
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return Poscar.from_file(candidate)
    raise FileNotFoundError(f"No POSCAR or CONTCAR found in {directory!r}.")


def _resolve_incar(incar, directory):
    """Return an :class:`Incar` or ``None`` when the file is absent."""
    if isinstance(incar, Incar):
        return incar
    if incar is not None:
        return Incar.from_file(incar)
    candidate = os.path.join(directory, "INCAR")
    return Incar.from_file(candidate) if os.path.exists(candidate) else None


def _resolve_potcar(potcar, directory):
    """Return a :class:`Potcar` or ``None`` when the file is absent."""
    if isinstance(potcar, Potcar):
        return potcar
    if potcar is not None:
        return Potcar.from_file(potcar)
    candidate = os.path.join(directory, "POTCAR")
    return Potcar.from_file(candidate) if os.path.exists(candidate) else None


def _resolve_charges(poscar, potcar, zval):
    """
    Build the ``{element: Z_val}`` map, POTCAR first and overrides last.

    Raises
    ------
    ValueError
        If a species present in the structure has no valence charge.
    """
    charges = dict(potcar.zval_map) if potcar is not None else {}
    if zval:
        charges.update({str(k).split("_")[0]: float(v) for k, v in zval.items()})

    missing = [
        symbol.split("_")[0] for symbol in poscar.symbols
        if symbol.split("_")[0] not in charges
    ]
    if missing:
        raise ValueError(
            f"No valence charge for {sorted(set(missing))}. Provide a POTCAR "
            f"or pass zval={{'X': charge, ...}} explicitly."
        )
    return charges


def _resolve_widths(poscar, potcar, sigma, rcore_factor, model):
    """
    Build the ``{element: sigma}`` map of Gaussian widths (Å).

    Precedence: explicit ``sigma`` (scalar or per-element) beats the
    ``POTCAR`` ``RCORE``, which beats :data:`DEFAULT_SIGMA`.
    """
    if model != "gaussian":
        return {}

    elements = [symbol.split("_")[0] for symbol in poscar.symbols]
    rcore = potcar.rcore_map if potcar is not None else {}

    widths = {}
    for element in elements:
        radius = rcore.get(element)
        widths[element] = (
            DEFAULT_SIGMA if radius is None else max(MIN_SIGMA, rcore_factor * radius)
        )

    if isinstance(sigma, dict):
        widths.update({str(k).split("_")[0]: float(v) for k, v in sigma.items()})
    elif sigma is not None:
        widths = {element: float(sigma) for element in elements}

    return widths
