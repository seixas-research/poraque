# -*- coding: utf-8 -*-
# file: dataset.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
One PyTorch ``Dataset`` over a mixture of data layouts.

:class:`MixedFieldDataset` takes a **list of directories**, works out what each
one is with :mod:`poraque.data.sources`, and serves aligned ``(input, target)``
field pairs across all of them::

    from poraque.data import MixedFieldDataset

    data = MixedFieldDataset(
        ["data/vasp/structures", "data/MP"],   # VASP runs and a download
        task="ext2chg",
        resolution=32,
    )

Each path is read independently, so the mixture can be any combination of DFT
run trees, Materials Project downloads and prepared caches. Nothing in the
training stack below this class knows which material came from where.

Three things this has to get right, and does
--------------------------------------------
**Identifiers must stay unique.** Two archives can easily both contain a
``struct_000``, and a collision would silently drop one material from the
dataset and corrupt every per-material report. Duplicates are prefixed with
their directory (``MP:mp-124``), and the run is told.

**Not every source has every field.** A public archive publishes no
:math:`\tau`; a calculation directory may or may not have written one. The
dataset reports what it can actually serve (:meth:`available_tasks`) and
refuses a task nothing supports, rather than failing on the first batch.

**Not every source defines** :math:`V_{\rm ext}` **the same way.** A calculation
with a ``POTCAR`` gives the tabulated local pseudopotential; a download
without one gives the Gaussian pseudo-ion model, which differs from it by
around 0.1 relative :math:`L_2`. Training across both means the input field is
*two different quantities* wearing one name, and the operator will spend
capacity reconciling them. That is sometimes what you want — it is a far larger
and more diverse dataset — and it is never what you want by accident, so the
dataset emits a warning naming both conventions when a mixture is built.

The clean fix is ``potcar_dir``: give the download the pseudopotentials its
densities were computed with and both sources use the *same* tabulated
construction, at which point the mixture is one quantity again and the warning
does not fire.
"""

import os
import warnings

import numpy as np

from ..ml.data import FieldPairDataset
from .sources import discover_records, resolve_source


class MixedFieldDataset(FieldPairDataset):
    r"""
    Aligned field pairs drawn from any mixture of data layouts.

    Parameters
    ----------
    paths : str or sequence of str
        One directory or several. Each holds one subdirectory per material --
        whatever produced it -- and all of them are pooled.
    task : str or TaskSpec, optional
        The mapping to serve; see :mod:`poraque.ml.tasks`.
    resolution : int, optional
        Longest grid axis after spectral downsampling — a Fourier truncation,
        exact for a band-limited plane-wave field. ``None`` keeps each
        material's native grid, which for a public archive is 48³ upwards.
    format : str, optional
        ``"vasp"``, or ``"auto"`` (default) to detect the code that wrote each
        directory. See :data:`~poraque.data.sources.DATA_FORMATS`.
    spin : bool, optional
        ``False`` (default) serves the total density as one channel. ``True``
        serves :math:`(\rho, m)` and requires every source to carry a
        magnetisation block.
    charges : dict, optional
        ``{element: Z_val}`` for materials with no ``POTCAR`` to read them
        from. Taken from ``potcar_dir`` when one is given, and inferred from
        the densities otherwise.
    potcar_dir : str, optional
        A ``POTCAR`` library, used wherever the data itself ships none. With it
        the external potential is VASP's exact tabulated one; without it, the
        Gaussian pseudo-ion model. This is what decides whether a download
        and a run tree define :math:`V_{\rm ext}` the same way — see
        the warning below.
    sigma : float or dict, optional
        Gaussian pseudo-ion width in Å, where a model potential is used.
    gaussian_blur : float, optional
        Blur width in Å applied to every computed potential.
    blur_method : {"spectral", "ndimage"}, optional
        How that blur is applied.
    pattern : str, optional
        Subdirectory prefix filter — the usual reason is a sibling directory of
        isolated-atom references that must not be trained on.
    cache : bool, optional
        Keep decoded fields in memory. Worth enabling with ``resolution`` set:
        the potential is otherwise recomputed every epoch.
    materials : list of MaterialRecord, optional
        Explicit record list, bypassing discovery — used for splits.
    log : callable, optional
        Receives one line per path describing what it contributed.
    warn_mixed_potentials : bool, optional
        Warn when the mixture spans more than one definition of
        :math:`V_{\rm ext}`. Leave it on unless the mixture is deliberate and
        already understood.
    **kwargs
        Passed to :class:`~poraque.ml.data.FieldPairDataset`.

    Attributes
    ----------
    sources : list of MaterialSource
        One per path, in the order given.

    Examples
    --------
    >>> data = MixedFieldDataset(["data/vasp/structures", "data/MP"],  # doctest: +SKIP
    ...                          task="ext2chg", resolution=32)
    >>> data.available_tasks()                                     # doctest: +SKIP
    ['ext2chg']
    """

    def __init__(self, paths, task="ext2chg", resolution=None, format="auto",
                 spin=False, charges=None, potcar_dir=None, sigma=None,
                 gaussian_blur=None, blur_method="spectral", pattern=None,
                 cache=False, materials=None, log=None,
                 warn_mixed_potentials=True, **kwargs):
        from ..ml.tasks import resolve_task

        self.paths = [str(paths)] if isinstance(paths, (str, os.PathLike)) \
            else [str(path) for path in paths]
        if not self.paths:
            raise ValueError("At least one data path is required.")

        self.resolution = int(resolution) if resolution else None
        self._format = format
        self._log = log or (lambda *_: None)
        self._source_options = {
            "charges": charges, "potcar_dir": potcar_dir, "sigma": sigma,
            "gaussian_blur": gaussian_blur, "blur_method": blur_method,
            "pattern": pattern, "log": self._log,
        }
        self.sources = self._build_sources(format)

        task = resolve_task(task)
        records = (materials if materials is not None
                   else self._discover(task))

        if warn_mixed_potentials and materials is None:
            self._warn_if_potentials_disagree(records, task)

        super().__init__(self.paths[0], task, materials=records, cache=cache,
                         spin=spin, **kwargs)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_sources(self, format):
        """One source per path. ``format`` names the code, or detects it."""
        return [resolve_source(path, format=format, **self._source_options)
                for path in self.paths]

    def _discover(self, task):
        """
        Enumerate every material this task can be trained on.

        Only materials supplying *both* of the task's fields are kept: a
        calculation directory with no ``TAUCAR`` belongs in an ``ext2chg``
        dataset and not in a ``chg2tau`` one, and that is a per-material
        question, not a per-directory one.
        """
        records = discover_records(self.sources, required=task.required_files,
                                   log=self._log)
        if not records:
            raise ValueError(
                f"No material under {self.paths} supplies both "
                f"{task.input_field} and {task.target_field}. Available "
                f"tasks here: {self.available_tasks() or 'none'}."
            )
        return records

    def _warn_if_potentials_disagree(self, records, task):
        """
        Warn when the mixture spans two definitions of the external potential.

        The test is on the *construction* each source uses, not on its layout.
        A run tree and a download that both build the tabulated
        potential — because ``potcar_dir`` supplies the pseudopotentials the
        latter lacks — are one quantity and warrant no warning; two archives of
        the same layout that disagree do.

        Only when the potential is actually used, too: a ``chg2tau`` dataset
        never touches it, and warning there would be noise.
        """
        if task.input_field != "EXTCAR" and task.target_field != "EXTCAR":
            return

        conventions = {}
        for record in records:
            conventions.setdefault(record.source.potential_model(), set()).add(
                record.source.root)
        if len(conventions) < 2:
            return

        warnings.warn(
            "This dataset mixes archives that define the external potential "
            "differently: "
            + "; ".join(f"{name} ({', '.join(sorted(roots))})"
                        for name, roots in sorted(conventions.items()))
            + ". The tabulated local pseudopotential and the Gaussian "
              "pseudo-ion model differ by roughly 0.1 relative L2, so the "
              "operator will be learning from two different input quantities "
              "under one name. Set data.potcar_dir so every source uses the "
              "tabulated construction, or pass warn_mixed_potentials=False "
              "once the mixture is a deliberate choice.",
            UserWarning,
            stacklevel=3,
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def available_tasks(self):
        """
        Task names at least one material under these paths can supply.

        Returns
        -------
        list of str
            In the registry's order. A dataset of lone densities returns
            ``["ext2chg"]``; a complete VASP archive returns both.
        """
        from ..ml.tasks import TASKS

        fields = set()
        for source in self.sources:
            for record in source.discover():
                fields.update(source.provides(record))
        return [name for name, task in TASKS.items()
                if set(task.required_files) <= fields]

    def contributions(self):
        """
        How many materials each path contributed.

        Returns
        -------
        dict
            ``{path: count}``, for the run header — a mixed dataset whose
            second archive silently matched nothing is otherwise invisible.
        """
        counts = dict.fromkeys(self.paths, 0)
        for record in self.materials:
            counts[record.source.root] = counts.get(record.source.root, 0) + 1
        return counts

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _resolve_spin(self, requested):
        r"""
        Whether densities are served as one channel or two.

        Relaxed relative to the base class, deliberately. There, ``spin=False``
        against a two-block file is an error, because a single VASP directory
        is expected to describe one calculation and a mismatch means the
        dataset is mislabelled. Across a *mixture* that reasoning does not
        hold: public archives are spin-polarised as a matter of policy while
        local runs often are not, and reading only the total block is a
        legitimate reduction of data that is present — unlike inventing a
        magnetisation channel that is not.

        So ``False`` serves :math:`\rho`, ``True`` requires and serves
        :math:`(\rho, m)` from every material, and ``"auto"`` follows the data
        and refuses a mixture it cannot serve consistently.
        """
        if requested is False:
            return False

        polarized = {record.source.is_spin_polarized(record)
                     for record in self.materials}

        if requested == "auto":
            if len(polarized) > 1:
                raise ValueError(
                    "Some materials in this mixture are spin-polarised and "
                    "some are not, so spin='auto' has no single answer. Pass "
                    "spin=False to train on the total density throughout, "
                    "which every one of them carries."
                )
            return polarized.pop()

        if not all(polarized):
            unpolarized = [record.identifier for record in self.materials
                           if not record.source.is_spin_polarized(record)]
            raise ValueError(
                f"spin=True was requested, but {len(unpolarized)} material(s) "
                f"carry no magnetisation block ({unpolarized[:5]}...). There "
                f"are no values for a second channel."
            )
        return True

    def load_fields(self, index):
        """
        Load one material's input and target fields in physical units.

        The source attached to the record decides how — read from disk,
        computed from a calculation's inputs, or computed from the density's
        own header. Both fields are placed on one shared grid, and downsampled
        together when :attr:`resolution` is set.

        Returns
        -------
        tuple
            ``(input_field, target_field)``.
        """
        if index in self._cache:
            return self._cache[index]

        record = self.materials[index]
        source = record.source
        native = source.grid(record)

        fields = tuple(
            source.read(record, name, native, spin=self.spin)
            for name in (self.task.input_field, self.task.target_field)
        )
        if self.resolution:
            fields = self._downsample(fields, native)

        record.shape = tuple(fields[0].grid.shape)
        if self.cache:
            self._cache[index] = fields
        return fields

    def _downsample(self, fields, native):
        """Fourier-truncate every field onto one reduced grid."""
        from ..fields.resample import downsampled_grid, resample_field

        reduced = downsampled_grid(native, self.resolution)
        return tuple(resample_field(field, reduced.shape, grid=reduced)
                     for field in fields)

    def shapes(self):
        """
        Grid shape of every material, from headers alone.

        Overridden because the base class peeks at the *input* file, and here
        the input file often does not exist — the potential is built on the
        density's grid, so that is the grid to report.
        """
        from ..fields.resample import downsample_shape

        shapes = []
        for record in self.materials:
            if record.shape is None:
                native = record.source.shape(record)
                record.shape = (downsample_shape(native,
                                                 target_max=self.resolution)
                                if self.resolution else tuple(native))
            shapes.append(record.shape)
        return shapes

    # ------------------------------------------------------------------ #
    def split(self, fraction=0.8, seed=0):
        """
        Split by material, carrying this dataset's settings to both halves.

        The split is at the **material** level and ignores which archive each
        came from, so a mixed dataset's validation set is a random sample of
        the mixture rather than one whole archive — which would measure
        transfer between archives and report it as generalisation.
        """
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.materials))
        cut = int(round(fraction * len(order)))
        if cut == 0 or cut == len(order):
            raise ValueError(
                f"fraction={fraction} splits {len(order)} materials into "
                f"{cut} and {len(order) - cut}: one side is empty. Adjust the "
                f"fraction or add materials."
            )

        def subset(indices):
            other = type(self)(
                self.paths, self.task, resolution=self.resolution,
                # The *resolved* format, not the "auto" default: a dataset
                # that needed an explicit format must not have its halves
                # silently re-detected.
                format=self._format, log=self._log,
                spin=self.spin, cache=self.cache,
                references=self.references,
                materials=[self.materials[i] for i in indices],
                input_transform=self.input_transform,
                target_transform=self.target_transform,
                dtype=self.dtype, warn_mixed_potentials=False,
                **{key: value for key, value in self._source_options.items()
                   if key != "log"},
            )
            return other

        return subset(order[:cut]), subset(order[cut:])

