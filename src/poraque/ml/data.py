# -*- coding: utf-8 -*-
# file: data.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Dataset plumbing for field-to-field learning across many materials.

Expected layout — one directory per material, each holding the three fields on
that material's own grid::

    dataset_root/
        Si_diamond/   EXTCAR  CHGCAR  TAUCAR
        GaAs/         EXTCAR  CHGCAR  TAUCAR
        MoS2/         EXTCAR  CHGCAR  TAUCAR

Grids differ *between* materials and are identical *within* one, which is
exactly the invariant :mod:`poraque.fields` establishes and the invariant this
module validates on load.

Batching ragged grids
---------------------
Samples of different spatial shape cannot be stacked into one tensor. Rather
than pad (which wastes compute and injects fake vacuum into the FFT) this
module groups samples of identical shape into batches with
:class:`ShapeBucketSampler`. Materials that share a shape train in real
batches; unusual shapes fall back to smaller batches automatically. Setting
``batch_size=1`` also works and needs no sampler at all.

Delta-density mode
------------------
With a ``baseline`` supplied, a density target becomes the **residual over a
superposition of isolated atoms**, :math:`\delta\rho = \rho - \rho_{\rm sup}`.
Most of a crystal's valence density is its free atoms placed side by side, so
this removes the part that was never in doubt — including nearly all of the
dynamic range the ``asinh`` transform exists to absorb — and leaves the bonding
charge. Measured on this project's own platinum cells, the residual is about 4.5 %
of the density in :math:`L^2`.

Two consequences, and neither is cosmetic. The target is now **signed**, so
positivity is a statement about :math:`\delta\rho + \rho_{\rm sup}` and not
about the target. And a relative :math:`L^2` reported on :math:`\delta\rho` is
**not comparable** with one reported on :math:`\rho`, because the denominator is
twenty times smaller. Every sample therefore carries its ``baseline``, so the
training loop can reconstruct the absolute density before any physics term sees
it, and so evaluation can quote the error where it means something. See
``DESIGN_PAW.md`` §3.1 and §3.3.
"""

import os
from dataclasses import dataclass, field as dataclass_field

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..fields import ChargeDensity, ExternalPotential, FieldGrid, KineticEnergyDensity
from .transforms import DEFAULT_TRANSFORMS, Identity

#: Field name -> the :class:`~poraque.fields.ScalarField` subclass handling it.
FIELD_CLASSES = {
    "EXTCAR": ExternalPotential,
    "CHGCAR": ChargeDensity,
    "TAUCAR": KineticEnergyDensity,
}

#: Above this estimated decoded size, ``cache="auto"`` declines to keep the
#: dataset in RAM.
#:
#: Four gibibytes is chosen to be uncontroversial rather than optimal: it is
#: comfortably below what a GPU node or a modern workstation has, and it is
#: enough for the datasets this package is actually used on --- 115 structures
#: at :math:`32^3` come to about 60 MiB, three orders of magnitude under it. The
#: budget exists for the other case, a set of :math:`128^3` grids over thousands
#: of materials, where silently caching would turn a slow run into a dead one.
#:
#: It deliberately does *not* consult the machine's free memory. A number read
#: from the host at construction time would make the same configuration cache
#: on one run and not on the next, so two runs of one experiment would differ in
#: speed by 10x for reasons nothing recorded.
CACHE_MEMORY_BUDGET = 4 * 1024 ** 3


@dataclass
class MaterialRecord:
    """
    One material's location on disk, resolved but not yet loaded.

    Attributes
    ----------
    identifier : str
        Directory name, used as the material's id.
    directory : str
        Path to the material's directory.
    files : dict
        ``{field_name: path}``.
    shape : tuple of int or None
        Grid shape, filled in on first load.
    source : MaterialSource or None
        Which layout this record came out of, and therefore how its fields are
        produced --- see :mod:`poraque.data.sources`. ``None`` for records
        discovered by :func:`discover_materials`, which reads one layout and
        needs no dispatch. It is what lets
        :class:`~poraque.data.dataset.MixedFieldDataset` hold materials from
        several directories of different shapes at once.
    """

    identifier: str
    directory: str
    files: dict = dataclass_field(default_factory=dict)
    shape: tuple = None
    source: object = None


def prepared_fields(directory, names):
    """
    ``{field: path}`` for one prepared material directory, in either layout.

    A cache stores its fields either as ``CHGCAR``-format text files or as one
    HDF5 store per material, and this is the only function in the codebase that
    has to know which::

        cache/mp-124/CHGCAR              -> "cache/mp-124/CHGCAR"
        cache/mp-124/fields.h5           -> "cache/mp-124/fields.h5::CHGCAR"

    Both spellings are paths every reader accepts, so the difference stops
    here. A field present in both layouts resolves to the **text** file: a
    directory holding both was converted by hand, the text file is the one
    every other tool can read, and picking silently between two disagreeing
    copies is not this function's call to make.

    Parameters
    ----------
    directory : str
    names : sequence of str
        Fields to look for.

    Returns
    -------
    dict
        Only the fields actually present.
    """
    files = {name: os.path.join(directory, name) for name in names
             if os.path.exists(os.path.join(directory, name))}

    from ..fields.hdf5 import HDF5_SUFFIXES

    store = next((os.path.join(directory, entry)
                  for entry in sorted(os.listdir(directory))
                  if entry.lower().endswith(HDF5_SUFFIXES)), None)
    if store is None:
        return files

    from ..fields.hdf5 import field_names, join_target

    try:
        available = set(field_names(store))
    except (OSError, ValueError):
        # An unreadable or half-written store is not a reason to lose the text
        # files beside it; the material is simply served with what parses.
        return files
    for name in names:
        if name in available and name not in files:
            files[name] = join_target(store, name)
    return files


def discover_materials(root, required=("EXTCAR", "CHGCAR", "TAUCAR")):
    """
    Find material directories under ``root`` that contain every required field.

    Both cache layouts are found: a directory of ``CHGCAR``-format files, or
    one holding an HDF5 field store — see :func:`prepared_fields`.

    Parameters
    ----------
    root : str or pathlib.Path
        Dataset root.
    required : sequence of str, optional
        Field names that must all be present.

    Returns
    -------
    list of MaterialRecord
        Sorted by identifier for reproducible ordering.
    """
    records = []
    for entry in sorted(os.listdir(root)):
        directory = os.path.join(root, entry)
        if not os.path.isdir(directory):
            continue
        files = prepared_fields(directory, required)
        if all(name in files for name in required):
            records.append(MaterialRecord(entry, directory, files))
    return records


class FieldPairDataset(Dataset):
    """
    Aligned ``(input field, target field)`` pairs over a set of materials.

    Parameters
    ----------
    root : str or pathlib.Path
        Dataset root laid out as described in the module docstring.
    task : TaskSpec or str
        Which mapping to serve, e.g. ``"ext2chg"`` or ``"chg2tau"``; see
        :mod:`poraque.ml.tasks`.
    input_transform, target_transform : FieldTransform, optional
        Normalizations. Use :meth:`fit_transforms` to derive them from the
        data.
    materials : list of MaterialRecord, optional
        Explicit record list, bypassing discovery (used for train/val splits).
    cache : {"auto", True, False}, optional
        Keep decoded fields --- and the tensors made from them --- in memory
        between epochs. ``"auto"``, the default, enables it when the decoded
        dataset fits in :data:`CACHE_MEMORY_BUDGET`.

        Uncached, **every epoch re-reads, decompresses and re-parses every
        field**, and the arithmetic waits on it. Measured on a V100 with 115
        structures at :math:`32^3`: :meth:`__getitem__` was 59 % of the
        training loop --- 78.9 s over 483 calls --- with the GPU at 2-4 %
        utilisation, and turning the cache on took twenty epochs from 168.0 s to
        16.3 s, a factor of **10.3**, with the validation error identical to
        five decimals.

        ``"auto"`` rather than a bare ``True`` because the risk the old
        ``False`` default was guarding is real and the default was still wrong:
        a set that fits is the common case, and a set of :math:`128^3` grids
        over thousands of materials is what the budget refuses. What ``"auto"``
        cannot see is the rest of the machine, so an explicit ``False`` remains
        the answer when the process shares its RAM with something else.
    dtype : torch.dtype, optional
        Output tensor dtype.
    references : ReferenceEnergies, str or dict, optional
        Isolated-atom energies. When given, each sample carries
        ``reference_energy`` --- :math:`\\sum_i E_{\\rm iso}(Z_i)` for that
        structure --- and :meth:`reference_energy` is available for reporting.

        .. note::

           This is **not** a training target and does not enter the loss. The
           operators map fields to fields (``EXTCAR -> CHGCAR``,
           ``CHGCAR -> TAUCAR``); no energy is regressed anywhere in
           :mod:`poraque.ml`, and the total energy is obtained afterwards by
           integrating the predicted fields with
           :class:`~poraque.physics.EnergyCalculator`. The value is carried
           here so that evaluation and reporting can quote a cohesive energy
           per structure without re-reading the reference directory, and so
           that a future energy-regression head has it to hand.

    baseline : AtomicReferenceLibrary, str or None, optional
        Isolated-atom database enabling **delta-density mode**: the target
        becomes :math:`\\rho - \\rho_{\\rm sup}` and every sample carries its
        ``baseline`` so the absolute density can be reconstructed downstream.
        Accepts a library, a path to one, or ``None`` (absolute mode, the
        default). Ignored unless the task's target is a ``CHGCAR`` --- there is
        no atomic superposition of a kinetic energy density, so ``chg2tau`` is
        unaffected whatever is passed.
    spin : {"auto", True, False}, optional
        Whether the ``CHGCAR`` fields carry two channels
        (:math:`\\rho`, :math:`m`) from an ``ISPIN = 2`` run. ``"auto"``, the
        default, inspects the first material's file: a spin-polarised
        ``CHGCAR`` has a second grid block and a collinear one does not, so the
        data answers the question and no flag can contradict it. Pass ``True``
        or ``False`` to require one or the other, which turns a mixed or
        mislabelled dataset into an error instead of a silent half-conversion.

    Attributes
    ----------
    spin : bool
        Resolved answer. :attr:`channels` follows from it.
    cache : bool
        Resolved answer, so a caller can log what ``"auto"`` decided rather
        than what it was asked. :attr:`cache_bytes` is the size it decided on.

    Notes
    -----
    On every load the input and target grids are compared and a mismatch is a
    hard error: a silently misaligned pair would train the operator on
    nonsense.

    A spin-polarised dataset is only meaningful where the field *is* a density.
    ``EXTCAR`` is the ionic potential, which does not depend on spin, so the
    ``ext2chg`` task is one channel in and two out; ``chg2tau`` is two in and,
    for now, one out — :math:`\\tau` is written by VASP as a single block even
    under ``ISPIN = 2``.
    """

    def __init__(self, root, task, input_transform=None, target_transform=None,
                 materials=None, cache="auto", dtype=torch.float32,
                 spin="auto", references=None, baseline=None):
        from .tasks import resolve_task

        self.root = str(root)
        self.task = resolve_task(task)
        self._requested_spin = spin
        self.references = _resolve_references(references)
        self.baseline = _resolve_baseline(baseline)
        self._baselines = {}
        self.materials = (materials if materials is not None
                          else discover_materials(root, self.task.required_files))
        if not self.materials:
            raise ValueError(
                f"No material directories with {list(self.task.required_files)} "
                f"found under {root!r}."
            )

        self.input_transform = input_transform or Identity()
        self.target_transform = target_transform or Identity()
        self.dtype = dtype
        self._cache = {}
        # Pre-transform tensors, keyed by index. A second level, because the
        # field cache above removes the parse and leaves the conversion:
        # `ascontiguousarray` plus `as_tensor` plus the delta-density
        # subtraction still ran on every access of every epoch.
        self._tensors = {}
        self._baseline_tensors = {}
        self.spin = self._resolve_spin(spin)
        # After spin, which decides the channel count the estimate is over.
        self._cache_bytes = None
        self.cache = self._resolve_cache(cache)

    @property
    def cache_bytes(self):
        """
        Estimated RAM cost of caching this dataset, from :meth:`estimate_cache_bytes`.

        Available whether or not the cache was enabled, because "caching was
        declined" is only actionable beside the number that declined it.

        Computed on first access rather than in ``__init__``: the estimate
        reads a header per material, and a caller that only wants
        ``len(dataset)`` should not pay for one.
        """
        if self._cache_bytes is None:
            self._cache_bytes = self.estimate_cache_bytes()
        return self._cache_bytes

    def estimate_cache_bytes(self):
        """
        What caching this dataset's decoded fields would cost in RAM.

        Grid points times itemsize times channels, summed over the input and
        target fields. Shapes come from :meth:`shapes`, which reads headers
        rather than data, so the estimate costs a few file seeks.

        Returns
        -------
        int
            Bytes. An estimate of the *decoded* size, which is what matters:
            a gzipped text ``CHGCAR`` on disk says nothing useful about the
            float64 array it becomes.
        """
        itemsize = torch.empty(0, dtype=self.dtype).element_size()
        in_channels, out_channels = self.channels
        # The pre-transform tensor cache holds the same two fields again, and
        # delta-density mode holds a baseline per material on top. Counting
        # them is the difference between an estimate and an underestimate.
        per_point = in_channels + out_channels
        per_point += in_channels + out_channels
        if self.baseline is not None and self.task.target_field == "CHGCAR":
            per_point += 1
        total = 0
        for shape in self.shapes():
            points = 1
            for extent in shape:
                points *= int(extent)
            total += points * itemsize * per_point
        return int(total)

    def _resolve_cache(self, requested):
        """
        Decide whether to keep decoded fields in memory.

        ``"auto"`` compares :attr:`cache_bytes` against
        :data:`CACHE_MEMORY_BUDGET`. An explicit ``True`` is honoured whatever
        the size --- the caller may know something the estimate does not, and
        overriding a deliberate flag would be worse than running out of memory
        where it was asked for.
        """
        if isinstance(requested, str):
            key = requested.strip().lower()
            if key == "auto":
                return self.cache_bytes <= CACHE_MEMORY_BUDGET
            # A YAML file can spell a boolean as a quoted word, and refusing
            # `cache: "true"` would be pedantry about a setting whose meaning
            # is not in doubt. Anything else is a typo and raises: silently
            # reading an unrecognised word as False would turn it into a 10x
            # slowdown with nothing said.
            if key in ("true", "yes", "on", "1"):
                return True
            if key in ("false", "no", "off", "0"):
                return False
            raise ValueError(
                f"cache={requested!r} is not a recognised setting; "
                f"expected 'auto', True or False.")
        return bool(requested)

    def _resolve_spin(self, requested):
        """Decide whether this dataset's densities are two-channel."""
        from poraque.fields import is_spin_polarized

        density_files = [name for name in
                         (self.task.input_field, self.task.target_field)
                         if name == "CHGCAR"]
        if not density_files:
            return False

        detected = is_spin_polarized(self.materials[0].files["CHGCAR"])
        if requested == "auto":
            return detected
        if bool(requested) != detected:
            raise ValueError(
                f"spin={requested!r} was requested but "
                f"{self.materials[0].files['CHGCAR']} "
                f"{'is' if detected else 'is not'} spin-polarised. A CHGCAR "
                f"either carries a magnetisation block or it does not; "
                f"overriding that would train on a channel the data has no "
                f"values for."
            )
        return bool(requested)

    @property
    def channels(self):
        """``(input_channels, target_channels)`` for this task and dataset."""
        if not self.spin:
            return (1, 1)
        return (2 if self.task.input_field == "CHGCAR" else 1,
                2 if self.task.target_field == "CHGCAR" else 1)

    def reference_energy(self, index):
        r"""
        :math:`E_{\rm ref} = \sum_i E_{\rm iso}(Z_i)` for material ``index``.

        Parameters
        ----------
        index : int

        Returns
        -------
        float or None
            ``None`` when no references were supplied, or when they do not
            cover every species in the structure.
        """
        if self.references is None:
            return None

        structure = self.load_fields(index)[0].structure
        if not self.references.covers(structure):
            return None
        return self.references.total_for(structure)

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.materials)

    def load_fields(self, index):
        """
        Load one material's input and target fields in **physical units**.

        Returns
        -------
        tuple
            ``(input_field, target_field)`` as
            :class:`~poraque.fields.ScalarField` instances sharing one grid.
        """
        if index in self._cache:
            return self._cache[index]

        record = self.materials[index]
        source_name, target_name = self.task.input_field, self.task.target_field

        # The shared grid is taken from the input file and *imposed* on the
        # target, so any inconsistency raises instead of passing silently.
        grid = FieldGrid.from_file(record.files[source_name])
        source = self._read_field(source_name, record.files[source_name], grid)
        target = self._read_field(target_name, record.files[target_name], grid)

        record.shape = grid.shape
        if self.cache:
            self._cache[index] = (source, target)
        return source, target

    def baseline_for(self, index):
        r"""
        The atomic-superposition baseline for material ``index``.

        Returns
        -------
        numpy.ndarray or None
            ``(Nx, Ny, Nz)`` in e/Å³, or ``None`` in absolute-density mode or
            for a task whose target is not a density.

        Notes
        -----
        Cached per material regardless of the ``cache`` flag. It is one FFT and
        a structure factor, the structure never changes, and recomputing it
        every epoch would be the single most wasteful thing in the loop.
        """
        if self.baseline is None or self.task.target_field != "CHGCAR":
            return None
        if index in self._baselines:
            return self._baselines[index]

        from ..fields.atomic import atomic_superposition

        target = self.load_fields(index)[1]
        values = atomic_superposition(target.structure, target.grid,
                                      self.baseline).data
        self._baselines[index] = values
        return values

    def target_values(self, index):
        r"""
        The physical target this dataset actually regresses.

        The reference field in absolute mode; :math:`\rho - \rho_{\rm sup}` in
        delta-density mode. One method rather than two so that
        :meth:`fit_transforms` and :meth:`__getitem__` cannot disagree about
        what is being normalized --- fitting an ``Asinh`` scale to
        :math:`\rho` and then feeding it :math:`\delta\rho` would be a silent
        twenty-fold scale error.

        Returns
        -------
        numpy.ndarray
        """
        values = np.asarray(self.load_fields(index)[1].data, dtype=float)
        baseline = self.baseline_for(index)
        if baseline is None:
            return values

        # A spin-polarised target is (rho, m). Only the total density has an
        # atomic superposition: the references are ISPIN = 1 free atoms, and
        # subtracting a zero moment from m would be a no-op dressed up as
        # physics. The magnetisation channel is left alone.
        if values.ndim == 4:
            values = values.copy()
            values[0] = values[0] - baseline
            return values
        return values - baseline

    def _read_field(self, name, path, grid):
        """Read one field, as a spin pair when the dataset is spin-polarised."""
        if self.spin and name == "CHGCAR":
            from poraque.fields import SpinDensity

            return SpinDensity.read(path, grid=grid)
        return FIELD_CLASSES[name].read(path, grid=grid)

    def sample_tensors(self, index):
        r"""
        The **pre-transform** ``(input, target, cell)`` tensors for a material.

        Memoised when :attr:`cache` is on, which removes the second layer of
        per-epoch work: caching the :class:`~poraque.fields.ScalarField`
        objects kills the parse, but every access still redid
        :func:`numpy.ascontiguousarray`, :func:`torch.as_tensor` and --- in
        delta-density mode --- the subtraction of the baseline.

        **Pre**-transform on purpose. ``input_transform`` and
        ``target_transform`` are fitted by :meth:`fit_transforms` *after* the
        dataset exists, and are replaced outright on a validation split, so a
        cache of normalized tensors would go stale the moment the scale it was
        built with was superseded --- silently, since the values would still be
        finite and plausible. What is cached here is the physical field, which
        no later call can invalidate.

        Returns
        -------
        tuple of torch.Tensor
            ``(source, target, cell)``. The first two carry a leading channel
            axis; the target is :math:`\delta\rho` in delta-density mode.
        """
        if index in self._tensors:
            return self._tensors[index]

        source, _ = self.load_fields(index)
        tensors = (
            _with_channel_axis(torch.as_tensor(
                np.ascontiguousarray(source.data), dtype=self.dtype)),
            _with_channel_axis(torch.as_tensor(
                np.ascontiguousarray(self.target_values(index)),
                dtype=self.dtype)),
            torch.as_tensor(source.grid.cell, dtype=self.dtype),
        )
        if self.cache:
            self._tensors[index] = tensors
        return tensors

    def __getitem__(self, index):
        """
        Return one sample.

        Returns
        -------
        dict
            ``input`` ``(C_in, Nx, Ny, Nz)``, ``target`` ``(C_out, Nx, Ny,
            Nz)``, ``cell`` ``(3, 3)`` in Å, plus ``shape`` and
            ``material``. Fields are normalized; ``target_physical`` carries
            the untransformed target for physics losses. The channel counts
            are :attr:`channels`, which is ``(1, 1)`` unless the dataset is
            spin-polarised.
        """
        # No `load_fields` here: `sample_tensors` is the only reader, so an
        # uncached dataset opens each file once per access rather than twice,
        # and a cached one touches no field object at all after the first pass.
        # `record.shape` is populated either by that first read or by the
        # header scan `shapes()` does at construction.
        source_values, target_values, cell = self.sample_tensors(index)
        record = self.materials[index]

        sample = {
            "input": self.input_transform(source_values),
            "target": self.target_transform(target_values),
            "target_physical": target_values,
            "cell": cell,
            "shape": tuple(record.shape),
            "material": record.identifier,
            "reference_energy": self.reference_energy(index),
        }

        values = self.baseline_tensor(index, target_values.shape[0])
        if values is not None:
            sample["baseline"] = values
        return sample

    def baseline_tensor(self, index, channels):
        r"""
        The atomic superposition as a tensor shaped like the target.

        Carried in the sample rather than recomputed downstream: the training
        loop adds it back before every physics term, and the loss would
        otherwise have to rebuild a structure factor per step.

        Memoised on the same terms as :meth:`sample_tensors`. The array behind
        it is already cached unconditionally by :meth:`baseline_for` --- it is
        an FFT over a structure that never changes --- so what this saves is
        only the conversion, but that conversion ran on every access of every
        epoch and the padding allocates a second full-size block.

        Parameters
        ----------
        index : int
        channels : int
            The target's channel count. A spin-polarised target is
            :math:`(\rho, m)` and only :math:`\rho` has a superposition, so the
            magnetisation channel is padded with zeros --- shaped like the
            target so the two add without broadcasting surprises.

        Returns
        -------
        torch.Tensor or None
            ``None`` in absolute-density mode, and for any task whose target is
            not a density.
        """
        key = (index, int(channels))
        if key in self._baseline_tensors:
            return self._baseline_tensors[key]

        baseline = self.baseline_for(index)
        if baseline is None:
            return None

        values = torch.as_tensor(np.ascontiguousarray(baseline),
                                 dtype=self.dtype).unsqueeze(0)
        if channels > 1:
            values = torch.cat(
                [values, torch.zeros_like(values).expand(
                    channels - 1, -1, -1, -1)])
        if self.cache:
            self._baseline_tensors[key] = values
        return values

    # ------------------------------------------------------------------ #
    def shapes(self):
        """
        Grid shape of every material, reading headers only.

        Returns
        -------
        list of tuple
        """
        shapes = []
        for record in self.materials:
            if record.shape is None:
                record.shape = _peek_shape(record.files[self.task.input_field])
            shapes.append(record.shape)
        return shapes

    def fit_transforms(self, max_materials=32, max_points=200_000, seed=0):
        """
        Derive normalizations from a subsample of the data.

        Statistics are gathered from randomly chosen grid points of up to
        ``max_materials`` materials, which is far cheaper than a full pass and
        entirely sufficient for setting a scale.

        Parameters
        ----------
        max_materials : int, optional
            Number of materials to sample.
        max_points : int, optional
            Grid points to draw per field.
        seed : int, optional
            RNG seed.

        Returns
        -------
        tuple of FieldTransform
            The fitted ``(input_transform, target_transform)``, also installed
            on this dataset.
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self))[:max_materials]
        per_material = max_points // max(len(indices), 1)

        # One bucket per channel, so a multi-channel field gets one transform
        # per channel rather than a single scale fitted across all of them.
        source_values, target_values = [], []
        for index in indices:
            source, _ = self.load_fields(int(index))
            for values, sink in ((source.data, source_values),
                                 (self.target_values(int(index)),
                                  target_values)):
                channels = values if values.ndim == 4 else values[None]
                if not sink:
                    sink.extend([] for _ in range(channels.shape[0]))
                for channel, bucket in zip(channels, sink):
                    flat = channel.ravel()
                    take = min(per_material, flat.size)
                    bucket.append(rng.choice(flat, size=take, replace=False))

        self.input_transform = _fit_transform(self.task.input_field,
                                              source_values)
        self.target_transform = _fit_transform(self.task.target_field,
                                               target_values)
        return self.input_transform, self.target_transform

    def split(self, fraction=0.8, seed=0):
        """
        Split into two datasets by material (never by grid point).

        Splitting at the material level is essential: two crops of the same
        material would leak information across the split and inflate the
        validation score.

        Parameters
        ----------
        fraction : float, optional
            Share of materials assigned to the first dataset.
        seed : int, optional
            RNG seed.

        Returns
        -------
        tuple of FieldPairDataset
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
            return type(self)(
                self.root, self.task,
                input_transform=self.input_transform,
                target_transform=self.target_transform,
                materials=[self.materials[i] for i in indices],
                cache=self.cache, dtype=self.dtype,
                # The resolved settings, not the constructor defaults: a
                # subset that re-detected spin or dropped the reference
                # energies would silently differ from its parent. The baseline
                # belongs to that list too -- a validation split in absolute
                # mode against a model trained on residuals would report an
                # error against the wrong field entirely.
                spin=self.spin, references=self.references,
                baseline=self.baseline,
            )

        return subset(order[:cut]), subset(order[cut:])


# ---------------------------------------------------------------------- #
# Ragged-shape batching
# ---------------------------------------------------------------------- #
class ShapeBucketSampler(Sampler):
    """
    Yield batches whose samples all share one grid shape.

    Materials are bucketed by ``(Nx, Ny, Nz)`` and each bucket is chunked into
    batches, so no padding is ever needed and the FFT sees only real data. The
    batch order is shuffled every epoch, and so is the content of each bucket,
    so shape does not correlate with training order.

    Parameters
    ----------
    dataset : FieldPairDataset
        Dataset to sample from.
    batch_size : int, optional
        Maximum samples per batch; the final batch of each bucket may be
        smaller.
    shuffle : bool, optional
        Shuffle within buckets and across batches.
    drop_last : bool, optional
        Drop trailing partial batches.
    seed : int, optional
        Base RNG seed; the epoch index is mixed in.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=False, seed=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self.buckets = {}
        for index, shape in enumerate(dataset.shapes()):
            self.buckets.setdefault(tuple(shape), []).append(index)

    def set_epoch(self, epoch):
        """Set the epoch so shuffling differs between passes."""
        self.epoch = int(epoch)

    def _batches(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = []
        for indices in self.buckets.values():
            indices = list(indices)
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                chunk = indices[start:start + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


class DistributedShapeBucketSampler(Sampler):
    """
    :class:`ShapeBucketSampler`, split across the ranks of a DDP group.

    The batches are distributed, not the samples, and that is the whole design.
    ``DataLoader`` takes a ``sampler`` **or** a ``batch_sampler`` and never
    both, so a plain :class:`~torch.utils.data.distributed.DistributedSampler`
    would have to displace the shape bucketing --- and a batch mixing
    :math:`32^3` with :math:`40^3` does not merely train badly, it raises in
    :func:`collate_fields`, because there is no padding anywhere in this
    pipeline and the FFT is the reason.

    So the bucketing runs first, unchanged and identically on every rank (it is
    a pure function of ``seed`` and ``epoch``, which is what makes the ranks
    agree without communicating), and the resulting *list of batches* is what
    a real ``DistributedSampler`` is then asked to partition. Every rank gets a
    unique, non-overlapping subset; no batch is split; no batch mixes shapes.

    **The padding is load-bearing.** ``DistributedSampler`` extends its index
    list to a multiple of the world size by wrapping around to the front, so
    every rank yields the same number of batches. That is not tidiness: DDP
    all-reduces gradients inside each ``backward()``, and a rank that runs out
    of batches first leaves the others waiting in a collective that will never
    complete. The job then burns its allocation in a hang rather than failing.
    The cost is that up to ``world_size - 1`` batches are seen twice in an
    epoch by somebody, which slightly over-weights a few materials --- against
    a deadlock, that is not a close call.

    Parameters
    ----------
    dataset : FieldPairDataset
    batch_size : int, optional
        Maximum samples per batch *per rank*. The effective global batch is
        ``batch_size * world_size``, which is worth remembering when comparing
        a four-GPU run against a one-GPU one: they are not the same optimiser.
    shuffle : bool, optional
    drop_last : bool, optional
        Passed to the inner :class:`ShapeBucketSampler` and applied to *batches
        within a bucket*, not to the rank partition --- the rank partition is
        padded rather than truncated, for the reason above.
    seed : int, optional
    num_replicas, rank : int, optional
        World size and this process's rank. Read from the initialised process
        group when omitted, which is what ``DistributedSampler`` does and what
        makes an explicit pair useful only in a test.

    See Also
    --------
    poraque.ml.distributed : the launch side, and why NCCL only.
    """

    def __init__(self, dataset, batch_size=1, shuffle=True, drop_last=False,
                 seed=0, num_replicas=None, rank=None):
        from torch.utils.data.distributed import DistributedSampler

        self.buckets_sampler = ShapeBucketSampler(
            dataset, batch_size=batch_size, shuffle=shuffle,
            drop_last=drop_last, seed=seed)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

        # The number of batches is a property of the buckets and the batch
        # size, not of the epoch: shuffling reorders batches and reorders
        # within a bucket, and neither changes how many chunks a bucket splits
        # into. So a fixed-length index range is a faithful stand-in for the
        # batch list, and `DistributedSampler` can partition it once.
        self._n_batches = len(self.buckets_sampler)
        self.partition = DistributedSampler(
            range(self._n_batches), num_replicas=num_replicas, rank=rank,
            shuffle=shuffle, seed=seed, drop_last=False,
        )
        self.num_replicas = self.partition.num_replicas
        self.rank = self.partition.rank

    def set_epoch(self, epoch):
        """
        Advance both halves of the sampler.

        Forwarded to the bucket sampler *and* to the ``DistributedSampler``.
        Missing either is a silent bug rather than a crash: forget the first
        and every epoch draws the same batches, forget the second and every
        epoch sends the same batches to the same rank, and in both cases
        training proceeds and merely learns less than the log claims.
        """
        self.epoch = int(epoch)
        self.buckets_sampler.set_epoch(epoch)
        self.partition.set_epoch(epoch)

    def _batches(self):
        # Rebuilt per epoch and identical on every rank, since it depends only
        # on `seed` and `epoch`. That identity is what allows the partition to
        # be agreed without a collective.
        batches = self.buckets_sampler._batches()
        if not batches:
            return []
        # `DistributedSampler`'s padding can wrap past the end when the batch
        # count is not a multiple of the world size; the modulo is what makes
        # that wrap land on a real batch.
        return [batches[index % len(batches)] for index in self.partition]

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self.partition)


def collate_fields(samples):
    """
    Collate samples that share one grid shape.

    Parameters
    ----------
    samples : list of dict
        Items produced by :meth:`FieldPairDataset.__getitem__`.

    Returns
    -------
    dict
        Batched tensors plus the shared ``shape`` and the list of material ids.

    Raises
    ------
    ValueError
        If the batch mixes grid shapes — the error names the offending shapes
        and points at :class:`ShapeBucketSampler`.
    """
    shapes = {sample["shape"] for sample in samples}
    if len(shapes) > 1:
        raise ValueError(
            f"Cannot batch mixed grid shapes {sorted(shapes)}. Use "
            f"ShapeBucketSampler as the DataLoader's batch_sampler, or set "
            f"batch_size=1."
        )

    batch = {
        "input": torch.stack([s["input"] for s in samples]),
        "target": torch.stack([s["target"] for s in samples]),
        "target_physical": torch.stack([s["target_physical"] for s in samples]),
        "cell": torch.stack([s["cell"] for s in samples]),
        "shape": samples[0]["shape"],
        "material": [s["material"] for s in samples],
        # Kept as a plain list, not stacked: the entries are None whenever no
        # reference energies were supplied, and torch.stack cannot batch that.
        # It is carried for reporting rather than consumed by the loss, so a
        # list is the honest container.
        "reference_energy": [s.get("reference_energy") for s in samples],
    }
    # Present only in delta-density mode, and then for every sample: a batch
    # mixing the two would mean two datasets were merged, which the shared-grid
    # invariant already forbids.
    if "baseline" in samples[0]:
        batch["baseline"] = torch.stack([s["baseline"] for s in samples])
    return batch


def make_dataloader(dataset, batch_size=1, shuffle=True, num_workers=0, seed=0,
                    pin_memory=False, distributed=None):
    """
    Build a :class:`torch.utils.data.DataLoader` that tolerates ragged grids.

    Parameters
    ----------
    dataset : FieldPairDataset
    batch_size : int, optional
    shuffle : bool, optional
    num_workers : int, optional
        Worker processes. **Prefer the dataset's in-memory cache to this**; see
        the note below.
    seed : int, optional
    pin_memory : bool, optional
        Stage batches in page-locked host memory. Only useful when the
        destination is CUDA --- it does nothing on MPS and some PyTorch
        versions warn there --- so the caller decides, since it is
        :func:`~poraque.ml.training.train` that knows the operator's device.
    distributed : DistributedContext, optional
        When it describes a real group, batches are partitioned across its
        ranks by :class:`DistributedShapeBucketSampler`. A disabled context and
        ``None`` are the same thing, so the caller passes it unconditionally.

    Returns
    -------
    torch.utils.data.DataLoader

    Notes
    -----
    ``num_workers`` and :attr:`FieldPairDataset.cache` are **alternatives, not
    complements**. Measured on 115 structures at :math:`32^3`, twenty epochs:
    the cache alone took 168.0 s to 16.3 s, four workers alone to 120.2 s, and
    the two together back up to **18.8 s**. Each worker is a process with a
    cache of its own, so the parse the cache removes is paid once per worker
    instead of once.

    That arithmetic gets worse under DDP, not better: each of the four ranks is
    already a process with a cache of its own, so ``num_workers`` on top
    multiplies the copies by another factor. One rank per GPU with the cache on
    and no workers is the configuration to start from.
    """
    from torch.utils.data import DataLoader

    if distributed:
        sampler = DistributedShapeBucketSampler(
            dataset, batch_size=batch_size, shuffle=shuffle, seed=seed,
            num_replicas=distributed.world_size, rank=distributed.rank)
    else:
        sampler = ShapeBucketSampler(dataset, batch_size=batch_size,
                                     shuffle=shuffle, seed=seed)
    options = {}
    if num_workers:
        # Rebuilding the worker pool every epoch costs more than the pool saves
        # on a dataset this small, and prefetching is the only reason to have
        # paid for the processes at all.
        options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fields,
                      num_workers=int(num_workers),
                      pin_memory=bool(pin_memory), **options)


def _resolve_references(references):
    """Accept a :class:`ReferenceEnergies`, a directory, a dict, or ``None``."""
    if references is None:
        return None

    from poraque.physics import ReferenceEnergies

    if isinstance(references, ReferenceEnergies):
        return references
    if isinstance(references, dict):
        return ReferenceEnergies(references)
    return ReferenceEnergies.from_directory(str(references))


def _resolve_baseline(baseline):
    """Accept an :class:`AtomicReferenceLibrary`, a path, or ``None``."""
    if baseline is None:
        return None

    from ..fields.atomic import AtomicReferenceLibrary

    if isinstance(baseline, AtomicReferenceLibrary):
        return baseline if len(baseline) else None
    library = AtomicReferenceLibrary.load(str(baseline))
    if not len(library):
        raise ValueError(
            f"{baseline!r} holds no isolated-atom references, so delta-density "
            f"mode has no baseline to subtract. Build one with `poraque-atoms`, "
            f"or set data.delta_density: false.")
    return library


def _fit_transform(field_name, per_channel):
    """
    Fit ``field_name``'s default transform to each channel's samples.

    A single-channel field gets the transform itself, not a
    :class:`~poraque.ml.transforms.Channelwise` of length one: wrapping it
    would change the serialized form of every existing checkpoint for no gain.
    """
    from .transforms import Channelwise

    fitted = [DEFAULT_TRANSFORMS[field_name](np.concatenate(samples))
              for samples in per_channel]
    return fitted[0] if len(fitted) == 1 else Channelwise(fitted)


def _with_channel_axis(values):
    """
    Ensure a leading channel axis.

    A scalar field arrives as ``(Nx, Ny, Nz)`` and a spin pair already as
    ``(2, Nx, Ny, Nz)``, so the axis is added from the rank rather than
    assumed — an unconditional ``unsqueeze(0)`` would turn the spin pair into a
    single sample of two "grids" and silently reinterpret the data.
    """
    return values.unsqueeze(0) if values.ndim == 3 else values


def _peek_shape(path):
    """
    The grid shape of a volumetric file, read from its header alone.

    Cheap by construction in both formats, which is what lets
    :class:`ShapeBucketSampler` bucket a dataset without decoding it: the text
    reader stops at the dimension line, and HDF5 keeps the shape in the object
    header.
    """
    from ..fields.hdf5 import is_hdf5_path

    if is_hdf5_path(path):
        from ..fields.hdf5 import peek_shape

        return peek_shape(path)

    from ..fields.io.compressed import open_text

    with open_text(path) as handle:
        blank_seen = False
        for line in handle:
            if not line.strip():
                blank_seen = True
                continue
            if blank_seen:
                return tuple(int(token) for token in line.split()[:3])
    raise ValueError(f"{path}: no grid-dimension line found.")
