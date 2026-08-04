# -*- coding: utf-8 -*-
# file: data.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
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
"""

import os
from dataclasses import dataclass, field as dataclass_field

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from ..fields import ChargeDensity, ExternalPotential, FieldGrid, KineticEnergyDensity
from .transforms import DEFAULT_TRANSFORMS, FieldTransform, Identity

#: Field name -> the :class:`~poraque.fields.ScalarField` subclass handling it.
FIELD_CLASSES = {
    "EXTCAR": ExternalPotential,
    "CHGCAR": ChargeDensity,
    "TAUCAR": KineticEnergyDensity,
}


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
    """

    identifier: str
    directory: str
    files: dict = dataclass_field(default_factory=dict)
    shape: tuple = None


def discover_materials(root, required=("EXTCAR", "CHGCAR", "TAUCAR")):
    """
    Find material directories under ``root`` that contain every required field.

    Parameters
    ----------
    root : str or pathlib.Path
        Dataset root.
    required : sequence of str, optional
        File names that must all be present.

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
        files = {name: os.path.join(directory, name) for name in required}
        if all(os.path.exists(path) for path in files.values()):
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
    cache : bool, optional
        Keep decoded arrays in memory. Volumetric files are large; enable only
        for datasets that fit.
    dtype : torch.dtype, optional
        Output tensor dtype.
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
    channels : tuple of int
        ``(input_channels, target_channels)`` — what an operator trained on
        this dataset must be built with.

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
                 materials=None, cache=False, dtype=torch.float32,
                 spin="auto"):
        from .tasks import resolve_task

        self.root = str(root)
        self.task = resolve_task(task)
        self._requested_spin = spin
        self.materials = (materials if materials is not None
                          else discover_materials(root, self.task.required_files))
        if not self.materials:
            raise ValueError(
                f"No material directories with {list(self.task.required_files)} "
                f"found under {root!r}."
            )

        self.input_transform = input_transform or Identity()
        self.target_transform = target_transform or Identity()
        self.cache = bool(cache)
        self.dtype = dtype
        self._cache = {}
        self.spin = self._resolve_spin(spin)

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

    def _read_field(self, name, path, grid):
        """Read one field, as a spin pair when the dataset is spin-polarised."""
        if self.spin and name == "CHGCAR":
            from poraque.fields import SpinDensity

            return SpinDensity.read(path, grid=grid)
        return FIELD_CLASSES[name].read(path, grid=grid)

    def __getitem__(self, index):
        """
        Return one sample.

        Returns
        -------
        dict
            ``input`` ``(C_in, Nx, Ny, Nz)``, ``target`` ``(C_out, Nx, Ny,
            Nz)``, ``cell`` ``(3, 3)`` in Å, plus ``volume``, ``shape`` and
            ``material``. Fields are normalized; ``target_physical`` carries
            the untransformed target for physics losses. The channel counts
            are :attr:`channels`, which is ``(1, 1)`` unless the dataset is
            spin-polarised.
        """
        source, target = self.load_fields(index)

        source_values = _with_channel_axis(
            torch.as_tensor(np.ascontiguousarray(source.data),
                            dtype=self.dtype))
        target_values = _with_channel_axis(
            torch.as_tensor(np.ascontiguousarray(target.data),
                            dtype=self.dtype))

        return {
            "input": self.input_transform(source_values),
            "target": self.target_transform(target_values),
            "target_physical": target_values,
            "cell": torch.as_tensor(source.grid.cell, dtype=self.dtype),
            "volume": torch.tensor(source.grid.volume, dtype=self.dtype),
            "shape": tuple(source.grid.shape),
            "material": self.materials[index].identifier,
        }

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
            source, target = self.load_fields(int(index))
            for values, sink in ((source.data, source_values),
                                 (target.data, target_values)):
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

        def subset(indices):
            return type(self)(
                self.root, self.task,
                input_transform=self.input_transform,
                target_transform=self.target_transform,
                materials=[self.materials[i] for i in indices],
                cache=self.cache, dtype=self.dtype,
            )

        return subset(order[:cut]), subset(order[cut:])

    def state_dict(self):
        """Serializable normalization state, to be stored with a checkpoint."""
        return {
            "task": self.task.name,
            "input_transform": self.input_transform.state_dict(),
            "target_transform": self.target_transform.state_dict(),
        }

    def load_state_dict(self, state):
        """Restore normalizations saved by :meth:`state_dict`."""
        self.input_transform = FieldTransform.from_state_dict(state["input_transform"])
        self.target_transform = FieldTransform.from_state_dict(state["target_transform"])


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

    return {
        "input": torch.stack([s["input"] for s in samples]),
        "target": torch.stack([s["target"] for s in samples]),
        "target_physical": torch.stack([s["target_physical"] for s in samples]),
        "cell": torch.stack([s["cell"] for s in samples]),
        "volume": torch.stack([s["volume"] for s in samples]),
        "shape": samples[0]["shape"],
        "material": [s["material"] for s in samples],
    }


def make_dataloader(dataset, batch_size=1, shuffle=True, num_workers=0, seed=0):
    """
    Build a :class:`torch.utils.data.DataLoader` that tolerates ragged grids.

    Parameters
    ----------
    dataset : FieldPairDataset
    batch_size : int, optional
    shuffle : bool, optional
    num_workers : int, optional
    seed : int, optional

    Returns
    -------
    torch.utils.data.DataLoader
    """
    from torch.utils.data import DataLoader

    sampler = ShapeBucketSampler(dataset, batch_size=batch_size, shuffle=shuffle,
                                 seed=seed)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fields,
                      num_workers=num_workers)


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
    """Read only the grid-dimension line of a volumetric file."""
    with open(path, "r") as handle:
        blank_seen = False
        for line in handle:
            if not line.strip():
                blank_seen = True
                continue
            if blank_seen:
                return tuple(int(token) for token in line.split()[:3])
    raise ValueError(f"{path}: no grid-dimension line found.")
