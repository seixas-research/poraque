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

    Notes
    -----
    On every load the input and target grids are compared and a mismatch is a
    hard error: a silently misaligned pair would train the operator on
    nonsense.
    """

    def __init__(self, root, task, input_transform=None, target_transform=None,
                 materials=None, cache=False, dtype=torch.float32):
        from .tasks import resolve_task

        self.root = str(root)
        self.task = resolve_task(task)
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
        source = FIELD_CLASSES[source_name].read(record.files[source_name], grid=grid)
        target = FIELD_CLASSES[target_name].read(record.files[target_name], grid=grid)

        record.shape = grid.shape
        if self.cache:
            self._cache[index] = (source, target)
        return source, target

    def __getitem__(self, index):
        """
        Return one sample.

        Returns
        -------
        dict
            ``input`` ``(1, Nx, Ny, Nz)``, ``target`` ``(1, Nx, Ny, Nz)``,
            ``cell`` ``(3, 3)`` in Å, plus ``volume``, ``shape`` and
            ``material``. Fields are normalized; ``target_physical`` carries
            the untransformed target for physics losses.
        """
        source, target = self.load_fields(index)

        source_values = torch.as_tensor(np.ascontiguousarray(source.data),
                                        dtype=self.dtype)
        target_values = torch.as_tensor(np.ascontiguousarray(target.data),
                                        dtype=self.dtype)

        return {
            "input": self.input_transform(source_values).unsqueeze(0),
            "target": self.target_transform(target_values).unsqueeze(0),
            "target_physical": target_values.unsqueeze(0),
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

        source_values, target_values = [], []
        for index in indices:
            source, target = self.load_fields(int(index))
            for values, sink in ((source.data, source_values),
                                 (target.data, target_values)):
                flat = values.ravel()
                take = min(max_points // max(len(indices), 1), flat.size)
                sink.append(rng.choice(flat, size=take, replace=False))

        self.input_transform = DEFAULT_TRANSFORMS[self.task.input_field](
            np.concatenate(source_values))
        self.target_transform = DEFAULT_TRANSFORMS[self.task.target_field](
            np.concatenate(target_values))
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
