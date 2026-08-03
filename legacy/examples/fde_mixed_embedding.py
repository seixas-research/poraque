# -*- coding: utf-8 -*-
"""Frozen-Density Embedding: a KS active region embedded in an OF region."""

import numpy as np

from poraque.backends.numpy import NumpyBackend
from poraque.core import Grid, SolverSettings, System
from poraque.fde import FDEEngine, Subsystem


cell = np.eye(3) * 12.0
grid = Grid((24, 24, 24), cell, pbc=True)

active = System([[5.0, 6.0, 6.0]], [1], cell, electrons=1)   # KS region
frozen = System([[7.5, 6.0, 6.0]], [1], cell, electrons=1)   # OF region

engine = FDEEngine(
    [
        Subsystem(active, method="ks", name="active", external_kwargs={"a": 0.8}),
        Subsystem(frozen, method="of", name="frozen", external_kwargs={"a": 0.8}),
    ],
    grid,
    NumpyBackend(),
    settings=SolverSettings(max_iter=8, tolerance=1e-4),
    inner_settings=SolverSettings(max_iter=40, tolerance=1e-5, mixing=0.2),
)

result = engine.freeze_and_thaw()
print(f"Converged:    {result.converged} ({result.iterations} cycles)")
print(f"Total energy: {result.total_energy:.6f} Hartree")
for name, value in result.energy_components.items():
    print(f"  {name:<24s} {value:12.6f}")
print(f"Total electrons: {result.density.integrate():.6f}")
