# -*- coding: utf-8 -*-
# file: conftest.py
"""Shared pytest fixtures for the Poraquê test-suite."""

import numpy as np
import pytest

from poraque.backends.numpy import NumpyBackend
from poraque.core import Grid, System


@pytest.fixture
def backend():
    """A NumPy reference backend instance."""
    return NumpyBackend()


@pytest.fixture
def cubic_cell():
    """A 10x10x10 Bohr cubic cell."""
    return np.eye(3) * 10.0


@pytest.fixture
def grid(cubic_cell):
    """A periodic 24^3 grid on the cubic cell."""
    return Grid((24, 24, 24), cubic_cell, pbc=True)


@pytest.fixture
def coarse_grid(cubic_cell):
    """A small periodic grid for cheap iterative tests."""
    return Grid((16, 16, 16), cubic_cell, pbc=True)


@pytest.fixture
def hydrogen_system(cubic_cell):
    """A single nucleus (Z=1, 1 electron) at the cell centre."""
    return System(
        positions=[[5.0, 5.0, 5.0]],
        atomic_numbers=[1],
        cell=cubic_cell,
        pbc=True,
        electrons=1,
    )
