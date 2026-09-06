# -*- coding: utf-8 -*-
# file: __init__.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
External data sources, and the ingestion that turns them into training sets.

:mod:`poraque.fields` reads what a DFT calculation left in a directory;
this package is about getting that data in the first place and reconciling
what a public archive publishes with what the pipeline expects.

Materials Project
-----------------
:class:`~poraque.data.materials_project.MPDataFetcher` downloads charge
densities for a chemical space; :mod:`poraque.data.mp_dataset` trains on them,
synthesising the external potential from the structure each density carries and
handling the fields MP does not publish.

::

    from poraque.data import MPDataFetcher, MPChargeDensityDataset

    with MPDataFetcher(["Pt", "Pd", "Ni"], outdir="data/MP") as mp:
        print(mp.estimate())            # exact size, nothing transferred
        mp.run(max_size_mb=20)

    data = MPChargeDensityDataset("data/MP", resolution=32)

Nothing here is imported by :mod:`poraque.ml` or :mod:`poraque.fields`, so the
Materials Project client stays out of the import path of a training run that
does not use it.
"""

from .cache import build_field_cache, build_paw_reference, load_paw_reference
from .dataset import MixedFieldDataset
from .materials_project import Estimate, MPDataFetcher, load_api_key
from .mp_dataset import (
    MPChargeDensityDataset,
    available_tasks,
    build_mp_cache,
    discover_mp_chgcars,
)
from .provenance import code_version, file_hash
from .sources import (
    DATA_FORMATS,
    CalculationSource,
    MaterialSource,
    discover_records,
    infer_valence_charges,
    resolve_source,
)

__all__ = [
    "DATA_FORMATS",
    "CalculationSource",
    "Estimate",
    "MPChargeDensityDataset",
    "MPDataFetcher",
    "MaterialSource",
    "MixedFieldDataset",
    "available_tasks",
    "build_field_cache",
    "build_mp_cache",
    "build_paw_reference",
    "code_version",
    "discover_mp_chgcars",
    "discover_records",
    "file_hash",
    "infer_valence_charges",
    "load_api_key",
    "load_paw_reference",
    "resolve_source",
]
