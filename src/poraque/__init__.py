# -*- coding: utf-8 -*-
# file: __init__.py

"""
Poraquê — machine-learned density functionals on three-dimensional scalar fields.

Two subpackages carry the work:

:mod:`poraque.fields`
    The shared-grid data model: the local external potential, the valence
    charge density and the kinetic energy density of a material, all on one
    mesh and in one file format, plus a code-agnostic ingestion layer.

:mod:`poraque.ml`
    Fourier neural operators that map between those fields, with the
    differentiable DFT operators needed to constrain them physically.

:mod:`poraque.physics`
    Energy functionals evaluated on the predicted fields: the Kohn-Sham
    total-energy components, integrated on the shared grid.

:mod:`poraque.vis`
    Figures and typeset reports for trained models.

Each is imported directly::

    from poraque.fields import ExternalPotential, ChargeDensity
    from poraque.ml import FieldOperator, FieldPairDataset, train
    from poraque.physics import EnergyCalculator

:class:`poraque.calculator.Poraque` wraps the whole chain as an ASE
calculator::

    from poraque.calculator import Poraque

It is *not* re-exported here: importing it pulls in ASE and PyTorch, which the
field and energy layers do not need.
"""

from .version import __version__

__all__ = ["__version__"]
