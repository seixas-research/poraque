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

:mod:`poraque.data`
    Where the training data comes from: fetching charge densities from the
    Materials Project, and reconciling what a public archive publishes with
    what the pipeline expects.

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


import warnings
warnings.filterwarnings("ignore")

import os
import platform
from socket import gethostname
from sys import version as __python_version__
from sys import executable as __python_executable__
from ase import __version__ as __ase_version__
from ase import __file__ as __ase_file__
from numpy import __version__ as __numpy_version__
from numpy import __file__ as __numpy_file__
from scipy import __version__ as __scipy_version__
from scipy import __file__ as __scipy_file__
from matplotlib import __version__ as __mpl_version__
from matplotlib import __file__ as __mpl_file__
from torch import __version__ as __torch_version__
from torch import __file__ as __torch_file__
from yaml import __version__ as __yaml_version__
from yaml import __file__ as __yaml_file__
from pytest import __version__ as __pytest_version__
from pytest import __file__ as __pytest_file__

# from ase.parallel import parprint as print

from .version import __version__
__all__ = ["__version__"]



def banner():
    print("                                                 ")
    print("    ████▄ ▄███▄ ████▄  ▀▀█▄ ▄████ ██ ██ ▄█▀█▄    ")
    print("    ██ ██ ██ ██ ██ ▀▀ ▄█▀██ ██ ██ ██ ██ ██▄█▀    ")
    print("    ████▀ ▀███▀ ██    ▀█▄██ ▀████ ▀██▀█ ▀█▄▄▄    ")
    print("    ██                         ██                ")
    print("    ▀▀                         ▀▀                ")
    print("                                                 ")
    print(f"    version: {__version__}                       ")
    print("        developed by: Leandro Seixas Rocha      ")
    print("        homepage: https://github.com/seixas-research/poraque")
    print("                                                  ")
    print("------------------------------------------------------------")
    print("                                                  ")
    print("System:")
    print(f" ├── architecture: {platform.machine()}")
    print(f" ├── platform: {platform.system()}")
    print(f" ├── user: {os.environ['USER']}")
    print(f" ├── hostname: {gethostname()}")
    print(f" ├── cwd: {os.getcwd()}")
    print(f" └── PID: {os.getpid()}")
    print("                                               ")
    print("Python:")
    print(f" ├── version: {__python_version__}      ")
    print(f" └── executable: {__python_executable__}      ")
    print("                                               ")
    print("Dependencies:")
    print(f" ├── ase version: {__ase_version__}    [{__ase_file__[:-11]}]")
    print(f" ├── numpy version: {__numpy_version__}    [{__numpy_file__[:-11]}]")
    print(f" ├── scipy version: {__scipy_version__}    [{__scipy_file__[:-11]}]")
    print(f" ├── matplotlib version: {__mpl_version__}    [{__mpl_file__[:-11]}]")
    print(f" ├── torch version: {__torch_version__}    [{__torch_file__[:-11]}]")
    print(f" ├── yaml version: {__yaml_version__}    [{__yaml_file__[:-11]}]")
    print(f" └── pytest version: {__pytest_version__}    [{__pytest_file__[:-11]}]")
    print("                                               ")


banner()