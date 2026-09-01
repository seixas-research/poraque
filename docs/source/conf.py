# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from unittest.mock import MagicMock

# On Read the Docs, mock the heavy scientific dependencies so that autodoc can
# import the package and read its docstrings without compiling/installing them.
if os.environ.get("READTHEDOCS") == "True":
    MOCK_MODULES = [
        "numpy", "numpy.fft", "numpy.linalg",
        "scipy", "scipy.ndimage", "scipy.interpolate", "scipy.special",
        "torch", "torch.nn", "torch.nn.functional", "torch.fft",
        "torch.utils", "torch.utils.data", "torch.optim",
        "ase", "matplotlib", "matplotlib.pyplot", "matplotlib.colors",
        "yaml",
    ]
    sys.modules.update((mod_name, MagicMock()) for mod_name in MOCK_MODULES)

sys.path.insert(0, os.path.abspath("../../src"))  # repository src/ layout

# -- Project information ------------------------------------------------------

project = "Poraquê"
copyright = "Leandro Seixas Rocha, 2026"
author = "Leandro Seixas Rocha"
release = "26.9.1"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",      # Generate docs from docstrings
    "sphinx.ext.napoleon",     # NumPy/Google-style docstring support
    "sphinx.ext.viewcode",     # Add links to highlighted source code
    "sphinx.ext.mathjax",      # Render LaTeX math
    "myst_parser",             # Markdown (.md) source support
    "sphinx_rtd_theme",        # Read the Docs theme (available alternative)
]

templates_path = ["_templates"]
exclude_patterns = []

# Enable LaTeX math (``$...$`` and ``$$...$$``) in Markdown sources.
myst_enable_extensions = ["dollarmath", "amsmath"]

autodoc_member_order = "bysource"
autodoc_mock_imports = []
napoleon_numpy_docstring = True
napoleon_google_docstring = True

# -- Options for HTML output -------------------------------------------------

html_title = "Poraquê"
html_theme = "furo"  # alternatives: 'sphinx_rtd_theme', 'shibuya'
html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "logo_light.png",
    "dark_logo": "logo_dark.png",
}

html_show_sphinx = False
html_show_copyright = True
html_show_sourcelink = False
