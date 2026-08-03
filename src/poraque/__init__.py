# -*- coding: utf-8 -*-
# file: __init__.py

"""
Poraquê — electronic structure and machine learning on 3D scalar fields.

Top-level names are resolved **lazily** (PEP 562): importing ``poraque`` itself
is cheap and does not pull in the SCF engine, so light-weight subpackages such
as :mod:`poraque.fields` and :mod:`poraque.ml` can be used on their own while
the legacy solver stack is being restructured.
"""

from .version import __version__

_LAZY = {
    "KSDFTCalculator": "poraque.calculator",
    "OFDFTCalculator": "poraque.calculator",
    "FDEEngine": "poraque.fde",
    "Subsystem": "poraque.fde",
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name):
    """Import legacy top-level names on first access."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
