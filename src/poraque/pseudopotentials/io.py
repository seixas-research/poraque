# -*- coding: utf-8 -*-
# file: io.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Reading pseudopotentials from disk.

Poraquê ships a deliberately small, human-readable local-pseudopotential format
so that the calculator can read a "standard" file rather than only its built-in
analytic prescriptions. Two on-disk encodings are accepted:

* **JSON** (``.json``) — a single object, e.g.::

      {"element": "Si", "z_valence": 4, "form": "soft_coulomb", "rc": 0.8}

* **key/value text** (``.psp`` or anything else) — one ``key: value`` per line,
  ``#`` comments allowed::

      # Silicon local pseudopotential
      element: Si
      z_valence: 4
      form: soft_coulomb
      rc: 0.8

Recognized ``form`` values map onto the analytic potentials in
:mod:`poraque.pseudopotentials.local` (``soft_coulomb`` -> :class:`SoftCoulombPP`,
``gaussian`` -> :class:`GaussianCorePP`).
"""

import json
import os

from .local import GaussianCorePP, SoftCoulombPP

_FORMS = {
    "soft_coulomb": SoftCoulombPP,
    "soft": SoftCoulombPP,
    "gaussian": GaussianCorePP,
    "gaussian_core": GaussianCorePP,
}


def _build_from_spec(spec):
    """Construct a pseudopotential from a parsed ``dict`` specification."""
    symbol = spec["element"]
    z_valence = float(spec["z_valence"])
    form = str(spec.get("form", "soft_coulomb")).lower()
    if form not in _FORMS:
        raise ValueError(
            f"Unknown pseudopotential form {form!r}; "
            f"expected one of {sorted(_FORMS)}."
        )
    rc = float(spec.get("rc", 0.8))
    return _FORMS[form](symbol, z_valence, rc=rc)


def _parse_text(text):
    """Parse the ``key: value`` text format into a ``dict``."""
    spec = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Malformed pseudopotential line: {raw!r}")
        key, value = line.split(":", 1)
        spec[key.strip().lower()] = value.strip()
    return spec


def read_pseudopotential(path):
    """
    Read a local pseudopotential from ``path``.

    The encoding is chosen from the file extension (``.upf`` -> UPF v2 reader,
    ``.json`` -> JSON, anything else -> key/value text).

    Returns
    -------
    LocalPseudopotential
        The reconstructed pseudopotential.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".upf":
        # UPF files (PseudoDojo / Quantum ESPRESSO) have their own binary-ish
        # tabulated reader.
        from .upf import read_upf
        return read_upf(path)
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if ext == ".json":
        spec = json.loads(text)
    else:
        spec = _parse_text(text)
    return _build_from_spec(spec)
