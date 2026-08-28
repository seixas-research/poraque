# -*- coding: utf-8 -*-
# file: validation.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
A physical gate on the kinetic energy density, applied where τ enters the cache.

Why this module exists
----------------------
Every ``TAUCAR`` in the platinum dataset was wrong, and nothing noticed. The files
parsed, the grids matched, the arrays were finite and positive, the training
loop converged and reported a small relative :math:`L^2`. The only symptom was
that integrated kinetic energies came out absurdly large — a number nobody
looks at while a loss is going down.

The lesson is not "check the data more carefully". It is that a scalar field on
a grid carries no unit, no convention and no provenance, so **a wrong τ is
indistinguishable from a right one by inspection**. What distinguishes them is
physics, and there are two statements about τ strong enough to be worth
enforcing and cheap enough to enforce on every sample:

**Scale.** :math:`\int\tau\,d^3r` must be within an order of magnitude of the
Thomas-Fermi estimate :math:`C_{\rm TF}\int\rho^{5/3}d^3r` built from the
*paired* density. This is not a tight bound and is not meant to be: τ_TF is the
uniform-gas limit and a real system departs from it by tens of percent, not by
factors. A unit error, a missing or spurious volume factor, or a
:math:`\rm Ha \leftrightarrow eV` confusion all move the integral by orders of
magnitude and are all caught here. The 1000× bug fails this check by a factor
of several hundred.

**Positivity, locally.** :math:`\tau(\mathbf r) \ge \tau_{\rm vW}(\mathbf r) =
|\nabla\rho|^2/(8\rho)` pointwise. Unlike the scale check this one is a
*theorem* — the von Weizsäcker form is the exact τ of a one-orbital system and
a rigorous lower bound in general — so a violation is not a warning sign, it is
a proof that the two fields are not a (ρ, τ) pair from the same calculation.

Neither check can confirm that a τ is *correct*. Both can prove that one is
wrong, which is the asymmetry worth building on.

Provenance
----------
The third check is not physics. The deleted files asked for τ with

.. code-block:: none

    TAUCAR = .TRUE.

which is not a VASP tag at all — the documented one is ``LTAU``, whose default
is ``.FALSE.`` outside meta-GGA — from a build of ``vasp.6.2.0`` that must have
been locally patched to honour it. There is no way to know what convention that
patch used, and no test on the numbers alone would have said so.

So τ is refused unless the run states who *asked* for it: the tag that
requested τ, set to ``.TRUE.``, and a hash of the complete input file. Both
come from the ``INCAR``, which is an input to the calculation and is therefore
present wherever the calculation is. This is the check that would have stopped
the bad data at the door — the deleted runs' ``INCAR``\ s say
``TAUCAR = .TRUE.`` and do not say ``LTAU``, and no amount of looking at the
numbers says that — and it is the only one of the three that does not depend on
the values being wrong in a detectable way.

The **code version is recorded when the run says it and required only on
request** (``require_code_version``, off by default). It is read from ``OUTCAR``
or ``vasprun.xml``, which are *outputs*: large, routinely dropped when a run is
archived, and not something a dataset can be expected to carry in order to be
trainable. What the version was ever guarding against is a build old enough to
predate ``LTAU``, and such a build does not honour ``LTAU`` — it ignores it and
writes no τ at all, so the tag check already excludes it. When a version *is*
recorded it is still compared against :data:`REQUIRED_VASP_VERSION` and a run
older than that is still refused, because that check costs nothing and has
teeth; when none is recorded the gate notes it in the verdict's ``warnings``
and proceeds on the physics.

The provenance record travels with the data. It is written to each material's
cache directory as ``tau_provenance.json``, so a cache rebuilt from a cache
still knows where its τ came from, three copies downstream.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field as dataclass_field

import numpy as np

#: Written into every cached material that carries a τ, and read back when that
#: cache is itself used as a source. This is what makes the chain of custody
#: survive a re-cache: a prepared cache has no ``INCAR`` to re-read.
PROVENANCE_FILENAME = "tau_provenance.json"

#: Roll-up of every material's gate record, at the cache root. The manifest the
#: instruction asks for: one place to answer "what τ is in here, and did it
#: pass?" without reading a single field.
MANIFEST_FILENAME = "tau_validation.json"

#: The VASP tag that legitimately requests a kinetic energy density, and the
#: first release whose behaviour this project has verified against. Both are
#: **user-specified** for this project; VASP itself has
#: written ``TAUCAR`` for meta-GGA runs for much longer.
REQUIRED_TAU_TAG = "LTAU"
REQUIRED_VASP_VERSION = "6.6.1"


class TauValidationError(ValueError):
    """
    Raised when a kinetic energy density fails the ingestion gate.

    Carries the whole record rather than only a message, so a caller that wants
    to log the failure or write it to a manifest does not have to re-run the
    checks to find out what the numbers were.
    """

    def __init__(self, message, record=None):
        super().__init__(message)
        self.record = record or {}


@dataclass
class TauValidationConfig:
    r"""
    What the gate checks and how strictly.

    Every threshold is deliberately loose. The gate is meant to catch errors of
    *kind* — wrong units, wrong convention, mismatched pair, unknown code — not
    to adjudicate physics. A bound tight enough to reject a merely unusual
    material would be a bound that gets switched off, and a gate that is off
    catches nothing.

    Attributes
    ----------
    enabled : bool
        Master switch. Turning it off is recorded in the manifest, so a dataset
        built without the gate cannot later be mistaken for one that passed it.
    tf_ratio_range : tuple of float
        Accepted range of :math:`\int\tau / C_{\rm TF}\!\int\rho^{5/3}`. The
        default ``(0.2, 5.0)`` spans a factor of 25 and still rejects the 1000×
        bug by more than an order of magnitude on either side.
    check_von_weizsacker : bool
        Enforce the pointwise lower bound.
    density_threshold : float
        Points with :math:`\rho < {\rm threshold}\times\max\rho` are exempt from
        the bound check. :math:`\tau_{\rm vW}` divides by ρ, so in the vacuum
        tail it is a ratio of two quantities that are both numerical noise, and
        testing it there measures the FFT rather than the physics.
    vw_relative_tolerance, vw_absolute_tolerance : float
        Slack on the bound, relative to :math:`\tau_{\rm vW}` and absolute
        (eV/Å³). Band-limited fields ring; the bound is exact for the continuum
        fields, not for their truncations.
    max_violation_fraction : float
        Fraction of *significant* points allowed to violate the bound before the
        sample is refused. Not zero, for the reason above: a handful of ringing
        points near a core is an artefact of the grid, while a percent of the
        cell is a different field.
    require_provenance : bool
        Refuse τ that cannot say which tag requested it, from which ``INCAR``.
    require_code_version : bool
        Also refuse τ whose run does not state a code version. **Off by
        default**: the version lives in ``OUTCAR``/``vasprun.xml``, which are
        outputs a dataset need not ship, and requiring them would make a
        complete and physically valid set untrainable for want of a file that
        says nothing about the field. A version that *is* present is checked
        against :attr:`minimum_version` either way.
    required_tag : str
        The input tag that must be set. ``LTAU`` for VASP.
    minimum_version : str or None
        Lowest acceptable code version, compared component-wise, applied to
        whatever version the run *did* record. ``None`` accepts any.
    """

    enabled: bool = True
    tf_ratio_range: tuple = (0.2, 5.0)
    check_von_weizsacker: bool = True
    density_threshold: float = 1e-3
    vw_relative_tolerance: float = 0.05
    vw_absolute_tolerance: float = 1e-6
    max_violation_fraction: float = 0.01
    require_provenance: bool = True
    require_code_version: bool = False
    required_tag: str = REQUIRED_TAU_TAG
    minimum_version: str = REQUIRED_VASP_VERSION

    @classmethod
    def from_mapping(cls, mapping):
        """
        Build from a plain dict, e.g. a config's ``data.tau_validation`` block.

        Unknown keys **raise** rather than being ignored, matching how the rest
        of the configuration behaves: a misspelled threshold that silently does
        nothing is worse here than anywhere else, because what it silently does
        nothing to is a safety check.

        Raises
        ------
        ValueError
            On a key that is not a setting.
        """
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping

        values = dict(mapping)
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                f"Unknown tau_validation key(s): {unknown}. "
                f"Valid keys: {sorted(known)}.")
        if values.get("tf_ratio_range") is not None:
            values["tf_ratio_range"] = tuple(float(v)
                                             for v in values["tf_ratio_range"])
        return cls(**values)

    def as_dict(self):
        """JSON-serialisable settings, for the manifest."""
        return {name: (list(value) if isinstance(value, tuple) else value)
                for name, value in self.__dict__.items()}


# ---------------------------------------------------------------------- #
# Provenance
# ---------------------------------------------------------------------- #
#: ``vasp.6.2.0 18Jan21 (build ...)`` — the first line of every OUTCAR.
_VERSION_PATTERN = re.compile(r"vasp\.(\d+(?:\.\d+)*)", re.IGNORECASE)

#: Fortran logicals, as an INCAR spells them.
_TRUE = {".TRUE.", "T", ".T.", "TRUE"}
_FALSE = {".FALSE.", "F", ".F.", "FALSE"}


def _logical(value):
    """Parse a Fortran logical; ``None`` when it is neither."""
    if value is None:
        return None
    token = str(value).strip().split()[0].upper() if str(value).strip() else ""
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    return None


def file_hash(path):
    """SHA-256 of a file, or ``None`` when it is not there."""
    if not path or not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version(directory):
    """
    The DFT code's version string, read from the run's own output.

    Parameters
    ----------
    directory : str
        A calculation directory.

    Returns
    -------
    str or None
        E.g. ``"6.6.1"``. ``None`` when no output file records it — which is
        the normal case for an archived run stripped of everything but its
        densities, and is exactly what :func:`validate_tau` refuses.

    Notes
    -----
    Only the first line of ``OUTCAR`` is read; the version is the first token
    on it. ``vasprun.xml`` is consulted as a fallback because a run archived
    for size often keeps the XML and drops the 300 MB ``OUTCAR``.
    """
    outcar = os.path.join(directory, "OUTCAR")
    if os.path.exists(outcar):
        try:
            with open(outcar, "r", errors="replace") as handle:
                match = _VERSION_PATTERN.search(handle.readline())
        except OSError:
            match = None
        if match:
            return match.group(1)

    xml = os.path.join(directory, "vasprun.xml")
    if os.path.exists(xml):
        try:
            with open(xml, "r", errors="replace") as handle:
                for _ in range(64):
                    line = handle.readline()
                    if not line:
                        break
                    if 'name="version"' in line:
                        text = re.sub(r"<[^>]+>", " ", line).strip()
                        if text:
                            return text.split()[0]
        except OSError:
            pass
    return None


def version_at_least(version, minimum):
    """
    Component-wise version comparison on dotted numeric strings.

    Returns ``None`` when either side cannot be parsed, so an unrecognised
    version string is reported as *unknown* rather than silently as *too old*.
    """
    def parts(text):
        if not text:
            return None
        found = re.findall(r"\d+", str(text))
        return [int(value) for value in found] if found else None

    left, right = parts(version), parts(minimum)
    if left is None or right is None:
        return None
    width = max(len(left), len(right))
    left += [0] * (width - len(left))
    right += [0] * (width - len(right))
    return left >= right


def read_tau_provenance(directory, tag=REQUIRED_TAU_TAG):
    r"""
    Everything that says who computed this τ, and how.

    Two layouts answer the question, and they are tried in that order:

    1. A **prepared cache** carries :data:`PROVENANCE_FILENAME`, written when
       the τ was first ingested. This is what lets a cache be re-cached — at
       another resolution, into another mixture — without the provenance
       evaporating at the first hop.
    2. A **calculation directory** is read directly: the code version off its
       own output, the τ tag and the hash off its ``INCAR``.

    Parameters
    ----------
    directory : str
        Cache material directory or calculation directory.
    tag : str, optional
        Input tag that requests τ. ``LTAU`` for VASP.

    Returns
    -------
    dict
        ``code``, ``version``, ``tau_tag``, ``tau_tag_value``, ``incar_sha256``,
        ``incar_path`` and ``source``. Values are ``None`` where the directory
        does not say, which is a finding rather than an error — it is
        :func:`validate_tau` that decides whether a gap is fatal.
    """
    stored = os.path.join(directory, PROVENANCE_FILENAME)
    if os.path.exists(stored):
        try:
            with open(stored) as handle:
                record = json.load(handle)
            record["source"] = "cache"
            return record
        except (OSError, ValueError):
            pass                        # fall through and re-derive it

    incar_path = os.path.join(directory, "INCAR")
    tag_value, incar_sha = None, None
    legacy_tags = {}

    if os.path.exists(incar_path):
        from ..fields.vasp.incar import Incar

        incar = Incar.from_file(incar_path)
        tag_value = incar.get(tag.upper())
        incar_sha = file_hash(incar_path)
        # Recorded, not honoured. `TAUCAR = .TRUE.` is not a VASP tag; a run
        # carrying it was produced by a patched build whose convention is
        # undocumented, and naming it in the failure message is far more useful
        # than "LTAU missing" on its own.
        for name in ("TAUCAR", "EXTCAR", "METAGGA", "LCHARG"):
            if name in incar:
                legacy_tags[name] = incar[name]

    return {
        "code": "vasp",
        "version": code_version(directory),
        "tau_tag": tag.upper(),
        "tau_tag_value": tag_value,
        "tau_tag_set": _logical(tag_value),
        "incar_sha256": incar_sha,
        "incar_path": incar_path if os.path.exists(incar_path) else None,
        "other_tags": legacy_tags,
        "source": "calculation",
    }


def write_tau_provenance(directory, provenance):
    """Persist a provenance record beside a cached τ."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, PROVENANCE_FILENAME)
    with open(path, "w") as handle:
        json.dump(provenance, handle, indent=1, sort_keys=True)
    return path


# ---------------------------------------------------------------------- #
# The physical checks
# ---------------------------------------------------------------------- #
def thomas_fermi_scale(tau, density, grid):
    r"""
    Compare :math:`\int\tau` with the Thomas-Fermi estimate from ρ.

    Parameters
    ----------
    tau : array_like
        Kinetic energy density in eV/Å³.
    density : array_like
        The **paired** electron density in e/Å³.
    grid : FieldGrid
        Their shared mesh.

    Returns
    -------
    dict
        ``tau_integral``, ``tf_integral``, ``vw_integral``, ``ratio`` and the
        electron count. ``ratio`` is ``None`` when the TF integral vanishes,
        which happens only for an all-zero density.
    """
    from ..fields.density import thomas_fermi_tau, von_weizsacker_tau

    tau = np.asarray(tau, dtype=float)
    density = np.asarray(density, dtype=float)
    if tau.shape != density.shape:
        raise ValueError(
            f"tau has shape {tau.shape} and the density {density.shape}; they "
            f"are not a pair on one grid.")

    tau_integral = float(grid.integrate(tau))
    tf_integral = float(grid.integrate(thomas_fermi_tau(density)))
    vw_integral = float(grid.integrate(von_weizsacker_tau(density, grid)))
    ratio = (tau_integral / tf_integral) if abs(tf_integral) > 1e-30 else None

    return {"tau_integral": tau_integral, "tf_integral": tf_integral,
            "vw_integral": vw_integral, "ratio": ratio,
            "electrons": float(grid.integrate(density))}


def von_weizsacker_violations(tau, density, grid, density_threshold=1e-3,
                              relative_tolerance=0.05,
                              absolute_tolerance=1e-6):
    r"""
    Where, and how badly, :math:`\tau \ge \tau_{\rm vW}` fails.

    Parameters
    ----------
    tau, density : array_like
        The pair, in eV/Å³ and e/Å³.
    grid : FieldGrid
        Shared mesh; supplies the spectral gradient.
    density_threshold : float, optional
        Fraction of :math:`\max\rho` below which a point is exempt.
    relative_tolerance, absolute_tolerance : float, optional
        Slack, relative to :math:`\tau_{\rm vW}` and absolute.

    Returns
    -------
    dict
        Counts, the violating fraction of significant points, and the worst
        violation with the ratio :math:`\tau/\tau_{\rm vW}` at that point.
    """
    from ..fields.density import von_weizsacker_tau

    tau = np.asarray(tau, dtype=float)
    density = np.asarray(density, dtype=float)
    vw = von_weizsacker_tau(density, grid)

    peak = float(np.max(density)) if density.size else 0.0
    significant = density > density_threshold * peak
    n_significant = int(np.count_nonzero(significant))

    allowed = vw * (1.0 - relative_tolerance) - absolute_tolerance
    violating = significant & (tau < allowed)
    n_violating = int(np.count_nonzero(violating))

    worst_ratio, worst_deficit = None, 0.0
    if n_violating:
        deficit = (vw - tau)[violating]
        index = int(np.argmax(deficit))
        worst_deficit = float(deficit[index])
        vw_here = float(vw[violating][index])
        if vw_here > 0:
            worst_ratio = float(tau[violating][index] / vw_here)

    return {
        "significant_points": n_significant,
        "total_points": int(tau.size),
        "violations": n_violating,
        "violation_fraction": (n_violating / n_significant
                               if n_significant else 0.0),
        "worst_deficit": worst_deficit,
        "worst_ratio": worst_ratio,
        "density_threshold": float(density_threshold),
    }


# ---------------------------------------------------------------------- #
# The gate
# ---------------------------------------------------------------------- #
def validate_tau(tau, density, grid, provenance=None, config=None,
                 identifier=""):
    r"""
    Run every check, and refuse the sample if any of them fails.

    Parameters
    ----------
    tau, density : array_like or ScalarField
        The pair to check. ``ScalarField`` instances are accepted and their
        ``.data`` used, so a caller does not have to unwrap them.
    grid : FieldGrid
        Their shared mesh.
    provenance : dict, optional
        From :func:`read_tau_provenance`. ``None`` is treated as "nothing is
        known", which fails the provenance check when it is enabled.
    config : TauValidationConfig or dict, optional
        Thresholds; the defaults if omitted.
    identifier : str, optional
        Material name, for the failure message.

    Returns
    -------
    dict
        The gate record: ``passed``, the ``failures`` list, the ``warnings``
        list, the ``scale`` and ``bound`` measurements, the ``provenance``, and
        the ``settings`` used. Written to the manifest whether or not it
        passed, so a sample that passed *with* a warning stays visible.

    Raises
    ------
    TauValidationError
        On any failure, with every check's numbers in
        :attr:`~TauValidationError.record`.
    """
    config = TauValidationConfig.from_mapping(config)
    tau_values = getattr(tau, "data", tau)
    density_values = getattr(density, "data", density)

    # A spin-polarised density arrives as (rho, m); only the total is a density
    # in the sense both bounds are written for.
    tau_values = np.asarray(tau_values, dtype=float)
    density_values = np.asarray(density_values, dtype=float)
    if density_values.ndim == 4:
        density_values = density_values[0]
    if tau_values.ndim == 4:
        tau_values = tau_values[0]

    record = {
        "material": identifier,
        "enabled": bool(config.enabled),
        "settings": config.as_dict(),
        "provenance": provenance,
        "failures": [],
        # Recorded, not raised. A sample that passes with a gap in its record
        # has to stay distinguishable afterwards from one that had none.
        "warnings": [],
        "passed": True,
    }

    if not config.enabled:
        # Recorded, not skipped silently. A cache built with the gate off must
        # be distinguishable afterwards from one that passed it.
        record["passed"] = None
        return record

    scale = thomas_fermi_scale(tau_values, density_values, grid)
    record["scale"] = scale

    low, high = config.tf_ratio_range
    ratio = scale["ratio"]
    if ratio is None:
        record["failures"].append(
            "the paired density integrates to zero, so no Thomas-Fermi scale "
            "can be formed; tau cannot be checked against it")
    elif not (low <= ratio <= high):
        record["failures"].append(
            f"scale check: integral(tau) = {scale['tau_integral']:.6g} eV is "
            f"{ratio:.4g}x the Thomas-Fermi estimate "
            f"C_TF*integral(rho^5/3) = {scale['tf_integral']:.6g} eV, outside "
            f"the accepted [{low:g}, {high:g}]. A ratio this far off is a unit "
            f"or convention error, not physics: check whether the file holds "
            f"tau or tau*Omega, and whether it is in eV or Hartree.")

    if config.check_von_weizsacker:
        bound = von_weizsacker_violations(
            tau_values, density_values, grid,
            density_threshold=config.density_threshold,
            relative_tolerance=config.vw_relative_tolerance,
            absolute_tolerance=config.vw_absolute_tolerance)
        record["bound"] = bound
        if bound["violation_fraction"] > config.max_violation_fraction:
            ratio_text = ("n/a" if bound["worst_ratio"] is None
                          else f"{bound['worst_ratio']:.4g}")
            record["failures"].append(
                f"von Weizsaecker bound: tau < tau_vW at "
                f"{bound['violations']} of {bound['significant_points']} "
                f"points carrying significant density "
                f"({100 * bound['violation_fraction']:.2f}%, allowed "
                f"{100 * config.max_violation_fraction:.2f}%); worst "
                f"tau/tau_vW = {ratio_text}. tau >= |grad rho|^2/(8 rho) is "
                f"exact for any density, so this proves tau and rho are not "
                f"the pair they claim to be.")

    if config.require_provenance:
        failures, warnings = _provenance_failures(provenance, config)
        record["failures"].extend(failures)
        record["warnings"].extend(warnings)

    record["passed"] = not record["failures"]
    if not record["passed"]:
        head = (f"{identifier}: " if identifier else "")
        raise TauValidationError(
            head + "the kinetic energy density failed the ingestion gate.\n  - "
            + "\n  - ".join(record["failures"])
            + "\n\nThis gate exists because an entire dataset of invalid tau "
              "once passed unnoticed. Fix the data, or "
              "relax data.tau_validation in the config deliberately and "
              "record why.",
            record=record)
    return record


def _provenance_failures(provenance, config):
    """
    The provenance half of the gate.

    Returns
    -------
    (list of str, list of str)
        Failures, which refuse the sample, and warnings, which are recorded in
        the verdict and let it through. A missing code version is a warning by
        default -- see :attr:`TauValidationConfig.require_code_version`.
    """
    tag = config.required_tag.upper()
    if not provenance:
        return ([
            f"provenance: nothing recorded. tau is only accepted from a run "
            f"whose INCAR sets {tag}; a bare field on a grid carries no unit "
            f"and no convention, and that is precisely how the invalid data "
            f"got in."], [])

    failures, warnings = [], []
    version = provenance.get("version")
    if not version:
        message = (
            "provenance: no code version recorded. The version is read from "
            "OUTCAR or vasprun.xml, which are outputs a dataset need not "
            "carry; LTAU and the INCAR hash come from the INCAR and are "
            "checked regardless.")
        if config.require_code_version:
            failures.append(message + " require_code_version is on.")
        else:
            warnings.append(message)
    elif config.minimum_version:
        newer = version_at_least(version, config.minimum_version)
        if newer is False:
            failures.append(
                f"provenance: written by version {version}, below the "
                f"required {config.minimum_version}. This project's tau data "
                f"must come from VASP {config.minimum_version} or newer with "
                f"{tag} = .TRUE. (user-specified requirement; see "
                f"the post-mortem for what the older, patched build produced).")

    if provenance.get("tau_tag_set") is not True:
        other = provenance.get("other_tags") or {}
        extra = ""
        if "TAUCAR" in other:
            extra = (f" The INCAR instead sets TAUCAR = {other['TAUCAR']}, "
                     f"which is not a VASP tag: a stock build ignores it and "
                     f"writes no tau at all, so a TAUCAR produced beside it "
                     f"came from a patched build with an undocumented "
                     f"convention.")
        failures.append(
            f"provenance: {tag} is not set to .TRUE. in the INCAR (found "
            f"{provenance.get('tau_tag_value')!r}).{extra}")

    if not provenance.get("incar_sha256"):
        failures.append(
            "provenance: no INCAR hash recorded, so the settings that produced "
            "this tau cannot be pinned to the file they came from.")
    return failures, warnings


# ---------------------------------------------------------------------- #
# Manifest
# ---------------------------------------------------------------------- #
@dataclass
class TauValidationManifest:
    """
    The gate's record for a whole cache build.

    Written to :data:`MANIFEST_FILENAME` at the cache root, and merged with any
    earlier build's so a resumed build does not lose the materials it skipped.
    """

    entries: dict = dataclass_field(default_factory=dict)

    @classmethod
    def load(cls, cache):
        path = os.path.join(cache, MANIFEST_FILENAME)
        if not os.path.exists(path):
            return cls()
        try:
            with open(path) as handle:
                return cls(json.load(handle))
        except (OSError, ValueError):
            return cls()

    def add(self, identifier, record):
        self.entries[identifier] = record

    def write(self, cache):
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, MANIFEST_FILENAME)
        with open(path, "w") as handle:
            json.dump(self.entries, handle, indent=1, sort_keys=True)
        return path

    def summary(self):
        """``(passed, failed, ungated)`` counts."""
        passed = sum(1 for r in self.entries.values() if r.get("passed") is True)
        failed = sum(1 for r in self.entries.values() if r.get("passed") is False)
        ungated = sum(1 for r in self.entries.values() if r.get("passed") is None)
        return passed, failed, ungated

    def warned(self):
        """
        Materials that passed with something noted against them.

        A warning that reaches only the JSON is a gap nobody sees, which is the
        condition this whole module exists to prevent. The build log names the
        count and the first note so the omission is at least legible at the
        moment it is made.

        Returns
        -------
        (int, str or None)
            How many materials carry a warning, and one representative note.
        """
        notes = [note for record in self.entries.values()
                 for note in (record.get("warnings") or [])]
        return len(set(
            identifier for identifier, record in self.entries.items()
            if record.get("warnings"))), (notes[0] if notes else None)
