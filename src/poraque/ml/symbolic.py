# -*- coding: utf-8 -*-
# file: symbolic.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Symbolic distillation: read a closed-form functional out of a trained operator.

A Fourier Neural Operator reproduces :math:`\rho \mapsto \tau` to a few percent
and explains nothing. Symbolic regression searches short algebraic expressions
for one that reproduces the same mapping, trading accuracy for a formula that
can be read, published, and checked against known physics.

.. code-block:: text

    trained FNO  --predict-->  tau(r) on a grid
                                    |
    rho(r) --> features: rho, p = |grad rho| / (2 k_F rho)
                              q = lap rho  / (4 k_F^2 rho)
                                    |
                            symbolic regression (PySR)
                                    |
                             tau = f(rho, p, q)

What this can and cannot find
-----------------------------
The features are evaluated **at a point**, so the hypothesis space is exactly
the semi-local functionals. The non-local part of what the operator learned
cannot be expressed in it *at all*, and shows up as irreducible residual.

That makes a poor fit informative rather than disappointing: it puts a number
on how much of the learned map is not semi-local. A near-perfect fit would say
the operator found nothing a GGA-level functional could not have.

Two known answers sit inside the search space and are the calibration for any
result:

.. math::

    \tau_{\rm TF} = C_{\rm TF}\,\rho^{5/3},
    \qquad
    \tau_{\rm vW} = \frac{|\nabla\rho|^2}{8\rho}.

In the reduced variables these read :math:`F = 1` and :math:`F = 5p^2/3`. If a
run cannot recover them on data generated from them, the *search* is
misconfigured, not the physics — :func:`build_features` is deliberately exact
for both so that check is available.

Vacuum
------
:math:`p` and :math:`q` divide by :math:`\rho`, which decays exponentially into
vacuum. Denominators are clamped at :data:`DEFAULT_EPSILON` and voxels at or
below it are dropped: in vacuum the reduced variables are ratios of two
vanishing numbers, so they carry noise at a plausible magnitude rather than
information, and a fit corrupted that way looks healthy.

Units
-----
Features and targets are built in **atomic units** (:math:`\rho` in
:math:`e/a_0^3`, :math:`\tau` in :math:`E_h/a_0^3`), not the eV/Å³ the files
carry. In atomic units the coefficients the search must discover are order
unity — :math:`C_{\rm TF} = 2.871`, the von Weizsäcker :math:`1/8` — whereas in
eV/Å³ they are arbitrary decimals that a genetic search wastes its budget
approximating.

Engine
------
`PySR <https://github.com/MilesCranmer/PySR>`_ is the backend. It is an
optional dependency (``pip install pysr``; it fetches a Julia toolchain on
first use), and the engine is injected rather than imported at module scope, so
everything except the search itself is usable and testable without it.
"""

import tempfile
from dataclasses import dataclass, field

import numpy as np

from ..fields.constants import (
    BOHR_TO_ANGSTROM,
    C_TF,
    HARTREE_TO_EV,
)

#: Operator alphabets a run defaults to when the config leaves them unset.
DEFAULT_UNARY = ("exp", "log", "sqrt", "abs")
DEFAULT_BINARY = ("+", "-", "*", "/", "^")

#: Feature schemes understood by :func:`build_features`.
FEATURE_SCHEMES = ("gga", "enhancement", "raw")

#: Density below which a voxel is vacuum: dropped, and used to clamp every
#: denominator. In atomic units (:math:`e/a_0^3`).
DEFAULT_EPSILON = 1e-8

#: eV/Å³ per Hartree/Bohr³.
_HA_PER_BOHR3_TO_EV_PER_ANG3 = HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3


@dataclass
class FeatureTable:
    """
    Design matrix for the search.

    Attributes
    ----------
    features : numpy.ndarray
        ``(n_points, n_features)``.
    target : numpy.ndarray
        ``(n_points,)``.
    feature_names : list of str
        Column names, used verbatim as the engine's variable names and so
        carried into the printed expression.
    target_name : str
        Name of the fitted quantity.
    scheme : str
        Which of :data:`FEATURE_SCHEMES` produced it.
    units : str
        Human-readable unit note for the report.
    source : str
        Where the fitted values came from — ``"model"`` or ``"reference"``.
        Carried on the table rather than read back off the config, so a table
        built by hand cannot be reported as something it is not.
    """

    features: np.ndarray
    target: np.ndarray
    feature_names: list
    target_name: str
    scheme: str
    units: str
    source: str = ""

    def __len__(self):
        return int(self.features.shape[0])


@dataclass
class SymbolicResult:
    """
    Outcome of one search.

    Attributes
    ----------
    expression : str
        Best expression, in the engine's own notation.
    latex : str
        The same expression as LaTeX, for the report. Falls back to the plain
        form when it cannot be parsed.
    complexity : int
        Node count of the chosen expression.
    loss : float
        The engine's reported loss for it.
    r2 : float
        Coefficient of determination against the fitted target, computed here
        rather than taken from the engine so it means the same thing across
        backends.
    relative_l2 : float
        The error measure the rest of the package quotes, for comparability
        with the operator's own score.
    pareto : list of dict
        The accuracy/complexity front: every candidate the engine kept, as
        ``{"complexity", "loss", "expression"}``. The front is the result —
        a single expression hides the trade that produced it.
    """

    expression: str
    latex: str
    complexity: int
    loss: float
    r2: float
    relative_l2: float
    pareto: list = field(default_factory=list)
    feature_names: list = field(default_factory=list)
    target_name: str = ""
    scheme: str = ""
    units: str = ""
    target: str = ""
    n_samples: int = 0
    engine: str = ""

    def summary(self):
        """Multi-line text block, for the log and the terminal."""
        lines = [
            f"  expression   : {self.target_name} = {self.expression}",
            f"  variables    : {', '.join(self.feature_names)}  [{self.units}]",
            f"  complexity   : {self.complexity} nodes",
            f"  fit          : R2 {self.r2:.4f}   relative L2 {self.relative_l2:.4f}",
            f"  fitted on    : {self.n_samples} voxels of the "
            f"{'operator prediction' if self.target == 'model' else 'DFT reference'}",
        ]
        if self.pareto:
            lines.append("  accuracy/complexity front:")
            lines.append(f"      {'nodes':>5s}  {'loss':>12s}  expression")
            for entry in self.pareto:
                lines.append(f"      {entry['complexity']:5d}  "
                             f"{entry['loss']:12.5g}  {entry['expression']}")
        return "\n".join(lines)


# ===================================================================== #
# Features
# ===================================================================== #
def spectral_laplacian(field, grid, length_unit="angstrom"):
    r"""
    Laplacian of a periodic field via FFT, :math:`\nabla^2 f \to -|G|^2 \hat f`.

    One transform pair rather than the six a gradient-of-gradient would cost,
    and exact for band-limited fields for the same reason
    :func:`~poraque.fields.density.spectral_gradient` is.

    Parameters
    ----------
    field : array_like
        Real array of shape ``grid.shape``.
    grid : FieldGrid
        Mesh supplying the reciprocal vectors (Å⁻¹).
    length_unit : {"angstrom", "bohr"}, optional
        Unit of the differentiation variable in the result.

    Returns
    -------
    numpy.ndarray
    """
    scale = 1.0 if length_unit == "angstrom" else BOHR_TO_ANGSTROM
    g_squared = sum((component * scale) ** 2
                    for component in grid.get_g_vectors())
    transformed = np.fft.fftn(np.asarray(field, dtype=float))
    return np.real(np.fft.ifftn(-g_squared * transformed))


def build_features(density, target, grid, scheme="gga",
                   epsilon=DEFAULT_EPSILON):
    r"""
    Assemble the design matrix from a density and a target field.

    The default variables are the ones a semi-local kinetic functional is
    actually written in — the density together with its **dimensionless**
    reduced derivatives,

    .. math::

        k_F = (3\pi^2\rho)^{1/3},
        \qquad
        p = \frac{|\nabla\rho|}{2 k_F \rho},
        \qquad
        q = \frac{\nabla^2\rho}{4 k_F^2 \rho}.

    :math:`p` and :math:`q` are the GGA and meta-GGA variables. They are
    dimensionless, and invariant under the coordinate scaling
    :math:`\rho_\lambda(\mathbf r) = \lambda^3\rho(\lambda\mathbf r)` that
    fixes :math:`T_s` — so a functional expressed in them holds at every
    density scale, rather than having to be rediscovered at each. Feeding the
    raw :math:`|\nabla\rho|` and :math:`\nabla^2\rho` instead makes every
    coefficient carry units, which a genetic search spends its budget
    approximating.

    Derivatives are spectral — :math:`\nabla \to i\mathbf{G}`,
    :math:`\nabla^2 \to -|\mathbf{G}|^2` — which is exact for the band-limited
    periodic fields a plane-wave grid carries, unlike any finite-difference
    stencil.

    Parameters
    ----------
    density : ChargeDensity or numpy.ndarray
        :math:`\rho` in e/Å³, as stored.
    target : KineticEnergyDensity or numpy.ndarray
        :math:`\tau` in eV/Å³, as stored — either the operator's prediction or
        the DFT reference.
    grid : FieldGrid
        Supplies the reciprocal vectors for the spectral derivatives.
    scheme : {"gga", "enhancement", "raw"}, optional
        ``"gga"`` gives :math:`(\rho, p, q)` against :math:`\tau`.

        ``"enhancement"`` drops :math:`\rho` and fits the dimensionless
        enhancement factor :math:`F = \tau/\tau_{\rm TF}` on :math:`(p, q)`.
        This is the form the literature writes kinetic functionals in, so the
        answer is directly comparable: Thomas-Fermi is :math:`F = 1` and von
        Weizsäcker is :math:`F = 5p^2/3`. Everything the search must find is
        then order unity.

        ``"raw"`` gives :math:`(\rho, |\nabla\rho|, \nabla^2\rho)` against
        :math:`\tau`, all in atomic units — dimensional, and kept only for
        checking the reduced forms against something unprocessed.
    epsilon : float, optional
        Vacuum threshold in atomic units (:math:`e/a_0^3`). Used twice, and
        both uses are needed:

        1. Every denominator is clamped at it, so :math:`k_F`, :math:`p` and
           :math:`q` are finite even where the density underflows.
        2. Voxels with :math:`\rho \le \epsilon` are then **dropped**. Vacuum
           carries no information about the functional — :math:`p` and
           :math:`q` there are ratios of two numerically tiny numbers, so they
           are noise with a plausible magnitude, which is worse than a gap.

        The mask also removes the slightly negative voxels that band-limiting
        leaves around the core peaks (Gibbs ringing), where a fractional power
        of :math:`\rho` would otherwise be complex.

    Returns
    -------
    FeatureTable
    """
    if scheme not in FEATURE_SCHEMES:
        raise ValueError(f"Unknown feature scheme {scheme!r}; "
                         f"expected one of {list(FEATURE_SCHEMES)}.")

    from ..fields.density import spectral_gradient

    rho_ang = np.asarray(getattr(density, "data", density), dtype=float)
    tau_ang = np.asarray(getattr(target, "data", target), dtype=float)
    if rho_ang.shape != tau_ang.shape:
        raise ValueError("density and target must share a grid: "
                         f"{rho_ang.shape} against {tau_ang.shape}.")

    # -> atomic units, where the constants worth finding are order unity.
    rho = rho_ang * BOHR_TO_ANGSTROM ** 3
    tau = tau_ang / _HA_PER_BOHR3_TO_EV_PER_ANG3

    gradient = spectral_gradient(rho, grid, length_unit="bohr")
    grad_norm = np.sqrt(sum(component ** 2 for component in gradient))
    laplacian = spectral_laplacian(rho, grid, length_unit="bohr")

    epsilon = float(epsilon)
    # Clamp first, mask second. Clamping alone would keep vacuum voxels whose
    # p and q are noise; masking alone would still evaluate the ratios on the
    # full grid and raise or emit NaN on the way there.
    rho_safe = np.clip(rho, epsilon, None)
    k_f = (3.0 * np.pi ** 2 * rho_safe) ** (1.0 / 3.0)
    p = grad_norm / (2.0 * k_f * rho_safe)
    q = laplacian / (4.0 * k_f ** 2 * rho_safe)

    keep = rho > epsilon
    if not keep.any():
        raise ValueError(
            f"Every voxel is at or below the vacuum threshold "
            f"epsilon={epsilon:g} e/a0^3; nothing to fit.")

    if scheme == "gga":
        return FeatureTable(
            features=np.column_stack([rho[keep], p[keep], q[keep]]),
            target=tau[keep], feature_names=["rho", "p", "q"],
            target_name="tau", scheme=scheme,
            units="rho in e/a0^3, tau in Ha/a0^3; p and q dimensionless")

    if scheme == "raw":
        return FeatureTable(
            features=np.column_stack(
                [rho[keep], grad_norm[keep], laplacian[keep]]),
            target=tau[keep],
            feature_names=["rho", "grad_rho", "lap_rho"],
            target_name="tau", scheme=scheme,
            units="atomic units: rho in e/a0^3, tau in Ha/a0^3")

    enhancement = tau[keep] / (C_TF * rho_safe[keep] ** (5.0 / 3.0))
    return FeatureTable(
        features=np.column_stack([p[keep], q[keep]]), target=enhancement,
        feature_names=["p", "q"], target_name="F", scheme=scheme,
        units="dimensionless (F = tau / tau_TF)")


def sample_rows(table, n_samples, seed=0):
    """
    Take a random subset of the rows, without replacement.

    The search cost is linear in the row count and a single 32³ structure is
    already 32 768 voxels, so the whole dataset is never handed over whole.
    """
    total = len(table)
    if n_samples <= 0 or n_samples >= total:
        return table
    picked = np.random.default_rng(seed).choice(total, n_samples, replace=False)
    return FeatureTable(
        features=table.features[picked], target=table.target[picked],
        feature_names=list(table.feature_names),
        target_name=table.target_name, scheme=table.scheme, units=table.units,
        source=table.source)


def stack_tables(tables):
    """Concatenate per-structure tables into one design matrix."""
    tables = [table for table in tables if len(table)]
    if not tables:
        raise ValueError("No usable voxels in any structure.")
    first = tables[0]
    if any(table.feature_names != first.feature_names for table in tables):
        raise ValueError("Cannot stack tables built with different schemes.")
    return FeatureTable(
        features=np.concatenate([t.features for t in tables]),
        target=np.concatenate([t.target for t in tables]),
        feature_names=list(first.feature_names),
        target_name=first.target_name, scheme=first.scheme, units=first.units,
        source=first.source)


# ===================================================================== #
# Engine
# ===================================================================== #
def pysr_engine(features, target, feature_names, parameters):
    """
    Run PySR and return its accuracy/complexity front.

    The configured operator alphabet is passed through untouched: an operator
    the user asked for is one the search gets, and PySR is the authority on
    which names it accepts.

    Returns
    -------
    list of dict
        ``{"complexity", "loss", "expression"}``, best-first is not assumed.
    """
    try:
        from pysr import PySRRegressor
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "Symbolic distillation requires PySR: `pip install pysr`. "
            "PySR installs a Julia toolchain the first time it runs, which "
            "needs network access and a few minutes."
        ) from error

    # PySR writes its hall-of-fame files to `outputs/` under the *working
    # directory* by default, which drops build artefacts into whatever
    # repository the run was launched from. The whole front is captured in the
    # returned value, the JSON summary and the report, so the scratch copies
    # go to a temporary directory that is removed with it.
    with tempfile.TemporaryDirectory(prefix="poraque_pysr_") as workdir:
        model = PySRRegressor(
            niterations=parameters["iterations"],
            binary_operators=list(parameters["binary_operations"]),
            unary_operators=list(parameters["unary_operations"]),
            population_size=parameters["population_size"],
            populations=parameters["populations"],
            maxsize=parameters["max_size"],
            maxdepth=parameters["max_depth"],
            parsimony=parameters["parsimony"],
            random_state=parameters["seed"],
            output_directory=workdir,
            progress=False,
            verbosity=0,
        )
        model.fit(features, target, variable_names=list(feature_names))

        front = []
        for _, row in model.equations_.iterrows():
            front.append({"complexity": int(row["complexity"]),
                          "loss": float(row["loss"]),
                          "expression": str(row["equation"])})
    return front


def expression_to_latex(expression, feature_names=()):
    """
    Render an expression as LaTeX, falling back to a verbatim box.

    SymPy is used rather than string surgery so that precedence and grouping
    survive. A failure here must not lose the result: an expression that cannot
    be parsed is still the answer, and is passed through as monospace text.
    """
    try:
        import sympy
    except ImportError:  # pragma: no cover - sympy ships with the package deps
        return rf"\texttt{{{expression}}}"

    try:
        symbols = {name: sympy.Symbol(name) for name in feature_names}
        return sympy.latex(sympy.sympify(str(expression), locals=symbols))
    except (sympy.SympifyError, TypeError, SyntaxError, AttributeError):
        escaped = str(expression).replace("\\", r"\textbackslash{}")
        return rf"\texttt{{{escaped}}}"


class SymbolicDistiller:
    """
    Search for a closed-form expression reproducing a fitted mapping.

    Parameters
    ----------
    config : SymbolicConfig, optional
        Supplies every search parameter. Omit it to take the defaults.
    engine : callable, optional
        ``(features, target, feature_names, parameters) -> front``. Defaults to
        :func:`pysr_engine`. Injected rather than imported so the pipeline can
        be exercised without a Julia toolchain, and so a different backend is a
        parameter rather than a rewrite.

    Examples
    --------
    >>> distiller = SymbolicDistiller()                       # doctest: +SKIP
    >>> result = distiller.fit(table)                         # doctest: +SKIP
    >>> print(result.expression)                              # doctest: +SKIP
    """

    def __init__(self, config=None, engine=None):
        from .config import SymbolicConfig

        self.config = config if config is not None else SymbolicConfig()
        self.engine = engine if engine is not None else pysr_engine

    def parameters(self):
        """Search settings as a plain dict, as handed to the engine."""
        config = self.config
        unary = list(config.unary_operations or DEFAULT_UNARY)
        binary = list(config.binary_operations or DEFAULT_BINARY)
        return {
            "unary_operations": unary,
            "binary_operations": binary,
            "iterations": int(config.iterations),
            "population_size": int(config.population_size),
            "populations": int(config.populations),
            "max_size": int(config.max_size),
            "max_depth": int(config.max_depth),
            "parsimony": float(config.parsimony),
            "seed": int(config.seed),
        }

    def fit(self, table):
        """
        Run the search over ``table`` and score the winner.

        Returns
        -------
        SymbolicResult

        Raises
        ------
        ValueError
            If the engine returns no candidate at all — a silent empty result
            would otherwise be reported as a successful distillation.
        """
        front = self.engine(table.features, table.target,
                            list(table.feature_names), self.parameters())
        if not front:
            raise ValueError("The symbolic engine returned no expression.")

        front = sorted(front, key=lambda entry: entry["complexity"])
        best = min(front, key=lambda entry: entry["loss"])

        predicted = self.evaluate(best["expression"], table)
        return SymbolicResult(
            expression=best["expression"],
            latex=expression_to_latex(best["expression"], table.feature_names),
            complexity=int(best["complexity"]),
            loss=float(best["loss"]),
            r2=_r2(predicted, table.target),
            relative_l2=_relative_l2(predicted, table.target),
            pareto=front,
            feature_names=list(table.feature_names),
            target_name=table.target_name,
            scheme=table.scheme,
            units=table.units,
            target=(table.source
                    or getattr(self.config, "target", "")),
            n_samples=len(table),
            engine=getattr(self.engine, "__name__", str(self.engine)),
        )

    @staticmethod
    def evaluate(expression, table):
        """
        Evaluate an expression on a table's features.

        Scoring is done here, on the same rows, rather than trusting the
        engine's own loss: engines differ in what they report (mean square,
        mean absolute, normalised) and the numbers in the report have to be
        comparable with the operator's.

        Returns
        -------
        numpy.ndarray or None
            ``None`` when the expression cannot be parsed or evaluated, which
            leaves the fit metrics as NaN instead of inventing them.
        """
        try:
            import sympy
        except ImportError:  # pragma: no cover
            return None

        try:
            symbols = [sympy.Symbol(name) for name in table.feature_names]
            parsed = sympy.sympify(str(expression),
                                   locals={s.name: s for s in symbols})
            function = sympy.lambdify(symbols, parsed, "numpy")
            columns = [table.features[:, i] for i in range(len(symbols))]
            with np.errstate(all="ignore"):
                values = np.asarray(function(*columns), dtype=float)
            if values.ndim == 0:                      # a constant expression
                values = np.full(table.target.shape, float(values))
            return values
        except Exception:                             # noqa: BLE001
            # Any parse or evaluation failure means "no score", never a score
            # computed from something other than the reported expression.
            return None


def _r2(predicted, reference):
    """Coefficient of determination, NaN when the prediction is unusable."""
    if predicted is None:
        return float("nan")
    finite = np.isfinite(predicted)
    if not finite.all():
        return float("nan")
    total = np.sum((reference - reference.mean()) ** 2)
    if total <= 0:
        return float("nan")
    return float(1.0 - np.sum((predicted - reference) ** 2) / total)


def _relative_l2(predicted, reference):
    """Relative L2, on the same convention as the rest of the package."""
    if predicted is None:
        return float("nan")
    finite = np.isfinite(predicted)
    if not finite.all():
        return float("nan")
    denominator = np.linalg.norm(reference)
    if denominator <= 0:
        return float("nan")
    return float(np.linalg.norm(predicted - reference) / denominator)


# ===================================================================== #
# Driver
# ===================================================================== #
def distill_dataset(dataset, config, operator=None, log=None, engine=None):
    r"""
    Build the design matrix over a dataset and search it.

    Parameters
    ----------
    dataset : FieldPairDataset
        Supplies :math:`(\rho, \tau)` pairs; only ``chg2tau`` is meaningful,
        since a functional of the density is what is being sought.
    config : SymbolicConfig
        Search settings, including which target to fit.
    operator : FieldOperator, optional
        Required when ``config.target == "model"`` — the point of distillation
        is to fit what the network predicts, not the data it was trained on.
    log : callable, optional
        Progress sink.
    engine : callable, optional
        Passed to :class:`SymbolicDistiller`.

    Returns
    -------
    SymbolicResult
    """
    emit = log if log is not None else (lambda *_: None)
    if config.target not in ("model", "reference"):
        raise ValueError(f"Unknown symbolic target {config.target!r}; "
                         f"expected 'model' or 'reference'.")
    if config.target == "model" and operator is None:
        raise ValueError("target='model' distils the trained operator, so an "
                         "operator is required. Use target='reference' to fit "
                         "the DFT data instead.")

    tables = []
    for index in range(len(dataset)):
        source, reference = dataset.load_fields(index)
        target = (operator.predict(source) if config.target == "model"
                  else reference)
        built = build_features(
            source, target, source.grid, scheme=config.features,
            epsilon=getattr(config, "epsilon", DEFAULT_EPSILON))
        built.source = config.target
        tables.append(built)

    table = stack_tables(tables)
    emit(f"  built {len(table)} usable voxels from {len(dataset)} structure(s)"
         f"  [scheme: {config.features}, target: {config.target}]")
    table = sample_rows(table, config.n_samples, seed=config.seed)
    emit(f"  searching on {len(table)} sampled voxels "
         f"({config.iterations} iterations)")

    return SymbolicDistiller(config, engine=engine).fit(table)
