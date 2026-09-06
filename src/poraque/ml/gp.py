# -*- coding: utf-8 -*-
# file: gp.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Genetic programming over expression trees, in NumPy.

This is the search engine behind :mod:`poraque.ml.symbolic`, and it is native
on purpose: NumPy and SciPy are already hard dependencies, so distillation adds
no toolchain to a training run. A second language runtime would be faster and
would also be, on a supercomputer, a network fetch from a compute node, a
writable depot on a filesystem that may be purged between jobs, a
precompilation pass per architecture and a second runtime inside an MPI job —
for minutes of work at the end of a run that took hours.

The objective
-------------
**The physics is enforced as fitness**, never as a filter applied afterwards —
filtering candidates after a search only reports how few of them were
physical. Every candidate is scored by

.. math::

    \mathcal{L} = \underbrace{\tfrac1n\sum_i \ell(F_i - y_i)}_{\text{data}}
    \;+\; w_{+}\,\tfrac1n\sum_i \min(F_i, 0)^2
    \;+\; \sum_{\rm probes} w_\ell\,
          \frac{|F(\mathbf x_\ell) - t_\ell|}{s_\ell},

every term clamped at :data:`~poraque.ml.symbolic.PENALTY_CEILING` so a
candidate that diverges at :math:`p = 10^6` is decisively worse than a data
term of order one rather than infinitely worse than every other violator.
The probe points come from :func:`~poraque.ml.symbolic.physics_probes` and are
passed in as data, which is what keeps this module free of any import from
:mod:`poraque.ml.symbolic` and the two free of a cycle.

**The engine contract.** ``(features, target, feature_names, parameters) ->
front``, a list of ``{"complexity", "loss", "expression"}``. Injected into
:class:`~poraque.ml.symbolic.SymbolicDistiller` rather than imported by it, so
a different backend stays a parameter and not a rewrite.

Three properties worth knowing
------------------------------
**Expressions come out in Python notation**, with ``**`` for exponentiation:
the string is valid Python, which is what every reader of a distilled
functional will paste it into. (:func:`sympy.sympify` reads ``^`` as a power
too, so a config's ``"^"`` operator is the same operation under another name.)

**The search is deterministic by construction**, and ``deterministic`` is
therefore accepted and ignored rather than being a trade: the search runs in
one process, so the same seed gives the same front whatever else is true.

**The operator alphabet is checked rather than forwarded.** An operator that
is not implemented raises and names the ones that are; silently dropping one
would change the search space the user asked for.

Design notes
------------
Trees are nested tuples — ``("var", i)``, ``("const", v)``, ``("un", name, a)``,
``("bin", name, a, b)`` — which are immutable, hashable, cheap to copy, and
free to share between individuals, so crossover never needs a deep copy.

Constants are fitted by Nelder-Mead (:mod:`scipy.optimize`, already a hard
dependency) on candidates *entering the front*, not on every individual and not
once at the end. Structure search is what genetic programming is good at and
constant fitting is what it is bad at; fitting every offspring costs far more
than it returns when most of them die in the same generation, and fitting only
at the end is worse still, because the front keys on complexity and a slot
taken early by a raw variable can never be displaced by a constant that was not
yet worth anything.
"""

import math

import numpy as np


# ---------------------------------------------------------------------- #
# The operator alphabet
# ---------------------------------------------------------------------- #
def _protected(function):
    """Evaluate elementwise, letting NumPy produce ``nan``/``inf`` quietly."""
    def call(*arguments):
        with np.errstate(all="ignore"):
            return function(*arguments)
    return call


#: Unary operators, by the name a config spells them with.
#:
#: ``log`` and ``sqrt`` are the real-valued functions, not protected variants:
#: a negative argument yields ``nan``, which propagates into the fitness and
#: the candidate is charged the penalty ceiling. Protecting them — ``log|x|``,
#: ``sqrt|x|`` — is a common trick and a bad one here, because it lets the
#: search build expressions whose physical reading is a different function from
#: the one being reported.
UNARY_OPERATIONS = {
    "exp": _protected(np.exp),
    "log": _protected(np.log),
    "sqrt": _protected(np.sqrt),
    "abs": _protected(np.abs),
    "sin": _protected(np.sin),
    "cos": _protected(np.cos),
    "tanh": _protected(np.tanh),
    "square": _protected(np.square),
    "cube": _protected(lambda x: x ** 3),
    "inv": _protected(lambda x: 1.0 / x),
    "neg": _protected(np.negative),
}

#: Binary operators, by the name a config spells them with.
BINARY_OPERATIONS = {
    "+": _protected(np.add),
    "-": _protected(np.subtract),
    "*": _protected(np.multiply),
    "/": _protected(np.divide),
    "^": _protected(np.power),
    "**": _protected(np.power),
}

#: How each binary operator prints. ``^`` becomes ``**`` so the rendered string
#: is valid Python as well as valid SymPy. Both parse: :func:`sympy.sympify`
#: defaults to ``convert_xor=True`` and reads ``^`` as a power, which is worth
#: recording here because the opposite is easy to assume and was assumed once.
_BINARY_SYMBOLS = {"+": "+", "-": "-", "*": "*", "/": "/",
                   "^": "**", "**": "**"}

#: Unary operators with no direct SymPy spelling, written out instead.
_UNARY_TEMPLATES = {
    "square": "({0})**2",
    "cube": "({0})**3",
    "inv": "1/({0})",
    "neg": "-({0})",
}


def resolve_operations(unary, binary):
    """
    Check a configured alphabet and return the callables for it.

    Parameters
    ----------
    unary, binary : sequence of str

    Returns
    -------
    tuple of (dict, dict)
        Name → callable, for each arity.

    Raises
    ------
    ValueError
        On an operator this engine does not implement, naming the ones it
        does. Dropping an unknown operator silently would shrink the search
        space the user asked for.
    """
    resolved = []
    for names, table, arity in ((unary, UNARY_OPERATIONS, "unary"),
                                (binary, BINARY_OPERATIONS, "binary")):
        chosen = {}
        for name in names:
            key = str(name)
            if key not in table:
                raise ValueError(
                    f"Unknown {arity} operator {key!r}. This engine "
                    f"implements {sorted(table)}.")
            chosen[key] = table[key]
        resolved.append(chosen)
    return tuple(resolved)


# ---------------------------------------------------------------------- #
# Trees
# ---------------------------------------------------------------------- #
def complexity(node):
    """Number of nodes, which is what ``max_size`` and ``parsimony`` count."""
    kind = node[0]
    if kind in ("var", "const"):
        return 1
    if kind == "un":
        return 1 + complexity(node[2])
    return 1 + complexity(node[2]) + complexity(node[3])


def depth(node):
    """Longest root-to-leaf path, which is what ``max_depth`` bounds."""
    kind = node[0]
    if kind in ("var", "const"):
        return 1
    if kind == "un":
        return 1 + depth(node[2])
    return 1 + max(depth(node[2]), depth(node[3]))


def subtrees(node, path=()):
    """Every subtree with the path that reaches it, root first."""
    yield path, node
    if node[0] == "un":
        yield from subtrees(node[2], path + (2,))
    elif node[0] == "bin":
        yield from subtrees(node[2], path + (2,))
        yield from subtrees(node[3], path + (3,))


def replace(node, path, replacement):
    """Return ``node`` with the subtree at ``path`` swapped out."""
    if not path:
        return replacement
    index, rest = path[0], path[1:]
    listed = list(node)
    listed[index] = replace(node[index], rest, replacement)
    return tuple(listed)


def constants(node):
    """Every constant in the tree, with the path that reaches it."""
    return [(path, sub[1]) for path, sub in subtrees(node)
            if sub[0] == "const"]


def with_constants(node, values):
    """Return ``node`` with its constants replaced, in traversal order."""
    remaining = list(values)

    def rebuild(current):
        if current[0] == "const":
            return ("const", float(remaining.pop(0)))
        if current[0] == "un":
            return ("un", current[1], rebuild(current[2]))
        if current[0] == "bin":
            return ("bin", current[1], rebuild(current[2]), rebuild(current[3]))
        return current

    return rebuild(node)


def evaluate(node, matrix, unary, binary):
    """
    Evaluate a tree on an ``(n, d)`` design matrix.

    Returns
    -------
    numpy.ndarray
        ``(n,)``, possibly containing ``nan`` or ``inf`` — which the fitness
        reads as a violation rather than an error, since a candidate that
        cannot be evaluated at the uniform-gas point does not have a
        Thomas-Fermi limit.
    """
    kind = node[0]
    if kind == "var":
        return matrix[:, node[1]]
    if kind == "const":
        return np.full(matrix.shape[0], node[1], dtype=float)
    if kind == "un":
        return unary[node[1]](evaluate(node[2], matrix, unary, binary))
    return binary[node[1]](evaluate(node[2], matrix, unary, binary),
                           evaluate(node[3], matrix, unary, binary))


def to_string(node, feature_names):
    """
    Render a tree as a string :func:`sympy.sympify` parses correctly.

    Fully parenthesised rather than minimally: every consumer re-parses this,
    and a precedence bug in a printer is a wrong expression that still looks
    plausible.
    """
    kind = node[0]
    if kind == "var":
        return str(feature_names[node[1]])
    if kind == "const":
        return repr(float(node[1]))
    if kind == "un":
        inner = to_string(node[2], feature_names)
        template = _UNARY_TEMPLATES.get(node[1])
        return (template.format(inner) if template
                else f"{node[1]}({inner})")
    left = to_string(node[2], feature_names)
    right = to_string(node[3], feature_names)
    return f"({left} {_BINARY_SYMBOLS[node[1]]} {right})"


def respects_constraints(node, limits):
    """
    Whether every operator's arguments are within their complexity limits.

    ``limits`` is the config's ``constraints`` block, e.g. ``{"^": (-1, 1)}``
    holding a power's exponent to a single node. ``-1`` means unlimited. An
    unconstrained exponent is the main source of nonsense from a power
    operator: a fractional power of a negative quantity leaves the reals, and
    an exponent that is itself a subtree is unreadable and almost never
    physical.
    """
    if not limits:
        return True
    for _, sub in subtrees(node):
        if sub[0] != "bin":
            continue
        limit = limits.get(sub[1])
        if limit is None:
            continue
        bounds = limit if isinstance(limit, (list, tuple)) else (limit, limit)
        for argument, bound in zip(sub[2:4], bounds):
            if bound is not None and int(bound) >= 0 \
                    and complexity(argument) > int(bound):
                return False
    return True


# ---------------------------------------------------------------------- #
# Random construction and variation
# ---------------------------------------------------------------------- #
class TreeFactory:
    """
    Builds and mutates trees within one configured search space.

    Parameters
    ----------
    rng : numpy.random.Generator
    n_features : int
    unary, binary : sequence of str
        Operator names, already checked by :func:`resolve_operations`.
    max_size, max_depth : int
    constraints : dict, optional
        Per-operator argument complexity limits.
    """

    def __init__(self, rng, n_features, unary, binary, max_size=30,
                 max_depth=10, constraints=None):
        self.rng = rng
        self.n_features = int(n_features)
        self.unary = list(unary)
        self.binary = list(binary)
        self.max_size = int(max_size)
        self.max_depth = int(max_depth)
        self.constraints = dict(constraints or {})

    # -- construction --------------------------------------------------- #
    def constant(self):
        """A constant drawn from a scale-free-ish prior.

        Log-uniform in magnitude over four decades with a random sign, rather
        than normal about zero: the constants a physical formula needs span
        ``5/3`` and ``1e-3``, and a normal prior reaches the second only by
        accident.
        """
        magnitude = 10.0 ** self.rng.uniform(-2.0, 2.0)
        return ("const", float(self.rng.choice([-1.0, 1.0]) * magnitude))

    def leaf(self):
        """A variable, or occasionally a constant."""
        if self.n_features and self.rng.random() < 0.7:
            return ("var", int(self.rng.integers(self.n_features)))
        return self.constant()

    def grow(self, depth_budget):
        """One random tree, at most ``depth_budget`` deep."""
        if depth_budget <= 1 or self.rng.random() < 0.3:
            return self.leaf()
        if self.unary and (not self.binary or self.rng.random() < 0.35):
            return ("un", str(self.rng.choice(self.unary)),
                    self.grow(depth_budget - 1))
        return ("bin", str(self.rng.choice(self.binary)),
                self.grow(depth_budget - 1), self.grow(depth_budget - 1))

    def random(self):
        """A random tree that satisfies every structural limit."""
        for _ in range(32):
            candidate = self.grow(min(4, self.max_depth))
            if self.acceptable(candidate):
                return candidate
        return self.leaf()

    def acceptable(self, node):
        """Whether a tree is inside the size, depth and argument limits."""
        return (complexity(node) <= self.max_size
                and depth(node) <= self.max_depth
                and respects_constraints(node, self.constraints))

    # -- variation ------------------------------------------------------ #
    def mutate(self, node):
        """One mutation, chosen among four kinds."""
        choice = self.rng.random()
        if choice < 0.35:
            return self._mutate_subtree(node)
        if choice < 0.70:
            return self._mutate_constant(node)
        if choice < 0.85:
            return self._mutate_operator(node)
        return self._simplify(node)

    def _mutate_subtree(self, node):
        paths = [path for path, _ in subtrees(node)]
        path = paths[int(self.rng.integers(len(paths)))]
        return replace(node, path, self.grow(min(3, self.max_depth)))

    def _mutate_constant(self, node):
        found = constants(node)
        if not found:
            return self._mutate_subtree(node)
        path, value = found[int(self.rng.integers(len(found)))]
        # Multiplicative, so a constant can travel between decades; the
        # additive term lets one cross zero, which a purely multiplicative
        # perturbation never can.
        scaled = value * float(self.rng.normal(1.0, 0.3)) \
            + float(self.rng.normal(0.0, 0.1))
        return replace(node, path, ("const", float(scaled)))

    def _mutate_operator(self, node):
        candidates = [path for path, sub in subtrees(node)
                      if (sub[0] == "un" and len(self.unary) > 1)
                      or (sub[0] == "bin" and len(self.binary) > 1)]
        if not candidates:
            return self._mutate_subtree(node)
        path = candidates[int(self.rng.integers(len(candidates)))]
        target = node
        for index in path:
            target = target[index]
        pool = self.unary if target[0] == "un" else self.binary
        swapped = (target[0], str(self.rng.choice(pool))) + tuple(target[2:])
        return replace(node, path, swapped)

    def _simplify(self, node):
        """Replace a random subtree with one of its own children."""
        inner = [(path, sub) for path, sub in subtrees(node)
                 if sub[0] in ("un", "bin")]
        if not inner:
            return self._mutate_constant(node)
        path, sub = inner[int(self.rng.integers(len(inner)))]
        child = sub[2] if sub[0] == "un" or self.rng.random() < 0.5 else sub[3]
        return replace(node, path, child)

    def crossover(self, first, second):
        """Swap a random subtree of ``first`` for one of ``second``."""
        paths = [path for path, _ in subtrees(first)]
        donors = [sub for _, sub in subtrees(second)]
        path = paths[int(self.rng.integers(len(paths)))]
        donor = donors[int(self.rng.integers(len(donors)))]
        return replace(first, path, donor)

    def vary(self, first, second):
        """One offspring, retried until it satisfies the structural limits."""
        for _ in range(8):
            child = (self.crossover(first, second) if self.rng.random() < 0.5
                     else self.mutate(first))
            if self.acceptable(child):
                return child
        return first


# ---------------------------------------------------------------------- #
# Fitness
# ---------------------------------------------------------------------- #
#: Generations a population may go without improving before it is reseeded
#: around its elite.
#:
#: Constant rather than a fraction of ``iterations``, so that a longer run is a
#: longer version of a shorter one rather than a different one; see the comment
#: at the restart itself.
STALL_LIMIT = 12


#: Data terms, by the name a config spells them with.
#:
#: Evaluated inside ``errstate``: a candidate whose residual squares to
#: infinity is one the search is *supposed* to reject, and it reaches that
#: verdict through ``inf``. Letting NumPy warn about it would put thousands of
#: RuntimeWarnings on stderr for arithmetic working exactly as intended -- and
#: this package removed its global warning filter deliberately, so the
#: quieting has to be local to the operation that means nothing by it.
DATA_TERMS = {
    "mse": lambda residual: float(np.mean(residual ** 2)),
    "mae": lambda residual: float(np.mean(np.abs(residual))),
}


class ConstrainedObjective:
    r"""
    The fitness a candidate is scored by: data plus the physics penalties.

    The constrained ``loss`` reported on a front is not comparable with an
    unconstrained run's, so every term here is clamped and weighted exactly as
    :func:`~poraque.ml.symbolic.physics_constraints` states, and two runs
    with the same settings are comparable with each other.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n, d)`` design matrix.
    target : numpy.ndarray
        ``(n,)`` values to reproduce.
    feature_names : sequence of str
        Column order, used to place each probe's coordinates.
    unary, binary : dict
        Name → callable.
    data_loss : {"mse", "mae"}
    probes : sequence of dict, optional
        As produced by :func:`~poraque.ml.symbolic.physics_probes`:
        ``{"limit", "point", "target", "scale"}``. Passed as data rather than
        imported, which is what keeps this module independent of
        :mod:`poraque.ml.symbolic`.
    weights : dict, optional
        Penalty weights keyed by ``"positivity"`` and by each probe's
        ``"limit"``.
    ceiling : float, optional
        Clamp on every individual penalty term.
    """

    def __init__(self, features, target, feature_names, unary, binary,
                 data_loss="mse", probes=(), weights=None, ceiling=1e9):
        if data_loss not in DATA_TERMS:
            raise ValueError(f"Unknown data loss {data_loss!r}; expected one "
                             f"of {sorted(DATA_TERMS)}.")
        self.features = np.asarray(features, dtype=float)
        self.target = np.asarray(target, dtype=float).ravel()
        self.unary = unary
        self.binary = binary
        self.data_term = DATA_TERMS[data_loss]
        self.ceiling = float(ceiling)
        self.weights = dict(weights or {})
        self.probes = list(probes)

        names = list(feature_names)
        if self.probes:
            self.probe_points = np.array(
                [[float(probe["point"][name]) for name in names]
                 for probe in self.probes], dtype=float)
            self.probe_targets = np.array(
                [float(probe["target"]) for probe in self.probes])
            self.probe_scales = np.array(
                [float(probe["scale"]) for probe in self.probes])
            # One limit carries one weight however many probes express it, so
            # the density sweep of the `gga` scheme raises the resolution of
            # the check rather than its price.
            share = {}
            for probe in self.probes:
                share[probe["limit"]] = share.get(probe["limit"], 0) + 1
            self.probe_weights = np.array(
                [self.weights.get(probe["limit"], 0.0) / share[probe["limit"]]
                 for probe in self.probes])
        else:
            self.probe_points = np.empty((0, len(names)))
            self.probe_targets = np.empty(0)
            self.probe_scales = np.empty(0)
            self.probe_weights = np.empty(0)

    @property
    def enforced(self):
        """Names of the constraints this objective actually imposes."""
        order = ["thomas_fermi", "von_weizsacker"]
        limits = sorted({probe["limit"] for probe in self.probes},
                        key=order.index)
        return ["positivity"] + limits

    def __call__(self, node):
        """Score one tree. ``inf`` when it cannot be evaluated at all."""
        with np.errstate(all="ignore"):
            values = evaluate(node, self.features, self.unary, self.binary)
            if not np.all(np.isfinite(values)):
                return math.inf
            loss = self.data_term(values - self.target)
            if not math.isfinite(loss):
                return math.inf

            shortfall = float(np.mean(np.minimum(values, 0.0) ** 2))
            loss += (self.weights.get("positivity", 0.0)
                     * min(shortfall, self.ceiling))

            if len(self.probes):
                probed = evaluate(node, self.probe_points, self.unary,
                                  self.binary)
                deviation = np.where(
                    np.isfinite(probed),
                    np.abs(probed - self.probe_targets) / self.probe_scales,
                    self.ceiling)
                loss += float(np.sum(self.probe_weights
                                     * np.minimum(deviation, self.ceiling)))
        return loss if math.isfinite(loss) else math.inf

    def fit_constants(self, node, iterations=60):
        """
        Refine a candidate's constants by Nelder-Mead, keeping any improvement.

        Structure search is what genetic programming is good at and constant
        fitting is what it is bad at. This is applied to hall-of-fame entries
        only: fitting every offspring's constants costs far more than it
        returns when most of them die in the same generation.
        """
        found = constants(node)
        if not found:
            return node
        from scipy.optimize import minimize

        start = np.array([value for _, value in found], dtype=float)
        base = self(node)

        def objective(vector):
            score = self(with_constants(node, vector))
            return score if math.isfinite(score) else 1e300

        try:
            result = minimize(objective, start, method="Nelder-Mead",
                              options={"maxiter": int(iterations),
                                       "xatol": 1e-6, "fatol": 1e-8})
        except Exception:                                   # noqa: BLE001
            return node
        if not np.all(np.isfinite(result.x)):
            return node
        improved = with_constants(node, result.x)
        return improved if self(improved) < base else node


# ---------------------------------------------------------------------- #
# The search
# ---------------------------------------------------------------------- #
class HallOfFame:
    """
    Best expression seen at each complexity — the accuracy/complexity front.

    Keyed by complexity rather than kept as a sorted list of the best *n*: the
    result the caller wants is a Pareto front, and a front is exactly "the best
    there is at every size". A candidate no better than the incumbent at its
    own size is dropped, so the front never grows past ``max_size`` entries.
    """

    def __init__(self):
        self.best = {}

    def improves(self, node, loss):
        """Whether ``node`` would beat the incumbent at its own complexity."""
        current = self.best.get(complexity(node))
        return current is None or loss < current[1]

    def offer(self, node, loss):
        """Record ``node`` if it beats the incumbent at its complexity."""
        size = complexity(node)
        current = self.best.get(size)
        if current is None or loss < current[1]:
            self.best[size] = (node, loss)
            return True
        return False

    def front(self, feature_names):
        """
        The front, as the engine contract's list of dicts.

        Dominated entries are dropped: a candidate that is both larger and
        worse than a smaller one is on no reasonable reading a member of an
        accuracy/complexity front, and leaving it in makes
        :func:`~poraque.ml.symbolic.pareto_knee` choose between points one of
        which is simply worse.
        """
        entries = []
        floor = math.inf
        for size in sorted(self.best):
            node, loss = self.best[size]
            if loss < floor:
                floor = loss
                entries.append({"complexity": int(size),
                                "loss": float(loss),
                                "expression": to_string(node, feature_names)})
        return entries


def native_engine(features, target, feature_names, parameters):
    """
    Search for an expression reproducing ``target``, and return its front.

    The engine contract of :class:`~poraque.ml.symbolic.SymbolicDistiller`.

    Parameters
    ----------
    features : numpy.ndarray
        ``(n, d)`` design matrix.
    target : numpy.ndarray
        ``(n,)`` values to reproduce.
    feature_names : sequence of str
        Column names, in column order.
    parameters : dict
        As built by :meth:`~poraque.ml.symbolic.SymbolicDistiller.parameters`:
        the operator alphabet, the population and iteration counts, the size
        and depth ceilings, ``parsimony``, ``seed``, ``constraints``, and a
        ``physics`` block holding the probes, weights and penalty ceiling.

    Returns
    -------
    list of dict
        ``{"complexity", "loss", "expression"}``, one per point of the
        accuracy/complexity front. Not sorted by quality; the caller ranks it.

    Notes
    -----
    ``deterministic`` is accepted and ignored: the search runs in one process,
    so the same ``seed`` gives the same front regardless.
    """
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float).ravel()
    names = list(feature_names)

    unary, binary = resolve_operations(parameters["unary_operations"],
                                       parameters["binary_operations"])
    physics = dict(parameters.get("physics") or {})
    objective = ConstrainedObjective(
        features, target, names, unary, binary,
        data_loss=str(parameters.get("data_loss", "mse")),
        probes=physics.get("probes", ()),
        weights=physics.get("weights", {}),
        ceiling=float(physics.get("ceiling", 1e9)),
    )

    rng = np.random.default_rng(int(parameters.get("seed", 0)))
    factory = TreeFactory(
        rng, features.shape[1], list(unary), list(binary),
        max_size=int(parameters.get("max_size", 30)),
        max_depth=int(parameters.get("max_depth", 10)),
        constraints=parameters.get("constraints") or {},
    )
    parsimony = float(parameters.get("parsimony", 0.0))
    n_populations = max(1, int(parameters.get("populations", 1)))
    size = max(4, int(parameters.get("population_size", 20)))
    iterations = max(1, int(parameters.get("iterations", 10)))

    hall = HallOfFame()

    def score(node):
        """Selection fitness: the objective, plus the size penalty."""
        return objective(node) + parsimony * complexity(node)

    def consider(node):
        """
        Offer a candidate to the front, fitting its constants if it belongs.

        The fit is gated on :meth:`HallOfFame.improves` rather than run
        unconditionally, and that gate is what makes it affordable: a
        generation produces hundreds of individuals and only a handful are
        ever an improvement on the incumbent at their own size.

        It has to happen *here* rather than once at the end, which is where it
        first was. The front keys on complexity, so a size-1 slot taken early
        by a raw variable -- ``p`` is one node and beats an unfitted random
        constant on almost any data -- was locked, and a constant fitted
        afterwards could never displace it. On a target of :math:`F = 1` the
        front then had no size-1 entry at all, which is the first one a reader
        looks for.
        """
        raw = objective(node)
        if not math.isfinite(raw) or not hall.improves(node, raw):
            return
        fitted = objective.fit_constants(node)
        hall.offer(fitted, objective(fitted))

    populations = []
    for index in range(n_populations):
        # Every variable on its own, and a bare constant, seeded into the
        # first population: they are the answers a reader checks first --
        # `F = 1` is the Thomas-Fermi limit -- and leaving a one-node tree to
        # be rediscovered by chance is a poor use of the budget.
        seeds = ([("var", i) for i in range(features.shape[1])]
                 + [("const", 1.0)]) if index == 0 else []
        seeds = seeds[:size]
        members = seeds + [factory.random() for _ in range(size - len(seeds))]
        populations.append([(node, score(node)) for node in members])
        for node, _ in populations[-1]:
            consider(node)

    def tournament(members):
        """Best of three, which keeps selection pressure mild."""
        picks = rng.integers(len(members), size=3)
        return min((members[int(i)] for i in picks), key=lambda pair: pair[1])[0]

    # Generational replacement with a small elite, rather than keeping the
    # best `size` of parents *and* offspring together.
    #
    # The greedy version was measured and is the reason for this one: on a
    # target of F = 1/(1 + p^2) it converged by generation 40 and then did not
    # move at all -- 40, 120 and 300 iterations returned the same expression to
    # the last digit, at 1.5 s, 3.3 s and 7.7 s. That is not a search that has
    # found the answer, it is a population that has collapsed onto one genotype
    # and is recombining it with itself. Discarding the non-elite parents each
    # generation is what turns the extra budget back into exploration.
    elite_count = max(1, size // 10)
    immigrants = max(1, size // 12)

    # Restart a population that has stopped improving, keeping its elite.
    #
    # `iterations` is described in the shipped config as the single biggest
    # quality/time knob, and without this it is not one: measured on
    # F = 1/(1 + p^2), the last improvement to the global best arrived on the
    # order of generation 3 and generations 40, 120 and 300 all returned the
    # same expression to the last digit. The cause is the objective rather
    # than the operators -- a constraint violation costs ~1e2 against a data
    # term of ~1e-3, so selection is almost entirely "does this satisfy the
    # limits", and the few candidates that do are quickly all there is to
    # recombine. Reseeding is what turns the rest of the budget back into
    # search instead of into recombining one genotype with itself.
    # A fixed cadence, deliberately not a fraction of `iterations`. Tying it
    # to the budget made a longer run take a *different* trajectory rather than
    # a longer one -- measured, 300 generations returned a worse expression
    # than 120, which is indefensible for a knob whose whole meaning is "spend
    # more and get more". With a constant, a run of N generations is a strict
    # prefix of a run of 2N at the same seed, so the front can only improve.
    stalled = [0] * n_populations
    previous = [math.inf] * n_populations

    for generation in range(iterations):
        for index, members in enumerate(populations):
            offspring = [factory.vary(tournament(members), tournament(members))
                         for _ in range(size)]
            offspring += [factory.random() for _ in range(immigrants)]

            # The elite carry the generation's best forward, so a population
            # can never move backwards; everything else is replaced.
            elite = sorted(members, key=lambda pair: pair[1])[:elite_count]
            pool = dict(elite)
            for child in offspring:
                value = score(child)
                if pool.get(child, math.inf) > value:
                    pool[child] = value

            # Deduplicated on the tree itself, which is a plain tuple and so
            # hashable for free. Without it a population fills with copies of
            # one winner and the tournaments stop choosing between anything.
            ranked = sorted(pool.items(), key=lambda pair: pair[1])[:size]
            while len(ranked) < size:
                fresh = factory.random()
                ranked.append((fresh, score(fresh)))
            populations[index] = ranked
            for node, _ in ranked:
                consider(node)

            best_now = ranked[0][1]
            if best_now < previous[index] - 1e-12:
                previous[index], stalled[index] = best_now, 0
            else:
                stalled[index] += 1
            if stalled[index] >= STALL_LIMIT:
                survivors = ranked[:elite_count]
                fresh = [factory.random()
                         for _ in range(size - len(survivors))]
                populations[index] = survivors + [(node, score(node))
                                                  for node in fresh]
                populations[index].sort(key=lambda pair: pair[1])
                stalled[index], previous[index] = 0, math.inf

        # Migration: the best of each population joins the next, which is what
        # makes several small populations behave differently from one large
        # one -- they explore separately and share only their winners.
        if n_populations > 1 and generation % 5 == 4:
            champions = [members[0] for members in populations]
            for index, members in enumerate(populations):
                members[-1] = champions[index - 1]
                members.sort(key=lambda pair: pair[1])

    return hall.front(names)
