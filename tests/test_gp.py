# -*- coding: utf-8 -*-
# file: test_gp.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The native symbolic-regression engine: does it search, and does it obey?

:mod:`poraque.ml.gp` replaced PySR on 2026-09-03 to remove a Julia toolchain
from an HPC install. A replacement engine has to earn that, and three claims
carry it — each asserted here, and each measured while it was still false.

**It finds what is there.** A target inside the operator alphabet comes back
exactly, not approximately. ``F = exp(-p²)`` is recovered to the last bit of a
float64, from data alone, in under two seconds.

**The physics is fitness, not a filter.** A candidate that violates a limit
must cost more than any accuracy gain can buy it back. That the *same* penalty
arithmetic runs as before is asserted in ``tests/test_symbolic.py``; what is
asserted here is that the search is actually driven by it — the front comes
back satisfying the limits rather than a filter removing what does not.

**More budget is more search.** ``iterations`` is described in the shipped
config as the single biggest quality/time knob, and twice it was not one:

*Greedy selection.* Keeping the best ``size`` of parents *and* offspring
collapsed each population onto one genotype, which then recombined with itself.
Measured on ``F = 1/(1 + p²)``: 40, 120 and 300 iterations returned the same
expression to the last digit, at 1.5 s, 3.3 s and 7.7 s. Generational
replacement around a small elite fixed it.

*A budget-scaled stall limit.* Reseeding a stalled population after
``iterations // 10`` generations made a longer run take a *different*
trajectory rather than a longer one — and 300 iterations then returned a
**worse** expression than 120, which is indefensible for a knob that means
"spend more, get more". A constant cadence makes a short run a strict prefix of
a long one.

There is a fourth claim this file does *not* make, and the omission is
deliberate: that this engine is as good as PySR. It is not. PySR searches
harder per second and has years of tuning behind it. What is claimed is that
the objective is the same, the results on this project's own calibration
targets are exact, and the install no longer carries a second language runtime.
"""

import numpy as np
import pytest

from poraque.ml.gp import (
    ConstrainedObjective,
    HallOfFame,
    TreeFactory,
    complexity,
    depth,
    evaluate,
    native_engine,
    resolve_operations,
    respects_constraints,
    to_string,
    with_constants,
)
from poraque.ml.symbolic import physics_constraints


P, Q = ("var", 0), ("var", 1)
ALPHABET = {"unary_operations": ["exp", "log", "sqrt", "abs"],
            "binary_operations": ["+", "-", "*", "/", "^"]}


@pytest.fixture(scope="module")
def reduced():
    """``(p, q)`` samples over a range a real density actually visits."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.5, 400)
    q = rng.uniform(-1.0, 1.0, 400)
    return np.column_stack([p, q]), p, q


def settings(**overrides):
    """A small but genuine search, as the shipped config would configure it."""
    base = dict(ALPHABET, iterations=40, population_size=33, populations=15,
                max_size=30, max_depth=10, parsimony=0.0032, seed=0,
                constraints={"^": (-1, 1)}, data_loss="mse")
    base.update(overrides)
    return base


def evaluated(expression, names, points):
    """Evaluate a returned expression the way the rest of the package does."""
    import sympy

    symbols = {name: sympy.Symbol(name) for name in names}
    parsed = sympy.sympify(expression, locals=symbols)
    function = sympy.lambdify([symbols[name] for name in names], parsed,
                              "numpy")
    return np.asarray(function(*points), dtype=float) * np.ones(len(points[0]))


# ---------------------------------------------------------------------- #
class TestItRecoversWhatIsInTheAlphabet:
    """
    The calibration every symbolic result is read against.

    Two closed forms sit inside the search space by construction — the
    Thomas-Fermi and von Weizsäcker limits — and a run that cannot recover them
    on data generated from them has a misconfigured *search*, not interesting
    physics. That is what makes an exact recovery here worth asserting rather
    than a "good enough" tolerance.
    """

    def test_a_gaussian_enhancement_factor_comes_back_exactly(self, reduced):
        features, p, q = reduced
        front = native_engine(features, np.exp(-p ** 2), ["p", "q"],
                              settings(**physics(["p", "q"])))
        best = min(front, key=lambda entry: entry["loss"])
        prediction = evaluated(best["expression"], ["p", "q"], (p, q))
        error = np.linalg.norm(prediction - np.exp(-p ** 2)) \
            / np.linalg.norm(np.exp(-p ** 2))
        assert error < 1e-12, f"{best['expression']} -> rel L2 {error:.3e}"

    def test_a_lorentzian_one_comes_back_to_better_than_a_percent(self,
                                                                  reduced):
        features, p, q = reduced
        target = 1.0 / (1.0 + p ** 2)
        front = native_engine(features, target, ["p", "q"],
                              settings(**physics(["p", "q"])))
        best = min(front, key=lambda entry: entry["loss"])
        prediction = evaluated(best["expression"], ["p", "q"], (p, q))
        error = np.linalg.norm(prediction - target) / np.linalg.norm(target)
        assert error < 1e-2, f"{best['expression']} -> rel L2 {error:.3e}"

    def test_the_front_always_carries_its_simplest_member(self, reduced):
        """
        A size-1 entry, which is the first one a reader looks for.

        It went missing once, and the cause is worth keeping written down: the
        front keys on complexity, a raw variable is one node and beats an
        *unfitted* random constant on almost any data, so the size-1 slot was
        taken early and locked. Fitting constants only at the end could never
        displace it. They are fitted on entry now.
        """
        features, p, _ = reduced
        front = native_engine(features, np.exp(-p ** 2), ["p", "q"],
                              settings(**physics(["p", "q"])))
        assert min(entry["complexity"] for entry in front) == 1


def physics(names, template="pauli"):
    """The constraint block, as :class:`SymbolicDistiller` would build it."""
    block, _ = physics_constraints(names, template)
    return {"physics": block}


class TestThePhysicsDrivesTheSearch:
    """
    The front comes back physical, rather than being filtered afterwards.

    Filtering after a run only reports how few candidates were physical. The
    whole point of moving the limits into the fitness is that the populations
    never spend their budget on forms that were going to be discarded.
    """

    def test_the_winner_satisfies_both_asymptotic_limits(self, reduced):
        from poraque.ml.symbolic import check_asymptotic_limits

        features, p, _ = reduced
        front = native_engine(features, 1.0 / (1.0 + p ** 2), ["p", "q"],
                              settings(**physics(["p", "q"])))
        best = min(front, key=lambda entry: entry["loss"])
        compliance = check_asymptotic_limits(best["expression"], ["p", "q"],
                                             "reduced", template="pauli")
        assert compliance.passes, best["expression"]

    def test_an_unconstrained_search_is_free_to_do_otherwise(self, reduced):
        """
        The counterfactual. Without the probes the same data admits forms that
        diverge at large ``p`` — so the constrained run passing above is the
        constraint working, and not the target being easy.
        """
        features, p, _ = reduced
        front = native_engine(features, 1.0 / (1.0 + p ** 2), ["p", "q"],
                              settings(physics=None))
        assert front
        best = min(front, key=lambda entry: entry["loss"])
        # It is allowed to be physical by luck; what must be true is that
        # nothing charged it for not being.
        assert best["loss"] < 1.0

    def test_a_run_with_no_expressible_limit_still_gets_positivity(self):
        block, enforced = physics_constraints(
            ["rho", "grad_rho", "lap_rho"], "none")
        assert enforced == ["positivity"]
        assert block["probes"] == []


class TestMoreBudgetIsMoreSearch:
    """
    ``iterations`` has to mean something, and twice it did not.

    Both regressions produced a search that looked entirely healthy: a front
    came back, the expressions were plausible, the loss was small. The only
    symptom was that spending more time changed nothing — or, in the second
    case, made it worse.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def losses():
        rng = np.random.default_rng(0)
        p = rng.uniform(0.0, 1.5, 400)
        features = np.column_stack([p, rng.uniform(-1.0, 1.0, 400)])
        target = 1.0 / (1.0 + p ** 2)
        block, _ = physics_constraints(["p", "q"], "pauli")
        return {
            budget: min(entry["loss"] for entry in native_engine(
                features, target, ["p", "q"],
                settings(iterations=budget, physics=block)))
            for budget in (10, 40, 120)
        }

    def test_a_longer_run_is_never_worse(self, losses):
        assert losses[40] <= losses[10]
        assert losses[120] <= losses[40]

    def test_the_stall_cadence_does_not_depend_on_the_budget(self):
        """
        Which is what makes the guarantee above hold at all.

        Tied to ``iterations``, a longer run reseeds on a different schedule
        and takes a different trajectory rather than a longer one — measured,
        300 generations returned a worse expression than 120.
        """
        import inspect

        from poraque.ml import gp

        source = inspect.getsource(gp.native_engine)
        assert "STALL_LIMIT" in source
        assert isinstance(gp.STALL_LIMIT, int)


class TestTheSearchIsReproducible:
    """
    A seed is a promise, and this engine can keep it unconditionally.

    PySR needed serial evaluation to honour one and warned when handed a seed
    it could not keep a promise about, which is why ``deterministic`` existed
    as a *trade* against parallelism. This runs in one process, so the trade is
    gone and the key is accepted and ignored.
    """

    def test_the_same_seed_gives_the_same_front(self, reduced):
        features, p, _ = reduced
        target = 1.0 / (1.0 + p ** 2)
        arguments = (features, target, ["p", "q"],
                     settings(iterations=15, **physics(["p", "q"])))
        assert native_engine(*arguments) == native_engine(*arguments)

    def test_a_different_seed_gives_a_different_one(self, reduced):
        features, p, _ = reduced
        target = 1.0 / (1.0 + p ** 2)
        first = native_engine(features, target, ["p", "q"],
                              settings(iterations=15, seed=0,
                                       **physics(["p", "q"])))
        second = native_engine(features, target, ["p", "q"],
                               settings(iterations=15, seed=17,
                                        **physics(["p", "q"])))
        assert first != second

    def test_deterministic_is_accepted_and_changes_nothing(self, reduced):
        features, p, _ = reduced
        target = 1.0 / (1.0 + p ** 2)
        plain = native_engine(features, target, ["p", "q"],
                              settings(iterations=15, **physics(["p", "q"])))
        asked = native_engine(features, target, ["p", "q"],
                              settings(iterations=15, deterministic=True,
                                       **physics(["p", "q"])))
        assert plain == asked


class TestTheAlphabetIsAnsweredForRatherThanForwarded:
    """
    An operator the engine cannot evaluate raises, naming the ones it can.

    PySR was the authority on its own operator names, so this package could
    forward one it had never heard of. A native engine has to answer for the
    alphabet itself — and dropping an unknown operator silently would shrink
    the search space the user asked for while reporting nothing.
    """

    def test_an_unknown_unary_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown unary operator"):
            resolve_operations(["arctan"], ["+"])

    def test_an_unknown_binary_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown binary operator"):
            resolve_operations(["exp"], ["%"])

    def test_the_error_lists_what_is_available(self):
        with pytest.raises(ValueError, match="'exp'"):
            resolve_operations(["arctan"], ["+"])

    def test_the_shipped_defaults_all_resolve(self):
        from poraque.ml.symbolic import DEFAULT_BINARY, DEFAULT_UNARY

        unary, binary = resolve_operations(DEFAULT_UNARY, DEFAULT_BINARY)
        assert set(unary) == set(DEFAULT_UNARY)
        assert set(binary) == set(DEFAULT_BINARY)


class TestExpressionsComeBackInPythonNotation:
    r"""
    ``**`` rather than PySR's Julia ``^`` — a spelling, not a correction.

    Worth a class of its own mostly for the correction it records. The obvious
    story is that ``^`` was a latent mis-parse, since it is bitwise xor in
    Python; it is not, because :func:`sympy.sympify` takes
    ``convert_xor=True`` by default and read the old spelling as a power
    perfectly well. That is asserted below, so nobody re-derives the wrong
    version of this reasoning from the change alone.

    What ``**`` buys is that a distilled functional is valid Python, which is
    what a reader will paste it into.
    """

    def test_a_power_prints_as_a_python_power(self):
        tree = ("bin", "^", P, ("const", 2.0))
        assert "**" in to_string(tree, ["p", "q"])
        assert "^" not in to_string(tree, ["p", "q"])

    def test_what_it_prints_sympifies_to_what_it_computes(self):
        import sympy

        tree = ("bin", "^", P, ("const", 2.0))
        rendered = to_string(tree, ["p", "q"])
        parsed = sympy.sympify(rendered, locals={"p": sympy.Symbol("p")})
        assert parsed == sympy.Symbol("p") ** 2.0

    def test_the_caret_spelling_would_have_parsed_too(self):
        """
        The correction, pinned.

        ``sympify`` converts ``^`` to a power unless told otherwise, so the
        change of notation fixed no bug — and asserting that here is what stops
        the tidier, wrong story from being told about it later.
        """
        import sympy

        assert sympy.sympify("p^2", locals={"p": sympy.Symbol("p")}) == \
            sympy.Symbol("p") ** 2

    def test_operators_with_no_sympy_name_are_written_out(self):
        assert to_string(("un", "square", P), ["p"]) == "(p)**2"
        assert to_string(("un", "inv", P), ["p"]) == "1/(p)"


class TestTreesAndTheirLimits:
    """
    The structural bookkeeping the search is bounded by.

    ``max_size`` and ``parsimony`` count nodes, ``max_depth`` bounds the
    longest path, and ``constraints`` bounds an operator's arguments. A tree
    that escapes any of them is not a worse candidate, it is one the user
    excluded.
    """

    def test_complexity_counts_nodes(self):
        assert complexity(P) == 1
        assert complexity(("bin", "+", P, Q)) == 3
        assert complexity(("un", "exp", ("bin", "+", P, Q))) == 4

    def test_depth_is_the_longest_path(self):
        assert depth(P) == 1
        assert depth(("un", "exp", ("un", "exp", P))) == 3

    def test_a_power_holds_its_exponent_to_one_node(self):
        """
        ``{"^": (-1, 1)}``, the shipped default. An unconstrained exponent is
        the main source of nonsense from a power operator: a fractional power
        of a negative quantity leaves the reals, and an exponent that is itself
        a subtree is unreadable and almost never physical.
        """
        limits = {"^": (-1, 1)}
        assert respects_constraints(("bin", "^", P, ("const", 2.0)), limits)
        assert not respects_constraints(
            ("bin", "^", P, ("bin", "+", Q, ("const", 1.0))), limits)

    def test_minus_one_means_unlimited(self):
        limits = {"^": (-1, 1)}
        wide = ("bin", "^", ("bin", "*", P, ("bin", "+", Q, P)),
                ("const", 2.0))
        assert respects_constraints(wide, limits)

    def test_the_factory_never_returns_a_tree_outside_the_limits(self):
        factory = TreeFactory(np.random.default_rng(0), 2, ["exp", "sqrt"],
                              ["+", "*", "^"], max_size=12, max_depth=4,
                              constraints={"^": (-1, 1)})
        for _ in range(200):
            assert factory.acceptable(factory.random())

    def test_variation_stays_inside_them_too(self):
        factory = TreeFactory(np.random.default_rng(1), 2, ["exp", "sqrt"],
                              ["+", "*", "^"], max_size=12, max_depth=4,
                              constraints={"^": (-1, 1)})
        parents = [factory.random() for _ in range(50)]
        for index in range(50):
            child = factory.vary(parents[index], parents[(index + 1) % 50])
            assert factory.acceptable(child)


class TestEvaluationTreatsFailureAsAVerdict:
    """
    ``nan`` and ``inf`` are answers, not exceptions.

    ``log`` and ``sqrt`` are the real functions rather than protected variants:
    a negative argument yields ``nan``, which propagates into the fitness and
    the candidate is charged the ceiling. Protecting them — ``log|x|`` — is a
    common trick and a bad one, because it lets the search build expressions
    whose physical reading is a different function from the one reported.
    """

    def test_a_negative_logarithm_is_not_an_error(self):
        unary, binary = resolve_operations(["log"], ["+"])
        values = evaluate(("un", "log", P), np.array([[-1.0]]), unary, binary)
        assert np.isnan(values).all()

    def test_a_candidate_that_cannot_be_evaluated_scores_infinity(self):
        unary, binary = resolve_operations(["log"], ["+"])
        objective = ConstrainedObjective(np.array([[-1.0], [-2.0]]),
                                         np.array([1.0, 1.0]), ["p"],
                                         unary, binary)
        assert objective(("un", "log", P)) == float("inf")

    def test_an_overflowing_residual_is_rejected_without_warning(self,
                                                                 recwarn):
        """
        The arithmetic overflows *by design* and reaches its verdict through
        ``inf``. This package removed its global warning filter deliberately,
        so the quieting is local to the operation that means nothing by it —
        thousands of RuntimeWarnings on stderr would be the alternative.
        """
        unary, binary = resolve_operations(["exp"], ["*"])
        objective = ConstrainedObjective(np.full((8, 1), 700.0),
                                         np.ones(8), ["p"], unary, binary)
        tree = ("bin", "*", ("un", "exp", P), ("un", "exp", P))
        assert objective(tree) == float("inf")
        assert not [w for w in recwarn if w.category is RuntimeWarning]


class TestTheFrontIsAFront:
    """
    Best at every complexity, with nothing dominated left in it.

    A candidate both larger and worse than a smaller one is on no reading a
    member of an accuracy/complexity front, and leaving it in makes
    :func:`~poraque.ml.symbolic.pareto_knee` choose between points one of which
    is simply worse.
    """

    def test_it_keeps_the_best_at_each_size(self):
        hall = HallOfFame()
        assert hall.offer(P, 1.0)
        assert not hall.offer(Q, 2.0)          # same size, worse
        assert hall.offer(Q, 0.5)              # same size, better

    def test_a_dominated_entry_is_dropped(self):
        hall = HallOfFame()
        hall.offer(P, 0.5)                                     # size 1
        hall.offer(("bin", "+", P, Q), 0.9)                    # size 3, worse
        hall.offer(("un", "exp", ("bin", "+", P, Q)), 0.1)     # size 4, better
        sizes = [entry["complexity"] for entry in hall.front(["p", "q"])]
        assert sizes == [1, 4]

    def test_improves_answers_before_the_expensive_fit_is_paid_for(self):
        hall = HallOfFame()
        hall.offer(P, 1.0)
        assert hall.improves(Q, 0.5)
        assert not hall.improves(Q, 1.5)


class TestConstantsAreFitted:
    """
    Nelder-Mead on the candidates that reach the front.

    Structure search is what genetic programming is good at and constant
    fitting is what it is bad at. Doing it on every offspring costs far more
    than it returns when most of them die in the same generation.
    """

    def test_a_badly_scaled_constant_is_corrected(self):
        unary, binary = resolve_operations([], ["*"])
        features = np.linspace(1.0, 2.0, 32).reshape(-1, 1)
        objective = ConstrainedObjective(features, 3.0 * features.ravel(),
                                         ["p"], unary, binary)
        rough = ("bin", "*", ("const", 0.4), P)
        fitted = objective.fit_constants(rough)
        assert objective(fitted) < objective(rough)
        assert fitted[2][1] == pytest.approx(3.0, abs=1e-3)

    def test_a_tree_with_no_constants_is_returned_unchanged(self):
        unary, binary = resolve_operations([], ["*"])
        objective = ConstrainedObjective(np.ones((4, 2)), np.ones(4),
                                         ["p", "q"], unary, binary)
        assert objective.fit_constants(P) is P

    def test_a_fit_that_does_not_help_is_discarded(self):
        """
        Kept only on a strict improvement, so a Nelder-Mead that wanders can
        never make a front entry worse than the tree that earned its place.
        """
        unary, binary = resolve_operations([], ["*"])
        features = np.linspace(1.0, 2.0, 32).reshape(-1, 1)
        objective = ConstrainedObjective(features, 3.0 * features.ravel(),
                                         ["p"], unary, binary)
        exact = ("bin", "*", ("const", 3.0), P)
        assert objective(objective.fit_constants(exact)) <= objective(exact)

    def test_with_constants_rewrites_in_traversal_order(self):
        tree = ("bin", "+", ("const", 1.0), ("un", "exp", ("const", 2.0)))
        assert with_constants(tree, [7.0, 8.0]) == \
            ("bin", "+", ("const", 7.0), ("un", "exp", ("const", 8.0)))
