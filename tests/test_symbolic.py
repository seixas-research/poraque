# -*- coding: utf-8 -*-
# file: test_symbolic.py

"""
Tests for symbolic distillation.

The engine is stochastic, so it is injected here rather than run — the search
itself is exercised in ``tests/test_gp.py``. That is not a gap: everything the
package is responsible for around it — building the features, the units they
are built in, sampling, scoring, LaTeX — is deterministic and lives outside the
search.

The featurisation carries the real risk, because an error there is silent: a
wrong unit or a wrong derivative still yields a plausible expression, just not
one about the physics. So it is pinned against the two functionals whose closed
forms are known exactly.
"""

import json

import numpy as np
import pytest

from poraque.fields import FieldGrid, thomas_fermi_tau, von_weizsacker_tau
from poraque.fields.constants import BOHR_TO_ANGSTROM, C_TF
from poraque.ml.config import SymbolicConfig
from poraque.ml.symbolic import (
    DEFAULT_BINARY,
    check_asymptotic_limits,
    DEFAULT_UNARY,
    PENALTY_CEILING,
    SymbolicDistiller,
    build_features,
    expression_to_latex,
    physics_constraints,
    physics_probes,
    sample_rows,
    spectral_laplacian,
    stack_tables,
)


@pytest.fixture
def cell():
    """A smooth, strictly positive, periodic density on a cubic cell."""
    length, n = 8.0, 16
    grid = FieldGrid((n, n, n), np.eye(3) * length)
    axis = np.arange(n) / n * length
    _, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    density = 0.5 + 0.2 * np.sin(2 * np.pi * x / length) * np.cos(
        2 * np.pi * y / length)
    return grid, density


def power_law_engine(features, target, names, parameters):
    """
    A real, tiny search: fit ``y = a * x0^b`` by least squares in log space.

    Enough to exercise the whole pipeline deterministically, and — on
    Thomas-Fermi data — to prove the features carry the physics, since the
    exponent it recovers must be 5/3.
    """
    slope, intercept = np.polyfit(np.log(features[:, 0]), np.log(target), 1)
    return [
        {"complexity": 1, "loss": float(np.var(target)),
         "expression": f"{target.mean():.6f}"},
        {"complexity": 5, "loss": 1e-12,
         "expression": f"{np.exp(intercept):.6f} * {names[0]}^{slope:.6f}"},
    ]


# ===================================================================== #
# Derivatives
# ===================================================================== #
class TestSpectralLaplacian:
    def test_matches_the_analytic_answer(self, cell):
        grid, _ = cell
        length, n = 8.0, grid.shape[0]
        axis = np.arange(n) / n * length
        _, _, x = np.meshgrid(axis, axis, axis, indexing="ij")
        field = np.sin(2 * np.pi * x / length)

        computed = spectral_laplacian(field, grid, length_unit="angstrom")
        exact = -(2 * np.pi / length) ** 2 * field
        assert np.abs(computed - exact).max() < 1e-10

    def test_bohr_scaling(self, cell):
        """Å⁻² against Bohr⁻²: exactly a factor of BOHR_TO_ANGSTROM squared."""
        grid, density = cell
        angstrom = spectral_laplacian(density, grid, length_unit="angstrom")
        bohr = spectral_laplacian(density, grid, length_unit="bohr")
        assert np.allclose(bohr, angstrom * BOHR_TO_ANGSTROM ** 2)


# ===================================================================== #
# Features — pinned against the known functionals
# ===================================================================== #
class TestFeatures:
    def test_raw_reproduces_thomas_fermi(self, cell):
        r"""
        On :math:`\tau_{\rm TF}` data, ``C_TF * rho^(5/3)`` must reproduce the
        target column exactly. A unit slip anywhere in the conversion breaks
        this and nothing else would catch it.
        """
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="raw")
        rho = table.features[:, 0]
        assert np.allclose(C_TF * rho ** (5.0 / 3.0), table.target, rtol=1e-12)

    def test_pauli_factor_on_thomas_fermi_data(self, cell):
        r"""
        :math:`F = (\tau_{\rm TF} - \tau_{\rm vW})/\tau_{\rm TF}
        = 1 - 5p^2/3` exactly.
        """
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        p = table.features[:, 0]
        assert np.allclose(table.target, 1.0 - 5.0 * p ** 2 / 3.0, atol=1e-10)

    def test_pauli_factor_vanishes_on_von_weizsacker_data(self, cell):
        """
        The Pauli term is what is left after von Weizsacker is removed, so on
        pure vW data there is nothing left. Exactly zero, not approximately.
        """
        grid, density = cell
        table = build_features(density, von_weizsacker_tau(density, grid), grid,
                               scheme="enhancement")
        assert np.abs(table.target).max() < 1e-12

    def test_tau_vw_matches_the_library_implementation(self, cell):
        """
        Computed here from |grad rho|^2 / 8 rho on the grid; pinned against
        the field-level implementation so the two cannot drift.
        """
        from poraque.fields.constants import BOHR_TO_ANGSTROM, HARTREE_TO_EV

        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="gga", template="pauli")
        conversion = HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 3
        assert np.allclose(table.tau_vw * conversion,
                           von_weizsacker_tau(density, grid).ravel())

    def test_the_pauli_template_round_trips(self, cell):
        r""":math:`\tau = \tau_{\rm vW} + \tau_{\rm TF} F` must give the
        original field back."""
        from poraque.ml.symbolic import reconstruct_tau

        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="gga", template="pauli")
        assert np.allclose(reconstruct_tau(table.target, table),
                           table.physical_target)

    def test_gga_is_the_default_and_names_are_rho_p_q(self, cell):
        """PySR uses these names verbatim, so they are part of the contract."""
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        assert table.scheme == "gga"
        assert table.feature_names == ["rho", "p", "q"]
        assert table.features.shape == (len(table), 3)

    def test_p_and_q_match_their_definitions(self, cell):
        r"""
        :math:`p = |\nabla\rho|/(2k_F\rho)` and
        :math:`q = \nabla^2\rho/(4k_F^2\rho)`, recomputed here from the raw
        scheme so the two paths must agree.
        """
        grid, density = cell
        gga = build_features(density, thomas_fermi_tau(density), grid,
                             scheme="gga")
        raw = build_features(density, thomas_fermi_tau(density), grid,
                             scheme="raw")

        rho, grad, laplacian = raw.features.T
        k_f = (3.0 * np.pi ** 2 * rho) ** (1.0 / 3.0)
        assert np.allclose(gga.features[:, 1], grad / (2.0 * k_f * rho))
        assert np.allclose(gga.features[:, 2],
                           laplacian / (4.0 * k_f ** 2 * rho))

    def test_p_and_q_are_homogeneous_of_the_right_degree(self, cell):
        r"""
        Under :math:`\rho \to \lambda\rho` the reduced variables must go as
        :math:`p \to \lambda^{-1/3}p` and :math:`q \to \lambda^{-2/3}q`, since
        :math:`k_F \propto \rho^{1/3}`.

        This pins the exponent structure of the definitions: a :math:`k_F`
        written without its cube root, or a :math:`q` with :math:`k_F` instead
        of :math:`k_F^2`, changes these degrees and nothing else would notice.

        (They are *invariant* under the coordinate scaling
        :math:`\rho_\lambda(r) = \lambda^3\rho(\lambda r)`, which is the
        property that makes them the natural GGA variables — but that one
        cannot be tested on a fixed grid, since it rescales the cell.)
        """
        grid, density = cell
        lam = 4.0
        base = build_features(density, thomas_fermi_tau(density), grid)
        scaled = build_features(lam * density, thomas_fermi_tau(lam * density),
                                grid)
        assert np.allclose(scaled.features[:, 1],
                           lam ** (-1.0 / 3.0) * base.features[:, 1])
        assert np.allclose(scaled.features[:, 2],
                           lam ** (-2.0 / 3.0) * base.features[:, 2])

    def test_names_and_shapes(self, cell):
        grid, density = cell
        raw = build_features(density, thomas_fermi_tau(density), grid,
                             scheme="raw")
        assert raw.feature_names == ["rho", "grad_rho", "lap_rho"]
        assert raw.features.shape == (len(raw), 3)
        enhancement = build_features(density, thomas_fermi_tau(density), grid,
                                     scheme="enhancement")
        assert enhancement.feature_names == ["p", "q"]
        assert enhancement.features.shape == (len(enhancement), 2)

    def test_rejects_an_unknown_scheme(self, cell):
        grid, density = cell
        with pytest.raises(ValueError, match="feature scheme"):
            build_features(density, thomas_fermi_tau(density), grid,
                           scheme="nonsense")

    def test_rejects_mismatched_grids(self, cell):
        grid, density = cell
        with pytest.raises(ValueError, match="share a grid"):
            build_features(density, np.ones((4, 4, 4)), grid)


class TestVacuumRegularisation:
    """
    Vacuum is where this breaks if it breaks: p and q there are ratios of two
    vanishing numbers, so they come out as noise with a plausible magnitude —
    which corrupts a fit far more quietly than a NaN would.
    """

    def test_vacuum_voxels_are_dropped(self, cell):
        grid, density = cell
        sparse = density.copy()
        sparse[:4] = 1e-12                       # well under epsilon in a.u.
        table = build_features(sparse, thomas_fermi_tau(sparse), grid)
        assert len(table) == int((sparse > 0).sum()) - sparse[:4].size

    def test_no_infinities_survive_a_zero_density(self, cell):
        """The clamp has to hold even where the mask would not reach."""
        grid, density = cell
        holed = density.copy()
        holed[0] = 0.0
        table = build_features(holed, thomas_fermi_tau(holed), grid)
        assert np.isfinite(table.features).all()
        assert np.isfinite(table.target).all()

    def test_negative_voxels_are_dropped(self, cell):
        """Band-limiting rings, and rho^(5/3) of a negative rho is complex."""
        grid, density = cell
        ringing = density.copy()
        ringing[0] = -1e-3
        table = build_features(ringing, thomas_fermi_tau(ringing), grid)
        assert np.isfinite(table.features).all()
        assert (table.features[:, 0] > 0).all()

    def test_epsilon_is_configurable(self, cell):
        grid, density = cell
        loose = build_features(density, thomas_fermi_tau(density), grid,
                               epsilon=1e-8)
        strict = build_features(density, thomas_fermi_tau(density), grid,
                                epsilon=0.05)
        assert len(strict) < len(loose)

    def test_an_all_vacuum_cell_is_an_error_not_an_empty_fit(self, cell):
        grid, density = cell
        with pytest.raises(ValueError, match="vacuum threshold"):
            build_features(density, thomas_fermi_tau(density), grid,
                           epsilon=10.0)


class TestSampling:
    def test_samples_without_replacement(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        sampled = sample_rows(table, 100, seed=0)
        assert len(sampled) == 100
        assert sampled.feature_names == table.feature_names

    def test_a_request_larger_than_the_table_is_a_no_op(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        assert sample_rows(table, len(table) * 10, seed=0) is table

    def test_sampling_is_reproducible(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        first = sample_rows(table, 50, seed=3).target
        second = sample_rows(table, 50, seed=3).target
        assert np.array_equal(first, second)

    def test_stacking_requires_one_scheme(self, cell):
        grid, density = cell
        semilocal = build_features(density, thomas_fermi_tau(density), grid)
        reduced = build_features(density, thomas_fermi_tau(density), grid,
                                 scheme="enhancement")
        with pytest.raises(ValueError, match="different schemes"):
            stack_tables([semilocal, reduced])

    def test_stacking_concatenates(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        stacked = stack_tables([table, table])
        assert len(stacked) == 2 * len(table)


# ===================================================================== #
# The search pipeline
# ===================================================================== #
class TestDistiller:
    def test_recovers_thomas_fermi_end_to_end(self, cell):
        """
        The whole path, with a deterministic engine: features -> search ->
        scoring. The exponent must come back as 5/3.
        """
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=power_law_engine).fit(table)

        exponent = float(result.expression.split("^")[1])
        coefficient = float(result.expression.split("*")[0])
        assert exponent == pytest.approx(5.0 / 3.0, rel=1e-6)
        assert coefficient == pytest.approx(C_TF, rel=1e-6)
        assert result.r2 == pytest.approx(1.0, abs=1e-9)
        assert result.relative_l2 < 1e-6

    def test_passes_the_configured_operators_through(self, cell):
        """An operator the user asked for is one the engine must receive."""
        seen = {}

        def recording_engine(features, target, names, parameters):
            seen.update(parameters)
            return [{"complexity": 1, "loss": 0.0, "expression": "1.0"}]

        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        config = SymbolicConfig(unary_operations=["sin", "cos"],
                                binary_operations=["+", "*"],
                                iterations=7, population_size=11,
                                max_depth=4, max_size=9)
        SymbolicDistiller(config, engine=recording_engine).fit(table)

        assert seen["unary_operations"] == ["sin", "cos"]
        assert seen["binary_operations"] == ["+", "*"]
        assert seen["iterations"] == 7
        assert seen["population_size"] == 11
        assert seen["max_depth"] == 4
        assert seen["max_size"] == 9

    def test_empty_operator_lists_fall_back_to_the_defaults(self, cell):
        seen = {}

        def recording_engine(features, target, names, parameters):
            seen.update(parameters)
            return [{"complexity": 1, "loss": 0.0, "expression": "1.0"}]

        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        SymbolicDistiller(SymbolicConfig(unary_operations=[],
                                         binary_operations=[]),
                          engine=recording_engine).fit(table)
        assert seen["unary_operations"] == list(DEFAULT_UNARY)
        assert seen["binary_operations"] == list(DEFAULT_BINARY)

    def test_an_empty_front_is_an_error(self, cell):
        """Reporting 'distilled' with nothing found would be a false result."""
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        with pytest.raises(ValueError, match="no expression"):
            SymbolicDistiller(SymbolicConfig(),
                              engine=lambda *a: []).fit(table)

    def test_the_best_expression_wins_on_loss(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        front = [{"complexity": 9, "loss": 5.0, "expression": "9.0"},
                 {"complexity": 3, "loss": 0.1, "expression": "3.0"}]
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=lambda *a: front).fit(table)
        assert result.expression == "3.0"
        assert result.complexity == 3
        assert [entry["complexity"] for entry in result.pareto] == [3, 9]

    def test_an_unparseable_expression_scores_nan_not_a_number(self, cell):
        """A score must never come from anything but the reported expression."""
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        result = SymbolicDistiller(
            SymbolicConfig(),
            engine=lambda *a: [{"complexity": 1, "loss": 0.0,
                                "expression": "!! not an expression !!"}],
        ).fit(table)
        assert np.isnan(result.r2)
        assert np.isnan(result.relative_l2)

    def test_summary_mentions_the_expression_and_the_fit(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid)
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=power_law_engine).fit(table)
        summary = result.summary()
        assert result.expression in summary
        assert "R2" in summary and "complexity" in summary


class TestAsymptoticLimits:
    r"""
    Pinned against functionals whose limits are known exactly.

    The discriminating property is that **neither** textbook functional passes
    both: Thomas-Fermi is the :math:`p\to0` answer and fails :math:`p\to\infty`,
    von Weizsäcker the reverse. A checker that passed both on either one would
    be measuring nothing.
    """

    def check(self, expression, scheme="reduced", template="pauli"):
        return check_asymptotic_limits(expression, ["rho", "p", "q"], scheme,
                                       template=template)

    def test_thomas_fermi_passes_only_its_own_limit(self):
        """As a Pauli factor Thomas-Fermi is 1 - 5p^2/3: right at 0, diverges."""
        result = self.check("1 - 5*p**2/3")
        assert result.thomas_fermi.passes
        assert not result.von_weizsacker.passes
        assert result.score == 0.5

    def test_von_weizsacker_passes_only_its_own_limit(self):
        """Pure von Weizsacker leaves no Pauli term at all: F = 0."""
        result = self.check("0")
        assert not result.thomas_fermi.passes
        assert result.von_weizsacker.passes
        assert result.score == 0.5

    def test_an_interpolating_form_passes_both(self):
        """1 at the origin, decaying to 0 — what a usable KEDF must do."""
        for expression in ("exp(-p**2)", "1/(1 + p**2)"):
            result = self.check(expression)
            assert result.passes, expression
            assert result.badge() == "TF/vW"

    def test_a_finite_non_zero_limit_is_bounded_but_not_compliant(self):
        """
        F -> 1 stays bounded yet never reduces to von Weizsacker. Reported
        apart from a divergence, because only this one is repairable.
        """
        result = self.check("1")
        assert result.thomas_fermi.passes
        assert not result.von_weizsacker.passes
        assert result.bounded_at_infinity
        assert result.von_weizsacker.value == pytest.approx(1.0)

    def test_the_old_convention_is_converted(self):
        r"""
        A ``thomas_fermi`` template fits :math:`\tau/\tau_{\rm TF}`, which is
        the Pauli factor plus :math:`5p^2/3`. Converting it must give the same
        verdicts as fitting the Pauli factor directly.
        """
        old = self.check("5*p**2/3", template="thomas_fermi")
        new = self.check("0", template="pauli")
        assert old.badge() == new.badge() == "--/vW"

    def test_works_on_the_gga_scheme(self):
        """tau = tau_vW + tau_TF exp(-p^2), written out in full."""
        result = self.check(
            f"{C_TF} * rho**(5/3) * (5*p**2/3 + exp(-p**2))",
            scheme="gga", template="none")
        assert result.passes

    def test_a_density_dependent_limit_fails_and_says_why(self):
        """
        `tau = 0.7 rho` gives F = 0.7 rho^(-2/3): no single F(0,0) exists, so
        it is not a functional however well it fits.
        """
        result = self.check("0.7 * rho", scheme="gga")
        assert not result.thomas_fermi.passes
        assert "rho" in result.thomas_fermi.detail

    def test_symbols_are_bound_when_parsing(self):
        """
        `sympify` mints fresh symbols; a limit taken against differently
        assumed ones silently returns the expression unchanged, which would
        read as a pass.
        """
        result = self.check("0.5 + 0.5*q")
        assert result.thomas_fermi.value == pytest.approx(0.5)

    def test_an_unparseable_expression_is_undetermined_not_passing(self):
        result = self.check("!! junk !!")
        assert not result.passes
        assert result.thomas_fermi.method == "undetermined"

    def test_tolerance_is_respected(self):
        assert check_asymptotic_limits("0.97", ["p", "q"], "enhancement",
                                       tolerance=0.05).thomas_fermi.passes
        assert not check_asymptotic_limits("0.97", ["p", "q"], "enhancement",
                                           tolerance=0.01).thomas_fermi.passes

    def test_the_real_front_is_checked_without_hanging(self):
        """A genetic search emits deeply nested expressions; SymPy must cope."""
        gnarly = ("(rho + -0.013253311) - (q / (((-0.9693314 / rho) + "
                  "(((0.6341423 / rho) + (((p ** p) * p) * exp(q))) - "
                  "-0.29007912)) / rho))")
        result = self.check(gnarly, scheme="gga")
        assert isinstance(result.score, float)


class TestComplianceInThePipeline:
    def test_every_candidate_is_checked(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        front = [{"complexity": 1, "loss": 0.5, "expression": "1"},
                 {"complexity": 5, "loss": 0.1, "expression": "exp(-p**2)"}]
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=lambda *a: front).fit(table)
        assert all("limits" in entry for entry in result.pareto)
        assert result.pareto[0]["limits"]["badge"] == "TF/--"
        assert result.pareto[1]["limits"]["badge"] == "TF/vW"

    def test_compliant_candidates_are_collected(self, cell):
        """The most accurate expression is often the least physical."""
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        front = [{"complexity": 3, "loss": 0.5, "expression": "exp(-p**2)"},
                 {"complexity": 9, "loss": 0.01, "expression": "0.4 + q"}]
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=lambda *a: front).fit(table)
        assert result.expression == "0.4 + q"          # best loss wins
        assert not result.limits["passes"]             # ... and fails physics
        assert result.compliant_expressions == ["exp(-p**2)"]

    def test_summary_shows_the_badges(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        front = [{"complexity": 3, "loss": 0.1, "expression": "exp(-p**2)"}]
        summary = SymbolicDistiller(SymbolicConfig(),
                                    engine=lambda *a: front).fit(table).summary()
        assert "TF/vW" in summary
        assert "limits" in summary

    def test_summary_says_so_when_nothing_complies(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        front = [{"complexity": 3, "loss": 0.1, "expression": "0.4 + q"}]
        summary = SymbolicDistiller(SymbolicConfig(),
                                    engine=lambda *a: front).fit(table).summary()
        assert "no candidate satisfies both limits" in summary

    def test_the_result_is_json_serialisable(self, cell):
        """The compliance travels into the run summary, so it must survive."""
        from dataclasses import asdict

        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        front = [{"complexity": 3, "loss": 0.1, "expression": "exp(-p**2)"}]
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=lambda *a: front).fit(table)
        payload = json.loads(json.dumps(asdict(result), default=float))
        assert payload["limits"]["thomas_fermi"]["passes"] is True


class TestPhysicsConstraints:
    r"""
    The limits as *fitness*, not as a filter applied afterwards.

    The property that matters is what the objective **charges** for: an
    expression violating a limit must cost more than any accuracy gain can
    recover, and one that satisfies it must cost nothing beyond its data term.
    The probe data is checked here and the charge it produces is measured in
    :class:`TestWhatTheObjectiveCharges`.
    """

    def test_probes_pin_both_limits_on_the_pauli_factor(self):
        probes = physics_probes(["p", "q"], "pauli", p_infinity=1e6)
        assert [probe["limit"] for probe in probes] == \
            ["thomas_fermi", "von_weizsacker"]

        thomas_fermi, von_weizsacker = probes
        assert thomas_fermi["point"] == {"p": 0.0, "q": 0.0}
        assert thomas_fermi["target"] == 1.0
        assert von_weizsacker["point"] == {"p": 1e6, "q": 0.0}
        assert von_weizsacker["target"] == 0.0

    def test_the_thomas_fermi_template_approaches_five_p_squared_over_three(self):
        r"""
        Under that template :math:`F = \tau/\tau_{\rm TF}`, whose large-$p$
        limit is :math:`5p^2/3` rather than zero. Comparing it against zero
        would reject every correct functional.
        """
        _, von_weizsacker = physics_probes(["p", "q"], "thomas_fermi",
                                           p_infinity=1e3)
        assert von_weizsacker["target"] == pytest.approx(5.0 / 3.0 * 1e6)
        # ... and relatively, since an absolute tolerance on 1e6 is finer than
        # the 32-bit float the engine searches in.
        assert von_weizsacker["scale"] == pytest.approx(von_weizsacker["target"])

    def test_no_limits_where_the_target_is_tau_itself(self):
        """Under `template: none` the fitted quantity is not an enhancement
        factor, so neither limit is a statement about it."""
        assert physics_probes(["rho", "p", "q"], "none") == []

    def test_no_limits_without_p_and_q_to_take_them_in(self):
        assert physics_probes(["rho", "grad_rho", "lap_rho"], "pauli") == []

    def test_the_density_is_swept_when_it_is_a_variable(self):
        """
        The limits hold at *every* density. A candidate satisfying them at one
        value and not another has not satisfied them, and sweeping turns that
        into a penalty rather than a lucky pass.
        """
        probes = physics_probes(["rho", "p", "q"], "pauli",
                               densities=(0.1, 0.5))
        assert len(probes) == 4
        assert {probe["point"]["rho"] for probe in probes} == {0.1, 0.5}

    def test_positivity_is_enforced_under_every_template(self):
        """tau >= 0 always, and tau - tau_vW >= 0 by Hoffmann-Ostenhof."""
        for template in ("none", "thomas_fermi", "pauli"):
            _, enforced = physics_constraints(["rho", "p", "q"], template)
            assert "positivity" in enforced

    def test_reports_only_the_constraints_it_can_express(self):
        """
        A limit absent because the variables cannot express it must not be
        reported as enforced — the run would otherwise claim a guarantee it
        does not deliver.
        """
        _, reduced = physics_constraints(["p", "q"], "pauli")
        _, raw = physics_constraints(["rho", "grad_rho", "lap_rho"], "none")
        assert reduced == ["positivity", "thomas_fermi", "von_weizsacker"]
        assert raw == ["positivity"]

    def test_the_probes_carry_a_coordinate_for_every_variable(self):
        """
        A probe is a point in feature space, so it has to give a value to every
        variable. One missing coordinate is not a smaller probe, it is a
        candidate that cannot be evaluated there — and a limit silently not
        checked reads exactly like a limit satisfied.
        """
        block, _ = physics_constraints(["rho", "p", "q"], "pauli",
                                       densities=(0.1,), p_infinity=1e6)
        assert [probe["point"] for probe in block["probes"]] == [
            {"rho": 0.1, "p": 0.0, "q": 0.0},
            {"rho": 0.1, "p": 1e6, "q": 0.0},
        ]

    def test_the_weights_and_the_ceiling_travel_with_them(self):
        block, _ = physics_constraints(["p", "q"], "pauli",
                                       weights={"positivity": 7.0})
        assert block["weights"]["positivity"] == 7.0
        assert block["weights"]["thomas_fermi"] == 100.0
        assert block["ceiling"] == PENALTY_CEILING

    def test_reaches_the_engine_through_the_distiller(self, cell):
        """The objective is worth nothing if it does not arrive."""
        seen = {}

        def recording_engine(features, target, names, parameters):
            seen.update(parameters)
            return [{"complexity": 1, "loss": 0.0, "expression": "1.0"}]

        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        SymbolicDistiller(SymbolicConfig(),
                          engine=recording_engine).fit(table)

        assert len(seen["physics"]["probes"]) == 2
        assert seen["physics"]["weights"]["positivity"] > 0.0
        assert seen["constraints_enforced"] == \
            ["positivity", "thomas_fermi", "von_weizsacker"]

    def test_can_be_switched_off(self, cell):
        seen = {}

        def recording_engine(features, target, names, parameters):
            seen.update(parameters)
            return [{"complexity": 1, "loss": 0.0, "expression": "1.0"}]

        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        SymbolicDistiller(SymbolicConfig(physics={"enable": False}),
                          engine=recording_engine).fit(table)

        assert seen["physics"] is None
        assert seen["constraints_enforced"] == []

    def test_the_configured_weights_reach_the_objective(self, cell):
        seen = {}

        def recording_engine(features, target, names, parameters):
            seen.update(parameters)
            return [{"complexity": 1, "loss": 0.0, "expression": "1.0"}]

        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        SymbolicDistiller(
            SymbolicConfig(physics={"enable": True,
                                    "positivity_weight": 7.0,
                                    "thomas_fermi_weight": 11.0,
                                    "von_weizsacker_weight": 13.0,
                                    "p_infinity": 1234.0}),
            engine=recording_engine).fit(table)

        physics = seen["physics"]
        assert physics["weights"] == {"positivity": 7.0,
                                      "thomas_fermi": 11.0,
                                      "von_weizsacker": 13.0}
        assert [probe["point"]["p"] for probe in physics["probes"]] == \
            [0.0, 1234.0]

    def test_the_result_records_what_was_enforced(self, cell):
        """
        Reported because a limit the variables cannot express is simply absent,
        and "constrained" would otherwise be assumed of the whole set.
        """
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        front = [{"complexity": 3, "loss": 0.1, "expression": "exp(-p**2)"}]
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=lambda *a: front).fit(table)

        assert result.constraints_enforced == \
            ["positivity", "thomas_fermi", "von_weizsacker"]
        assert "penalised in-loop" in result.summary()

    def test_an_unconstrained_run_says_so_in_the_summary(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        front = [{"complexity": 3, "loss": 0.1, "expression": "exp(-p**2)"}]
        result = SymbolicDistiller(SymbolicConfig(physics={"enable": False}),
                                   engine=lambda *a: front).fit(table)
        assert "none in-loop" in result.summary()

    def test_defaults_to_on(self):
        assert SymbolicConfig().physics["enable"] is True

    def test_a_partial_block_keeps_the_other_defaults(self):
        """
        `physics: {enable: false}` is a valid config: the accessor fills in
        the weights rather than raising a KeyError inside the search.
        """
        from poraque.ml.symbolic import symbolic_physics

        physics = symbolic_physics(SymbolicConfig(physics={"enable": False}))
        assert physics["enable"] is False
        assert physics["p_infinity"] > 0
        assert physics["positivity_weight"] > 0

    def test_it_does_not_share_names_with_the_operator_block(self):
        """
        `training.physics_informed_setup` constrains the FNO over voxels;
        `symbolic.physics`
        constrains an algebraic expression over probe points. Two key names
        appear in both and mean different things, which is why they are nested
        separately instead of sharing a prefix.
        """
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig()
        shared = (set(config.training.physics_informed_setup)
                  & set(config.symbolic.physics))
        assert shared == {"positivity_weight", "von_weizsacker_weight"}, (
            "the collision is real, which is the reason for the nesting")
        assert (config.training.physics_informed_setup
                is not config.symbolic.physics)


class TestWhatTheObjectiveCharges:
    """
    The penalties, measured rather than read.

    These assertions used to live in ``TestPhysicsConstraintsInJulia`` and were
    **skipped unless ``PORAQUE_TEST_PYSR=1``**, because they needed a Julia
    toolchain that took minutes to precompile — so in practice they ran almost
    nowhere. They are the same five statements, against
    :class:`~poraque.ml.gp.ConstrainedObjective`, and they now run every time.

    That is the real dividend of dropping the second runtime, and it is worth
    stating plainly: this repository has already been bitten once by a fixture
    that skipped everywhere and quietly let six assertions rot for a month.
    """

    @pytest.fixture
    def charge(self):
        """
        The objective's value for one expression, on limit-satisfying data.

        ``F = 1/(1 + p^2)`` has ``F(0,0) = 1`` and ``F -> 0`` as ``p -> inf``,
        so the data and both limits agree and any penalty seen is the
        objective's own doing rather than a disagreement between them.
        """
        from poraque.ml.gp import ConstrainedObjective, resolve_operations

        rng = np.random.default_rng(0)
        p = np.abs(rng.normal(0, 0.4, 200))
        features = np.column_stack([p, np.zeros_like(p)])
        target = 1.0 / (1.0 + p ** 2)
        block, _ = physics_constraints(["p", "q"], "pauli")
        unary, binary = resolve_operations(["exp"], ["+", "-", "*", "/"])
        objective = ConstrainedObjective(
            features, target, ["p", "q"], unary, binary,
            probes=block["probes"], weights=block["weights"],
            ceiling=block["ceiling"])

        def score(tree):
            return objective(tree)

        return score

    @staticmethod
    def constant(value):
        return ("const", float(value))

    @property
    def _p(self):
        return ("var", 0)

    def test_a_form_satisfying_both_limits_pays_no_penalty(self, charge):
        # 1 / (1 + p * p)
        tree = ("bin", "/", self.constant(1.0),
                ("bin", "+", self.constant(1.0),
                 ("bin", "*", self._p, self._p)))
        assert charge(tree) < 1e-6

    def test_a_constant_pays_for_the_limit_it_does_not_have(self, charge):
        """``F = 1`` has the Thomas-Fermi limit and not the von Weizsacker one."""
        assert charge(self.constant(1.0)) == pytest.approx(100.0, rel=1e-3)

    def test_a_wrong_uniform_gas_limit_is_charged(self, charge):
        """``F(0,0) = 0.5`` is off by 0.5, at weight 100."""
        tree = ("bin", "/", self.constant(0.5),
                ("bin", "+", self.constant(1.0),
                 ("bin", "*", self._p, self._p)))
        assert charge(tree) == pytest.approx(50.0, rel=1e-2)

    def test_a_divergent_form_is_charged_the_ceiling(self, charge):
        """``F = p^2`` diverges at ``p = 1e6`` and must be decisively rejected."""
        assert charge(("bin", "*", self._p, self._p)) > 1e6

    def test_a_negative_prediction_is_charged(self, charge):
        assert charge(self.constant(-1.0)) > charge(self.constant(1.0))

    def test_one_limit_carries_one_weight_however_many_probes_express_it(self):
        """
        The density sweep of the ``gga`` scheme raises the *resolution* of a
        check, not its price. Three densities means six probes and still two
        limits, so a functional that satisfies both is not charged three times
        for a Thomas-Fermi limit it has.
        """
        from poraque.ml.gp import ConstrainedObjective, resolve_operations

        block, _ = physics_constraints(
            ["rho", "p", "q"], "pauli", densities=(0.1, 0.5, 0.9),
            weights={"thomas_fermi": 300.0, "von_weizsacker": 300.0})
        unary, binary = resolve_operations([], ["+", "*"])
        objective = ConstrainedObjective(
            np.ones((4, 3)), np.ones(4), ["rho", "p", "q"], unary, binary,
            probes=block["probes"], weights=block["weights"])
        assert len(block["probes"]) == 6
        assert objective.probe_weights.tolist() == [100.0] * 6

    def test_the_data_term_is_selectable_and_checked(self):
        from poraque.ml.gp import ConstrainedObjective, resolve_operations

        unary, binary = resolve_operations([], ["+"])
        features, target = np.zeros((3, 1)), np.array([1.0, 2.0, 3.0])
        squared = ConstrainedObjective(features, target, ["p"], unary, binary,
                                       data_loss="mse")
        absolute = ConstrainedObjective(features, target, ["p"], unary, binary,
                                        data_loss="mae")
        zero = ("const", 0.0)
        assert squared(zero) == pytest.approx((1 + 4 + 9) / 3)
        assert absolute(zero) == pytest.approx((1 + 2 + 3) / 3)

        with pytest.raises(ValueError, match="Unknown data loss"):
            ConstrainedObjective(features, target, ["p"], unary, binary,
                                 data_loss="huber")


class TestLatex:
    def test_renders_an_expression(self):
        rendered = expression_to_latex("2.871234 * rho^1.666667", ["rho"])
        assert r"\rho" in rendered

    def test_falls_back_rather_than_losing_the_result(self):
        """An unparseable expression is still the answer."""
        rendered = expression_to_latex("!! nonsense !!", ["rho"])
        assert r"\texttt{" in rendered

    def test_survives_a_backslash(self):
        assert expression_to_latex("a \\ b", ["a", "b"])


class TestConfig:
    def test_is_off_by_default(self):
        assert SymbolicConfig().enable_symbolic_distillation is False

    def test_round_trips_through_the_training_config(self, tmp_path):
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig()
        config.symbolic.enable_symbolic_distillation = True
        config.symbolic.unary_operations = ["sin"]
        path = tmp_path / "config.yaml"
        config.to_yaml(str(path))

        restored = TrainingConfig.from_yaml(str(path))
        assert restored.symbolic.enable_symbolic_distillation is True
        assert restored.symbolic.unary_operations == ["sin"]

    def test_rejects_an_unknown_symbolic_key(self):
        from poraque.ml.config import TrainingConfig

        with pytest.raises(ValueError, match="symbolic"):
            TrainingConfig.from_dict({"symbolic": {"iteratons": 10}})


class TestExpressionRounding:
    """
    A search returns constants at full float precision.

    ``0.33333334326744079589843750`` is a real example, and three of those in
    one formula overrun the width of a PDF page. Three places is also what the
    numbers are worth: the search is stochastic and its constants move in the
    third place between seeds.
    """

    def test_constants_are_rounded(self):
        from poraque.ml.symbolic import round_expression

        rounded = round_expression(
            "0.33333334326744079589843750*p**2 + 1.9999999523162841796875",
            feature_names=["p"])
        assert "0.333" in rounded
        assert "0.33333334" not in rounded

    def test_latex_rounds_by_default(self):
        from poraque.ml.symbolic import expression_to_latex

        latex = expression_to_latex("0.142857142857142857*p", ["p"])
        assert "0.143" in latex and "0.1428571" not in latex

    def test_rounding_can_be_switched_off(self):
        from poraque.ml.symbolic import expression_to_latex

        latex = expression_to_latex("0.142857142857142857*p", ["p"],
                                    decimals=None)
        assert "0.1428571" in latex

    def test_an_unparsable_expression_survives(self):
        """A display convenience must never lose the result."""
        from poraque.ml.symbolic import round_expression

        assert round_expression("this is not ) an expression") == \
            "this is not ) an expression"

    def test_integers_are_untouched(self):
        from poraque.ml.symbolic import round_expression

        assert "2" in round_expression("2*p + 3", feature_names=["p"])


class TestParetoKnee:
    r"""
    The knee: the front entry nearest the ideal corner :math:`(0, 0)` once
    complexity and log-loss are each rescaled to :math:`[0, 1]`.
    """

    FRONT = [
        {"complexity": 1, "loss": 1.0},
        {"complexity": 3, "loss": 0.1},
        {"complexity": 7, "loss": 0.02},
        {"complexity": 20, "loss": 0.019},
    ]

    def test_it_finds_the_elbow(self):
        """
        Complexity 20 buys 0.001 over complexity 7. The knee is 7.
        """
        from poraque.ml.symbolic import pareto_knee

        assert pareto_knee(self.FRONT)["complexity"] == 7

    def test_the_distance_is_written_onto_the_front(self):
        """
        So the report's table can show the column it ranked on, and the figure
        and the table cannot disagree.
        """
        from poraque.ml.symbolic import pareto_knee

        front = [dict(entry) for entry in self.FRONT]
        knee = pareto_knee(front)
        assert all("distance" in entry for entry in front)
        assert min(entry["distance"] for entry in front) == pytest.approx(
            knee["distance"])

    def test_the_loss_is_compared_logarithmically(self):
        """
        On a linear axis a front spanning decades collapses: every candidate
        but the most accurate sits at ~1, and the knee becomes the shortest
        expression whatever it costs. This front is exactly that case.
        """
        from poraque.ml.symbolic import pareto_knee

        front = [{"complexity": 1, "loss": 1.0},
                 {"complexity": 5, "loss": 1e-3},
                 {"complexity": 9, "loss": 1e-6}]
        assert pareto_knee(front)["complexity"] == 5, (
            "a linear comparison would pick the 1-node expression")

    def test_a_single_candidate_is_its_own_knee(self):
        from poraque.ml.symbolic import pareto_knee

        assert pareto_knee([{"complexity": 4, "loss": 0.1}])["complexity"] == 4

    def test_an_empty_front_gives_nothing(self):
        from poraque.ml.symbolic import pareto_knee

        assert pareto_knee([]) == {}

    def test_non_finite_losses_are_skipped(self):
        from poraque.ml.symbolic import pareto_knee

        front = [{"complexity": 1, "loss": float("inf")},
                 {"complexity": 5, "loss": 0.1}]
        assert pareto_knee(front)["complexity"] == 5

    def test_it_is_marked_as_the_knee(self):
        from poraque.ml.symbolic import pareto_knee

        assert pareto_knee(self.FRONT)["knee"] is True

    def test_the_result_carries_it(self, cell):
        """The distiller computes the knee with the front, not separately."""
        from poraque.ml.symbolic import SymbolicDistiller, build_features

        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        front = [{"complexity": 1, "loss": 0.5, "expression": "1.0"},
                 {"complexity": 5, "loss": 0.01, "expression": "exp(-p)"},
                 {"complexity": 15, "loss": 0.009, "expression": "exp(-p*p)"}]
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=lambda *a: front).fit(table)
        assert result.knee.get("complexity") == 5
        assert result.knee_expression() == "exp(-p)"


class TestResultIsSerialisable:
    """
    The summary is written *after* training, the report and the figures.

    A field that ``json.dump`` cannot encode therefore destroys the record of a
    run that already succeeded — which is what happened when the knee sections
    were added carrying the same voxel arrays as ``fitted`` and ``validation``
    while the stripper kept a hand-written list of two key names.
    """

    @staticmethod
    def _result():
        import numpy as np

        from poraque.ml.symbolic import SymbolicResult

        result = SymbolicResult(expression="1-p", latex="1-p", complexity=3,
                                loss=0.1, r2=0.9, relative_l2=0.1)
        scored = {"n_points": 10, "relative_l2": 0.1, "r2": 0.9,
                  "predicted": np.arange(5.0), "reference": np.arange(5.0)}
        result.fitted = dict(scored)
        result.validation = dict(scored)
        result.knee_fitted = dict(scored)
        result.knee_validation = dict(scored)
        result.knee = {"complexity": 2, "loss": 0.2, "expression": "1"}
        result.pareto = [{"complexity": 2, "loss": 0.2, "expression": "1",
                          "distance": 0.3}]
        return result

    def test_it_encodes_as_json(self):
        import json

        from poraque.ml.symbolic import result_to_dict

        json.dumps(result_to_dict(self._result()), default=float)

    def test_no_array_survives_anywhere(self):
        """
        Checked over *every* section rather than a named few, because the
        named-few version is the bug.
        """
        import numpy as np

        from poraque.ml.symbolic import result_to_dict

        payload = result_to_dict(self._result())

        def arrays(node, path=""):
            if isinstance(node, np.ndarray):
                return [path]
            if isinstance(node, dict):
                return [p for key, value in node.items()
                        for p in arrays(value, f"{path}.{key}")]
            if isinstance(node, (list, tuple)):
                return [p for i, value in enumerate(node)
                        for p in arrays(value, f"{path}[{i}]")]
            return []

        assert arrays(payload) == []

    def test_the_scores_themselves_survive(self):
        """Stripping the arrays must not take the metrics with them."""
        from poraque.ml.symbolic import result_to_dict

        payload = result_to_dict(self._result())
        for key in ("fitted", "validation", "knee_fitted", "knee_validation"):
            assert payload[key]["relative_l2"] == pytest.approx(0.1)
            assert "predicted" not in payload[key]
