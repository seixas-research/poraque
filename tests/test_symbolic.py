# -*- coding: utf-8 -*-
# file: test_symbolic.py

"""
Tests for symbolic distillation.

The engine (PySR) is optional and stochastic, so it is injected here rather
than run. That is not a gap: everything the package is responsible for —
building the features, the units they are built in, sampling, scoring, LaTeX —
is deterministic and lives outside the search.

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
    FeatureTable,
    SymbolicDistiller,
    build_features,
    expression_to_latex,
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

    def test_enhancement_is_unity_for_thomas_fermi(self, cell):
        """Thomas-Fermi is F = 1 by definition; anything else is a bug."""
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        assert np.allclose(table.target, 1.0, atol=1e-10)

    def test_enhancement_gives_the_von_weizsacker_form(self, cell):
        r""":math:`\tau_{\rm vW}/\tau_{\rm TF} = 5p^2/3` exactly."""
        grid, density = cell
        table = build_features(density, von_weizsacker_tau(density, grid), grid,
                               scheme="enhancement")
        p = table.features[:, 0]
        assert np.allclose(table.target, 5.0 * p ** 2 / 3.0, atol=1e-10)

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

    def check(self, expression, scheme="enhancement"):
        return check_asymptotic_limits(expression, ["rho", "p", "q"], scheme)

    def test_thomas_fermi_passes_only_its_own_limit(self):
        result = self.check("1")
        assert result.thomas_fermi.passes
        assert not result.von_weizsacker.passes
        assert result.score == 0.5

    def test_von_weizsacker_passes_only_its_own_limit(self):
        result = self.check("5*p**2/3")
        assert not result.thomas_fermi.passes
        assert result.von_weizsacker.passes
        assert result.score == 0.5

    def test_an_interpolating_form_passes_both(self):
        result = self.check("1 + 5*p**2/3")
        assert result.passes
        assert result.score == 1.0
        assert result.badge() == "TF/vW"

    def test_gradient_expansion_has_the_scaling_but_not_the_coefficient(self):
        r"""
        The second-order gradient expansion goes as :math:`p^2` with
        coefficient :math:`1/9`. Right shape, wrong size — and reporting it as
        "von Weizsäcker satisfied" would be wrong, so the two are separate.
        """
        result = self.check("1 + 5*p**2/27 + 20*q/9")
        assert result.thomas_fermi.passes
        assert not result.von_weizsacker.passes
        assert result.quadratic_scaling
        assert result.von_weizsacker.value == pytest.approx(1.0 / 9.0, rel=1e-6)

    def test_works_on_the_gga_scheme(self):
        """tau = C_TF rho^(5/3) (1 + 5p^2/3) is the same functional."""
        result = self.check(f"{C_TF} * rho**(5/3) * (1 + 5*p**2/3)",
                            scheme="gga")
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
                 {"complexity": 5, "loss": 0.1, "expression": "1 + 5*p**2/3"}]
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
        front = [{"complexity": 3, "loss": 0.5, "expression": "1 + 5*p**2/3"},
                 {"complexity": 9, "loss": 0.01, "expression": "0.4 + q"}]
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=lambda *a: front).fit(table)
        assert result.expression == "0.4 + q"          # best loss wins
        assert not result.limits["passes"]             # ... and fails physics
        assert result.compliant_expressions == ["1 + 5*p**2/3"]

    def test_summary_shows_the_badges(self, cell):
        grid, density = cell
        table = build_features(density, thomas_fermi_tau(density), grid,
                               scheme="enhancement")
        front = [{"complexity": 3, "loss": 0.1, "expression": "1 + 5*p**2/3"}]
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
        front = [{"complexity": 3, "loss": 0.1, "expression": "1 + 5*p**2/3"}]
        result = SymbolicDistiller(SymbolicConfig(),
                                   engine=lambda *a: front).fit(table)
        payload = json.loads(json.dumps(asdict(result), default=float))
        assert payload["limits"]["thomas_fermi"]["passes"] is True


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
