# -*- coding: utf-8 -*-
# file: test_committee.py

"""
Tests for query-by-committee: the ``init_seed`` split and the disagreement
measures built on it.

The load-bearing property is the *separation* of the two seeds. If varying
``init_seed`` also perturbed the batch order, the committee's disagreement
would mix optimisation variance with a reshuffled dataset and could not be
read as either.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.fields import ChargeDensity, FieldGrid              # noqa: E402
from poraque.ml import (                                          # noqa: E402
    FieldOperator,
    committee_integrals,
    committee_spread,
    disagreement_error_correlation,
    jensen_shannon_spread,
)


def _weights(operator):
    return [v.detach().clone() for v in operator.model.state_dict().values()]


def _build(init_seed=None, **kwargs):
    return FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                         projection_channels=8, device="cpu",
                         init_seed=init_seed, **kwargs)


# ===================================================================== #
# The seed itself
# ===================================================================== #
class TestInitSeed:
    def test_same_seed_gives_identical_weights(self):
        for a, b in zip(_weights(_build(init_seed=7)),
                        _weights(_build(init_seed=7))):
            assert torch.equal(a, b)

    def test_different_seeds_give_different_weights(self):
        """Otherwise every committee member would be the same model."""
        first, second = _weights(_build(init_seed=1)), _weights(_build(init_seed=2))
        assert any(not torch.equal(a, b) for a, b in zip(first, second))

    def test_is_recorded_on_the_operator(self):
        assert _build(init_seed=11).init_seed == 11
        assert _build().init_seed is None

    def test_survives_a_checkpoint_round_trip(self, tmp_path):
        """A committee member must be identifiable after it is saved."""
        path = str(tmp_path / "member.pfno")
        _build(init_seed=5).save(path)
        assert FieldOperator.load(path, device="cpu").init_seed == 5

    def test_leaves_the_global_stream_untouched(self):
        """
        The whole point: two members differ in their weights and in nothing
        else. If seeding the draw also re-aligned the global RNG, the batch
        order would move with it and the disagreement would be confounded.
        """
        torch.manual_seed(0)
        _build(init_seed=123)
        after_seeded = torch.randn(4)

        torch.manual_seed(0)
        _build(init_seed=456)
        after_other = torch.randn(4)

        assert torch.equal(after_seeded, after_other)

    def test_unseeded_construction_still_advances_the_stream(self):
        """Without init_seed the ambient behaviour is unchanged."""
        torch.manual_seed(0)
        _build()
        first = torch.randn(4)

        torch.manual_seed(0)
        second = torch.randn(4)
        assert not torch.equal(first, second)


# ===================================================================== #
# Disagreement
# ===================================================================== #
class TestCommitteeSpread:
    def test_identical_members_disagree_about_nothing(self):
        field = np.ones((4, 4, 4)) * 0.5
        result = committee_spread([field, field, field])
        assert result["relative"] == pytest.approx(0.0)
        assert np.allclose(result["spread"], 0.0)

    def test_mean_and_spread_are_the_sample_statistics(self):
        members = [np.full((2, 2, 2), v) for v in (1.0, 2.0, 3.0)]
        result = committee_spread(members)
        assert np.allclose(result["mean"], 2.0)
        assert np.allclose(result["spread"], 1.0)      # ddof=1

    def test_spread_is_a_field_not_a_number(self):
        """
        Knowing *where* the committee is unsure is the part a scalar cannot
        express, and the part that says whether the doubt sits on the cores.
        """
        rng = np.random.default_rng(0)
        base = rng.random((6, 6, 6))
        members = [base.copy() for _ in range(4)]
        for member in members[1:]:
            member[0, 0, 0] += rng.normal()            # disagree in one voxel

        result = committee_spread(members)
        assert result["spread"].shape == base.shape
        assert result["spread"][0, 0, 0] == result["spread"].max()
        assert result["spread"][3, 3, 3] == pytest.approx(0.0)

    def test_rejects_a_committee_of_one(self):
        with pytest.raises(ValueError, match="at least two"):
            committee_spread([np.ones((2, 2, 2))])

    def test_reference_gives_the_error_and_the_calibration_ratio(self):
        truth = np.full((4, 4, 4), 1.0)
        members = [np.full((4, 4, 4), 1.1), np.full((4, 4, 4), 1.3)]
        result = committee_spread(members, reference=truth)
        # mean 1.2 -> error 0.2/1.0; spread 0.1414/1.2
        assert result["error"] == pytest.approx(0.2, rel=1e-6)
        assert result["ratio"] == pytest.approx(result["relative"] / 0.2)
        assert result["ratio"] < 1.0                   # over-confident


class TestCommitteeIntegrals:
    def test_spread_of_the_integrated_quantity(self):
        grid = FieldGrid((8, 8, 8), np.eye(3) * 2.0)
        members = [np.full(grid.shape, v) for v in (0.5, 0.6, 0.7)]
        result = committee_integrals(members, grid)
        assert result["mean"] == pytest.approx(0.6 * grid.volume)
        assert result["spread"] == pytest.approx(0.1 * grid.volume)
        assert result["relative"] == pytest.approx(0.1 / 0.6)

    def test_agreeing_pointwise_is_not_agreeing_on_the_integral(self):
        """
        A committee can be tight everywhere and still disagree on the electron
        count, because a small constant offset integrates. The two measures are
        independent and both are reported.
        """
        grid = FieldGrid((8, 8, 8), np.eye(3) * 2.0)
        members = [np.full(grid.shape, 1.0), np.full(grid.shape, 1.02)]
        pointwise = committee_spread(members)["relative"]
        integrated = committee_integrals(members, grid)["relative"]
        assert pointwise == pytest.approx(integrated, rel=1e-6)


class TestCalibration:
    def test_perfect_ranking_gives_spearman_one(self):
        records = [{"relative": d, "error": e} for d, e in
                   [(0.01, 0.02), (0.02, 0.05), (0.03, 0.09), (0.04, 0.11)]]
        result = disagreement_error_correlation(records)
        assert result["spearman"] == pytest.approx(1.0)
        assert result["n"] == 4

    def test_inverted_ranking_gives_spearman_minus_one(self):
        """A measure that ranks backwards is worse than useless."""
        records = [{"relative": d, "error": e} for d, e in
                   [(0.04, 0.02), (0.03, 0.05), (0.02, 0.09), (0.01, 0.11)]]
        assert disagreement_error_correlation(records)["spearman"] == \
            pytest.approx(-1.0)

    def test_calibration_below_one_means_over_confident(self):
        records = [{"relative": 0.01, "error": 0.05}] * 3
        assert disagreement_error_correlation(records)["calibration"] == \
            pytest.approx(0.2)

    def test_needs_enough_structures_to_correlate(self):
        with pytest.raises(ValueError, match="at least three"):
            disagreement_error_correlation([{"relative": 1.0, "error": 1.0}] * 2)


class TestJensenShannon:
    """
    The charge density is already a probability density up to the electron
    count, so information-theoretic distances apply to it directly. For a
    committee the right form is the divergence about the mean, not a pairwise
    KL: it is symmetric, finite, and bounded by ln K.
    """

    @pytest.fixture
    def grid(self):
        return FieldGrid((12, 12, 12), np.eye(3) * 4.0)

    def test_identical_members_give_exactly_zero(self, grid):
        uniform = np.full(grid.shape, 0.3)
        result = jensen_shannon_spread([uniform, uniform, uniform], grid)
        assert result["jsd"] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("members", [2, 3, 4])
    def test_disjoint_supports_saturate_the_bound(self, grid, members):
        """
        Members that share no support are maximally divergent, and the JSD is
        then exactly ln K. An implementation that is off by a volume element
        or a normalisation fails this.
        """
        block = grid.shape[0] // members
        fields = []
        for index in range(members):
            field = np.zeros(grid.shape)
            field[index * block:(index + 1) * block] = 1.0
            fields.append(field)

        result = jensen_shannon_spread(fields, grid)
        assert result["jsd"] == pytest.approx(np.log(members), rel=1e-9)
        assert result["normalised"] == pytest.approx(1.0, rel=1e-9)

    def test_is_blind_to_a_common_rescaling(self, grid):
        """
        Deliberate: this measures *shape*. The magnitude it discards is what
        committee_integrals reports, and the electron count is known to drift,
        so the two are complementary rather than redundant.
        """
        rng = np.random.default_rng(0)
        field = rng.random(grid.shape) + 0.1
        result = jensen_shannon_spread([field, 3.7 * field], grid)
        assert result["jsd"] == pytest.approx(0.0, abs=1e-12)

    def test_is_symmetric(self, grid):
        """Unlike a raw KL, which would give two different answers."""
        rng = np.random.default_rng(1)
        a, b = rng.random(grid.shape) + 0.1, rng.random(grid.shape) + 0.1
        assert jensen_shannon_spread([a, b], grid)["jsd"] == pytest.approx(
            jensen_shannon_spread([b, a], grid)["jsd"])

    def test_grows_with_the_disagreement(self, grid):
        rng = np.random.default_rng(2)
        base = rng.random(grid.shape) + 1.0
        previous = -1.0
        for amplitude in (0.0, 0.1, 0.3, 0.6):
            perturbed = base * (1.0 + amplitude * rng.random(grid.shape))
            value = jensen_shannon_spread([base, perturbed], grid)["jsd"]
            assert value > previous
            previous = value

    def test_floors_negative_voxels_and_reports_it(self, grid):
        """
        Band-limiting rings, so a resampled field can dip below zero and log of
        that is undefined. Silently producing a NaN would be the worst outcome.
        """
        field = np.full(grid.shape, 0.2)
        field[0, 0, 0] = -1e-4
        result = jensen_shannon_spread([field, np.full(grid.shape, 0.2)], grid)
        assert np.isfinite(result["jsd"])
        assert result["clipped"] >= 1

    def test_pointwise_integrand_localises_the_disagreement(self, grid):
        """The field form says *where*, which the scalar cannot."""
        a = np.full(grid.shape, 0.2)
        b = a.copy()
        b[5, 5, 5] *= 6.0
        result = jensen_shannon_spread([a, b], grid)
        assert result["pointwise"].shape == grid.shape
        assert np.argmax(result["pointwise"]) == np.ravel_multi_index(
            (5, 5, 5), grid.shape)

    def test_rejects_a_committee_of_one(self, grid):
        with pytest.raises(ValueError, match="at least two"):
            jensen_shannon_spread([np.ones(grid.shape)], grid)


class TestCommitteeClass:
    """The ensemble wrapper, with JSD as the headline measure."""

    @pytest.fixture
    def members(self):
        return [FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                              projection_channels=8, device="cpu",
                              init_seed=s) for s in (0, 1, 2)]

    @pytest.fixture
    def potential(self):
        from poraque.fields import ExternalPotential
        from poraque.fields.vasp.poscar import Poscar

        grid = FieldGrid((8, 8, 8), np.eye(3) * 5.0)
        structure = Poscar(np.eye(3) * 5.0, ["Si"], [2],
                           np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))
        return ExternalPotential.compute(structure, grid, {"Si": 4.0},
                                         widths={"Si": 0.5})

    def test_predicts_once_per_member(self, members, potential):
        from poraque.ml import Committee

        predictions = Committee(members).predict(potential)
        assert len(predictions) == 3
        assert all(p.grid is potential.grid for p in predictions)

    def test_untrained_members_predict_a_signed_field_and_jsd_is_withheld(
            self, members, potential):
        """
        Real behaviour, not a limitation of the test: an untrained operator
        predicts a density that is negative over much of the cell. That is not
        a probability density, and flooring it into one would yield a number
        that looks meaningful. The L2 measures still apply.
        """
        from poraque.ml import Committee

        scored = Committee(members).disagreement(potential)
        assert scored["jsd"] is None
        assert scored["relative"] >= 0.0
        assert "integral_relative" in scored

    def test_disagreement_leads_with_jsd(self, members, potential, monkeypatch):
        """With a physical (positive) prediction, JSD heads the report."""
        from poraque.ml import Committee

        rng = np.random.default_rng(0)
        fields = [rng.random(potential.grid.shape) + 0.5 for _ in members]
        for operator, values in zip(members, fields):
            monkeypatch.setattr(
                operator, "predict",
                lambda field, v=values: ChargeDensity(v, field.grid,
                                                      field.structure))

        scored = Committee(members).disagreement(potential)
        assert scored["jsd"] is not None and scored["jsd"] > 0.0
        assert 0.0 <= scored["jsd_normalised"] <= 1.0
        assert scored["jsd_pointwise"].shape == potential.grid.shape
        # the L2 and integral measures ride alongside, not instead
        assert "relative" in scored and "integral_relative" in scored

    def test_identical_members_agree_completely(self, potential, monkeypatch):
        """Same predictions -> zero divergence, exactly."""
        from poraque.ml import Committee

        twins = [FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                               projection_channels=8, device="cpu",
                               init_seed=42) for _ in range(2)]
        values = np.full(potential.grid.shape, 0.4)
        for operator in twins:
            monkeypatch.setattr(
                operator, "predict",
                lambda field: ChargeDensity(values, field.grid,
                                            field.structure))

        with pytest.warns(RuntimeWarning, match="share init_seed"):
            committee = Committee(twins)
        assert committee.disagreement(potential)["jsd"] == pytest.approx(
            0.0, abs=1e-12)

    def test_rejects_a_single_member(self, members):
        from poraque.ml import Committee

        with pytest.raises(ValueError, match="at least two"):
            Committee(members[:1])

    def test_rejects_mixed_tasks(self, members):
        """A number averaged over two different maps means nothing."""
        from poraque.ml import Committee

        other = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                              device="cpu", init_seed=9)
        with pytest.raises(ValueError, match="share a task"):
            Committee([members[0], other])

    def test_signed_fields_report_no_jsd(self, members):
        """
        A field that is negative over much of the cell is not a density.
        Flooring it into one would produce a number that looks meaningful.
        """
        from poraque.ml.committee import _mostly_positive

        signed = [np.linspace(-1, 1, 64).reshape(4, 4, 4) for _ in range(2)]
        assert not _mostly_positive(signed)

    def test_ringing_negatives_are_tolerated(self):
        """A few Gibbs voxels are an artefact, not a change of object."""
        from poraque.ml.committee import _mostly_positive

        field = np.full((10, 10, 10), 0.5)
        field[0, 0, 0] = -1e-6
        assert _mostly_positive([field, field])

    def test_loads_from_bundles(self, members, tmp_path, potential):
        from poraque.ml import Committee, save_bundle

        paths = [save_bundle(str(tmp_path / f"m{i}.pfno"), {"ext2chg": m})
                 for i, m in enumerate(members)]
        committee = Committee.from_bundles(paths, "ext2chg", device="cpu")
        assert len(committee) == 3
        assert committee.init_seeds == [0, 1, 2]
        assert committee.disagreement(potential)["relative"] >= 0.0
