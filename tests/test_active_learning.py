# -*- coding: utf-8 -*-
# file: test_active_learning.py

"""
Tests for the active-learning round: score a pool, rank it, select the top K.

Two properties carry the module. The first is that the **ranking is by
disagreement** — a candidate the committee argues about must outrank one it
agrees on, since that ordering is the only thing a DFT budget is spent from.
The second is that the loop never mutates the filesystem it was not told to:
a scoring run that quietly moves directories out of a pool is unrecoverable.
"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.fields import ChargeDensity, ExternalPotential, FieldGrid  # noqa: E402
from poraque.fields.vasp.poscar import Poscar                           # noqa: E402
from poraque.ml import FieldOperator                                    # noqa: E402
from poraque.ml.active_learning import (                                # noqa: E402
    discover_pool,
    format_ranking,
    format_statistics,
    jsd_statistics,
    promote,
    round_to_dict,
    run_round,
    score_candidate,
    score_pool,
    select_top_k,
)
from poraque.ml.committee import Committee                              # noqa: E402


# ===================================================================== #
# Fixtures: a pool on disk and a committee that predicts what we tell it
# ===================================================================== #
@pytest.fixture
def grid():
    return FieldGrid((8, 8, 8), np.eye(3) * 5.0)


@pytest.fixture
def structure():
    return Poscar(np.eye(3) * 5.0, ["Si"], [2],
                  np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))


@pytest.fixture
def pool(tmp_path, grid, structure):
    """Four candidates carrying only an EXTCAR — an unlabelled pool."""
    root = tmp_path / "pool"
    rng = np.random.default_rng(0)
    for name in ("mp-001", "mp-002", "mp-003", "mp-004"):
        directory = root / name
        directory.mkdir(parents=True)
        ExternalPotential(rng.normal(size=grid.shape), grid,
                          structure).write(str(directory / "EXTCAR"))
    return str(root)


@pytest.fixture
def members():
    return [FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                          projection_channels=8, device="cpu", init_seed=s)
            for s in (0, 1, 2)]


def scripted(committee, monkeypatch, per_material):
    """
    Make each member predict a chosen field, keyed by the structure's name.

    Untrained operators predict a signed field, for which the divergence is
    undefined by design. Scripting the predictions is what lets the *ranking*
    be tested against a known answer rather than against noise.
    """
    def install(operator, index):
        def predict(field, index=index):
            name = field.metadata["material"]
            return ChargeDensity(per_material[name][index], field.grid,
                                 field.structure)
        monkeypatch.setattr(operator, "predict", predict)

    for index, operator in enumerate(committee.operators):
        install(operator, index)


@pytest.fixture
def tagged_load(monkeypatch):
    """Carry the material name onto the loaded field, for `scripted`."""
    from poraque.ml import active_learning

    original = active_learning.load_input

    def load_input(record, task):
        field = original(record, task)
        field.metadata["material"] = record.identifier
        return field

    monkeypatch.setattr(active_learning, "load_input", load_input)


# ===================================================================== #
# The pool
# ===================================================================== #
class TestDiscoverPool:
    def test_finds_candidates_carrying_only_the_input_field(self, pool):
        """
        The whole point of an unlabelled pool: the target is what the DFT run
        would produce. Requiring a complete pair would find nothing.
        """
        found = discover_pool(pool, "ext2chg")
        assert [record.identifier for record in found] == \
            ["mp-001", "mp-002", "mp-003", "mp-004"]

    def test_excludes_what_is_already_in_training(self, pool):
        found = discover_pool(pool, "ext2chg", exclude={"mp-002", "mp-004"})
        assert [r.identifier for r in found] == ["mp-001", "mp-003"]

    def test_a_chg2tau_pool_needs_a_density_not_a_potential(self, pool):
        """The task selects which field a candidate must carry."""
        assert discover_pool(pool, "chg2tau") == []


# ===================================================================== #
# Scoring
# ===================================================================== #
class TestScoreCandidate:
    def test_agreeing_members_score_zero(self, members, grid, structure,
                                         monkeypatch):
        field = ExternalPotential(np.zeros(grid.shape), grid, structure)
        values = np.full(grid.shape, 0.4)
        for operator in members:
            monkeypatch.setattr(
                operator, "predict",
                lambda f, v=values: ChargeDensity(v, f.grid, f.structure))

        record = score_candidate(Committee(members), field)
        assert record["jsd"] == pytest.approx(0.0, abs=1e-12)
        assert record["n_members"] == 3

    def test_disagreeing_members_score_more(self, members, grid, structure,
                                            monkeypatch):
        """
        Monotone in the *shape* disagreement, which is what the measure is for.
        A member scaled up uniformly is deliberately not a disagreement — see
        the electron-count test below.
        """
        field = ExternalPotential(np.zeros(grid.shape), grid, structure)
        rng = np.random.default_rng(1)
        base = rng.random(grid.shape) + 0.5

        def score(amplitude):
            for index, operator in enumerate(members):
                values = base.copy()
                values[index] += amplitude     # each member peaks elsewhere
                monkeypatch.setattr(
                    operator, "predict",
                    lambda f, v=values: ChargeDensity(v, f.grid, f.structure))
            return score_candidate(Committee(members), field)["jsd"]

        assert score(0.0) < score(0.4) < score(2.0)

    def test_the_l2_and_integral_spreads_ride_alongside(self, members, grid,
                                                        structure, monkeypatch):
        """
        The JSD normalises each member, so it is blind to the electron-count
        drift. Reporting it alone would hide exactly the quantity the energy
        needs.
        """
        field = ExternalPotential(np.zeros(grid.shape), grid, structure)
        for index, operator in enumerate(members):
            values = np.full(grid.shape, 0.4 * (1.0 + 0.1 * index))
            monkeypatch.setattr(
                operator, "predict",
                lambda f, v=values: ChargeDensity(v, f.grid, f.structure))

        record = score_candidate(Committee(members), field)
        assert record["jsd"] == pytest.approx(0.0, abs=1e-12)   # same shape
        assert record["integral_relative"] > 0.0                # different N

    def test_a_signed_prediction_withholds_the_jsd(self, members, grid,
                                                   structure):
        """
        Real behaviour: an untrained operator predicts a density negative over
        much of the cell. That is not a probability distribution, and flooring
        it into one would return a number that looks meaningful.
        """
        field = ExternalPotential(np.zeros(grid.shape), grid, structure)
        record = score_candidate(Committee(members), field)
        assert record["jsd"] is None
        assert record["relative"] >= 0.0

    def test_keeps_no_grid_sized_array(self, members, grid, structure,
                                       monkeypatch):
        """
        The memory contract. One pointwise field per candidate is what turns a
        pool sweep into a bottleneck, so a scored candidate keeps scalars only.
        """
        field = ExternalPotential(np.zeros(grid.shape), grid, structure)
        values = np.full(grid.shape, 0.4)
        for operator in members:
            monkeypatch.setattr(
                operator, "predict",
                lambda f, v=values: ChargeDensity(v, f.grid, f.structure))

        record = score_candidate(Committee(members), field)
        assert not any(isinstance(value, np.ndarray)
                       for value in record.values())

    def test_predicts_with_gradients_disabled(self, members, grid, structure,
                                              monkeypatch):
        """
        Stated here rather than assumed from the decorator on predict(): a
        retained graph in this loop is M times a 3D field per candidate.
        """
        field = ExternalPotential(np.zeros(grid.shape), grid, structure)
        seen = []
        values = np.full(grid.shape, 0.4)
        for operator in members:
            def predict(f, v=values):
                seen.append(torch.is_grad_enabled())
                return ChargeDensity(v, f.grid, f.structure)
            monkeypatch.setattr(operator, "predict", predict)

        score_candidate(Committee(members), field)
        assert seen and not any(seen)

    def test_residency_one_still_scores_the_same(self, members, grid,
                                                 structure, monkeypatch):
        """It is a memory knob. If it moved the numbers it would be a bug."""
        field = ExternalPotential(np.zeros(grid.shape), grid, structure)
        rng = np.random.default_rng(5)
        for operator in members:
            values = rng.random(grid.shape) + 0.5
            monkeypatch.setattr(
                operator, "predict",
                lambda f, v=values: ChargeDensity(v, f.grid, f.structure))

        committee = Committee(members)
        assert score_candidate(committee, field, residency="one")["jsd"] == \
            pytest.approx(score_candidate(committee, field)["jsd"])


class TestScorePool:
    def test_scores_every_candidate(self, pool, members, monkeypatch,
                                    tagged_load):
        committee = Committee(members)
        rng = np.random.default_rng(2)
        shape = (8, 8, 8)
        scripted(committee, monkeypatch, {
            name: [rng.random(shape) + 0.5 for _ in members]
            for name in ("mp-001", "mp-002", "mp-003", "mp-004")})

        records = score_pool(committee, discover_pool(pool, "ext2chg"),
                             "ext2chg", batch_size=2)
        assert len(records) == 4
        assert {r["material"] for r in records} == \
            {"mp-001", "mp-002", "mp-003", "mp-004"}
        assert all(r["jsd"] is not None for r in records)

    def test_the_chunk_size_does_not_change_the_answer(self, pool, members,
                                                       monkeypatch, tagged_load):
        """It is a memory knob. If it moved the numbers it would be a bug."""
        committee = Committee(members)
        rng = np.random.default_rng(3)
        fields = {name: [rng.random((8, 8, 8)) + 0.5 for _ in members]
                  for name in ("mp-001", "mp-002", "mp-003", "mp-004")}
        scripted(committee, monkeypatch, fields)

        records = discover_pool(pool, "ext2chg")
        one = score_pool(committee, records, "ext2chg", batch_size=1)
        many = score_pool(committee, records, "ext2chg", batch_size=99)
        assert [r["jsd"] for r in one] == pytest.approx([r["jsd"] for r in many])

    def test_rejects_a_chunk_size_that_would_never_advance(self, pool, members):
        with pytest.raises(ValueError, match="at least 1"):
            score_pool(Committee(members), discover_pool(pool, "ext2chg"),
                       "ext2chg", batch_size=0)

    def test_reports_progress_per_chunk(self, pool, members, monkeypatch,
                                        tagged_load):
        committee = Committee(members)
        rng = np.random.default_rng(4)
        scripted(committee, monkeypatch, {
            name: [rng.random((8, 8, 8)) + 0.5 for _ in members]
            for name in ("mp-001", "mp-002", "mp-003", "mp-004")})

        lines = []
        score_pool(committee, discover_pool(pool, "ext2chg"), "ext2chg",
                   batch_size=2, log=lines.append)
        assert len(lines) == 2
        assert "4/4" in lines[-1]


@pytest.mark.skipif(
    not [name for name in ("cuda", "mps")
         if (torch.cuda.is_available() if name == "cuda"
             else bool(getattr(torch.backends, "mps", None)
                       and torch.backends.mps.is_built()
                       and torch.backends.mps.is_available()))],
    reason="no CUDA or MPS device available")
class TestResidencyOnAnAccelerator:
    """
    Where cycling members off the device is not a no-op.

    On CPU every ``.to()`` here does nothing and the parking cannot be
    observed, so the contract has to be checked on real hardware: during a
    sweep the committee is parked, and when the sweep returns it is exactly
    where it was found. A member left on the CPU with ``operator.device``
    still naming the accelerator fails the *next* prediction, far from here.
    """

    @pytest.fixture
    def accelerated(self):
        from poraque.ml.device import resolve_device

        device = resolve_device("auto")
        return [FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                              projection_channels=8, device=device,
                              init_seed=seed) for seed in (0, 1)]

    @staticmethod
    def _devices(operators):
        return [{p.device.type for p in op.model.parameters()}
                for op in operators]

    def test_the_committee_is_restored_after_a_sweep(self, accelerated, pool,
                                                     monkeypatch, tagged_load):
        committee = Committee(accelerated)
        rng = np.random.default_rng(6)
        scripted(committee, monkeypatch, {
            name: [rng.random((8, 8, 8)) + 0.5 for _ in accelerated]
            for name in ("mp-001", "mp-002", "mp-003", "mp-004")})

        before = self._devices(accelerated)
        score_pool(committee, discover_pool(pool, "ext2chg"), "ext2chg",
                   batch_size=2, residency="one")
        assert self._devices(accelerated) == before
        assert before[0] == {accelerated[0].device.type}

    def test_the_committee_is_restored_even_when_scoring_raises(
            self, accelerated, grid, structure, monkeypatch):
        """A half-parked committee is worse than a failed round."""
        field = ExternalPotential(np.zeros(grid.shape), grid, structure)
        before = self._devices(accelerated)

        def explode(_):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(accelerated[1], "predict", explode)
        with pytest.raises(RuntimeError, match="out of memory"):
            score_candidate(Committee(accelerated), field, residency="one")
        assert self._devices(accelerated) == before

    def test_only_one_member_is_resident_at_a_time(self, accelerated, grid,
                                                   structure, monkeypatch):
        """The saving itself: peak residency is one member, not the committee."""
        field = ExternalPotential(np.zeros(grid.shape), grid, structure)
        resident = []
        values = np.full(grid.shape, 0.4)

        def watch(operator):
            def predict(f, v=values, op=operator):
                resident.append(sum(
                    next(o.model.parameters()).device.type != "cpu"
                    for o in accelerated))
                return ChargeDensity(v, f.grid, f.structure)
            monkeypatch.setattr(operator, "predict", predict)

        for operator in accelerated:
            watch(operator)

        score_candidate(Committee(accelerated), field, residency="one")
        assert resident == [1, 1]


# ===================================================================== #
# Statistics and selection
# ===================================================================== #
class TestStatistics:
    def test_min_max_and_mean_over_the_pool(self):
        records = [{"jsd": v} for v in (0.1, 0.2, 0.6)]
        stats = jsd_statistics(records)
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.6)
        assert stats["mean"] == pytest.approx(0.3)
        assert stats["median"] == pytest.approx(0.2)
        assert stats["n"] == 3 and stats["n_scored"] == 3

    def test_unscored_candidates_are_counted_not_averaged(self):
        """Treating a withheld JSD as zero would drag the mean down."""
        stats = jsd_statistics([{"jsd": 0.4}, {"jsd": None}, {"jsd": 0.6}])
        assert stats["n"] == 3 and stats["n_scored"] == 2
        assert stats["mean"] == pytest.approx(0.5)

    def test_an_unscorable_pool_gives_nan_not_a_number(self):
        stats = jsd_statistics([{"jsd": None}, {"jsd": None}])
        assert stats["n_scored"] == 0
        assert all(np.isnan(stats[key]) for key in ("min", "max", "mean"))

    def test_the_spread_says_whether_the_ranking_means_anything(self):
        flat = jsd_statistics([{"jsd": 0.5}, {"jsd": 0.5}, {"jsd": 0.5}])
        informative = jsd_statistics([{"jsd": 0.01}, {"jsd": 0.5}, {"jsd": 1.0}])
        assert flat["spread"] == pytest.approx(1.0)
        assert informative["spread"] > 10.0

    def test_formats_the_round_metrics(self):
        text = format_statistics(jsd_statistics([{"jsd": v}
                                                 for v in (0.1, 0.2, 0.6)]))
        assert "min" in text and "max" in text and "mean" in text

    def test_formats_an_empty_round_without_inventing_numbers(self):
        assert "unavailable" in format_statistics(jsd_statistics([]))


class TestSelection:
    def test_takes_the_most_uncertain_first(self):
        records = [{"material": name, "jsd": value} for name, value in
                   [("a", 0.1), ("b", 0.9), ("c", 0.5), ("d", 0.7)]]
        assert [r["material"] for r in select_top_k(records, 2)] == ["b", "d"]

    def test_a_request_larger_than_the_pool_returns_the_pool(self):
        records = [{"material": "a", "jsd": 0.1}, {"material": "b", "jsd": 0.2}]
        assert len(select_top_k(records, 99)) == 2

    def test_unscored_candidates_are_never_selected(self):
        """
        Spending a DFT run on a candidate whose measure did not apply is the
        one outcome worse than selecting at random.
        """
        records = [{"material": "a", "jsd": None},
                   {"material": "b", "jsd": None},
                   {"material": "c", "jsd": 0.01}]
        assert [r["material"] for r in select_top_k(records, 3)] == ["c"]

    def test_ranking_table_is_ordered_and_truncated(self):
        records = [{"material": name, "jsd": value, "jsd_normalised": value,
                    "relative": 0.1, "integral_relative": 0.2}
                   for name, value in [("a", 0.1), ("b", 0.9), ("c", 0.5)]]
        rows = format_ranking(records, limit=2).splitlines()[2:]
        assert [row.split()[0] for row in rows] == ["b", "c"]   # ordered
        assert len(rows) == 2                                   # and truncated


# ===================================================================== #
# Promotion: the part that touches the filesystem
# ===================================================================== #
class TestPromote:
    @pytest.fixture
    def selection(self, pool):
        return [{"material": "mp-001",
                 "directory": os.path.join(pool, "mp-001")},
                {"material": "mp-003",
                 "directory": os.path.join(pool, "mp-003")}]

    def test_dry_run_touches_nothing(self, selection, pool, tmp_path):
        train = tmp_path / "train"
        moved = promote(selection, str(train), mode="move", dry_run=True)
        assert not train.exists()
        assert os.path.isdir(os.path.join(pool, "mp-001"))
        assert all(not entry["transferred"] for entry in moved)

    def test_move_leaves_the_pool_holding_only_the_unlabelled(self, selection,
                                                              pool, tmp_path):
        train = tmp_path / "train"
        promote(selection, str(train), mode="move")
        assert sorted(os.listdir(train)) == ["mp-001", "mp-003"]
        assert sorted(os.listdir(pool)) == ["mp-002", "mp-004"]

    def test_copy_leaves_the_pool_intact(self, selection, pool, tmp_path):
        train = tmp_path / "train"
        promote(selection, str(train), mode="copy")
        assert sorted(os.listdir(train)) == ["mp-001", "mp-003"]
        assert len(os.listdir(pool)) == 4

    def test_symlink_keeps_one_copy_on_disk(self, selection, pool, tmp_path):
        train = tmp_path / "train"
        promote(selection, str(train), mode="symlink")
        assert os.path.islink(train / "mp-001")
        assert (train / "mp-001" / "EXTCAR").exists()

    def test_an_existing_destination_is_skipped_not_overwritten(
            self, selection, pool, tmp_path):
        """
        A half-written training structure is worse than a missing one, and a
        candidate promoted in an earlier round is the ordinary way this happens.
        """
        train = tmp_path / "train"
        (train / "mp-001").mkdir(parents=True)
        (train / "mp-001" / "marker").write_text("kept")

        moved = promote(selection, str(train), mode="move")
        assert (train / "mp-001" / "marker").read_text() == "kept"
        assert os.path.isdir(os.path.join(pool, "mp-001"))     # not moved
        assert [entry["transferred"] for entry in moved] == [False, True]

    def test_rejects_an_unknown_mode(self, selection, tmp_path):
        with pytest.raises(ValueError, match="Unknown transfer mode"):
            promote(selection, str(tmp_path / "train"), mode="teleport")


# ===================================================================== #
# The round
# ===================================================================== #
class TestRunRound:
    @pytest.fixture
    def committee(self, members, monkeypatch, tagged_load):
        """
        A committee scripted so the ranking is known: mp-002 is the one the
        members disagree about, mp-004 the next, and mp-001 the one they agree
        on completely.
        """
        committee = Committee(members)
        shape = (8, 8, 8)
        base = np.full(shape, 0.5)

        def spread(amplitude):
            fields = []
            for index in range(len(members)):
                values = base.copy()
                values[index] += amplitude          # differ in shape
                fields.append(values)
            return fields

        scripted(committee, monkeypatch, {
            "mp-001": spread(0.0),
            "mp-002": spread(3.0),
            "mp-003": spread(0.2),
            "mp-004": spread(1.0),
        })
        return committee

    def test_ranks_the_pool_by_disagreement(self, committee, pool):
        result = run_round(committee, pool, "ext2chg", n_select=2)
        assert [r["material"] for r in result["selection"]] == \
            ["mp-002", "mp-004"]

    def test_reports_the_round_metrics(self, committee, pool):
        result = run_round(committee, pool, "ext2chg", n_select=2)
        stats = result["statistics"]
        assert stats["n"] == 4 and stats["n_scored"] == 4
        assert stats["min"] < stats["mean"] < stats["max"]

    def test_scoring_alone_never_touches_the_pool(self, committee, pool):
        """No --train, no filesystem change. The default has to be safe."""
        run_round(committee, pool, "ext2chg", n_select=2)
        assert len(os.listdir(pool)) == 4

    def test_dry_run_is_the_default_even_with_a_training_root(
            self, committee, pool, tmp_path):
        train = tmp_path / "train"
        result = run_round(committee, pool, "ext2chg", n_select=2,
                           train_root=str(train))
        assert not train.exists()
        assert len(os.listdir(pool)) == 4
        assert [entry["material"] for entry in result["promoted"]] == \
            ["mp-002", "mp-004"]

    def test_promotes_when_asked(self, committee, pool, tmp_path):
        train = tmp_path / "train"
        run_round(committee, pool, "ext2chg", n_select=2,
                  train_root=str(train), dry_run=False)
        assert sorted(os.listdir(train)) == ["mp-002", "mp-004"]
        assert sorted(os.listdir(pool)) == ["mp-001", "mp-003"]

    def test_excluded_structures_never_reach_the_selection(self, committee,
                                                           pool):
        result = run_round(committee, pool, "ext2chg", n_select=2,
                           exclude={"mp-002"})
        assert [r["material"] for r in result["selection"]] == \
            ["mp-004", "mp-003"]

    def test_an_empty_pool_is_an_error_not_an_empty_round(self, committee,
                                                          tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="no candidates"):
            run_round(committee, str(empty), "ext2chg")

    def test_an_unscorable_pool_is_an_error(self, members, pool):
        """
        Untrained members predict a signed field. Selecting from that would
        spend DFT runs on an ordering that does not exist.
        """
        with pytest.raises(ValueError, match="no candidate could be scored"):
            run_round(Committee(members), pool, "ext2chg")

    def test_the_result_is_json_serialisable(self, committee, pool):
        result = run_round(committee, pool, "ext2chg", n_select=2)
        payload = json.dumps(round_to_dict(result), default=float)
        assert "mp-002" in payload


# ===================================================================== #
# The command line
# ===================================================================== #
class TestCommandLine:
    def test_promotion_is_opt_in(self):
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "scripts"))
        from poraque_active_learning import build_parser

        assert build_parser().parse_args([]).promote is None

    def test_transfer_modes_match_the_library(self):
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "scripts"))
        from poraque_active_learning import build_parser

        from poraque.ml.active_learning import TRANSFERS

        args = build_parser().parse_args(["--promote", "copy"])
        assert args.promote == "copy" and "copy" in TRANSFERS
