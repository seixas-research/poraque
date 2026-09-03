# -*- coding: utf-8 -*-
# file: test_distributed.py
"""
Tests for :mod:`poraque.ml.distributed` and the distributed batch sampler.

**Nothing here forms a process group.** NCCL is a CUDA library, the module
refuses to initialise without one, and the machine this was written on has no
CUDA at all — so what is tested is everything that can be decided *before* a
collective: the Slurm hostlist expansion, the port derivation, which launcher
is believed, the fallbacks, and the partitioning of shape-bucketed batches
across a world whose size is stated rather than discovered.

That partition is the part worth testing hardest, because its failure mode is a
hang rather than an error. DDP all-reduces gradients inside every
``backward()``, so if two ranks are handed different numbers of batches the one
with more waits forever on a collective the other has already left — and the
job spends its allocation inside a barrier instead of failing. The properties
asserted below are therefore *equal counts*, *no overlap*, and *complete
coverage*, in that order of importance.
"""

import os

import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.ml.distributed import (  # noqa: E402
    PORT_RANGE,
    DistributedContext,
    _parse_nodelist,
    all_reduce_mean,
    barrier,
    default_master_port,
    describe,
    discover,
    expand_nodelist,
    initialize,
    shutdown,
    unwrap,
)


@pytest.fixture
def clean_environment(monkeypatch):
    """A process with no launcher variables at all."""
    for name in ("SLURM_JOB_ID", "SLURM_PROCID", "SLURM_LOCALID",
                 "SLURM_NTASKS", "SLURM_NTASKS_PER_NODE", "SLURM_NODELIST",
                 "SLURM_JOB_NODELIST", "SLURM_STEP_NODELIST",
                 "SLURM_GPUS_ON_NODE", "RANK", "LOCAL_RANK", "WORLD_SIZE",
                 "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class TestTheSlurmNodelistIsExpandedWithoutSlurm:
    """
    ``scontrol show hostnames`` is Slurm's own answer and is asked first, but
    it is absent on a login node, in a container, and in this test suite. The
    fallback parser has to handle the forms Santos Dumont actually emits, and
    the first host of the list is ``MASTER_ADDR`` — so getting the *order*
    wrong is as bad as getting the names wrong: the ranks would rendezvous at
    different addresses and the job would hang in ``init_process_group``.
    """

    def test_a_bare_host_is_itself(self):
        assert _parse_nodelist("sdumont1234") == ["sdumont1234"]

    def test_a_range_expands_in_order(self):
        assert _parse_nodelist("sdumont[1234-1236]") == [
            "sdumont1234", "sdumont1235", "sdumont1236"]

    def test_a_bracketed_list_mixes_ranges_and_singletons(self):
        assert _parse_nodelist("sdumont[1234-1236,1240]") == [
            "sdumont1234", "sdumont1235", "sdumont1236", "sdumont1240"]

    def test_zero_padding_is_part_of_the_name(self):
        """``node1`` and ``node01`` are different hosts, and Slurm pads."""
        assert _parse_nodelist("node[01-03]") == ["node01", "node02", "node03"]

    def test_commas_inside_brackets_do_not_separate_entries(self):
        """The reason the string cannot simply be split on commas."""
        assert _parse_nodelist("a[1,2],b[3-4]") == ["a1", "a2", "b3", "b4"]

    def test_a_suffix_after_the_brackets_survives(self):
        assert _parse_nodelist("gpu[1-2].local") == ["gpu1.local", "gpu2.local"]

    def test_an_empty_list_is_empty(self):
        assert expand_nodelist("") == []
        assert expand_nodelist(None) == []


class TestTheRendezvousPortIsDerivedNotHardCoded:
    """
    Two jobs can land on one node, and two runs agreeing on a hard-coded 29500
    is a bind failure that reads as a network problem. The port has to be
    distinct per job, identical across the ranks of one job — which is the only
    property that must hold — and stable across a requeue.
    """

    def test_it_lands_inside_the_reserved_range(self):
        low, high = PORT_RANGE
        for job in ("1", "999999", "24601", "0"):
            assert low <= default_master_port(job) <= high

    def test_the_same_job_always_gets_the_same_port(self):
        """Every rank derives it independently; they must agree."""
        assert default_master_port("8675309") == default_master_port("8675309")

    def test_different_jobs_generally_differ(self):
        assert default_master_port("1000") != default_master_port("1001")

    def test_a_non_numeric_job_id_does_not_raise(self):
        """Array jobs are ``12345_7``; the digits are what matters."""
        assert PORT_RANGE[0] <= default_master_port("12345_7") <= PORT_RANGE[1]


class TestNoLauncherMeansNoGroup:
    """
    A workstation must reach the single-device path by falling *through*, not
    by catching something. Every branch below returns a disabled context, and a
    disabled context is falsy so the call sites read ``if context:``.
    """

    def test_a_bare_process_gets_a_disabled_context(self, clean_environment):
        context = discover("auto")
        assert not context
        assert context.is_main and context.world_size == 1
        assert context.device == "auto"

    def test_a_one_task_allocation_is_not_a_group(self, clean_environment):
        """
        The commonest misconfiguration: four GPUs requested, one task
        launched. It is not an error — it is a perfectly good single-GPU run —
        but the log has to say so, which is what :func:`describe` is for.
        """
        clean_environment.setenv("SLURM_PROCID", "0")
        clean_environment.setenv("SLURM_NTASKS", "1")
        clean_environment.setenv("SLURM_JOB_ID", "424242")
        assert not discover("auto")

    def test_off_refuses_a_real_allocation(self, clean_environment):
        """How a four-task allocation is bisected against single-GPU."""
        clean_environment.setenv("SLURM_PROCID", "2")
        clean_environment.setenv("SLURM_LOCALID", "2")
        clean_environment.setenv("SLURM_NTASKS", "4")
        assert not discover("off")
        assert not discover(False)

    def test_without_cuda_it_declines_and_says_why(self, clean_environment,
                                                   monkeypatch):
        """
        Falling back to Gloo would let a misconfigured job train distributed
        across CPU cores at a fraction of one GPU's speed — the same silent
        waste ``strict_device`` exists to stop.
        """
        clean_environment.setenv("SLURM_PROCID", "1")
        clean_environment.setenv("SLURM_LOCALID", "1")
        clean_environment.setenv("SLURM_NTASKS", "4")
        clean_environment.setenv("SLURM_STEP_NODELIST", "sdumont[1,2]")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.warns(RuntimeWarning, match="no CUDA device"):
            assert not discover("auto")


class TestTheSlurmTopologyIsRead:
    """
    The topology is read from the step, never hard-coded and never asked of the
    user. What the tests below pin is *which* variable answers each question,
    because the wrong one is silently plausible: ``SLURM_JOB_NODELIST`` is the
    whole allocation while ``SLURM_STEP_NODELIST`` is the nodes of this step,
    and they differ when a script runs several steps over subsets of its nodes.
    """

    @staticmethod
    def _allocate(monkeypatch, devices=4, **overrides):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: devices)
        environment = {
            "SLURM_JOB_ID": "5150",
            "SLURM_PROCID": "3",
            "SLURM_LOCALID": "3",
            "SLURM_NTASKS": "4",
            "SLURM_STEP_NODELIST": "sdumont[1234-1235]",
        }
        environment.update(overrides)
        for name, value in environment.items():
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, str(value))
        return discover("auto")

    def test_the_rank_and_world_come_from_slurm(self, clean_environment):
        context = self._allocate(clean_environment)
        assert (context.rank, context.world_size) == (3, 4)
        assert context.launcher == "slurm"
        assert context

    def test_the_local_rank_is_the_cuda_ordinal(self, clean_environment):
        """
        Every rank on ``cuda:0`` is four processes contending for one device,
        and it reports itself as a scaling failure rather than as an error.
        """
        context = self._allocate(clean_environment, SLURM_LOCALID="2")
        assert context.local_rank == 2
        assert context.device == "cuda:2"

    def test_a_missing_local_id_is_derived_from_the_tasks_per_node(
            self, clean_environment):
        context = self._allocate(clean_environment, SLURM_LOCALID=None,
                                 SLURM_PROCID="5", SLURM_NTASKS="8",
                                 SLURM_NTASKS_PER_NODE="4")
        assert context.local_rank == 1

    def test_the_step_nodelist_wins_over_the_job_nodelist(self,
                                                          clean_environment):
        context = self._allocate(clean_environment,
                                 SLURM_STEP_NODELIST="sdumont[900-901]",
                                 SLURM_JOB_NODELIST="sdumont[100-199]")
        assert context.master_addr == "sdumont900"

    def test_the_master_is_the_first_host_not_this_one(self,
                                                       clean_environment):
        """Every rank must name the same address, so it is the list's first."""
        context = self._allocate(clean_environment)
        assert context.master_addr == "sdumont1234"

    def test_only_rank_zero_writes(self, clean_environment):
        assert not self._allocate(clean_environment, SLURM_PROCID="1",
                                  SLURM_LOCALID="1").is_main
        assert self._allocate(clean_environment, SLURM_PROCID="0",
                              SLURM_LOCALID="0").is_main

    def test_more_tasks_than_devices_warns_and_wraps(self, clean_environment):
        """One task per GPU is the launch this is written for."""
        with pytest.warns(RuntimeWarning, match="exceeds"):
            context = self._allocate(clean_environment, devices=2,
                                     SLURM_LOCALID="3")
        assert context.local_rank == 1

    def test_the_present_variables_are_recorded_for_the_log(self,
                                                            clean_environment):
        """
        The log prints what it saw rather than what it expected, because the
        usual failure looks identical from inside the process to the run that
        was asked for.
        """
        context = self._allocate(clean_environment)
        assert context.present["SLURM_NTASKS"] == "4"
        assert any("SLURM_NTASKS" in line for line in describe(context))


class TestSlurmIsBelievedBeforeTorchrun:
    """
    A ``torchrun`` launch *inside* an allocation carries both sets of
    variables, and its own describe the group it actually formed. Reversing the
    order would make that case read the allocation's task count instead of the
    launcher's — which is right only by coincidence.
    """

    def test_torchrun_alone_is_used(self, clean_environment, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
        for name, value in {"RANK": "5", "LOCAL_RANK": "1", "WORLD_SIZE": "8",
                            "MASTER_ADDR": "10.0.0.1",
                            "MASTER_PORT": "29411"}.items():
            monkeypatch.setenv(name, value)
        context = discover("auto")
        assert context.launcher == "torchrun"
        assert (context.rank, context.world_size) == (5, 8)
        assert context.master_addr == "10.0.0.1"
        assert context.master_port == 29411

    def test_slurm_wins_when_both_are_set(self, clean_environment, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
        for name, value in {"SLURM_PROCID": "1", "SLURM_LOCALID": "1",
                            "SLURM_NTASKS": "4", "SLURM_JOB_ID": "7",
                            "SLURM_STEP_NODELIST": "sdumont1",
                            "RANK": "0", "LOCAL_RANK": "0",
                            "WORLD_SIZE": "2"}.items():
            monkeypatch.setenv(name, value)
        assert discover("auto").launcher == "slurm"


class TestADisabledContextIsANoOp:
    """
    Every collective is written so that the single-device path reaches it
    unconditionally. If any of them needed a guard at the call site, the guard
    would eventually be forgotten in one place and the failure would be a hang.
    """

    def test_initialize_does_nothing(self):
        context = DistributedContext()
        assert initialize(context) is context
        assert not context.initialized

    def test_barrier_and_shutdown_return(self):
        context = DistributedContext()
        barrier(context)                    # no raise
        shutdown(context)                   # no raise

    def test_all_reduce_returns_the_value_unchanged(self):
        context = DistributedContext()
        value = torch.tensor(3.5)
        assert all_reduce_mean(value, context) is value

    def test_unwrap_passes_a_bare_module_through(self):
        module = torch.nn.Linear(2, 2)
        assert unwrap(module) is module

    def test_unwrap_reaches_inside_a_wrapper(self):
        """
        DDP prefixes every ``state_dict`` key with ``module.``, exactly as
        ``torch.compile`` prefixes them with ``_orig_mod.``, so a checkpoint
        written from the wrapper reloads nowhere else.
        """
        class _Wrapper:
            def __init__(self, module):
                self.module = module

        inner = torch.nn.Linear(2, 2)
        assert unwrap(_Wrapper(inner)) is inner

    def test_it_describes_itself_as_single_device(self, clean_environment):
        """
        ``clean_environment`` is the whole point of this one.

        Without it the assertion passes on a workstation and **fails inside
        any Slurm allocation**, because :func:`describe` then correctly appends
        the lines saying an allocation is present with no group formed — the
        behaviour the very next test asserts. Reproduced on the LNCC login
        node, where it was the only failure outside the ``gpu`` marker and so
        made ``pytest -m "not gpu"``, the CI command, fail on the cluster for a
        reason unrelated to GPUs. This test is about a disabled context, not
        about the environment it was collected in.
        """
        assert describe(DistributedContext()) == \
            ["single device (no distributed group)"]

    def test_an_unformed_group_inside_an_allocation_is_flagged(self,
                                                               monkeypatch):
        """
        Four GPUs requested and one task launched is invisible from inside the
        process, so the log says the allocation was there and no group formed.
        """
        monkeypatch.setenv("SLURM_NTASKS", "1")
        monkeypatch.setenv("SLURM_JOB_ID", "31337")
        lines = describe(DistributedContext())
        assert any("no group was formed" in line for line in lines)


class TestTheBatchesAreSplitNotTheSamples:
    """
    :class:`~poraque.ml.data.DistributedShapeBucketSampler` distributes
    *batches*, and that is forced rather than chosen: ``DataLoader`` takes a
    ``sampler`` or a ``batch_sampler`` and never both, so a plain
    ``DistributedSampler`` would have to displace the shape bucketing — and a
    batch mixing two grid shapes does not train badly, it raises in
    ``collate_fields``, because there is no padding anywhere in this pipeline.

    The properties below are what keep a four-GPU run from hanging. Equal
    counts first: DDP all-reduces inside each ``backward()``, so a rank that
    runs out of batches early leaves the others in a collective that never
    completes, and the job burns its wall clock in a barrier rather than
    failing.
    """

    @staticmethod
    def _samplers(shapes, batch_size=2, world=4, seed=7):
        from poraque.ml.data import DistributedShapeBucketSampler

        class _Dataset:
            def __init__(self, shapes):
                self._shapes = [tuple(s) for s in shapes]

            def shapes(self):
                return list(self._shapes)

            def __len__(self):
                return len(self._shapes)

        dataset = _Dataset(shapes)
        return [DistributedShapeBucketSampler(
            dataset, batch_size=batch_size, shuffle=True, seed=seed,
            num_replicas=world, rank=rank) for rank in range(world)]

    @staticmethod
    def _ragged(n_small=9, n_large=5):
        return [(8, 8, 8)] * n_small + [(12, 12, 12)] * n_large

    def test_every_rank_gets_the_same_number_of_batches(self):
        """The property whose absence is a hang rather than an error."""
        samplers = self._samplers(self._ragged())
        counts = {len(list(sampler)) for sampler in samplers}
        assert len(counts) == 1
        assert counts.pop() > 0

    def test_the_count_holds_when_batches_do_not_divide_the_world(self):
        """
        Five batches over four ranks. ``DistributedSampler`` pads by wrapping
        to the front, which is exactly the property being borrowed.
        """
        samplers = self._samplers([(8, 8, 8)] * 5, batch_size=1, world=4)
        assert {len(list(s)) for s in samplers} == {2}

    def test_no_batch_mixes_grid_shapes(self):
        """
        The invariant the whole design exists to keep. A mixed batch is not a
        worse batch: ``collate_fields`` raises on it.
        """
        shapes = self._ragged()
        for sampler in self._samplers(shapes):
            for batch in sampler:
                assert len({shapes[index] for index in batch}) == 1

    def test_the_ranks_do_not_overlap_when_the_batches_divide_evenly(self):
        """Otherwise four GPUs compute the same gradient four times."""
        seen = [set(sum(list(sampler), [])) for sampler in self._samplers(
            self._ragged(n_small=16, n_large=16), batch_size=2, world=4)]
        # 8 + 8 batches over 4 ranks: nothing to pad, so the partition is a
        # partition in the strict sense.
        for left in range(len(seen)):
            for right in range(left + 1, len(seen)):
                assert not seen[left] & seen[right]

    def test_the_only_repetition_is_the_padding(self):
        """
        The honest version of the property above, and the one that holds on
        real data: 43 + 11 materials at ``batch_size`` 10 is 7 batches, which
        four ranks do not divide. ``DistributedSampler`` wraps to the front to
        equalise the counts, so somebody sees a batch twice — and the excess is
        exactly the padding, never more.

        Stated as a bound rather than as "no overlap" because the alternative
        to the padding is a deadlock, not a tidier epoch.
        """
        shapes = self._ragged(n_small=43, n_large=11)
        samplers = self._samplers(shapes, batch_size=10, world=4)
        drawn = [tuple(sorted(batch))
                 for sampler in samplers for batch in sampler]
        distinct = set(drawn)
        assert len(drawn) - len(distinct) == len(drawn) % 4 or \
            len(drawn) - len(distinct) < 4
        # And nothing is invented: every batch a rank yields is one the
        # undistributed sampler would have produced.
        whole = {tuple(sorted(batch))
                 for batch in self._samplers(shapes, batch_size=10,
                                             world=1)[0]}
        assert distinct <= whole

    def test_together_they_cover_the_dataset(self):
        shapes = self._ragged(n_small=16, n_large=16)
        covered = set()
        for sampler in self._samplers(shapes, batch_size=2, world=4):
            covered |= set(sum(list(sampler), []))
        assert covered == set(range(len(shapes)))

    def test_they_cover_the_dataset_when_padding_is_needed_too(self):
        """The padding must not cost coverage; it only adds repetition."""
        shapes = self._ragged(n_small=43, n_large=11)
        covered = set()
        for sampler in self._samplers(shapes, batch_size=10, world=4):
            covered |= set(sum(list(sampler), []))
        assert covered == set(range(len(shapes)))

    def test_set_epoch_reaches_both_halves(self):
        """
        Forwarded to the bucket sampler *and* to the ``DistributedSampler``.
        Forget the first and every epoch draws the same batches; forget the
        second and every epoch sends the same batches to the same rank. Neither
        raises, and training merely learns less than the log claims.
        """
        sampler = self._samplers(self._ragged())[0]
        first = list(sampler)
        sampler.set_epoch(1)
        assert sampler.buckets_sampler.epoch == 1
        assert sampler.partition.epoch == 1
        assert list(sampler) != first

    def test_one_rank_reproduces_the_undistributed_content(self):
        """
        A world of one must see every material exactly once, so that a
        distributed run and a single-device run differ in the *partition* and
        in nothing else.
        """
        shapes = self._ragged(n_small=10, n_large=6)
        sampler = self._samplers(shapes, batch_size=2, world=1)[0]
        drawn = sorted(sum(list(sampler), []))
        assert drawn == list(range(len(shapes)))

    def test_an_empty_dataset_yields_nothing(self):
        """Reached by a task whose target field no material carries."""
        assert list(self._samplers([], world=2)[0]) == []


class TestTheLoaderTakesTheContextUnconditionally:
    """
    ``make_dataloader`` is called with ``distributed=`` in both training paths
    whether or not a group exists, so ``None`` and a disabled context have to
    mean the same thing — the single-device sampler.
    """

    @staticmethod
    def _dataset():
        class _Dataset(torch.utils.data.Dataset):
            def shapes(self):
                return [(4, 4, 4)] * 6

            def __len__(self):
                return 6

            def __getitem__(self, index):        # pragma: no cover - unused
                raise NotImplementedError

        return _Dataset()

    def test_none_gives_the_plain_sampler(self):
        from poraque.ml.data import ShapeBucketSampler, make_dataloader

        loader = make_dataloader(self._dataset(), batch_size=2,
                                 distributed=None)
        assert isinstance(loader.batch_sampler, ShapeBucketSampler)

    def test_a_disabled_context_gives_the_plain_sampler(self):
        from poraque.ml.data import ShapeBucketSampler, make_dataloader

        loader = make_dataloader(self._dataset(), batch_size=2,
                                 distributed=DistributedContext())
        assert isinstance(loader.batch_sampler, ShapeBucketSampler)

    def test_a_group_gives_the_distributed_sampler(self):
        from poraque.ml.data import (
            DistributedShapeBucketSampler,
            make_dataloader,
        )

        context = DistributedContext(enabled=True, rank=1, local_rank=1,
                                     world_size=2)
        loader = make_dataloader(self._dataset(), batch_size=2,
                                 distributed=context)
        assert isinstance(loader.batch_sampler, DistributedShapeBucketSampler)
        assert loader.batch_sampler.rank == 1
        assert loader.batch_sampler.num_replicas == 2


@pytest.fixture
def toy(tmp_path):
    """A two-material dataset on one grid shape, enough to drive `train`."""
    import numpy as np

    from poraque.fields import (
        ChargeDensity,
        ExternalPotential,
        FieldGrid,
        KineticEnergyDensity,
    )
    from poraque.fields.vasp.poscar import Poscar
    from poraque.ml import FieldPairDataset

    root = tmp_path / "toy"
    root.mkdir()
    rng = np.random.default_rng(0)
    for index in range(2):
        directory = root / f"mat_{index}"
        directory.mkdir()
        grid = FieldGrid((8, 8, 8), np.eye(3) * 5.0)
        structure = Poscar(np.eye(3) * 5.0, ["Si"], [2], rng.random((2, 3)))
        ExternalPotential.compute(structure, grid, {"Si": 4.0},
                                  widths={"Si": 0.5}).write(directory / "EXTCAR")
        density = rng.random(grid.shape) * 0.1 + 0.01
        ChargeDensity(density, grid, structure).write(directory / "CHGCAR")
        KineticEnergyDensity(density * 50.0, grid,
                             structure).write(directory / "TAUCAR")
    return FieldPairDataset(str(root), task="chg2tau")


class TestTheTrainingLoopSilencesEveryRankButTheFirst:
    """
    Four ranks appending to one log truncate each other's lines, and four
    calling ``operator.save`` on one path race on the same inode — which does
    not raise, it leaves a checkpoint that loads and holds a mixture.

    ``enabled`` without ``initialized`` is the state these use: every
    collective is a no-op so the loop runs single-device on the CPU, but
    ``is_main`` is already ``False``. That is exactly what a non-zero rank
    looks like before anything has been reduced, and it is what lets the
    rank-gating be tested on a machine with no CUDA.
    """

    @staticmethod
    def _operator():
        from poraque.ml import FieldOperator

        torch.manual_seed(0)
        return FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                             projection_channels=8, device="cpu")

    def test_a_non_main_rank_writes_no_checkpoint(self, toy, tmp_path):
        from poraque.ml.training import train

        destination = tmp_path / "should_not_appear.poraque"
        context = DistributedContext(enabled=True, rank=2, local_rank=0,
                                     world_size=4)
        train(self._operator(), toy, epochs=1, batch_size=1, eval_every=1,
              validation=toy, checkpoint=str(destination),
              distributed=context)
        assert not destination.exists()

    def test_a_non_main_rank_prints_nothing(self, toy):
        """Interleaved progress tables from four ranks are unreadable."""
        from poraque.ml.training import train

        lines = []
        context = DistributedContext(enabled=True, rank=3, local_rank=0,
                                     world_size=4)
        train(self._operator(), toy, epochs=1, batch_size=1, eval_every=1,
              verbose=True, log=lines.append, distributed=context)
        assert lines == []

    def test_rank_zero_writes_normally(self, toy, tmp_path):
        from poraque.ml.training import train

        destination = tmp_path / "written.poraque"
        context = DistributedContext(enabled=True, rank=0, local_rank=0,
                                     world_size=4)
        train(self._operator(), toy, epochs=1, batch_size=1, eval_every=1,
              validation=toy, checkpoint=str(destination),
              distributed=context)
        assert destination.exists()

    def test_a_disabled_context_trains_exactly_as_before(self, toy):
        """
        The single-device path must be untouched by any of this, to five
        digits: `None` and a disabled context are the same run.
        """
        from poraque.ml.training import train

        first = train(self._operator(), toy, epochs=2, batch_size=1,
                      eval_every=1, validation=toy, verbose=False,
                      distributed=None)
        second = train(self._operator(), toy, epochs=2, batch_size=1,
                       eval_every=1, validation=toy, verbose=False,
                       distributed=DistributedContext())
        assert first["train_loss"] == pytest.approx(second["train_loss"],
                                                    rel=1e-9)


class TestTheSubmissionScriptLaunchesOneTaskPerGpu:
    """
    Poraquê cannot invent ranks: the launcher decides the topology and the
    config decides only whether to believe it. A job that requests four GPUs
    and launches one task runs single-GPU and looks, from inside the process,
    exactly like the single-GPU run somebody asked for — so the shipped script
    is the documentation of the one line that matters.
    """

    @staticmethod
    def _script():
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "scripts", "slurm", "poraque_ddp.sbatch")
        with open(path) as handle:
            return handle.read()

    def test_the_task_count_matches_the_gpu_count(self):
        text = self._script()
        assert "--ntasks-per-node=4" in text
        assert "--gres=gpu:4" in text

    def test_it_launches_with_srun(self):
        """One process per task is what sets SLURM_PROCID at all."""
        assert "srun" in self._script()

    def test_it_is_strict_about_the_device(self):
        """A silent CPU fallback inside a GPU allocation is a lost day."""
        assert "--strict-device" in self._script()

    def test_it_pins_torch_to_the_environment(self):
        """A ~/.local torch outranks the activated environment on a cluster."""
        assert "PYTHONNOUSERSITE=1" in self._script()
