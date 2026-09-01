# -*- coding: utf-8 -*-
# file: test_device.py
"""
Tests for :mod:`poraque.ml.device` and cross-backend numerical agreement.

The selection logic is tested everywhere. The numerical-agreement tests skip
unless an accelerator is actually present, because they exist to catch *silent*
backend bugs — and this codebase has already hit two of them on MPS:

* ``torch.einsum`` over the real view of a **non-contiguous** complex tensor
  returns wrong values with no error (40-90 % off);
* ``torch.linalg.det`` and float64 are unavailable, which the cell-metric
  helpers work around by evaluating on the host.

Both are guarded below, since a regression would corrupt every prediction on
Apple Silicon without raising anything.
"""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.ml import device as device_module  # noqa: E402
from poraque.ml.device import (  # noqa: E402
    PREFERENCE_ORDER,
    available_devices,
    cuda_available,
    cuda_capability_supported,
    describe_device,
    device_report,
    empty_cache,
    enable_tf32,
    mps_available,
    resolve_device,
    supports_float64,
    synchronize,
)
from poraque.ml.fno import FNO3d, SpectralConv3d, complex_contract  # noqa: E402
from poraque.ml.physics import (  # noqa: E402
    cell_reciprocal,
    cell_volume,
    reciprocal_vectors,
)

ACCELERATORS = [name for name in ("cuda", "mps") if
                (cuda_available() if name == "cuda" else mps_available())]
requires_accelerator = pytest.mark.skipif(
    not ACCELERATORS, reason="no CUDA or MPS device available"
)


# --------------------------------------------------------------------- #
# Selection logic
# --------------------------------------------------------------------- #
class TestResolveDevice:
    def test_cpu_is_always_available(self):
        assert "cpu" in available_devices()
        assert available_devices()[-1] == "cpu"

    def test_auto_picks_the_preferred_backend(self):
        chosen = resolve_device("auto")
        assert chosen.type == available_devices()[0]

    def test_none_behaves_like_auto(self):
        assert resolve_device(None).type == resolve_device("auto").type

    def test_explicit_cpu_is_honoured(self):
        assert resolve_device("cpu").type == "cpu"

    def test_accepts_a_torch_device(self):
        assert resolve_device(torch.device("cpu")).type == "cpu"

    def test_preference_order(self):
        assert PREFERENCE_ORDER == ("cuda", "mps", "cpu")

    @pytest.mark.skipif(cuda_available(), reason="CUDA is present here")
    def test_missing_cuda_warns_and_falls_back(self):
        with pytest.warns(RuntimeWarning, match="CUDA was requested"):
            assert resolve_device("cuda").type == "cpu"

    @pytest.mark.skipif(mps_available(), reason="MPS is present here")
    def test_missing_mps_warns_and_falls_back(self):
        with pytest.warns(RuntimeWarning, match="MPS was requested"):
            assert resolve_device("mps").type == "cpu"

    def test_unknown_backend_warns_and_falls_back(self):
        with pytest.warns(RuntimeWarning, match="Unknown device"):
            assert resolve_device("tpu").type == "cpu"

    def test_describe_device_mentions_the_backend(self):
        assert "cpu" in describe_device("cpu")
        for name in ACCELERATORS:
            assert name in describe_device(name)

    def test_supports_float64(self):
        assert supports_float64("cpu") is True
        assert supports_float64("mps") is False

    def test_synchronize_and_empty_cache_are_safe_on_cpu(self):
        synchronize("cpu")
        empty_cache("cpu")


# --------------------------------------------------------------------- #
# Is this build able to run on this GPU at all?
# --------------------------------------------------------------------- #
class TestBuildCapability:
    """
    `torch.cuda.is_available()` answers "is there a driver and a device", not
    "can this binary generate code for this device". CUDA 13 dropped Volta, so
    a `+cu130` wheel on a V100 reports available and then aborts on the first
    kernel launch — after the job has spent its time in the GPU queue. That is
    what this pair of checks exists to make cheap.
    """

    @pytest.mark.skipif(not cuda_available(), reason="no CUDA here")
    @pytest.mark.gpu
    def test_this_build_has_kernels_for_this_gpu(self):
        """The failure this catches costs a queue slot, not a test run."""
        assert cuda_capability_supported()

    @pytest.mark.skipif(cuda_available(), reason="CUDA is present here")
    def test_capability_check_is_false_without_cuda(self):
        """Safe to call anywhere, so callers need no CUDA guard of their own."""
        assert cuda_capability_supported() is False

    def test_a_capability_below_every_listed_one_is_unsupported(self, monkeypatch):
        """sm_70 against a build listing sm_75 upwards: the V100 case."""
        monkeypatch.setattr(device_module, "cuda_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_arch_list",
                            lambda: ["sm_75", "sm_80", "sm_90", "compute_90"])
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (7, 0))
        assert cuda_capability_supported() is False

    def test_a_capability_above_the_oldest_listed_one_is_supported(self, monkeypatch):
        """A build ships PTX for its newest architecture, which JITs forward."""
        monkeypatch.setattr(device_module, "cuda_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_arch_list",
                            lambda: ["sm_50", "sm_70", "sm_80"])
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (9, 0))
        assert cuda_capability_supported() is True

    def test_an_empty_arch_list_is_not_read_as_a_refusal(self, monkeypatch):
        """Absence of evidence: a build that lists nothing blocks nothing."""
        monkeypatch.setattr(device_module, "cuda_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["compute_80"])
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (3, 5))
        assert cuda_capability_supported() is True


class TestStrictDevice:
    """
    Graceful degradation is right on a workstation and expensive in a queue:
    the job waits for its GPU, `resolve_device` warns into a log nobody reads
    until afterwards, and the run trains on the CPU inside the GPU allocation
    until the wall clock ends. `strict=True` turns that into an error at the
    first second instead of the last.
    """

    @pytest.mark.skipif(cuda_available(), reason="CUDA is present here")
    def test_strict_raises_instead_of_falling_back(self):
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda", strict=True)

    @pytest.mark.skipif(cuda_available(), reason="CUDA is present here")
    def test_non_strict_still_warns_and_falls_back(self):
        with pytest.warns(RuntimeWarning, match="CUDA was requested"):
            assert resolve_device("cuda").type == "cpu"

    @pytest.mark.skipif(cuda_available(), reason="CUDA is present here")
    def test_the_message_names_a_probable_cause(self):
        """"It did not work" and "install cu126" are different messages."""
        with pytest.raises(RuntimeError) as error:
            resolve_device("cuda", strict=True)
        text = str(error.value)
        assert "torch" in text
        assert any(word in text for word in
                   ("CPU-only", "driver", "CUDA_VISIBLE_DEVICES", "sm_"))

    def test_an_unknown_backend_raises_too(self):
        """Every branch that warns has to be a branch that can refuse."""
        with pytest.raises(RuntimeError, match="Unknown device"):
            resolve_device("tpu", strict=True)

    @pytest.mark.skipif(mps_available(), reason="MPS is present here")
    def test_a_missing_mps_raises_too(self):
        with pytest.raises(RuntimeError, match="MPS was requested"):
            resolve_device("mps", strict=True)

    def test_an_available_device_is_unaffected_by_strict(self):
        assert resolve_device("cpu", strict=True).type == "cpu"
        for name in ACCELERATORS:
            assert resolve_device(name, strict=True).type == name


class TestTheDeviceReport:
    """
    Written for a log read after the fact, when the question is "why did this
    run on the CPU?". Two of its lines answer questions no other part of the
    banner does: which CUDA the wheel was built against, and where torch was
    imported from — on a cluster a `~/.local` install outranks the activated
    environment and is the one that gets used.
    """

    def test_it_names_the_build_and_where_it_came_from(self):
        text = "\n".join(device_report())
        assert torch.__version__ in text
        assert "cuda build" in text
        assert "installed" in text

    def test_it_says_what_the_request_resolved_to(self):
        text = "\n".join(device_report("cpu"))
        assert "cpu" in text.split("requested")[-1]

    def test_it_does_not_warn_while_reporting(self, recwarn):
        """A report about a fallback must not itself trigger the fallback's
        warning: the reader asked a question, not for the run to proceed."""
        device_report("tpu")
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]

    def test_every_gpu_is_labelled_usable_or_not(self):
        lines = device_report()
        if not cuda_available():
            assert any("cuda devices : none" in line for line in lines)
            return
        entries = [line for line in lines if line.strip().startswith("cuda:")]
        assert entries
        assert all("usable" in line.lower() for line in entries)


class TestTf32:
    """
    Ampere and later only, and a no-op on the V100 this was measured on. It is
    here for the next cluster, and the return value exists so a run can log
    what happened rather than what was asked for.
    """

    def test_it_declines_off_cuda(self):
        assert enable_tf32("cpu", True) is False

    @pytest.mark.skipif(not cuda_available(), reason="no CUDA here")
    @pytest.mark.gpu
    def test_it_sets_both_flags_on_cuda(self):
        try:
            assert enable_tf32("cuda", True) is True
            assert torch.backends.cuda.matmul.allow_tf32 is True
            assert torch.backends.cudnn.allow_tf32 is True
        finally:
            enable_tf32("cuda", False)


# --------------------------------------------------------------------- #
# Cell metric: the float64 / linalg.det workaround
# --------------------------------------------------------------------- #
class TestCellMetric:
    @pytest.mark.parametrize("device", ["cpu"] + ACCELERATORS)
    def test_reciprocal_is_dual_to_the_cell(self, device):
        r"""``A B^T = 2 pi I`` must hold on every backend."""
        cell = torch.tensor(np.diag([5.0, 6.0, 7.0]), dtype=torch.float32,
                            device=device).unsqueeze(0)
        reciprocal = cell_reciprocal(cell, device=device)
        product = cell.cpu()[0] @ reciprocal.cpu()[0].T
        assert torch.allclose(product, 2 * np.pi * torch.eye(3), atol=1e-4)
        assert reciprocal.device.type == torch.device(device).type

    @pytest.mark.parametrize("device", ["cpu"] + ACCELERATORS)
    def test_volume_matches_the_determinant(self, device):
        cell = torch.tensor(np.diag([5.0, 6.0, 7.0]), dtype=torch.float32,
                            device=device).unsqueeze(0)
        volume = cell_volume(cell, device=device)
        assert float(volume) == pytest.approx(210.0, rel=1e-5)
        assert volume.device.type == torch.device(device).type

    @pytest.mark.parametrize("device", ["cpu"] + ACCELERATORS)
    def test_handles_a_triclinic_cell(self, device):
        cell = torch.tensor([[4.0, 0.2, 0.1], [0.3, 5.0, 0.4], [0.1, 0.2, 6.0]],
                            dtype=torch.float32, device=device).unsqueeze(0)
        reciprocal = cell_reciprocal(cell, device=device)
        product = cell.cpu()[0] @ reciprocal.cpu()[0].T
        assert torch.allclose(product, 2 * np.pi * torch.eye(3), atol=1e-3)

    def test_accepts_an_unbatched_cell(self):
        cell = torch.eye(3) * 4.0
        assert cell_reciprocal(cell).shape == (1, 3, 3)
        assert float(cell_volume(cell)) == pytest.approx(64.0, rel=1e-5)


# --------------------------------------------------------------------- #
# The memoised integer mesh
#
# There is deliberately no test here asserting that the cell metric stays on
# the host, or that any other operation happens on a particular device. That
# was measured (keeping it on the device is 2 % *slower* on a V100, because a
# 3x3 `linalg.det` costs more in kernel launch than in the copy it avoids), and
# a test pinning the rejected choice would turn the next measurement into
# "fixing a broken test". Where the device changes the *answer* -- float64 and
# complex einsum on MPS -- there are tests above. Where it changes only the
# speed, the place is a benchmark.
# --------------------------------------------------------------------- #
class TestTheIntegerMeshIsMemoised:
    """
    `reciprocal_vectors` rebuilds `(3, Nx, Ny, Nz)` of integers on every call,
    and it is called several times per batch while depending only on
    `(shape, device, dtype)`. Memoising it is 1.6 % of training time -- small,
    consistent, and free. What it must not change is a single number.
    """

    @pytest.mark.parametrize("device", ["cpu"] + ACCELERATORS)
    def test_reciprocal_vectors_match_cpu(self, device):
        """The memoised mesh must be identical to the rebuilt one.

        On a triclinic cell, which is the ill-conditioned case the float64
        host round-trip exists for.
        """
        cell = torch.tensor([[4.0, 0.2, 0.1], [0.3, 5.0, 0.4], [0.1, 0.2, 6.0]],
                            dtype=torch.float32).unsqueeze(0)
        reference = reciprocal_vectors(cell, (8, 10, 12), device="cpu")
        result = reciprocal_vectors(cell.to(device), (8, 10, 12), device=device)
        torch.testing.assert_close(result.cpu(), reference,
                                   rtol=1e-6, atol=1e-6)

    def test_a_second_call_returns_the_same_mesh(self):
        """The point of the cache, stated as an identity rather than a timing."""
        from poraque.ml.physics import _integer_mesh

        first = _integer_mesh((6, 6, 6), torch.device("cpu"), torch.float32)
        second = _integer_mesh((6, 6, 6), torch.device("cpu"), torch.float32)
        assert first is second

    def test_a_list_shape_keys_the_same_entry_as_a_tuple(self):
        """A list is unhashable, so `reciprocal_vectors` must normalise before
        it reaches the cache -- and a `torch.Size` must not key a second one."""
        cell = torch.eye(3).unsqueeze(0) * 5.0
        as_list = reciprocal_vectors(cell, [4, 4, 4])
        as_size = reciprocal_vectors(cell, torch.Size([4, 4, 4]))
        torch.testing.assert_close(as_list, as_size)

    def test_the_shapes_of_one_dataset_fit_in_the_cache(self):
        """
        `ShapeBucketSampler` batches by grid shape, so the number of distinct
        shapes is small and stable -- 19 in the set this was measured on. The
        cache is sized against that, and an entry per shape must not evict the
        one before it.
        """
        from poraque.ml.physics import _integer_mesh

        _integer_mesh.cache_clear()
        cell = torch.eye(3).unsqueeze(0) * 5.0
        for n in range(4, 23):
            reciprocal_vectors(cell, (n, n, n))
        assert _integer_mesh.cache_info().currsize == 19


# --------------------------------------------------------------------- #
# Cross-backend numerical agreement
# --------------------------------------------------------------------- #
@requires_accelerator
@pytest.mark.gpu
@pytest.mark.parametrize("device", ACCELERATORS)
class TestBackendAgreement:
    def test_complex_contract_on_contiguous_operands(self, device):
        torch.manual_seed(0)
        x = torch.randn(1, 4, 6, 6, 6, dtype=torch.cfloat)
        w = torch.randn(4, 4, 6, 6, 6, dtype=torch.cfloat)
        reference = torch.einsum("bixyz,ioxyz->boxyz", x, w)
        result = complex_contract("bixyz,ioxyz->boxyz", x.to(device),
                                  w.to(device)).cpu()
        assert (reference - result).abs().max() < 1e-4 * reference.abs().max()

    def test_complex_contract_on_strided_operands(self, device):
        """The regression guard: strided views silently broke on MPS.

        ``x_ft[..., nx-m1:, ny-m2:, :m3]`` is exactly the high-frequency corner
        the spectral convolution reads. Without ``.contiguous()`` this returned
        values 40-90 % wrong, with no error raised.
        """
        torch.manual_seed(0)
        nx, ny, nzh, m = 10, 10, 6, 3
        x = torch.randn(1, 4, nx, ny, nzh, dtype=torch.cfloat)
        w = torch.randn(4, 4, 4, m, m, m, dtype=torch.cfloat)
        equation = "bixyz,ioxyz->boxyz"

        corners = [(slice(None, m), slice(None, m)),
                   (slice(None, m), slice(ny - m, None)),
                   (slice(nx - m, None), slice(None, m)),
                   (slice(nx - m, None), slice(ny - m, None))]
        for index, (s1, s2) in enumerate(corners):
            reference = torch.einsum(equation, x[:, :, s1, s2, :m],
                                     w[index, :, :, :m, :m, :m])
            result = complex_contract(
                equation, x.to(device)[:, :, s1, s2, :m],
                w.to(device)[index, :, :, :m, :m, :m],
            ).cpu()
            assert (reference - result).abs().max() < 1e-4 * reference.abs().max()

    def test_spectral_conv_matches_cpu(self, device):
        torch.manual_seed(0)
        layer = SpectralConv3d(8, 8, modes=(4, 4, 4)).eval()
        x = torch.randn(1, 8, 16, 16, 16)
        with torch.no_grad():
            reference = layer(x)
            result = copy.deepcopy(layer).to(device)(x.to(device)).cpu()
        assert (reference - result).abs().max() < 1e-4 * reference.abs().max()

    @pytest.mark.parametrize("shape,cell_diagonal", [
        ((16, 16, 16), (6.0, 6.0, 6.0)),
        ((12, 16, 20), (5.0, 6.0, 7.0)),
    ])
    def test_full_model_matches_cpu(self, device, shape, cell_diagonal):
        torch.manual_seed(0)
        model = FNO3d(width=12, modes=4, n_layers=2, projection_channels=24).eval()
        x = torch.randn(1, 1, *shape)
        cell = torch.diag(torch.tensor(cell_diagonal)).unsqueeze(0)
        with torch.no_grad():
            reference = model(x, cell)
            result = copy.deepcopy(model).to(device)(x.to(device),
                                                     cell.to(device)).cpu()
        assert (reference - result).abs().max() < 1e-3 * reference.abs().max()

    def test_training_runs_and_stays_finite(self, device):
        """A full optimiser step, including complex-parameter gradient clipping."""
        from poraque.ml.training import clip_gradients

        torch.manual_seed(0)
        model = FNO3d(width=8, modes=4, n_layers=2, projection_channels=16).to(device)
        x = torch.randn(1, 1, 16, 16, 16, device=device)
        cell = torch.eye(3, device=device).unsqueeze(0) * 6.0
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(x, cell).pow(2).mean().backward()
            norm = clip_gradients(model.parameters(), 1.0)
            assert np.isfinite(norm)
            optimizer.step()

        synchronize(device)
        assert all(torch.isfinite(p).all() for p in model.parameters())


# --------------------------------------------------------------------- #
# Gradient clipping with complex parameters
# --------------------------------------------------------------------- #
class TestClipGradients:
    def test_matches_torch_for_real_parameters(self):
        from poraque.ml.training import clip_gradients

        torch.manual_seed(0)
        def make():
            return [torch.nn.Parameter(torch.randn(4, 4)) for _ in range(3)]


        ours, theirs = make(), make()
        for a, b in zip(ours, theirs):
            b.data.copy_(a.data)
            gradient = torch.randn(4, 4) * 10
            a.grad, b.grad = gradient.clone(), gradient.clone()

        our_norm = clip_gradients(ours, 1.0)
        their_norm = float(torch.nn.utils.clip_grad_norm_(theirs, 1.0))
        assert our_norm == pytest.approx(their_norm, rel=1e-5)
        for a, b in zip(ours, theirs):
            assert torch.allclose(a.grad, b.grad, atol=1e-6)

    def test_handles_complex_parameters(self):
        """torch's helper raises here; the FNO's spectral weights are complex."""
        from poraque.ml.training import clip_gradients

        torch.manual_seed(0)
        parameter = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.cfloat))
        parameter.grad = torch.randn(4, 4, dtype=torch.cfloat) * 100

        expected = torch.view_as_real(parameter.grad).pow(2).sum().sqrt()
        norm = clip_gradients([parameter], 1.0)
        assert norm == pytest.approx(float(expected), rel=1e-5)
        # After clipping the norm is at most max_norm.
        clipped = torch.view_as_real(parameter.grad).pow(2).sum().sqrt()
        assert float(clipped) <= 1.0 + 1e-4

    def test_no_op_below_the_threshold(self):
        from poraque.ml.training import clip_gradients

        parameter = torch.nn.Parameter(torch.zeros(4))
        parameter.grad = torch.full((4,), 0.1)
        before = parameter.grad.clone()
        clip_gradients([parameter], 100.0)
        assert torch.allclose(parameter.grad, before)

    def test_handles_missing_gradients(self):
        from poraque.ml.training import clip_gradients

        assert clip_gradients([torch.nn.Parameter(torch.zeros(4))], 1.0) == 0.0
        assert clip_gradients([], 1.0) == 0.0
