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

from poraque.ml.device import (  # noqa: E402
    PREFERENCE_ORDER,
    available_devices,
    cuda_available,
    describe_device,
    empty_cache,
    mps_available,
    resolve_device,
    supports_float64,
    synchronize,
)
from poraque.ml.fno import FNO3d, SpectralConv3d, complex_contract  # noqa: E402
from poraque.ml.physics import cell_reciprocal, cell_volume  # noqa: E402

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
# Cross-backend numerical agreement
# --------------------------------------------------------------------- #
@requires_accelerator
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
