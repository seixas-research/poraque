# -*- coding: utf-8 -*-
# file: test_backend.py

"""
Tests for the optional C backend of the spectral contraction.

Two properties matter and they pull against each other. The kernel must be
*fast*, which is why it exists; and it must be *invisible*, which is harder --
an optimisation that changes a number, or that quietly detaches the autograd
graph, is worse than no optimisation at all, because nothing downstream
reports it.

Most of what follows therefore checks the second property.
"""

import os

import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.ml import backend                              # noqa: E402
from poraque.ml.fno import FNO3d, SpectralConv3d, complex_contract  # noqa: E402

EQUATION = "bixyz,ioxyz->boxyz"


@pytest.fixture
def compiled():
    """The loaded backend, or a skip when this machine cannot build it."""
    backend.reset()
    if not backend.available():
        pytest.skip(f"C backend unavailable: {backend.describe()}")
    yield backend
    backend.reset()


@pytest.fixture
def operands():
    def build(batch=1, in_channels=8, out_channels=8, modes=(4, 4, 3),
              dtype=torch.cfloat, seed=0):
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(batch, in_channels, *modes, dtype=dtype,
                        generator=generator)
        weight = torch.randn(in_channels, out_channels, *modes, dtype=dtype,
                             generator=generator)
        return x, weight

    return build


# ===================================================================== #
# It computes the right thing
# ===================================================================== #
class TestNumericalAgreement:
    def test_selftest_passes(self, compiled):
        report = compiled.selftest()
        assert report["available"]
        assert report["max_relative"] < 2e-5

    @pytest.mark.parametrize("batch,in_ch,out_ch,modes", [
        (1, 8, 8, (4, 4, 3)),
        (1, 32, 32, (12, 12, 12)),
        (3, 16, 8, (5, 6, 7)),
        (1, 1, 1, (1, 1, 1)),
        (2, 4, 9, (2, 3, 1)),
    ])
    def test_matches_einsum(self, compiled, operands, batch, in_ch, out_ch,
                            modes):
        x, weight = operands(batch, in_ch, out_ch, modes)
        expected = torch.einsum(EQUATION, x, weight)
        got = compiled.contract(x, weight)
        assert got is not None
        assert got.shape == expected.shape
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_double_precision_is_exact_to_double_rounding(self, compiled,
                                                          operands):
        x, weight = operands(1, 8, 8, (4, 4, 4), dtype=torch.cdouble)
        expected = torch.einsum(EQUATION, x, weight)
        got = compiled.contract(x, weight)
        assert got is not None and got.dtype == torch.cdouble
        assert torch.allclose(got, expected, rtol=1e-12, atol=1e-14)

    def test_a_strided_operand_is_materialised_first(self, compiled):
        """
        The failure this module must not have.

        The caller hands the kernel *corner views* of the spectral tensor --
        ``x_ft[:, :, nx-m:, ny-m:, :m]`` -- which are strided. The kernel walks
        raw memory, so reading a strided operand as if it were contiguous is
        silently wrong rather than an error.
        """
        generator = torch.Generator().manual_seed(1)
        full = torch.randn(1, 8, 10, 10, 6, dtype=torch.cfloat,
                           generator=generator)
        weight = torch.randn(8, 8, 3, 3, 2, dtype=torch.cfloat,
                             generator=generator)

        view = full[:, :, 7:, 7:, :2]
        assert not view.is_contiguous(), "test no longer exercises striding"

        expected = torch.einsum(EQUATION, view, weight)
        got = compiled.contract(view, weight)
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_it_writes_rather_than_accumulates(self, compiled, operands):
        """Calling twice with the same operands must give the same answer."""
        x, weight = operands()
        first = compiled.contract(x, weight)
        second = compiled.contract(x, weight)
        assert torch.equal(first, second)


# ===================================================================== #
# It refuses what it cannot serve
# ===================================================================== #
class TestFallback:
    def test_an_unsupported_dtype_is_declined(self, compiled):
        x = torch.randn(1, 4, 2, 2, 2)
        weight = torch.randn(4, 4, 2, 2, 2)
        assert compiled.contract(x, weight) is None

    def test_mismatched_channels_are_declined(self, compiled):
        x = torch.randn(1, 4, 2, 2, 2, dtype=torch.cfloat)
        weight = torch.randn(5, 4, 2, 2, 2, dtype=torch.cfloat)
        assert compiled.contract(x, weight) is None

    def test_mismatched_modes_are_declined(self, compiled):
        x = torch.randn(1, 4, 2, 2, 2, dtype=torch.cfloat)
        weight = torch.randn(4, 4, 3, 2, 2, dtype=torch.cfloat)
        assert compiled.contract(x, weight) is None

    def test_mixed_dtypes_are_declined(self, compiled):
        x = torch.randn(1, 4, 2, 2, 2, dtype=torch.cfloat)
        weight = torch.randn(4, 4, 2, 2, 2, dtype=torch.cdouble)
        assert compiled.contract(x, weight) is None

    def test_the_environment_can_switch_it_off(self, monkeypatch):
        monkeypatch.setenv("PORAQUE_C_BACKEND", "0")
        backend.reset()
        try:
            assert not backend.enabled()
            assert not backend.available()
            assert "disabled" in backend.describe()
        finally:
            backend.reset()

    def test_describe_never_raises(self, monkeypatch):
        monkeypatch.setenv("PORAQUE_C_BACKEND", "0")
        backend.reset()
        try:
            assert isinstance(backend.describe(), str)
        finally:
            backend.reset()


# ===================================================================== #
# It is invisible to the model
# ===================================================================== #
class TestModelIsUnchanged:
    @staticmethod
    def _predict(shape=(16, 16, 16), width=8, modes=4, layers=2, seed=0):
        torch.manual_seed(seed)
        model = FNO3d(in_channels=1, out_channels=1, width=width, modes=modes,
                      n_layers=layers).eval()
        generator = torch.Generator().manual_seed(seed + 100)
        x = torch.randn(1, 1, *shape, generator=generator)
        cell = torch.eye(3).unsqueeze(0) * 8.0
        with torch.no_grad():
            return model(x, cell)

    def test_inference_agrees_with_the_pytorch_path(self, monkeypatch):
        monkeypatch.setenv("PORAQUE_C_BACKEND", "0")
        backend.reset()
        reference = self._predict()

        monkeypatch.setenv("PORAQUE_C_BACKEND", "1")
        backend.reset()
        if not backend.available():
            pytest.skip(f"C backend unavailable: {backend.describe()}")
        accelerated = self._predict()
        backend.reset()

        scale = reference.abs().max().item() or 1.0
        assert (accelerated - reference).abs().max().item() / scale < 1e-5

    def test_a_spectral_layer_agrees(self, monkeypatch):
        torch.manual_seed(3)
        conv = SpectralConv3d(8, 8, modes=(4, 4, 4)).eval()
        x = torch.randn(1, 8, 16, 16, 16)

        monkeypatch.setenv("PORAQUE_C_BACKEND", "0")
        backend.reset()
        with torch.no_grad():
            reference = conv(x)

        monkeypatch.setenv("PORAQUE_C_BACKEND", "1")
        backend.reset()
        if not backend.available():
            pytest.skip("C backend unavailable")
        with torch.no_grad():
            accelerated = conv(x)
        backend.reset()

        assert torch.allclose(accelerated, reference, rtol=1e-4, atol=1e-6)


# ===================================================================== #
# It must never touch training
# ===================================================================== #
class TestAutogradIsUntouched:
    """
    The kernel records nothing on the autograd tape.

    If it ran while a graph was being built, the spectral weights would receive
    no gradient and the model would train to a constant -- with no error raised
    anywhere, and a loss curve that merely looks disappointing. The guard is
    ``torch.is_grad_enabled()``; these tests hold it in place.
    """

    def test_the_c_path_is_skipped_while_grad_is_enabled(self, compiled):
        x = torch.randn(1, 8, 4, 4, 4, dtype=torch.cfloat, requires_grad=True)
        weight = torch.randn(8, 8, 4, 4, 4, dtype=torch.cfloat,
                             requires_grad=True)
        assert torch.is_grad_enabled()
        result = complex_contract(EQUATION, x, weight)
        assert result.grad_fn is not None, (
            "the contraction returned a tensor with no autograd history")

    def test_the_c_path_is_taken_under_no_grad(self, compiled):
        x = torch.randn(1, 8, 4, 4, 4, dtype=torch.cfloat)
        weight = torch.randn(8, 8, 4, 4, 4, dtype=torch.cfloat)
        with torch.no_grad():
            result = complex_contract(EQUATION, x, weight)
        assert result.grad_fn is None
        assert torch.allclose(result, torch.einsum(EQUATION, x, weight),
                              rtol=1e-4, atol=1e-5)

    def test_spectral_weights_still_receive_gradients(self, compiled):
        torch.manual_seed(0)
        model = FNO3d(in_channels=1, out_channels=1, width=8, modes=4,
                      n_layers=2).train()
        x = torch.randn(1, 1, 16, 16, 16)
        cell = torch.eye(3).unsqueeze(0) * 5.0
        model(x, cell).pow(2).mean().backward()

        spectral = [parameter for parameter in model.parameters()
                    if parameter.is_complex()]
        assert spectral, "no complex spectral weights in the model"
        for parameter in spectral:
            assert parameter.grad is not None
            assert parameter.grad.abs().max().item() > 0

    def test_a_training_step_changes_the_spectral_weights(self, compiled):
        torch.manual_seed(0)
        model = FNO3d(in_channels=1, out_channels=1, width=8, modes=4,
                      n_layers=1).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        x = torch.randn(1, 1, 16, 16, 16)
        cell = torch.eye(3).unsqueeze(0) * 5.0

        spectral = next(p for p in model.parameters() if p.is_complex())
        before = spectral.detach().clone()
        model(x, cell).pow(2).mean().backward()
        optimizer.step()
        assert not torch.equal(spectral.detach(), before)


# ===================================================================== #
# Build and cache behaviour
# ===================================================================== #
class TestBuild:
    def test_the_fingerprint_tracks_the_source(self, compiled, monkeypatch,
                                               tmp_path):
        first = backend.source_fingerprint()

        decoy = tmp_path / "_spectral.c"
        decoy.write_text("/* different */\n")
        monkeypatch.setattr(backend, "SOURCE", str(decoy))
        assert backend.source_fingerprint() != first, (
            "editing the kernel must not reuse the cached library")

    def test_the_library_lives_under_the_cache_directory(self, compiled):
        assert backend.library_path().startswith(backend.cache_dir())

    def test_the_cache_directory_is_configurable(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PORAQUE_CACHE_DIR", str(tmp_path))
        assert backend.cache_dir() == str(tmp_path)

    def test_a_missing_source_reports_rather_than_raises(self, monkeypatch,
                                                        tmp_path):
        monkeypatch.setattr(backend, "SOURCE", str(tmp_path / "absent.c"))
        monkeypatch.setenv("PORAQUE_CACHE_DIR", str(tmp_path))
        backend.reset()
        try:
            messages = []
            assert backend.build(force=True, log=messages.append) is None
            assert any("no source" in line for line in messages)
        finally:
            backend.reset()

    def test_the_source_ships_with_the_package(self):
        assert os.path.exists(backend.SOURCE), (
            "the C source must be installed alongside the module, or the "
            "backend can never be built from a wheel")


class TestThreading:
    """
    Threading is pthreads, not OpenMP.

    PyTorch already links an OpenMP runtime; a second one in the same process
    is what libomp reports as ``OMP: Error #15 ... can degrade performance or
    cause incorrect results``, whose only workaround its own documentation
    calls unsafe. pthreads have no runtime to collide with.
    """

    @pytest.mark.parametrize("threads", [1, 2, 4, 8])
    def test_every_thread_count_gives_the_same_answer(self, compiled, operands,
                                                      threads):
        x, weight = operands(2, 16, 16, (5, 4, 3))
        expected = torch.einsum(EQUATION, x, weight)
        got = compiled.contract(x, weight, threads=threads)
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_more_threads_than_rows_is_harmless(self, compiled, operands):
        x, weight = operands(1, 4, 2, (2, 2, 2))
        expected = torch.einsum(EQUATION, x, weight)
        got = compiled.contract(x, weight, threads=64)
        assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_zero_or_negative_threads_is_clamped(self, compiled, operands):
        x, weight = operands()
        expected = torch.einsum(EQUATION, x, weight)
        for threads in (0, -4):
            got = compiled.contract(x, weight, threads=threads)
            assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_the_build_reports_which_it_got(self, compiled):
        assert ("pthreads" in compiled.describe()
                or "serial" in compiled.describe())
