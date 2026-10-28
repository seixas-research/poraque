# -*- coding: utf-8 -*-
# file: test_truncated_fft.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The spectral layers transform only the modes they keep.

A layer retaining ``m`` of ``N`` modes per axis used to take the whole
``rfftn``, slice it, and ``irfftn`` a zero-padded spectrum back: most of the
transform spent on coefficients it discarded or zeros it invented.
:func:`truncated_rfftn` and :func:`padded_irfftn` do the separable transform in
stages, so only the first (the real FFT along z) touches the full grid. That is
a change of algorithm, not of function, and the tests hold it to that: the
block is the one the full transform contains, and the gradients, which the
training loop uses, are the adjoint of the same operation.
"""

import pytest
import torch

from poraque.ml.fno import (PointwiseConv3d, RadialSpectralConv3d,
                            SpectralConv3d, padded_irfftn, truncated_rfftn)

SHAPES = [(16, 16, 16), (20, 24, 28), (17, 18, 15), (16, 16, 6)]


def full_block(x, m1, m2, m3):
    """The retained block cut out of the whole ``rfftn``: the old route."""
    spectrum = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="forward")
    rows = torch.cat([spectrum[..., :m1, :, :], spectrum[..., -m1:, :, :]],
                     dim=-3)
    return torch.cat([rows[..., :m2, :], rows[..., -m2:, :]], dim=-2)[..., :m3]


def full_inverse(block, shape):
    """``irfftn`` of the block zero-padded into a whole spectrum: the old route."""
    nx, ny, nz = shape
    batch, channels, rows, columns, m3 = block.shape
    m1, m2 = rows // 2, columns // 2
    spectrum = torch.zeros(batch, channels, nx, ny, nz // 2 + 1,
                           dtype=block.dtype)
    spectrum[:, :, :m1, :m2, :m3] = block[:, :, :m1, :m2]
    spectrum[:, :, :m1, ny - m2:, :m3] = block[:, :, :m1, m2:]
    spectrum[:, :, nx - m1:, :m2, :m3] = block[:, :, m1:, :m2]
    spectrum[:, :, nx - m1:, ny - m2:, :m3] = block[:, :, m1:, m2:]
    return torch.fft.irfftn(spectrum, s=shape, dim=(-3, -2, -1),
                            norm="forward")


class TestTheTruncatedTransformIsTheRetainedBlockOfTheFullOne:

    @pytest.mark.parametrize("shape", SHAPES)
    def test_forward_block(self, shape):
        torch.manual_seed(0)
        x = torch.randn(2, 3, *shape, dtype=torch.float64)
        m = (4, 3, min(4, shape[2] // 2 + 1))
        assert torch.allclose(truncated_rfftn(x, *m), full_block(x, *m),
                              atol=1e-13)

    @pytest.mark.parametrize("shape", SHAPES)
    def test_inverse(self, shape):
        torch.manual_seed(1)
        m = (4, 3, min(4, shape[2] // 2 + 1))
        block = torch.randn(2, 3, 2 * m[0], 2 * m[1], m[2],
                            dtype=torch.complex128)
        assert torch.allclose(padded_irfftn(block, shape),
                              full_inverse(block, shape), atol=1e-13)


class TestTheLayersAreUnchangedByIt:
    """Outputs, input gradients and weight gradients, against the full route."""

    @staticmethod
    def reference(layer, x, cell):
        """The layer with ``truncated_rfftn``/``padded_irfftn`` swapped for the
        full transform, by substituting the module functions."""
        from poraque.ml import fno

        saved = fno.truncated_rfftn, fno.padded_irfftn
        fno.truncated_rfftn = lambda t, a, b, c: full_block(t, a, b, c)
        fno.padded_irfftn = full_inverse
        try:
            return layer(x) if cell is None else layer(x, cell=cell)
        finally:
            fno.truncated_rfftn, fno.padded_irfftn = saved

    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("radial", [False, True])
    def test_forward_and_backward(self, shape, radial):
        from poraque.ml.fno import set_precision

        torch.manual_seed(2)
        modes = (5, 4, 4)
        layer = set_precision(
            (RadialSpectralConv3d if radial else SpectralConv3d)(3, 4, modes),
            "float64")
        cell = (torch.eye(3, dtype=torch.float64) * 7.0).expand(2, 3, 3) \
            if radial else None
        x = torch.randn(2, 3, *shape, dtype=torch.float64, requires_grad=True)

        def run(use_reference):
            layer.zero_grad()
            x.grad = None
            y = (self.reference(layer, x, cell) if use_reference else
                 (layer(x, cell=cell) if radial else layer(x)))
            y.square().sum().backward()
            return (y.detach().clone(), x.grad.clone(),
                    [p.grad.clone() for p in layer.parameters()])

        fast, slow = run(False), run(True)
        assert torch.allclose(fast[0], slow[0], atol=1e-12)
        assert torch.allclose(fast[1], slow[1], atol=1e-11)
        for a, b in zip(fast[2], slow[2]):
            assert torch.allclose(a, b, atol=1e-11)


class TestThePointwiseConvolutionIsStillAConvolution:
    """
    ``PointwiseConv3d`` contracts over channels on MPS (1.4-3x faster than the
    convolution kernel there) and calls ``conv3d`` elsewhere. It must remain,
    to everything that looks, an ``nn.Conv3d``: same state dict, same shapes,
    which is what LoRA and old checkpoints rely on.
    """

    def test_it_is_a_conv3d_with_the_same_parameters(self):
        layer = PointwiseConv3d(5, 7)
        reference = torch.nn.Conv3d(5, 7, kernel_size=1)
        assert isinstance(layer, torch.nn.Conv3d)
        assert tuple(layer.kernel_size) == (1, 1, 1)
        assert {k: v.shape for k, v in layer.state_dict().items()} == \
            {k: v.shape for k, v in reference.state_dict().items()}
        reference.load_state_dict(layer.state_dict())
        x = torch.randn(2, 5, 4, 4, 4)
        assert torch.equal(layer(x), reference(x))

    @pytest.mark.gpu
    def test_the_accelerator_path_agrees_with_conv3d(self):
        if not torch.backends.mps.is_available():
            pytest.skip("the contraction path is MPS-only")
        torch.manual_seed(0)
        cpu = PointwiseConv3d(5, 7)
        mps = PointwiseConv3d(5, 7).to("mps")
        mps.load_state_dict(cpu.state_dict())
        x = torch.randn(2, 5, 6, 6, 6, requires_grad=True)
        xm = x.detach().to("mps").requires_grad_(True)
        y, ym = cpu(x), mps(xm)
        y.square().sum().backward()
        ym.square().sum().backward()
        assert torch.allclose(ym.cpu(), y, atol=1e-5)
        assert torch.allclose(xm.grad.cpu(), x.grad, atol=1e-4, rtol=1e-4)
        assert torch.allclose(mps.weight.grad.cpu(), cpu.weight.grad,
                              atol=1e-3, rtol=1e-4)
