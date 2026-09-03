# -*- coding: utf-8 -*-
# file: test_cuequivariance_gpu.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Rotational equivariance, asserted on an NVIDIA GPU. **Not run locally.**

Run it on the cluster, deliberately::

    pytest tests/test_cuequivariance_gpu.py -v -m gpu

Every test here skips without CUDA, so the ordinary suite passes on a machine
with no accelerator and ``pytest -m "not gpu"`` deselects the file outright.

**It has now been run**, on a Tesla V100 at LNCC, and it failed on first
contact — five tests, of which four were one bug in :func:`rotate_field`
(``addmm_cuda`` has no ``Long`` kernel, so the exact voxel permutation could
not be built on the device at all) and one was a real defect in the shipped
layer that nothing else had caught. Both are fixed and both are commented at
the site, because "written, linted, collected and never run" is a state with a
characteristic failure mode and this file is the record of it.

Why the file is called this, and does not import ``cuequivariance``
------------------------------------------------------------------
It was asked for under that name, and the name is worth keeping because it is
the question people arrive with. The answer is that there is nothing here for
the library to accelerate, and the reason is structural rather than a matter of
effort.

``cuequivariance`` implements arithmetic on *irreducible representations of
O(3)* — Clebsch-Gordan tensor products, spherical harmonics, symmetric
contractions — for models whose hidden features carry angular momentum. This
operator maps a scalar field to a scalar field on a grid. Every feature it
carries is :math:`\ell = 0`, every tensor product between two of them is a
scalar multiply, and the Fourier layer is a complex ``einsum`` diagonal in the
mode index and dense in channels, which is not an expressible CG contraction at
all. Profiled by CUDA kernel on a V100, the share of a training step that
belongs to irrep arithmetic is 0.0 %.

Equivariance here is therefore obtained *by construction* rather than by a
kernel library: constrain the Fourier multiplier to a function of
:math:`|\mathbf{G}|` and the layer is a convolution with a radial kernel, which
commutes with every rotation exactly. See
:class:`~poraque.ml.fno.RadialSpectralConv3d`, and
``tests/test_equivariance.py`` for the device-agnostic half of these
assertions — the property is a property of the architecture, so the bulk of it
is asserted where it will actually be run.

What this file adds over that one
---------------------------------
Three things that only a GPU can answer:

**TF32 is on by default and it is not a rounding detail.** ``training.tf32``
defaults to true, and TF32 carries ten explicit mantissa bits — a relative
precision near :math:`10^{-3}`. An equivariance check under TF32 measures the
matmul, not the architecture. Every test below turns it off and one of them
measures what happens when it is left on, so a future reader who sees
:math:`10^{-3}` on a cluster and :math:`10^{-7}` on a laptop knows immediately
which of the two is the anomaly.

**Batched cells.** The radial basis is built per sample from that sample's own
cell. On CPU the tests use a batch of one; here the batch mixes cells, which is
where a wrongly broadcast ``(B, 3, 3)`` would show up.

**Scale.** Widths and grids that are worth the transfer, and a check that the
same weights still serve several shapes on the device.

Where the irreps would come in, if ever
---------------------------------------
There is one honest way for this architecture to acquire :math:`\ell > 0`
features, and it is already half-present: the three fractional-coordinate
channels of :attr:`~poraque.ml.fno.FNO3d.use_coordinates` *are* an
:math:`\ell = 1` object, which is exactly why they have to be switched off for
the equivariant variant rather than merely being unhelpful. A design that
carried vector and rank-2 hidden features — obtained in Fourier space as
:math:`\hat{\mathbf{G}}` and :math:`\hat{\mathbf{G}}\hat{\mathbf{G}}` times the
scalar channels — would have genuine CG products to compute, and *that* is the
architecture ``cuequivariance`` exists for. It is a research question, not a
configuration flag, and nothing in this package implements it.
"""

import itertools

import pytest
import torch

from poraque.ml.fno import FNO3d

pytestmark = pytest.mark.gpu

CUDA = pytest.mark.skipif(not torch.cuda.is_available(),
                          reason="requires a CUDA device")

#: TF32 is an Ampere feature, and the machine this file was addressed to has
#: none. `sequana_gpu` at LNCC is Tesla V100, `sm_70`, and
#: :func:`~poraque.ml.device.enable_tf32`'s own docstring says the flag is a
#: no-op before Ampere — so the test that measures TF32's effect asserted a
#: tenfold degradation on hardware physically incapable of producing one, and
#: reported a failure on precisely the cluster it was written for.
AMPERE = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() < (8, 0),
    reason="TF32 needs compute capability 8.0; a V100 is sm_70 and has none")


# ---------------------------------------------------------------------- #
# Helpers — deliberately duplicated from tests/test_equivariance.py.
#
# This file has to be runnable on a cluster checkout in isolation, and a
# cross-import between two test modules is the kind of coupling that turns one
# broken collection into two. They are twelve lines.
# ---------------------------------------------------------------------- #
def octahedral_rotations():
    """The 24 proper rotations of the cube, as integer 3×3 matrices."""
    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            matrix = torch.zeros(3, 3, dtype=torch.long)
            for axis, target in enumerate(permutation):
                matrix[target, axis] = signs[axis]
            if int(round(float(torch.linalg.det(matrix.double())))) == 1:
                matrices.append(matrix)
    return matrices


def rotate_field(field, rotation):
    r"""
    Apply ``rotation`` to a periodic ``(B, C, Nx, Ny, Nz)`` field.

    :math:`f'(\mathbf r) = f(R^{-1}\mathbf r)`, which on a periodic grid is a
    permutation of voxels: exact, with no interpolation to be mistaken for the
    error being measured.
    """
    shape = field.shape[-3:]
    # The permutation is built on the **host** and moved once, finished.
    # `addmm_cuda` is not implemented for Long: an int64 matmul works on the
    # CPU and on MPS and has no CUDA kernel at all, so doing this arithmetic on
    # the device raised before any test could measure anything, and every
    # rotation test in this file failed for that one reason. Casting to float
    # to reach a CUDA kernel is not the fix -- it would put a rounding step
    # inside the exactness this permutation exists to provide.
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(n) for n in shape], indexing="ij"))
    inverse = torch.linalg.inv(rotation.double().cpu()).round().long()
    source = (inverse @ grid.reshape(3, -1)).reshape(3, *shape)
    source = torch.stack([source[i] % shape[i]
                          for i in range(3)]).to(field.device)
    return field[..., source[0], source[1], source[2]]


@pytest.fixture
def exact_matmuls():
    """
    TF32 off for the duration of a test, and restored afterwards.

    ``training.tf32`` defaults to true and Poraquê's own training script turns
    it on process-wide, so a test that merely *assumed* full precision would
    measure whatever the last run left behind.
    """
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = matmul
    torch.backends.cudnn.allow_tf32 = cudnn


def sample_field(shape=(32, 32, 32), seed=0, batch=1,
                 dtype=torch.float32):
    """
    A test field on the device, drawn from a **stated** seed.

    The dense layer's rotation error is a property of the field as much as of
    the weights — across seeds it runs from 0.086 to 0.93 — so a counterfactual
    compared against an unseeded ``torch.randn`` is a threshold measured
    against whatever the rest of the session left in the global RNG. The same
    repair was made in ``tests/test_equivariance.py`` and did not reach this
    file, which is what a file written and never run costs.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 1, *shape, generator=generator,
                       dtype=dtype).cuda()


def build(equivariant, width=16, seed=7, **kwargs):
    """A model on the GPU, deterministically initialised."""
    torch.manual_seed(seed)
    kwargs.setdefault("use_coordinates", not equivariant)
    kwargs.setdefault("cell_conditioning", True)
    return FNO3d(in_channels=1, out_channels=1, width=width, modes=8,
                 n_layers=3, projection_channels=64,
                 equivariant=equivariant, **kwargs).eval().cuda()


def worst_rotation_error(model, field, cell, rotations):
    """Largest relative deviation of ``f(Rx)`` from ``R f(x)``."""
    worst = 0.0
    with torch.no_grad():
        reference = model(field, cell)
        scale = reference.abs().max()
        for rotation in rotations:
            deviation = (model(rotate_field(field, rotation), cell)
                         - rotate_field(reference, rotation)).abs().max()
            worst = max(worst, float(deviation / scale))
    return worst


def cubic(edge=8.0, batch=1, device="cuda"):
    return (torch.eye(3, device=device) * edge).unsqueeze(0).repeat(batch, 1, 1)


# ---------------------------------------------------------------------- #
@CUDA
class TestTheRotationPropertyHoldsOnTheDevice:
    """
    The claim itself, at a width and a grid worth putting on a GPU.

    The dense baseline is asserted to fail in the same breath. Equivariance is
    a claim a test can pass for the wrong reason — a model whose output barely
    depends on its input is trivially equivariant — and the counterfactual is
    what rules that out.
    """

    def test_all_24_rotations_of_the_cube(self, exact_matmuls):
        error = worst_rotation_error(build(True), sample_field(), cubic(),
                                     octahedral_rotations())
        assert error < 1e-4, f"worst relative deviation {error:.3e}"

    def test_the_dense_operator_does_not(self, exact_matmuls):
        """
        The counterfactual, stated as a **ratio** rather than a threshold.

        The dense layer's rotation error is a property of the field as well as
        of the weights — 0.086 at one seed, 0.93 at another — so a bare
        ``> 0.1`` is a bound compared against whatever the session left in the
        global RNG. The claim that survives is scale-free: on the *same* field,
        the dense layer is wrong by orders of magnitude more than the radial
        one.
        """
        field = sample_field()
        rotations = octahedral_rotations()
        dense = worst_rotation_error(build(False, use_coordinates=False),
                                     field, cubic(), rotations)
        radial = worst_rotation_error(build(True), field, cubic(), rotations)
        assert dense > 1e-3, (
            f"the dense spectral layer came out equivariant to {dense:.3e}, "
            f"which it has no reason to be -- if the field or the model has "
            f"become degenerate the equivariant test above passes for nothing")
        assert dense > 1e3 * radial, f"dense {dense:.3e}, radial {radial:.3e}"

    def test_float64_on_the_device_reaches_machine_precision(self,
                                                             exact_matmuls):
        """
        Separating round-off from a real asymmetry, on the hardware.

        A residual that shrinks with the float is arithmetic; one that does not
        is the retained mode set being lopsided, which is a real bug that once
        sat at 2e-3 and looked entirely plausible.
        """
        model = build(True).double()
        field = sample_field((24, 24, 24), dtype=torch.float64)
        error = worst_rotation_error(model, field, cubic().double(),
                                     octahedral_rotations()[:6])
        assert error < 1e-12, f"worst relative deviation {error:.3e}"


@CUDA
class TestTF32IsTheThingThatWillLookLikeAFailure:
    """
    Ten mantissa bits, and the equivariance measured through them.

    ``training.tf32`` is on by default and the training script sets it
    process-wide, so anyone who checks this property inside a real run will see
    :math:`\\sim10^{-3}` and reasonably conclude the architecture is broken.
    It is not: it is the matmul. Written down as an assertion so the number is
    on record rather than rediscovered.

    **And it is Ampere and later only.** Run on `sequana_gpu` at LNCC — Tesla
    V100, ``sm_70`` — the flag does nothing at all: measured, ``exact`` is
    8.7e-07 and turning ``allow_tf32`` on does not move it. So this test could
    only ever fail there, on the one machine it was addressed to, while being
    right about the hazard everywhere else. Hence :data:`AMPERE`. The
    consequence for anyone debugging on Volta is worth carrying: **you will
    never see the 1e-3 warned about here**, and a 1e-3 measured there has a
    different cause — see
    ``tests/test_equivariance.py::TestTheCutoffDoesNotTieWithItsOwnRoundOff``.
    """

    @AMPERE
    def test_the_error_is_dominated_by_the_matmul_when_tf32_is_on(self):
        matmul = torch.backends.cuda.matmul.allow_tf32
        try:
            field = sample_field()
            rotations = octahedral_rotations()[:6]

            torch.backends.cuda.matmul.allow_tf32 = False
            exact = worst_rotation_error(build(True), field, cubic(),
                                         rotations)
            torch.backends.cuda.matmul.allow_tf32 = True
            approximate = worst_rotation_error(build(True), field, cubic(),
                                               rotations)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul

        assert exact < 1e-4
        # Not a hard bound on TF32 itself -- the point is only that the two
        # regimes are different, and by orders of magnitude rather than by a
        # factor. If this ever stops holding, TF32 has stopped being used.
        assert approximate > 10.0 * max(exact, 1e-9), (
            f"exact={exact:.3e}, tf32={approximate:.3e}: TF32 appears not to "
            f"be in use, so the fixture that disables it is protecting "
            f"nothing")


@CUDA
class TestABatchOfDifferentCells:
    """
    The radial basis is per sample, and a batch is where that goes wrong.

    :meth:`~poraque.ml.fno.RadialSpectralConv3d.radial_basis` builds
    :math:`|\\mathbf{G}|` from each sample's own cell. A wrongly broadcast
    ``(B, 3, 3)`` — one cell silently applied to the whole batch — is invisible
    at batch one, and this operator is trained with a shape-bucketed sampler
    that puts several materials in a batch as a matter of course.
    """

    def test_each_sample_gets_its_own_cell(self, exact_matmuls):
        model = build(True)
        field = sample_field((24, 24, 24), batch=3)
        cells = torch.stack([torch.eye(3, device="cuda") * edge
                             for edge in (6.0, 8.0, 11.0)])
        with torch.no_grad():
            batched = model(field, cells)
            separate = torch.cat([
                model(field[i:i + 1], cells[i:i + 1]) for i in range(3)])
        assert torch.allclose(batched, separate, atol=1e-5)

    def test_the_rotation_property_survives_a_mixed_batch(self, exact_matmuls):
        """
        The test that found a defect it was not written to find.

        It failed at 4.485e-04 on a V100 and the batch was innocent: an
        11 Ang cell fails **on its own**, because its face modes tie with the
        spherical cutoff's own round-off. Batch-of-three-identical equalled
        batch-of-one throughout, so the per-sample basis was broadcast
        correctly all along. The cause and its fix are
        :data:`~poraque.ml.fno.CUTOFF_SLACK`; the 11 Ang cell stays in this
        batch deliberately, as the cheapest possible guard against it coming
        back.
        """
        model = build(True)
        field = sample_field((24, 24, 24), batch=3)
        cells = torch.stack([torch.eye(3, device="cuda") * edge
                             for edge in (6.0, 8.0, 11.0)])
        error = worst_rotation_error(model, field, cells,
                                     octahedral_rotations()[:8])
        assert error < 1e-4, f"worst relative deviation {error:.3e}"


@CUDA
class TestOneSetOfWeightsStillServesEveryGrid:
    """
    The property the whole architecture exists for, on the device.

    Worth re-checking here rather than trusting the CPU test: the radial basis
    allocates a ``(B, R, 2m1, 2m2, m3)`` intermediate whose shape changes with
    the grid, and a device-side cache keyed on the wrong thing would fail only
    on the second shape.
    """

    @pytest.mark.parametrize("shape", [(24, 24, 24), (32, 32, 32),
                                       (20, 28, 36), (48, 48, 48)])
    def test_a_single_model_runs_every_shape(self, shape, exact_matmuls):
        model = build(True)
        with torch.no_grad():
            out = model(torch.randn(1, 1, *shape, device="cuda"), cubic())
        assert tuple(out.shape[-3:]) == shape
        assert torch.isfinite(out).all()


@CUDA
class TestTheGradientFlowsThroughTheRadialKernel:
    """
    Equivariance is worthless if the layer cannot be trained.

    The contraction splits the spectrum into real and imaginary parts and
    contracts each against the same real coefficients. That is arithmetic
    autograd handles, but ``torch.complex`` in the middle of a graph is exactly
    where a backward pass stops being obvious, and a layer that silently
    received no gradient would train to its initialisation and read as a bad
    dataset.
    """

    def test_every_radial_coefficient_receives_a_gradient(self,
                                                          exact_matmuls):
        model = build(True)
        out = model(torch.randn(2, 1, 24, 24, 24, device="cuda"),
                    cubic(batch=2))
        out.square().mean().backward()
        for index, block in enumerate(model.blocks):
            gradient = block.spectral.weight.grad
            assert gradient is not None, f"block {index} received none"
            assert torch.isfinite(gradient).all()
            assert float(gradient.abs().max()) > 0.0


@CUDA
class TestCuequivarianceIsNotOnThisPath:
    """
    The claim in the README, asserted rather than left as prose.

    "Poraquê has no irreps to accelerate" is a statement about the package that
    would quietly stop being true if someone added an irrep layer, and about
    the *dependency* that would stop being true the moment an import appeared.
    Both are cheap to check and neither is checkable by reading.
    """

    def test_no_module_of_the_package_imports_it(self):
        import sys

        import poraque.ml.fno            # noqa: F401  (the import is the test)
        import poraque.ml.training       # noqa: F401

        assert "cuequivariance" not in sys.modules

    def test_the_equivariant_model_carries_only_scalar_features(self):
        """
        Every hidden channel is :math:`\\ell = 0`, which is why there is no
        tensor product to compute anywhere in the forward pass.

        Stated as a shape assertion: a hidden feature map is
        ``(B, C, Nx, Ny, Nz)`` with no angular index. An architecture with
        :math:`\\ell > 0` features would have to carry one.
        """
        model = build(True)
        captured = []
        model.blocks[0].register_forward_hook(
            lambda module, inputs, output: captured.append(output.shape))
        with torch.no_grad():
            model(torch.randn(1, 1, 24, 24, 24, device="cuda"), cubic())
        assert len(captured[0]) == 5
        assert captured[0][1] == model.width
