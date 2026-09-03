# -*- coding: utf-8 -*-
# file: test_equivariance.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Rotational equivariance of the Fourier layer, and the three things it needs.

The Kohn-Sham map is a map between *functions on space*, and space has no
preferred axes: rotate a crystal and its charge density rotates with it. A
grid-based operator has no such guarantee — :class:`SpectralConv3d` learns one
complex number per retained mode and can therefore treat :math:`\mathbf{G}` and
:math:`R\mathbf{G}` differently, which it does, by 93 % of the field's own
amplitude in the very first test below.

Constraining the multiplier to a function of :math:`|\mathbf{G}|` alone fixes
that, and **three** separate conditions have to hold together before the
network as a whole is equivariant. Each one was measured while it was still
broken, and each is asserted here because each fails silently — a model that is
almost equivariant trains, converges, and reports a perfectly ordinary
validation curve.

**The kernel must be radial.** That is :class:`RadialSpectralConv3d`, and it is
the part everyone thinks of.

**The retained set must be a ball, not a box.** A radial multiplier applied
over a box of modes is equivariant only under the box's own symmetry group.
On a cubic cell with :math:`m_1 = m_2 = m_3` the box happens to be invariant
under the twenty-four rotations of the cube, which is exactly why this is easy
to miss; on a tetragonal cell the error is :math:`5\times10^{-2}`.

**The lifting stage must carry no coordinate channels.** The three fractional
coordinates are not three scalar fields. Under a rotation they turn into each
other — they are an :math:`\ell = 1` object handed to a network that treats
every channel as :math:`\ell = 0` — and being absolute positions they cost
translation equivariance as well.

There is a fourth condition that needed no work: every *other* operation in a
Fourier block is already equivariant. The 1×1×1 convolutions and the
activation are pointwise in the voxel index; ``GroupNorm`` reduces over
statistics that a permutation of voxels leaves alone; and
:class:`~poraque.ml.fno.CellEncoder` conditions on lengths, angle cosines and a
volume, which are rotation-invariant by construction.
"""

import itertools

import pytest
import torch

from poraque.ml.fno import FNO3d, RadialSpectralConv3d


# ---------------------------------------------------------------------- #
# Helpers
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

    A rotation of a *function* is :math:`f'(\mathbf r) = f(R^{-1}\mathbf r)`,
    which on a periodic grid is a permutation of voxels and therefore exact —
    no interpolation, no resampling, nothing that could be mistaken for the
    error being measured. It is only available for rotations that map the grid
    to itself, which is the honest limit of what a lattice can be asked: the
    radial construction guarantees the *continuum* operator commutes with every
    rotation, and a discrete grid can only ever exercise its own point group.
    """
    shape = field.shape[-3:]
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(n) for n in shape], indexing="ij"))
    inverse = torch.linalg.inv(rotation.double()).round().long()
    source = (inverse @ grid.reshape(3, -1)).reshape(3, *shape)
    source = torch.stack([source[i] % shape[i] for i in range(3)])
    return field[..., source[0], source[1], source[2]]


def build(equivariant, seed=7, **kwargs):
    """A small operator, deterministically initialised."""
    torch.manual_seed(seed)
    kwargs.setdefault("use_coordinates", not equivariant)
    kwargs.setdefault("cell_conditioning", True)
    return FNO3d(in_channels=1, out_channels=1, width=8, modes=4, n_layers=2,
                 projection_channels=16, equivariant=equivariant,
                 **kwargs).eval()


CUBIC = (torch.eye(3) * 8.0).unsqueeze(0)


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


# ---------------------------------------------------------------------- #
class TestARotatedFieldGivesARotatedPrediction:
    """
    The property itself, over the whole point group of the cube.

    The dense baseline is asserted to *fail* rather than merely left
    unmentioned. Equivariance is the kind of claim that a test can pass for the
    wrong reason — a model whose output barely depends on its input is
    trivially equivariant — and the counterfactual is what rules that out.
    """

    def test_the_equivariant_operator_commutes_with_all_24_rotations(self):
        field = torch.randn(1, 1, 16, 16, 16)
        error = worst_rotation_error(build(True), field, CUBIC,
                                     octahedral_rotations())
        assert error < 1e-5, f"worst relative deviation {error:.3e}"

    def test_the_dense_operator_does_not(self):
        field = torch.randn(1, 1, 16, 16, 16)
        error = worst_rotation_error(build(False, use_coordinates=False),
                                     field, CUBIC, octahedral_rotations())
        assert error > 0.1, (
            f"the dense spectral layer came out equivariant to {error:.3e}, "
            f"which it has no reason to be -- if the test field or the model "
            f"has become degenerate the equivariant test above passes for "
            f"nothing")

    def test_the_residual_is_float_round_off_and_not_a_small_asymmetry(self):
        """
        In float64 the deviation drops by nine orders of magnitude.

        It matters that this is round-off rather than a small real violation,
        because a small real violation is what the retained set being lopsided
        produced: ``rfftn`` gives axis *i* the frequencies ``[0, m)`` and
        ``[-m, 0)``, so mode :math:`-m` is kept and :math:`+m` is not, and the
        equivariance was good to 2e-3 — plausible, stable, and wrong.
        """
        model = build(True).double()
        field = torch.randn(1, 1, 16, 16, 16, dtype=torch.float64)
        error = worst_rotation_error(model, field, CUBIC.double(),
                                     octahedral_rotations()[:6])
        assert error < 1e-12, f"worst relative deviation {error:.3e}"


class TestTheRetainedModesMustBeASphereNotABox:
    """
    Masking to the inscribed sphere, and what it is worth.

    A radial multiplier over a box of modes is equivariant under the box's
    symmetry group and no more. The cubic case hides this completely, so the
    measurement that matters is on a cell where the retained box is not a cube.
    """

    TETRAGONAL = torch.diag(torch.tensor([8.0, 8.0, 12.0])).unsqueeze(0)
    QUARTER_TURN = [torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])]

    def test_the_cutoff_is_what_makes_a_non_cubic_cell_equivariant(self):
        field = torch.randn(1, 1, 16, 16, 24)
        error = worst_rotation_error(build(True), field, self.TETRAGONAL,
                                     self.QUARTER_TURN)
        assert error < 1e-5, f"worst relative deviation {error:.3e}"

    def test_without_it_the_same_model_is_not(self):
        field = torch.randn(1, 1, 16, 16, 24)
        error = worst_rotation_error(build(True, spherical_cutoff=False),
                                     field, self.TETRAGONAL,
                                     self.QUARTER_TURN)
        assert error > 1e-3, (
            f"the box-truncated kernel came out equivariant to {error:.3e} on "
            f"a tetragonal cell, which would make spherical_cutoff pointless")

    def test_the_mask_is_strict_so_the_kept_set_is_closed_under_inversion(self):
        """
        ``|G| < g_inscribed``, never ``<=``.

        On the sphere itself sits the face :math:`n_i = -m_i`, whose partner
        :math:`+m_i` ``rfftn`` never stored. Keeping it makes the retained set
        lopsided by one face per axis; excluding it forces
        :math:`|n_i| < m_i` on every axis at once and the remainder is a
        genuine ball.
        """
        layer = RadialSpectralConv3d(1, 1, modes=(4, 4, 4), n_radial=8)
        basis = layer.radial_basis(CUBIC, (4, 4, 4), torch.device("cpu"),
                                   torch.float32)
        # Frequency -4 along the first axis sits exactly at 2*pi*4/8.
        assert float(basis[0, :, 4, 0, 0].abs().max()) == 0.0
        assert float(basis[0, :, 0, 0, 0].abs().max()) > 0.0


class TestTheCoordinateChannelsAreNotThreeScalars:
    """
    ``use_coordinates`` and ``equivariant`` are refused together.

    They are not merely unhelpful in combination: the fractional coordinates
    rotate into each other, so appending them undoes the constraint the
    spectral layer was built to satisfy. Silently disabling them would be
    worse than refusing — the run would train the architecture the user did not
    ask for and report nothing.
    """

    def test_the_constructor_refuses_and_names_the_key(self):
        with pytest.raises(ValueError, match="use_coordinates"):
            FNO3d(width=8, modes=4, n_layers=2, equivariant=True,
                  use_coordinates=True)

    def test_the_config_refuses_before_the_cache_is_built(self):
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig.from_dict({"model": {"equivariant": True}})
        with pytest.raises(ValueError, match="use_coordinates"):
            config.model_kwargs()


class TestTheRadialCoefficientsAreReal:
    """
    Not a storage choice — the only spelling that gives a real output field.

    A radial multiplier satisfies :math:`R(-\\mathbf G) = R(\\mathbf G)`, and a
    real field needs :math:`R(-\\mathbf G) = \\overline{R(\\mathbf G)}`. Both
    hold together only for a real :math:`R`. Nothing is lost by it: a
    real-space kernel that is a function of :math:`|\\mathbf r|` is real and
    centrosymmetric already, so the constrained class *is* the radial one —
    which is also why the layer is equivariant under the full :math:`O(3)`,
    inversion included, and not only the :math:`SE(3)` it was asked for.
    """

    def test_the_weight_is_a_real_parameter(self):
        layer = RadialSpectralConv3d(2, 3, modes=(4, 4, 4), n_radial=8)
        assert not layer.weight.is_complex()
        assert tuple(layer.weight.shape) == (2, 3, 8)

    def test_no_einsum_in_the_layer_ever_sees_a_complex_operand(self, monkeypatch):
        """
        Which is why it needs none of :func:`complex_contract`'s workaround.

        That function exists for one reason — MPS aborts the *process*, not the
        call, on a complex ``einsum`` — and a layer whose coefficients are real
        reaches the same result through real arithmetic and pays none of it.
        Asserted on the calls rather than on the source, because a comment
        naming the function it avoids would satisfy a search for the name.
        """
        seen = []
        original = torch.einsum

        def recording(equation, *operands):
            seen.append(any(t.is_complex() for t in operands))
            return original(equation, *operands)

        monkeypatch.setattr(torch, "einsum", recording)
        layer = RadialSpectralConv3d(2, 2, modes=(4, 4, 4), n_radial=8)
        layer(torch.randn(1, 2, 16, 16, 16), CUBIC)

        assert seen, "no einsum was called at all; the test proves nothing"
        assert not any(seen)

    def test_the_output_field_is_real(self):
        layer = RadialSpectralConv3d(1, 1, modes=(4, 4, 4), n_radial=8)
        out = layer(torch.randn(1, 1, 16, 16, 16), CUBIC)
        assert not out.is_complex()
        assert out.shape == (1, 1, 16, 16, 16)


class TestTheCellIsRequired:
    """
    A radius in Å⁻¹ does not exist without one.

    The dense layer's weights are indexed by mode and need no lattice, so this
    is a genuine difference in contract between the two and is raised rather
    than defaulted. Defaulting to a unit cell would make every prediction
    quietly wrong by the cell's own scale.
    """

    def test_the_layer_refuses_a_missing_cell(self):
        layer = RadialSpectralConv3d(1, 1, modes=(4, 4, 4), n_radial=8)
        with pytest.raises(ValueError, match="cell"):
            layer(torch.randn(1, 1, 16, 16, 16), None)

    def test_the_model_refuses_a_missing_cell(self):
        model = build(True, cell_conditioning=False)
        with pytest.raises(ValueError, match="cell"):
            model(torch.randn(1, 1, 16, 16, 16), None)


class TestTheConstraintIsPaidForInCapacity:
    """
    An equivariant model of the same width is a much smaller model.

    Worth asserting because it is the honest cost of the method and the thing a
    reader will want to know before choosing a width: the dense layer holds
    :math:`4 C_{\\rm in}C_{\\rm out}m_1m_2m_3` complex numbers, this one holds
    :math:`C_{\\rm in}C_{\\rm out}R` real ones.
    """

    def test_the_radial_layer_is_far_smaller(self):
        dense = build(False, use_coordinates=False).n_parameters()
        radial = build(True).n_parameters()
        assert radial < dense / 10

    def test_n_radial_is_the_knob_that_buys_it_back(self):
        small = RadialSpectralConv3d(4, 4, modes=(8, 8, 8), n_radial=8)
        large = RadialSpectralConv3d(4, 4, modes=(8, 8, 8), n_radial=32)
        assert large.weight.numel() == 4 * small.weight.numel()

    def test_a_single_basis_function_is_refused(self):
        with pytest.raises(ValueError, match="n_radial"):
            RadialSpectralConv3d(1, 1, modes=(4, 4, 4), n_radial=1)


class TestTheOperatorStillServesEveryGrid:
    """
    The property the whole architecture exists for is not spent on this one.

    Weights live in mode space, so one model serves every grid shape; the
    radial kernel keeps that and adds a second, stronger claim — the same
    coefficient means the same *physical* wavevector in every material,
    because the radius comes from the sample's own cell rather than from an
    index.
    """

    @pytest.mark.parametrize("shape", [(16, 16, 16), (12, 20, 24), (24, 24, 24)])
    def test_one_set_of_weights_serves_many_shapes(self, shape):
        model = build(True)
        with torch.no_grad():
            out = model(torch.randn(1, 1, *shape), CUBIC)
        assert tuple(out.shape[-3:]) == shape

    def test_the_same_coefficients_mean_the_same_wavevector_in_two_cells(self):
        layer = RadialSpectralConv3d(1, 1, modes=(4, 4, 4), n_radial=8)
        small = layer.radial_basis((torch.eye(3) * 4.0).unsqueeze(0),
                                   (4, 4, 4), torch.device("cpu"),
                                   torch.float32)
        large = layer.radial_basis((torch.eye(3) * 8.0).unsqueeze(0),
                                   (4, 4, 4), torch.device("cpu"),
                                   torch.float32)
        # |G| at mode (1,0,0) in the 4 A cell equals |G| at (2,0,0) in the 8 A
        # one, so the basis reads the same there and nowhere else by accident.
        assert torch.allclose(small[0, :, 1, 0, 0], large[0, :, 2, 0, 0],
                              atol=1e-6)
        assert not torch.allclose(small[0, :, 1, 0, 0], large[0, :, 1, 0, 0],
                                  atol=1e-3)


class TestAnEquivariantCheckpointReloadsAsOne:
    """
    The variant lives in a tensor *rank*, and in the architecture record too.

    ``projection_activation`` is the precedent: a hyper-parameter that no
    tensor shape encodes reloads as the constructor default and silently
    changes what the model computes. Here the rank of the spectral weight does
    encode it — three indices instead of six — which is the one architectural
    fact that survives even a checkpoint written with no record at all.
    """

    def test_the_reloaded_model_predicts_identically(self, tmp_path):
        from poraque.ml.training import FieldOperator

        torch.manual_seed(3)
        operator = FieldOperator(
            "ext2chg", device="cpu", width=8, modes=4, n_layers=2,
            projection_channels=16, use_coordinates=False, equivariant=True,
            n_radial=12, g_basis=6.0)
        field, cell = torch.randn(1, 1, 16, 16, 16), CUBIC
        with torch.no_grad():
            before = operator.model(field, cell)

        path = tmp_path / "equivariant.poraque"
        operator.save(path)
        reloaded = FieldOperator.load(path, device="cpu").model

        assert reloaded.equivariant is True
        assert reloaded.n_radial == 12
        assert reloaded.g_basis == 6.0
        assert reloaded.modes == (4, 4, 4)
        with torch.no_grad():
            assert torch.equal(before, reloaded(field, cell))

    def test_the_variant_is_inferable_from_the_tensors_alone(self):
        from poraque.ml.training import infer_backbone_kwargs

        torch.manual_seed(3)
        model = FNO3d(width=8, modes=4, n_layers=2, projection_channels=16,
                      use_coordinates=False, equivariant=True, n_radial=12)
        inferred = infer_backbone_kwargs(model.state_dict())
        assert inferred["equivariant"] is True
        assert inferred["n_radial"] == 12
        assert inferred["width"] == 8
        assert "modes" not in inferred      # radial: no mode index in a shape

    def test_a_dense_checkpoint_is_still_read_as_dense(self):
        from poraque.ml.training import infer_backbone_kwargs

        model = FNO3d(width=8, modes=4, n_layers=2, projection_channels=16)
        inferred = infer_backbone_kwargs(model.state_dict())
        assert inferred.get("equivariant") is None
        assert inferred["modes"] == 4


class TestTheSetupBlockBehavesLikeKanSetup:
    """
    One flag plus one block, read only when the flag is on.

    The same shape as ``model.activation`` / ``model.kan_setup``, and for the
    same reason: all three settings are read by one architecture and by no
    other, so as flat keys they would sit beside ``width`` and ``modes`` and
    read as decisions every run should be making.
    """

    def test_an_unknown_key_raises(self):
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig.from_dict({"model": {
            "equivariant": True, "use_coordinates": False,
            "equivariant_setup": {"nradial": 4}}})
        with pytest.raises(ValueError, match="equivariant_setup"):
            config.model_kwargs()

    def test_a_block_beside_a_dense_model_warns_rather_than_acting(self):
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig.from_dict({"model": {
            "equivariant_setup": {"n_radial": 4}}})
        with pytest.warns(RuntimeWarning, match="equivariant_setup"):
            kwargs = config.model_kwargs()
        assert "n_radial" not in kwargs

    def test_the_block_reaches_the_constructor(self):
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig.from_dict({"model": {
            "equivariant": True, "use_coordinates": False,
            "equivariant_setup": {"n_radial": 24, "g_basis": 5.0,
                                  "spherical_cutoff": False}}})
        kwargs = config.model_kwargs()
        assert kwargs["n_radial"] == 24
        assert kwargs["g_basis"] == 5.0
        assert kwargs["spherical_cutoff"] is False


class TestTheEquivariantOperatorCanActuallyBeTrained:
    """
    Equivariance is worthless if the layer receives no gradient.

    The contraction splits the spectrum into real and imaginary halves and
    contracts each against the same real coefficients, rejoining them with
    ``torch.complex``. That is arithmetic autograd handles, but a complex
    constructor in the middle of a graph is exactly where a backward pass stops
    being obvious — and a layer that silently received nothing would train to
    its initialisation and read as a bad dataset.

    Asserted here rather than only in ``tests/test_cuequivariance_gpu.py``,
    because that file does not run on a machine without CUDA and this claim has
    nothing to do with the device.
    """

    def test_every_radial_coefficient_receives_a_gradient(self):
        model = build(True)
        field = torch.randn(2, 1, 16, 16, 16)
        cell = CUBIC.repeat(2, 1, 1)
        model(field, cell).square().mean().backward()
        for index, block in enumerate(model.blocks):
            gradient = block.spectral.weight.grad
            assert gradient is not None, f"block {index} received none"
            assert torch.isfinite(gradient).all()
            assert float(gradient.abs().max()) > 0.0

    def test_a_few_steps_move_the_loss(self):
        torch.manual_seed(0)
        model = build(True)
        field = torch.randn(2, 1, 16, 16, 16)
        target = torch.randn(2, 1, 16, 16, 16)
        cell = CUBIC.repeat(2, 1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        first = None
        for _ in range(20):
            optimizer.zero_grad()
            loss = (model(field, cell) - target).square().mean()
            loss.backward()
            optimizer.step()
            first = float(loss.detach()) if first is None else first
        assert float(loss.detach()) < first

    def test_the_equivariance_survives_the_training(self):
        """
        It is a property of the architecture rather than of the weights, so no
        amount of optimisation can spend it — but a weight update that touched
        the wrong tensor could, and nothing else would notice.
        """
        torch.manual_seed(0)
        model = build(True)
        field = torch.randn(1, 1, 16, 16, 16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        for _ in range(10):
            optimizer.zero_grad()
            model(field, CUBIC).square().mean().backward()
            optimizer.step()

        model.eval()
        error = worst_rotation_error(model, field, CUBIC,
                                     octahedral_rotations()[:6])
        assert error < 1e-5, f"worst relative deviation {error:.3e}"


class TestTheBasisRadiusFollowsTheBandTheRunNamed:
    """
    ``g_basis`` defaults to ``g_max`` when a physical band has been stated.

    A radial basis needs a scale, and a run using ``mode_selection:
    physical`` has already named one. Taking an unrelated default there would
    put the basis nodes somewhere other than the modes the run retains, which
    costs resolution exactly where the model is being asked to work.
    """

    def test_it_takes_g_max_when_the_mode_selection_is_physical(self):
        model = FNO3d(width=8, modes=4, n_layers=2, use_coordinates=False,
                      equivariant=True, mode_selection="physical", g_max=3.5)
        assert model.g_basis == 3.5

    def test_an_explicit_value_still_wins(self):
        model = FNO3d(width=8, modes=4, n_layers=2, use_coordinates=False,
                      equivariant=True, mode_selection="physical", g_max=3.5,
                      g_basis=9.0)
        assert model.g_basis == 9.0

    def test_a_fixed_run_falls_back_to_the_documented_default(self):
        from poraque.ml.fno import DEFAULT_G_BASIS

        model = FNO3d(width=8, modes=4, n_layers=2, use_coordinates=False,
                      equivariant=True)
        assert model.g_basis == DEFAULT_G_BASIS
