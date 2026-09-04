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

from poraque.ml import fno
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


def sample_field(shape=(16, 16, 16), seed=0, device="cpu", dtype=torch.float32):
    """
    A test field drawn from a **stated** seed.

    Not decoration. The dense baseline's rotation error is a property of the
    field as well as of the weights — measured across seeds it runs from 0.086
    to 0.93, an order of magnitude — so a counterfactual asserting "the dense
    layer is *not* equivariant" against an unseeded ``torch.randn`` is a
    threshold compared with a number that depends on what the rest of the
    session left in the global RNG. It passed; it would have failed under a
    different test order, and the failure would have looked like a real
    regression in the thing it exists to disprove.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 1, *shape, generator=generator,
                       dtype=dtype).to(device)


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

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_equivariant_operator_commutes_with_all_24_rotations(self, seed):
        error = worst_rotation_error(build(True), sample_field(seed=seed),
                                     CUBIC, octahedral_rotations())
        assert error < 1e-5, f"worst relative deviation {error:.3e}"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_the_dense_operator_does_not(self, seed):
        """
        The counterfactual, stated as a **ratio** rather than a threshold.

        The dense layer's rotation error depends on the field as well as on the
        weights — 0.086 at one seed, 0.93 at another — so an absolute bound
        here is a number that has to be re-tuned whenever anything upstream
        touches the RNG. The claim that actually matters is scale-free: on the
        *same* field, the dense layer is wrong by orders of magnitude more than
        the radial one. Measured, that ratio is ~1e5.
        """
        field = sample_field(seed=seed)
        rotations = octahedral_rotations()
        dense = worst_rotation_error(build(False, use_coordinates=False),
                                     field, CUBIC, rotations)
        radial = worst_rotation_error(build(True), field, CUBIC, rotations)
        assert dense > 1e-3, (
            f"the dense spectral layer came out equivariant to {dense:.3e}, "
            f"which it has no reason to be -- if the field or the model has "
            f"become degenerate the equivariant test above passes for nothing")
        assert dense > 1e3 * radial, f"dense {dense:.3e}, radial {radial:.3e}"

    def test_the_residual_is_float_round_off_not_a_small_asymmetry(self):
        """
        In float64 the deviation drops by nine orders of magnitude.

        It matters that this is round-off rather than a small real violation,
        because a small real violation is what the retained set being lopsided
        produced: ``rfftn`` gives axis *i* the frequencies ``[0, m)`` and
        ``[-m, 0)``, so mode :math:`-m` is kept and :math:`+m` is not, and the
        equivariance was good to 2e-3 — plausible, stable, and wrong.
        """
        model = build(True).double()
        field = sample_field(dtype=torch.float64)
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
        field = sample_field((16, 16, 24))
        error = worst_rotation_error(build(True), field, self.TETRAGONAL,
                                     self.QUARTER_TURN)
        assert error < 1e-5, f"worst relative deviation {error:.3e}"

    def test_without_it_the_same_model_is_not(self):
        field = sample_field((16, 16, 24))
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


class TestTheCutoffDoesNotTieWithItsOwnRoundOff:
    r"""
    The strict ``<`` is exact in intent and inexact in arithmetic.

    Found on a V100 at LNCC by the mixed-batch test in
    ``tests/test_cuequivariance_gpu.py`` — written to catch a wrongly broadcast
    ``(B, 3, 3)``, and instead catching a cell that fails on its own — then
    reproduced here on the CPU, so it is not CUDA-specific and never was.

    The two sides of the comparison are computed by different routes:
    ``radius`` from the reciprocal cell contracted against the integer
    frequencies, ``inscribed`` from :math:`2\pi m_i/|a_i|`. They agree
    analytically and disagree in the last bits, and the face-centre modes
    :math:`(\pm m, 0, 0)` and their rotations sit *exactly* on the boundary —
    so round-off decides which side each lands on, and it need not decide the
    same way for every axis. One face kept where its own rotation image was
    dropped is a discrete change in the retained set and therefore in the
    operator. **6 of 20 cubic cells swept from 4 Å to 13.5 Å lost their
    equivariance in float32** (31 % over a finer 200-cell sweep), at
    :math:`3\times10^{-4}` to :math:`1.7\times10^{-3}` against the
    :math:`3\times10^{-7}` of a cell that happens not to tie. Two of the same
    twenty tie in float64 as well, so raising the precision is not the fix.

    :data:`~poraque.ml.fno.CUTOFF_SLACK` is: it lifts the boundary off the
    shell by a relative amount far above the arithmetic disagreement and far
    below the spacing between mode shells.
    """

    # Six edges measured to tie as shipped, and two that do not. Both halves
    # are asserted: the counterfactual has to distinguish the cells the
    # mechanism predicts from the cells it does not, or it is measuring
    # something else.
    TIED = (4.5, 5.5, 6.5, 9.0, 11.0, 13.0)
    UNTIED = (8.0, 12.0)

    @staticmethod
    def cubic(edge, dtype=torch.float32):
        return (torch.eye(3, dtype=dtype) * edge).unsqueeze(0)

    @staticmethod
    def retained(layer, cell, modes, dtype):
        """
        How many *modes* the cutoff keeps — reduced over the radial axis.

        Counting non-zero basis *entries* instead is a trap, and one this test
        fell into on the way: the Gaussians underflow to exactly zero far
        sooner in float32 than in float64, so an entry count differs between
        the precisions on most cells for a reason that has nothing to do with
        the mask. A mode that survives the cutoff always has its nearest
        centre within half a spacing, which never underflows, so the maximum
        over the radial axis is a clean membership indicator in either dtype.
        """
        basis = layer.radial_basis(cell, modes, torch.device("cpu"), dtype)
        return int((basis[0].amax(dim=0) > 0.0).sum())

    def test_every_cell_in_the_sweep_is_equivariant_in_float32(self):
        model = build(True)
        field = sample_field()
        rotations = octahedral_rotations()
        for edge in self.TIED + self.UNTIED:
            error = worst_rotation_error(model, field, self.cubic(edge),
                                         rotations)
            assert error < 1e-4, (
                f"a {edge} Ang cubic cell deviates by {error:.3e}")

    def test_without_the_slack_exactly_those_cells_fail(self, monkeypatch):
        """
        The counterfactual, and the diagnosis with it.

        Removing the slack has to break the six cells the tie mechanism names
        and leave the other two at float32 round-off. A slack that improved
        everything uniformly would be hiding a different defect.
        """
        monkeypatch.setattr(fno, "CUTOFF_SLACK", 0.0)
        model = build(True)
        field = sample_field()
        rotations = octahedral_rotations()
        for edge in self.TIED:
            error = worst_rotation_error(model, field, self.cubic(edge),
                                         rotations)
            assert error > 1e-4, (
                f"a {edge} Ang cell was expected to tie at the cutoff and "
                f"came out equivariant to {error:.3e}")
        for edge in self.UNTIED:
            error = worst_rotation_error(model, field, self.cubic(edge),
                                         rotations)
            assert error < 1e-5, (
                f"a {edge} Ang cell does not tie and should be unaffected, "
                f"but deviates by {error:.3e}")

    def test_the_retained_set_stops_depending_on_the_precision(self):
        """
        The fix stated as what it actually does, one level below the operator.

        A mask decided by round-off is one whose *size* differs between float32
        and float64 on the same cell. Counting is a sharper instrument than a
        deviation: it says the retained set changed, not merely that some
        number moved — and here it says exactly *how*. Without the slack these
        six cells keep **150** modes in float32 against **148** in float64, and
        the two extra are the faces :math:`(-m, 0, 0)` and :math:`(0, -m, 0)`.
        The third face is the one ``rfftn`` never stored, which is what leaves
        the set asymmetric rather than merely larger. Over 200 cubic cells from
        3 Å to 13 Å the two precisions disagree on 69 to 83 of them as shipped,
        depending on ``modes``, and on none of them with the slack.
        """
        single = RadialSpectralConv3d(1, 1, modes=(4, 4, 4), n_radial=8)
        double = RadialSpectralConv3d(1, 1, modes=(4, 4, 4),
                                      n_radial=8).double()
        for edge in self.TIED + self.UNTIED:
            kept = self.retained(single, self.cubic(edge), (4, 4, 4),
                                 torch.float32)
            reference = self.retained(double,
                                      self.cubic(edge, torch.float64),
                                      (4, 4, 4), torch.float64)
            assert kept == reference, (
                f"a {edge} Ang cell keeps {kept} modes in float32 and "
                f"{reference} in float64")

    def test_the_slack_removes_the_shell_and_nothing_else(self, monkeypatch):
        """
        A relative 1e-5 has to be too small to reach the next shell in.

        The nearest genuine gap below the cutoff is :math:`1/2m^2` — 8e-3 at
        ``modes=4``, and shrinking only as :math:`m^{-2}` — so the slack is
        two orders clear of it. Asserted in float64, where the tie itself is
        absent from these cells, so any change in the count is the slack
        overreaching rather than the round-off it exists to absorb.
        """
        layer = RadialSpectralConv3d(1, 1, modes=(4, 4, 4),
                                     n_radial=8).double()
        cells = [self.cubic(edge, torch.float64)
                 for edge in self.TIED + self.UNTIED]
        with_slack = [self.retained(layer, cell, (4, 4, 4), torch.float64)
                      for cell in cells]
        monkeypatch.setattr(fno, "CUTOFF_SLACK", 0.0)
        exact = [self.retained(layer, cell, (4, 4, 4), torch.float64)
                 for cell in cells]
        assert with_slack == exact


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

        config = TrainingConfig.from_dict(
            {"model": {"equivariant": {"enable": True}}})
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


class TestTheEquivarianceBlockIsOneKey:
    """
    The switch lives *inside* the settings it governs.

    It did not always. Until 26.9.8 this was ``equivariant: true`` beside a
    separately-named ``equivariant_setup`` group, and the two halves drifted in
    exactly the way a two-key layout invites: a config with ``equivariant:
    false`` above a populated ``equivariant_setup`` reads at a glance as an
    equivariant run and is not one, and nothing brings the flag and the
    settings into the same field of view. The block is now what
    ``fine_tuning``, ``symbolic.physics`` and ``training.physics_informed``
    already were.

    What has *not* changed is the constructor: :class:`FNO3d` still takes flat
    ``equivariant=`` / ``n_radial=`` keywords, and every checkpoint's
    ``architecture`` record is unaffected. The grouping is a property of the
    configuration file.
    """

    @staticmethod
    def _config(**model):
        from poraque.ml.config import TrainingConfig

        return TrainingConfig.from_dict({"model": model})

    def test_an_unknown_setting_raises_and_names_the_block(self):
        """
        At load, and the message says which block --- there are three.

        "Unknown key: nradal" is true of a schema with one such block and
        useless in one with several.
        """
        with pytest.raises(ValueError, match=r"model\.equivariant.*nradial"):
            self._config(equivariant={"enable": True, "nradial": 4},
                         use_coordinates=False)

    def test_a_setting_beside_a_dense_model_warns_rather_than_acting(self):
        config = self._config(equivariant={"n_radial": 4})
        with pytest.warns(RuntimeWarning, match="enable false"):
            kwargs = config.model_kwargs()
        assert "n_radial" not in kwargs
        assert kwargs["equivariant"] is False

    def test_the_block_reaches_the_constructor(self):
        config = self._config(
            use_coordinates=False,
            equivariant={"enable": True, "n_radial": 24, "g_basis": 5.0,
                         "spherical_cutoff": False})
        kwargs = config.model_kwargs()
        assert kwargs["equivariant"] is True
        assert kwargs["n_radial"] == 24
        assert kwargs["g_basis"] == 5.0
        assert kwargs["spherical_cutoff"] is False

    def test_an_unstated_setting_stays_unstated(self):
        """
        Not filled in with a default, and that is load-bearing.

        ``g_basis`` unstated means "follow ``g_max`` if
        ``mode_selection: physical`` named a band, else the module default" ---
        a resolution :class:`FNO3d` performs and the config cannot, because it
        depends on a key in the same section. A block that helpfully passed
        ``g_basis: 8.0`` would bypass it and silently ignore ``g_max``.
        """
        config = self._config(use_coordinates=False,
                              equivariant={"enable": True, "n_radial": 24})
        kwargs = config.model_kwargs()
        assert "g_basis" not in kwargs and "spherical_cutoff" not in kwargs

    def test_the_old_two_key_spelling_says_what_to_write_instead(self):
        """A config written against the old schema is refused, not reinterpreted."""
        from poraque.ml.config import TrainingConfig

        with pytest.raises(ValueError, match="must be a block"):
            TrainingConfig.from_dict({"model": {"equivariant": True}})

        with pytest.raises(ValueError, match="equivariant_setup"):
            TrainingConfig.from_dict(
                {"model": {"equivariant_setup": {"n_radial": 4}}})

    def test_a_bad_switch_value_is_refused_at_load(self):
        """
        ``from_dict`` validates the block, so a run says so on the command line.

        Not at ``model_kwargs()`` time: the model is built after the field
        cache, which on a real dataset is minutes.
        """
        from poraque.ml.config import TrainingConfig

        with pytest.raises(ValueError, match="enable is 'yes'"):
            TrainingConfig.from_dict(
                {"model": {"equivariant": {"enable": "yes"}}})


def accelerator():
    """The device this machine actually has, or ``None``."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


@pytest.mark.gpu
@pytest.mark.skipif(accelerator() is None, reason="requires MPS or CUDA")
class TestItHoldsOnWhateverAcceleratorIsHere:
    """
    The same property, on the device — and on **MPS specifically**.

    Worth its own class because Metal is where this could plausibly have gone
    wrong and not said so. :func:`~poraque.ml.fno.complex_contract` exists for
    two documented MPS defects: a complex ``einsum`` *aborts the process*
    rather than raising, and ``.real``/``.imag`` of a **strided** complex view
    silently computes the wrong einsum — no error, results off by 40-90 %. The
    radial layer's claim is that it meets neither, because its coefficients are
    real and it contracts real tensors gathered into a contiguous block.

    That claim is exactly the kind that is cheap to make and expensive to be
    wrong about: a silently wrong contraction on Metal produces finite,
    plausible numbers. ``tests/test_cuequivariance_gpu.py`` covers the CUDA
    side and needs a cluster; this runs wherever there is an accelerator at
    all, which on the development machine means it runs every time.
    """

    @staticmethod
    def _device():
        return accelerator()

    def test_the_rotation_property_survives_the_device(self):
        device = self._device()
        model = build(True).to(device)
        field = sample_field(device=device)
        error = worst_rotation_error(model, field, CUBIC.to(device),
                                     octahedral_rotations())
        assert error < 1e-5, f"{device.type}: worst deviation {error:.3e}"

    def test_the_device_agrees_with_the_cpu_on_the_prediction_itself(self):
        """
        Not only on the symmetry.

        A contraction that were wrong on Metal in a *rotation-invariant* way
        would pass the test above while computing a different operator. This is
        the assertion that would catch it, and it is the one the strided-view
        defect would have broken.
        """
        device = self._device()
        field, cell = sample_field(), CUBIC
        with torch.no_grad():
            on_cpu = build(True)(field, cell)
            on_device = build(True).to(device)(field.to(device),
                                               cell.to(device)).cpu()
        relative = float((on_cpu - on_device).abs().max()
                         / on_cpu.abs().max())
        assert relative < 1e-5, f"{device.type}: {relative:.3e}"

    def test_it_trains_there_and_stays_equivariant(self):
        device = self._device()
        torch.manual_seed(0)
        model = build(True).to(device)
        field = sample_field(device=device)
        target = sample_field(seed=1, device=device)
        cell = CUBIC.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        first = None
        for _ in range(10):
            optimizer.zero_grad()
            loss = (model(field, cell) - target).square().mean()
            loss.backward()
            for index, layer in enumerate(model.blocks):
                gradient = layer.spectral.weight.grad
                assert gradient is not None, f"block {index} received none"
                assert torch.isfinite(gradient).all()
            optimizer.step()
            first = float(loss.detach()) if first is None else first

        assert float(loss.detach()) < first
        model.eval()
        error = worst_rotation_error(model, field, cell,
                                     octahedral_rotations()[:6])
        assert error < 1e-5, f"{device.type}: worst deviation {error:.3e}"


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


class TestGMaxDoesNotSizeTheBasisUnderFixedModeSelection:
    """
    A ``g_max`` left in a ``fixed`` config silently rebuilt the architecture.

    ``FNO3d``'s docstring says ``g_max`` is *required by* ``mode_selection:
    physical``, so a reader of a ``fixed`` config concludes the key is inert.
    It was inert for truncation and not for the radial basis: the fallback
    ``elif self.g_max is not None`` fired under either mode selection, so
    ``g_max: 6`` beside ``mode_selection: fixed`` gave a basis of 6.0 rather
    than the 8.0 default, and nothing in the log, the resolved config or the
    run record said so.

    LNCC found it after a 90-run architecture study had already been run
    through it, on Santos Dumont, 2026-09-03. The measured accuracy cost was
    within seed spread --- ``g_basis`` 6/8/12/16 all landed inside 5 % of each
    other at ``modes`` 8, disagreeing in sign between seeds --- so this is a
    usability defect and not an accuracy one. That is the reason it survived
    twenty runs.

    The coupling under ``physical`` is correct and stays: there
    ``m_i = floor(g_max*L_i/2*pi)``, so the inscribed radius
    ``min_i 2*pi*m_i/L_i`` is at most ``g_max`` and a basis spanning ``g_max``
    spans the retained band by construction.
    """

    def test_a_leftover_g_max_no_longer_resizes_the_basis(self):
        from poraque.ml.fno import DEFAULT_G_BASIS

        with pytest.warns(RuntimeWarning, match="does not size the radial"):
            model = FNO3d(width=8, modes=4, n_layers=2, use_coordinates=False,
                          equivariant=True, mode_selection="fixed", g_max=6.0)
        assert model.g_basis == DEFAULT_G_BASIS

    def test_the_rule_lives_in_one_function_and_says_the_same_thing(self):
        """
        The counterfactual: the pre-fix rule read ``g_max`` in both modes.

        ``default_g_basis`` exists so the constructor and ``poraque-train``'s
        startup report cannot disagree about what was built --- a diagnostic
        that describes a different architecture is the failure being fixed,
        not a milder version of it.
        """
        from poraque.ml.fno import DEFAULT_G_BASIS, default_g_basis

        assert default_g_basis(6.0, "physical") == 6.0
        assert default_g_basis(6.0, "fixed") == DEFAULT_G_BASIS
        assert default_g_basis(None, "physical") == DEFAULT_G_BASIS

        model = FNO3d(width=8, modes=4, n_layers=2, use_coordinates=False,
                      equivariant=True, mode_selection="physical", g_max=6.0)
        assert model.g_basis == default_g_basis(6.0, "physical")

    def test_it_is_silent_when_there_is_nothing_to_say(self):
        """
        The warning fires on the one configuration that changed behaviour.

        A dense model has no radial basis to resize, and a ``physical`` run
        keeps the coupling, so neither is told anything. Warning on either
        would put a notice in front of every reader of a config the change
        does not touch.
        """
        import warnings

        for kwargs in (dict(equivariant=True, use_coordinates=False,
                            mode_selection="physical", g_max=6.0),
                       dict(equivariant=True, use_coordinates=False,
                            mode_selection="fixed", g_max=6.0, g_basis=5.0),
                       dict(mode_selection="fixed", g_max=6.0)):
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                FNO3d(width=8, modes=4, n_layers=2, **kwargs)

    def test_an_existing_checkpoint_is_unaffected(self):
        """
        ``state()`` records the resolved ``g_basis``, so a reload is a number.

        Every model trained through the defect carries ``g_basis: 6.0`` in its
        architecture record and reloads with 6.0 --- the change moves what a
        *new* model is built with and nothing about an old one.
        """
        from poraque.ml.training import FieldOperator

        # `g_basis=6.0` is what the defect resolved `g_max: 6` to under
        # `fixed`; stating it is how a model trained through it is reproduced.
        operator = FieldOperator(
            "ext2chg", device="cpu", width=8, modes=4, n_layers=2,
            projection_channels=16, use_coordinates=False, equivariant=True,
            mode_selection="fixed", g_max=6.0, g_basis=6.0)
        state = operator.state()
        assert state["architecture"]["g_basis"] == 6.0

        reloaded = FieldOperator.from_state(state, device="cpu")
        assert reloaded.model.g_basis == 6.0
        assert reloaded.model.g_max == 6.0


class TestTheRadialBasisIsBuiltOncePerForwardPass:
    r"""
    Every layer in the stack rebuilt the same tensor, and it was not cheap.

    The basis depends on the batch's cells, the retained mode counts, the
    device and the dtype --- and on the layer's own ``n_radial``, ``g_basis``
    and ``spherical_cutoff``, which :meth:`FNO3d.__init__` gives every block
    from one ``radial_kwargs``. So an ``n_layers=3`` model computed three
    bit-identical tensors per forward pass and threw two away.

    The cost is not the arithmetic. Building it calls
    :func:`~poraque.ml.physics.cell_reciprocal`, which moves the cell **to the
    host** to widen it to float64 --- MPS cannot represent one --- inverts it
    there and copies it back. That is a device-to-host-to-device round trip
    per Fourier layer per batch, against a dense backbone that makes none at
    all; on CUDA the device-to-host half also drains the queue, so what is
    being paid for is a synchronisation rather than a 3x3 inverse.

    Measured on MPS, batch 4 at 32^3, ``width 32 / n_radial 32``: 55.8 ms per
    forward down to 47.2, **15 %**; at ``width 16 / n_radial 16`` where the
    FFTs hide less of it, 32.9 down to 25.8, **21 %**. The saving is forward
    only --- the basis carries no gradient, so the backward pass never
    traversed the copies either way.
    """

    @staticmethod
    def _model(n_layers=3, **kwargs):
        torch.manual_seed(0)
        return FNO3d(1, 1, width=8, modes=4, n_layers=n_layers,
                     equivariant=True, use_coordinates=False,
                     n_radial=6, **kwargs)

    @staticmethod
    def _batch(batch=2, n=16):
        x = torch.randn(batch, 1, n, n, n)
        cell = torch.eye(3).mul(6.0).unsqueeze(0).repeat(batch, 1, 1)
        # Not all alike: a shared basis that quietly used the first sample's
        # cell for all of them would pass on a batch of identical cells.
        cell[1] *= 1.4
        return x, cell

    def test_the_geometry_is_resolved_once_however_deep_the_stack(self,
                                                                  monkeypatch):
        from poraque.ml import physics

        calls = []
        original = physics.cell_reciprocal
        monkeypatch.setattr(
            physics, "cell_reciprocal",
            lambda *a, **k: (calls.append(1), original(*a, **k))[1])

        x, cell = self._batch()
        for n_layers in (1, 3, 6):
            calls.clear()
            self._model(n_layers=n_layers)(x, cell)
            assert len(calls) == 1, (
                f"{len(calls)} host round trips for {n_layers} layers")

    def test_sharing_it_changes_nothing_about_the_answer(self):
        """
        Bit-identical, not merely close.

        A shared basis is an optimisation: it earns its place only if the
        operator it computes is the same operator, so the assertion is
        `torch.equal` and not a tolerance.
        """
        model = self._model().eval()
        x, cell = self._batch()
        with torch.no_grad():
            shared = model(x, cell)
            # `basis=None` per block: exactly what the code did before.
            v = model.lift(x)
            for block in model.blocks:
                v = block(v, embedding=None, max_modes=None, cell=cell,
                          basis=None)
            per_layer = model.project(v)
        assert torch.equal(shared, per_layer)

    def test_a_layer_on_its_own_still_builds_its_own(self):
        """The sharing is `FNO3d`'s doing; the layer is complete without it."""
        layer = RadialSpectralConv3d(2, 2, (4, 4, 4), n_radial=6)
        x = torch.randn(2, 2, 16, 16, 16)
        cell = torch.eye(3).mul(6.0).unsqueeze(0).repeat(2, 1, 1)
        assert layer(x, cell).shape == x.shape

    def test_a_basis_from_the_wrong_geometry_raises(self):
        """
        Every axis but the radial one would broadcast, and silently.

        A basis built for another grid, another mode count or a batch of one
        does not fail to multiply against the mode block -- it broadcasts, and
        returns a finite field computed from a geometry that is not the
        sample's. That is the failure this optimisation makes possible, so it
        is the one it has to refuse.
        """
        layer = RadialSpectralConv3d(2, 2, (4, 4, 4), n_radial=6)
        x = torch.randn(2, 2, 16, 16, 16)
        cell = torch.eye(3).mul(6.0).unsqueeze(0).repeat(2, 1, 1)
        good = layer.radial_basis(cell, (4, 4, 4), x.device, x.dtype)
        assert layer(x, cell, basis=good).shape == x.shape

        for wrong in (good[:1],                       # one sample's basis
                      good[:, :3],                    # too few radial functions
                      good[..., :2]):                 # a smaller mode block
            with pytest.raises(ValueError, match="basis has shape"):
                layer(x, cell, basis=wrong)

    def test_the_mode_counts_are_not_rebuilt_on_the_host_every_call(self):
        """
        Three numbers, and building them was a host-to-device copy per call.

        ``torch.tensor([...], device=...)`` is a transfer whatever its size,
        and on an accelerator a small transfer costs its launch rather than
        its bytes. They are constants of the run.
        """
        from poraque.ml.fno import _mode_counts

        _mode_counts.cache_clear()
        first = _mode_counts((4, 4, 4), torch.device("cpu"), torch.float32)
        second = _mode_counts((4, 4, 4), torch.device("cpu"), torch.float32)
        assert first is second
        assert _mode_counts.cache_info().hits == 1
        assert torch.equal(first, torch.tensor([4.0, 4.0, 4.0]))


class TestTheRetainedBandIsMeasuredRatherThanAssumed:
    r"""
    ``g_basis`` spans ``[0, g_basis]``; the band a material retains is its own.

    The two are unrelated until somebody checks, and nothing checked. Over
    LNCC's six-metal set the retained band edge ran 1.94 to 20.96 1/Ang at
    ``modes`` 8 --- cell lengths spanning 2.29 to 25.87 Ang --- so no constant
    fits it, and a run said nothing about which end of the spread it had
    landed on.

    Missing is not symmetric, which is the whole point of measuring:

    * too **narrow** clamps the outer modes onto one basis function, and cost
      nothing measurable --- 71.3 % of modes clamped moved the error less than
      the seed spread did;
    * too **wide** leaves radial functions at radii no mode reaches, which is
      dead capacity in a layer whose capacity *is* the radial bank, and cost
      14 % to 43 % where it was measured (a basis of 48 against a median band
      edge of 31.2, orphaning six of sixteen functions).
    """

    @staticmethod
    def _cubes(lengths, n=24):
        cells = [torch.eye(3, dtype=torch.float64) * length
                 for length in lengths]
        return cells, [(n, n, n)] * len(lengths)

    def test_the_band_edge_is_the_inscribed_radius(self):
        r"""
        The largest retained ``|G|`` sits just under ``min_i 2*pi*m_i/L_i``.

        Which is the quantity the spherical cutoff is stated in, so a report
        that disagreed with it would be describing a different retained set
        from the one the layer masks to.
        """
        import numpy as np

        cells, shapes = self._cubes([5.0, 9.0, 14.0])
        report = fno.retained_band(cells, shapes, 6)
        for length, edge in zip([5.0, 9.0, 14.0], report["edges"]):
            inscribed = 2.0 * np.pi * 6 / length
            assert edge < inscribed
            assert edge > 0.85 * inscribed

    def test_nothing_is_clamped_by_a_basis_wider_than_every_band(self):
        cells, shapes = self._cubes([5.0, 9.0, 14.0])
        report = fno.retained_band(cells, shapes, 6, g_basis=100.0)
        assert report["clamped_fraction"] == 0.0
        assert report["inside_samples"] == report["n_samples"] == 3
        assert report["clamped_samples"] == 0

    def test_the_clamped_fraction_rises_as_the_basis_narrows(self):
        cells, shapes = self._cubes([5.0, 9.0, 14.0])
        fractions = [fno.retained_band(cells, shapes, 6, g_basis=g)[
            "clamped_fraction"] for g in (12.0, 6.0, 3.0, 1.0)]
        assert fractions == sorted(fractions)
        assert fractions[0] < fractions[-1]

    def test_the_report_counts_the_modes_the_layer_actually_keeps(self):
        """
        The diagnostic and the kernel read one geometry, not two.

        ``retained_radii`` was extracted from ``radial_basis`` rather than
        reimplemented beside it: two copies would be two places for the report
        to describe a retained set the operator does not use, which is the
        class of defect the report exists to catch.
        """
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 7.0
        layer = RadialSpectralConv3d(1, 1, (5, 5, 5), n_radial=4,
                                     g_basis=8.0, spherical_cutoff=True)
        layer.double()
        radius, kept = fno.retained_radii(cell, (5, 5, 5))
        basis = layer.radial_basis(cell, (5, 5, 5), cell.device,
                                   torch.float64)

        # A masked mode contributes nothing through any basis function, and an
        # unmasked one contributes through all of them.
        assert torch.all(basis[:, :, ~kept[0]] == 0.0)
        assert torch.all(basis[:, :, kept[0]] > 0.0)
        assert int(kept.sum()) == fno.retained_band(
            [cell[0]], [(24, 24, 24)], 5)["retained"][0]
        assert radius.shape == kept.shape

    def test_the_physical_branch_needs_the_cutoff_it_truncates_at(self):
        cells, shapes = self._cubes([5.0])
        with pytest.raises(ValueError, match="requires g_max"):
            fno.retained_band(cells, shapes, 6, mode_selection="physical")
        with pytest.raises(ValueError, match="Unknown mode_selection"):
            fno.retained_band(cells, shapes, 6, mode_selection="physcial")

    def test_a_physical_truncation_keeps_the_band_inside_g_max(self):
        """
        The one coupling that is sound: ``m_i = floor(g_max*L_i/2*pi)`` puts
        the inscribed radius at or below ``g_max`` for every cell, which is why
        ``g_basis`` may follow ``g_max`` there and nowhere else.
        """
        cells, shapes = self._cubes([4.0, 7.0, 11.0, 19.0], n=48)
        report = fno.retained_band(cells, shapes, 24, mode_selection="physical",
                                   g_max=9.0, g_basis=9.0)
        assert report["edges"].max() <= 9.0
        assert report["clamped_fraction"] == 0.0


class TestTheBasisCanBeSizedFromTheTrainingSplit:
    """
    ``g_basis: auto`` takes the median band the training materials retain.

    The median is what fits the measurements rather than what sounds tidy. Of
    the configurations LNCC ran where basis and band could be compared, the
    four that performed well had ``g_basis`` within a few percent of the median
    band edge, and the one that performed badly --- 14 % worse at one
    resolution, 43 % at another --- had a basis of 48 against a median edge of
    31.2. Half the set is then clamped, and that is the deliberate half:
    clamping is free and orphaning is not.
    """

    def test_it_returns_the_median_band_edge(self):
        import numpy as np

        cells = [torch.eye(3, dtype=torch.float64) * length
                 for length in (4.0, 6.0, 9.0, 13.0, 21.0)]
        shapes = [(24, 24, 24)] * 5
        edges = fno.retained_band(cells, shapes, 6)["edges"]
        assert fno.resolve_g_basis(cells, shapes, 6) == float(np.median(edges))

    def test_an_empty_split_falls_back_rather_than_raising(self):
        """
        A diagnostic must not be the thing that stops a run.

        There is no band to measure over no materials, and the dataset layer
        has a far better error for that case a moment later.
        """
        assert fno.resolve_g_basis([], [], 6) == fno.DEFAULT_G_BASIS

    def test_the_model_refuses_the_word_and_names_who_resolves_it(self):
        """
        ``auto`` never reaches the constructor: a checkpoint must record a
        radius, and a constructor has seen no cells to derive one from.
        """
        with pytest.raises(ValueError, match="resolve_g_basis"):
            FNO3d(width=8, modes=4, n_layers=2, use_coordinates=False,
                  equivariant=True, g_basis="auto")

    def test_the_config_accepts_auto_and_refuses_a_typo(self):
        from poraque.ml.config import TrainingConfig

        def model(**equivariant):
            return TrainingConfig.from_dict({"model": {
                "use_coordinates": False,
                "equivariant": {"enable": True, **equivariant}}}).model

        assert model(g_basis="auto").equivariant_kwargs()["g_basis"] == "auto"
        assert model(g_basis=12.5).equivariant_kwargs()["g_basis"] == 12.5
        assert "g_basis" not in model().equivariant_kwargs()
        for wrong in ("atuo", -3.0, 0.0, True, "8.0"):
            with pytest.raises(ValueError, match="g_basis"):
                model(g_basis=wrong).equivariant_kwargs()


class TestTheRunSaysWhatBandItRetains:
    """
    The clamped fraction goes in the log, on run 1 rather than run 20.

    LNCC's 90-run study trained every equivariant arm with a 6.0 basis instead
    of the 8.0 default, 71.3 % of its retained modes beyond that basis, and
    found out by reading the source. One line at startup --- what the basis
    spans, what the data retains, how much of the data falls outside it ---
    would have said so immediately, and it is cheap: the geometry is the cells
    and grids the dataset already holds.

    The reporter reads ``g_basis`` off the **model**, not the config: it is
    resolved in three places (stated, ``auto``, defaulted) and only the built
    model knows which one applied.
    """

    @staticmethod
    def _train_script():
        import importlib.util
        import os
        import sys

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "scripts", "poraque_train.py")
        spec = importlib.util.spec_from_file_location("_poraque_train_band",
                                                      path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_poraque_train_band"] = module
        spec.loader.exec_module(module)
        return module

    class _Dataset:
        """Only the cells and the grid shape are read out of a sample."""

        def __init__(self, lengths, n=24):
            self.lengths, self.n = list(lengths), n

        def __len__(self):
            return len(self.lengths)

        def __getitem__(self, index):
            return {"cell": torch.eye(3) * self.lengths[index],
                    "input": torch.zeros(1, self.n, self.n, self.n)}

    @staticmethod
    def _config(**equivariant):
        from poraque.ml.config import TrainingConfig

        return TrainingConfig.from_dict({"model": {
            "modes": 6, "use_coordinates": False,
            "equivariant": {"enable": True, "n_radial": 16, **equivariant}}})

    def test_auto_becomes_a_number_before_the_model_is_built(self):
        script = self._train_script()
        data = self._Dataset([4.0, 7.0, 12.0, 20.0])
        lines = []

        resolved = script.resolve_radial_basis(
            data, self._config(g_basis="auto"), lines.append)

        expected = fno.resolve_g_basis(
            [torch.eye(3) * length for length in data.lengths],
            [(24, 24, 24)] * 4, 6)
        assert resolved == {"g_basis": pytest.approx(expected)}
        assert any("g_basis: auto ->" in line for line in lines)

    def test_a_stated_basis_is_left_alone(self):
        script = self._train_script()
        data = self._Dataset([4.0, 7.0])
        assert script.resolve_radial_basis(
            data, self._config(g_basis=9.0), lambda line: None) == {}
        assert script.resolve_radial_basis(
            data, self._config(), lambda line: None) == {}

    def test_a_dense_run_is_told_nothing(self):
        """
        The report is about a layer a dense backbone does not have, and the
        gate is the *model* rather than the config --- ``build_operator`` may
        be handed a pre-trained operator whose architecture came off disk.
        """
        from poraque.ml.config import TrainingConfig

        script = self._train_script()
        data = self._Dataset([4.0, 7.0])
        config = TrainingConfig.from_dict({"model": {"modes": 6}})
        lines = []
        script.report_radial_basis(
            data, FNO3d(width=8, modes=6, n_layers=1), lines.append)
        assert lines == []
        assert script.resolve_radial_basis(data, config, lines.append) == {}
        assert lines == []

    def test_it_names_the_clamped_fraction_and_the_band(self):
        script = self._train_script()
        data = self._Dataset([4.0, 7.0, 12.0, 20.0])
        model = FNO3d(width=8, modes=6, n_layers=1, use_coordinates=False,
                      equivariant=True, n_radial=16, g_basis=4.0)
        lines = []
        script.report_radial_basis(data, model, lines.append)
        text = "\n".join(lines)
        assert "radial basis: 16 functions over |G| < 4 1/Ang" in text
        assert "retained band:" in text
        assert "% of retained modes" in text
        # A 4 1/Ang basis is narrower than the 4 Ang cell's band, so something
        # is clamped -- the free direction, and no NOTE.
        assert "NOTE" not in text

    def test_only_the_expensive_direction_earns_a_note(self):
        """
        Clamping cost nothing measurable; orphaned basis functions cost 14 % to
        43 %. A warning on the free direction would be noise on the setting
        ``auto`` deliberately chooses.
        """
        script = self._train_script()
        data = self._Dataset([4.0, 5.0, 6.0])
        wide = FNO3d(width=8, modes=6, n_layers=1, use_coordinates=False,
                     equivariant=True, n_radial=16, g_basis=40.0)
        lines = []
        script.report_radial_basis(data, wide, lines.append)
        text = "\n".join(lines)
        assert "NOTE" in text
        assert "radial functions sit where no mode exists" in text

        narrow = FNO3d(width=8, modes=6, n_layers=1, use_coordinates=False,
                       equivariant=True, n_radial=16, g_basis=1.0)
        lines = []
        script.report_radial_basis(data, narrow, lines.append)
        assert "NOTE" not in "\n".join(lines)

    def test_the_geometry_is_read_once_however_many_reporters_ask(self):
        """
        Three reporters want the same cells, and without ``cache_in_memory``
        each pass is a full decode of every field.
        """
        script = self._train_script()

        class _Counting(self._Dataset):
            reads = 0

            def __getitem__(self, index):
                type(self).reads += 1
                return super().__getitem__(index)

        data = _Counting([4.0, 7.0, 12.0])
        script.training_geometry(data)
        script.training_geometry(data)
        script.training_geometry(data)
        assert _Counting.reads == 3
