# -*- coding: utf-8 -*-
# file: test_inference.py

"""
End-to-end tests of ``scripts/poraque_inference.py``.

**The gap these fill was coverage, not knowledge.** ``poraque-inference``
crashed on every spin-polarised model — every model this project has trained on
its own platinum data and on the six-metal Materials Project set — at
``density.integrate()``, which a :class:`~poraque.fields.SpinDensity`
deliberately does not offer. Two other files in the tree, ``poraque_train.py``
and ``calculator.py``, had already met that and each written its own
``hasattr`` check; the idiom was known and simply never reached here. Nothing
drove ``run()`` with a two-channel model at all, so neither that crash nor its
twin forty lines down on the normalisation branch was ever seen locally: both
were found by running the CLI on a V100 at LNCC against a real 400-epoch model.

The models are built here with untrained weights. What is under test is the
plumbing — which object reaches which reporting line, and in how many channels
— not the physics of the prediction.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from poraque.fields import ChargeDensity, FieldGrid, SpinDensity      # noqa: E402
from poraque.fields.structure import Structure                        # noqa: E402
from poraque.ml import BUNDLE_FILENAME, FieldOperator, save_bundle    # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_inference():
    """Import ``scripts/poraque_inference.py`` as a module."""
    path = os.path.join(_ROOT, "scripts", "poraque_inference.py")
    spec = importlib.util.spec_from_file_location("_poraque_inference_e2e", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_poraque_inference_e2e"] = module
    spec.loader.exec_module(module)
    return module


poraque_inference = _load_inference()


POSCAR = """Pt2
1.0
  4.0800000 0.0000000 0.0000000
  0.0000000 4.0800000 0.0000000
  0.0000000 0.0000000 4.0800000
  Pt
   2
Direct
  0.0000000 0.0000000 0.0000000
  0.5000000 0.5000000 0.5000000
"""


def _structure():
    """The same two-atom cell as :data:`POSCAR`, as a :class:`Structure`."""
    return Structure(np.eye(3) * 4.08, ["Pt"], [2],
                     np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))


@pytest.fixture
def structure_directory(tmp_path):
    """A bare two-atom cell: a POSCAR and nothing else, as a user would have."""
    directory = tmp_path / "Pt2"
    directory.mkdir()
    (directory / "POSCAR").write_text(POSCAR)
    return str(directory)


def _bundle(tmp_path, channels, name=BUNDLE_FILENAME):
    """
    A two-task bundle whose density channel count is ``channels``.

    ``ext2chg`` maps one channel (the external potential) to ``channels``;
    ``chg2tau`` takes those back to the single channel a kinetic energy density
    always has. That is exactly the shape ``data.spin`` resolves to — 1 -> 2
    and 2 -> 1 — so a bundle built this way exercises the real path.
    """
    built = {
        "ext2chg": FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu",
                                 in_channels=1, out_channels=channels,
                                 training_resolution=8),
        "chg2tau": FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu",
                                 in_channels=channels, out_channels=1,
                                 training_resolution=8),
    }
    # An untrained operator predicts a signed field that integrates to a
    # negative electron count, and `run()` skips the renormalisation when it
    # does (`electrons > 0`) -- so the branch these tests exist to reach would
    # never be entered. Offsetting the read-out bias makes the density
    # positive. It is not physics; it is the smallest thing that puts the code
    # path under test without training a model to get there.
    with torch.no_grad():
        built["ext2chg"].model.project[-1].bias.fill_(4.0)
    return save_bundle(str(tmp_path / name), built)


def _run(structure_directory, bundle, tmp_path, *extra):
    """Drive the CLI exactly as the console script does, and return its record."""
    return poraque_inference.predict([
        structure_directory,
        "--models", bundle,
        "--output", str(tmp_path / "predictions"),
        "--grid", "8", "8", "8",
        "--zval", "Pt=10",
        "--functional", "skip",
        "--device", "cpu",
        *extra,
    ])


# ===================================================================== #
# The crash, and its twin
# ===================================================================== #
class TestInferenceRunsOnASpinPolarisedModel:
    """
    ``poraque-inference`` on any model trained with ``data.spin`` resolved on.

    Reproduced at LNCC on `mp_6metals`, a 400-epoch six-metal model::

        File ".../poraque_inference.py", line 665, in run
            electrons = density.integrate()
        AttributeError: 'SpinDensity' object has no attribute 'integrate'

    after ``ext2chg`` had already predicted successfully — so the model and the
    bundle were fine and only the reporting around them was not. Fixing that
    one line moved the crash forty-four lines down to the identical call on the
    ``--normalize`` branch, which is why there are two tests here and not one.
    """

    def test_it_completes_and_writes_every_field(self, structure_directory,
                                                 tmp_path):
        results = _run(structure_directory, _bundle(tmp_path, 2), tmp_path)
        for name in ("EXTCAR", "CHGCAR", "TAUCAR"):
            assert os.path.exists(results["outputs"][name])

    def test_the_normalisation_branch_is_reached_too(self, structure_directory,
                                                     tmp_path):
        """
        Line 709 is on the ``--normalize`` branch, which is the default. A test
        that only ran ``--no-normalize`` would have declared the bug fixed
        while leaving the common path broken.
        """
        results = _run(structure_directory, _bundle(tmp_path, 2), tmp_path)
        assert results["normalized"] is True
        assert results["normalization_factor"] > 0.0

    def test_no_normalize_completes_as_well(self, structure_directory,
                                            tmp_path):
        results = _run(structure_directory, _bundle(tmp_path, 2), tmp_path,
                       "--no-normalize")
        assert results["normalized"] is False
        assert os.path.exists(results["outputs"]["CHGCAR"])

    def test_a_one_channel_model_still_works(self, structure_directory,
                                             tmp_path):
        """The control run, which is what said the two-channel path was the
        whole of the defect: the identical invocation on a one-channel model
        completed before any of this was fixed."""
        results = _run(structure_directory, _bundle(tmp_path, 1), tmp_path)
        assert os.path.exists(results["outputs"]["CHGCAR"])
        assert "magnetic_moment" not in results

    def test_the_written_density_still_carries_both_channels(
            self, structure_directory, tmp_path):
        """
        The repair must not be "convert to one channel and carry on". A
        spin-polarised prediction is written as a spin-polarised CHGCAR, and
        the file VASP reads back has both grid blocks.
        """
        results = _run(structure_directory, _bundle(tmp_path, 2), tmp_path)
        written = SpinDensity.read(results["outputs"]["CHGCAR"])
        assert written.magnetization.shape == (8, 8, 8)

    def test_the_magnetic_moment_is_reported(self, structure_directory,
                                             tmp_path):
        """A channel that is predicted and never mentioned is a channel nobody
        can check. It gets its own log line and its own record key."""
        results = _run(structure_directory, _bundle(tmp_path, 2), tmp_path)
        assert np.isfinite(results["magnetic_moment"])

    def test_the_optional_passes_survive_two_channels_too(
            self, structure_directory, tmp_path):
        """
        ``--compare`` subtracts a one-channel reference from the prediction and
        ``--plot-dir`` hands it to Matplotlib. Neither was named in the report
        from LNCC, because neither is reachable until the crash above is fixed:
        the subtraction would have broadcast a ``(2, ...)`` stack against a
        ``(...)`` reference and reported a relative L2 that averaged the
        density error together with the magnetisation error, and the figure
        would have died on a three-axis "image".
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")

        grid = FieldGrid((8, 8, 8), np.eye(3) * 4.08)
        values = np.random.default_rng(0).random((8, 8, 8)) + 0.1
        reference = ChargeDensity(values, grid, _structure())
        reference.write(os.path.join(structure_directory, "CHGCAR"))

        results = _run(structure_directory, _bundle(tmp_path, 2), tmp_path,
                       "--compare", "--plot-dir", str(tmp_path / "figs"))

        entry = results["comparison"]["CHGCAR"]
        assert np.isfinite(entry["relative_l2"])
        assert results["figures"]


# ===================================================================== #
# The silent half
# ===================================================================== #
class TestRenormalisationGoesThroughTheFieldsOwnMethod:
    """
    ``density.data = density.data * factor`` was the line after the crash.

    It is worth being precise about what it did, because the report from LNCC
    predicted a silent wrong answer and the truth is less forgiving and more
    convenient: :attr:`SpinDensity.data` is a read-only property returning a
    freshly stacked array, so the assignment raises
    ``AttributeError: property 'data' of 'SpinDensity' object has no setter``.
    A third crash, not a wrong number.

    The repair is to call the class's own :meth:`normalized`, which already
    knows what the hand-rolled multiply did not: that scaling :math:`\\rho`
    alone would change the local polarisation :math:`m/\\rho` everywhere, which
    is a different prediction rather than a normalisation, and that scaling the
    pair is what keeps :math:`\\rho_\\uparrow` and :math:`\\rho_\\downarrow`
    non-negative.
    """

    def test_the_electron_count_is_the_one_the_pseudopotentials_fix(
            self, structure_directory, tmp_path):
        results = _run(structure_directory, _bundle(tmp_path, 2), tmp_path)
        assert results["electrons"] == pytest.approx(20.0, rel=1e-9)

    def test_the_local_polarisation_is_what_is_preserved(self):
        """
        Stated as physics, on the class rather than through the CLI, because
        this is the decision the inference path now inherits instead of making
        its own. ``m/rho`` is pointwise invariant under the rescaling; the
        moment is not, and is not meant to be — nothing fixes it the way the
        pseudopotentials fix the electron count.
        """
        grid = FieldGrid((8, 8, 8), np.eye(3) * 4.08)
        rng = np.random.default_rng(3)
        rho = rng.random((8, 8, 8)) + 1.0
        m = 0.2 * (rng.random((8, 8, 8)) - 0.5)
        density = SpinDensity(rho, m, grid, _structure())

        scaled = density.normalized(20.0, clip_negative=False)
        assert scaled.electron_count() == pytest.approx(20.0)
        np.testing.assert_allclose(scaled.magnetization / scaled.total,
                                   m / rho, rtol=1e-12)

    def test_the_data_stack_cannot_be_assigned_to(self):
        """The regression, pinned as the property it is: a call site that
        reaches for ``.data =`` on a prediction is reaching for the wrong
        thing, whichever channel count it happens to get."""
        grid = FieldGrid((4, 4, 4), np.eye(3) * 4.08)
        density = SpinDensity(np.ones((4, 4, 4)), np.zeros((4, 4, 4)),
                              grid, _structure())
        with pytest.raises(AttributeError, match="no setter"):
            density.data = density.data * 2.0


# ===================================================================== #
# The diagnostics that were measuring the wrong field
# ===================================================================== #
class TestTheDiagnosticsAreAboutTheDensityAlone:
    """
    A valence density must be non-negative; a magnetisation must not.

    On a two-channel prediction the negative-voxel count ran over the stack, so
    every spin-down-majority voxel was reported as a defect and the ``range``
    line's minimum was the most negative magnetisation rather than anything in
    e/Ang^3. Loud and wrong, which is its own kind of harm: the natural
    reaction is to distrust a model that is fine.
    """

    def test_the_negative_voxel_count_ignores_the_magnetisation(self, capsys,
                                                                structure_directory,
                                                                tmp_path):
        _run(structure_directory, _bundle(tmp_path, 2), tmp_path)
        printed = capsys.readouterr().out

        # The magnetisation gets its own line, in mu_B, and the density's own
        # range line is still there and is still about e/Ang^3.
        assert "magnetization: integral" in printed
        assert "e/A^3" in printed

    def test_the_von_weizsacker_bound_is_taken_against_rho(self, capsys,
                                                           structure_directory,
                                                           tmp_path):
        """
        ``von_weizsacker_tau(density.data, grid)`` on a two-channel field
        returns a ``(2, ...)`` bound, which then broadcasts against a
        one-channel tau and doubles the number of points the constraint is
        checked at. The reported violation fraction is over the grid, so it is
        arithmetic that produces a plausible number from the wrong comparison.
        """
        results = _run(structure_directory, _bundle(tmp_path, 2), tmp_path)
        assert results["constraint"]["fraction"] <= 1.0
        # 8**3 points, not 2 * 8**3.
        violations = results["constraint"]["violations"]
        assert violations <= 8 ** 3


# ===================================================================== #
# One helper, not three
# ===================================================================== #
class TestOneHelperDecidesTheChannelQuestion:
    """
    ``poraque_train.py`` had a named helper, ``calculator.py`` had the same
    decision spelled the other way round, and ``poraque_inference.py`` had
    neither. That is how line 709 came to be missed: the fix was applied by
    reading, three times, in three files. There is one implementation now, in
    :mod:`poraque.fields`, beside the classes it discriminates between.
    """

    @pytest.fixture
    def fields(self):
        grid = FieldGrid((6, 6, 6), np.eye(3) * 4.08)
        rho = np.full((6, 6, 6), 0.5)
        plain = ChargeDensity(rho, grid, _structure())
        spin = SpinDensity(rho, np.zeros((6, 6, 6)), grid, _structure())
        return plain, spin

    def test_it_gives_the_same_answer_for_both_channel_counts(self, fields):
        from poraque.fields import field_integral

        plain, spin = fields
        assert field_integral(plain) == pytest.approx(field_integral(spin))

    def test_the_charge_channel_of_a_plain_field_is_itself(self, fields):
        from poraque.fields import charge_channel

        plain, _ = fields
        assert charge_channel(plain) is plain

    def test_the_charge_channel_of_a_pair_is_one_channel(self, fields):
        from poraque.fields import charge_channel

        _, spin = fields
        assert charge_channel(spin).data.shape == (6, 6, 6)

    def test_the_script_and_the_calculator_import_it_rather_than_copy_it(self):
        """
        Asserted on the source, because the failure this guards against is a
        fourth copy being written next year rather than anything a run would
        show. ``hasattr(..., "integrate")`` appearing again in either file is
        the shape of that mistake.
        """
        for name in ("scripts/poraque_train.py", "src/poraque/calculator.py",
                     "scripts/poraque_inference.py"):
            text = open(os.path.join(_ROOT, name)).read()
            assert "field_integral" in text
            assert 'hasattr(field, "integrate")' not in text
