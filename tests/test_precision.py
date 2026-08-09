# -*- coding: utf-8 -*-
# file: test_precision.py

"""
Tests for selectable numeric precision.

Two settings, deliberately separate: ``data.precision`` is how a field is
*stored*, ``model.precision`` is what the operator *computes* in. A field may
be held in float64 and fed to a float32 operator, which is the default.

Most of what follows guards conversions that fail quietly. An FNO carries
complex parameters, and both obvious PyTorch idioms mishandle them -- one
leaves them behind, the other deletes their imaginary part -- so the tests
that pin ``set_precision`` are the ones that matter.
"""

import numpy as np
import pytest

from poraque.fields import (
    FIELD_DTYPES,
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    SpinDensity,
    get_default_dtype,
    resolve_dtype,
    set_default_dtype,
)
from poraque.fields.vasp.poscar import Poscar

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.ml.fno import (  # noqa: E402
    FNO3d,
    PRECISIONS,
    model_precision,
    resolve_precision,
    set_precision,
)


@pytest.fixture
def material():
    cell = np.eye(3) * 5.0
    grid = FieldGrid((8, 8, 8), cell)
    structure = Poscar(cell, ["Si"], [2],
                       np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))
    values = np.random.default_rng(0).random(grid.shape) + 0.1
    return values, grid, structure


# ===================================================================== #
# Fields
# ===================================================================== #
class TestFieldDtype:
    def test_the_default_is_float64(self, material):
        values, grid, structure = material
        assert ChargeDensity(values, grid, structure).dtype == np.float64
        assert get_default_dtype() == np.dtype(np.float64)

    @pytest.mark.parametrize("name", sorted(FIELD_DTYPES))
    def test_every_named_precision_is_honoured(self, name, material):
        values, grid, structure = material
        field = ChargeDensity(values, grid, structure, dtype=name)
        assert field.dtype == FIELD_DTYPES[name]

    def test_float32_halves_the_memory(self, material):
        values, grid, structure = material
        double = ChargeDensity(values, grid, structure, dtype="float64")
        single = ChargeDensity(values, grid, structure, dtype="float32")
        assert single.nbytes() * 2 == double.nbytes()

    def test_astype_actually_converts(self, material):
        """
        The constructor applies the process default, so a naive ``astype``
        that forwarded only the cast converted straight back to float64.
        """
        values, grid, structure = material
        field = ChargeDensity(values, grid, structure)
        assert field.astype("float32").dtype == np.float32
        assert field.astype("float32").astype("float64").dtype == np.float64

    def test_astype_leaves_the_original_alone(self, material):
        values, grid, structure = material
        field = ChargeDensity(values, grid, structure)
        field.astype("float32")
        assert field.dtype == np.float64, "fields are shared between members"

    def test_an_integer_dtype_is_refused(self, material):
        values, grid, structure = material
        with pytest.raises(TypeError, match="floating type"):
            ChargeDensity(values, grid, structure, dtype="int32")

    def test_resolve_accepts_names_and_numpy_dtypes(self):
        assert resolve_dtype("float32") == np.dtype(np.float32)
        assert resolve_dtype(np.float64) == np.dtype(np.float64)
        assert resolve_dtype(None) == get_default_dtype()

    def test_the_default_can_be_set_and_restored(self, material):
        values, grid, structure = material
        previous = set_default_dtype("float32")
        try:
            assert ChargeDensity(values, grid, structure).dtype == np.float32
        finally:
            set_default_dtype(previous)
        assert ChargeDensity(values, grid, structure).dtype == np.float64

    def test_the_grid_stays_double_whatever_the_field_does(self, material):
        values, grid, structure = material
        field = ChargeDensity(values, grid, structure, dtype="float32")
        assert field.grid.cell.dtype == np.float64, (
            "geometry is nine numbers per material and a rounding error there "
            "moves an atom")

    def test_a_float32_field_still_integrates_correctly(self, material):
        values, grid, structure = material
        double = ChargeDensity(values, grid, structure)
        single = double.astype("float32")
        assert single.electron_count() == pytest.approx(
            double.electron_count(), rel=1e-6)

    def test_spin_keeps_both_channels_in_one_dtype(self, material):
        values, grid, structure = material
        spin = SpinDensity(values, values * 0.1, grid, structure,
                           dtype="float32")
        assert spin.total.dtype == spin.magnetization.dtype == np.float32
        assert spin.astype("float64").dtype == np.float64
        assert spin.as_charge_density().dtype == np.float32


class TestRoundTripThroughDisk:
    def test_a_float32_field_survives_a_write_and_read(self, material,
                                                        tmp_path):
        """
        Written values are widened to double first. The format prints eleven
        significant digits and a float32 product would fill seven of them with
        noise.
        """
        values, grid, structure = material
        field = ChargeDensity(values, grid, structure, dtype="float32")
        path = tmp_path / "CHGCAR"
        field.write(path)

        back = ChargeDensity.read(path, grid=grid)
        assert back.dtype == np.float64, "read defaults to the process default"
        np.testing.assert_allclose(back.data, field.data, rtol=1e-6)

    def test_read_honours_an_explicit_dtype(self, material, tmp_path):
        values, grid, structure = material
        path = tmp_path / "CHGCAR"
        ChargeDensity(values, grid, structure).write(path)
        assert ChargeDensity.read(path, grid=grid,
                                  dtype="float32").dtype == np.float32


# ===================================================================== #
# The model
# ===================================================================== #
class TestModelPrecision:
    @staticmethod
    def _model():
        torch.manual_seed(0)
        return FNO3d(in_channels=1, out_channels=1, width=8, modes=4,
                     n_layers=2)

    def test_the_default_is_float32(self):
        assert model_precision(self._model()) == "float32"

    def test_conversion_reaches_the_complex_weights(self):
        model = set_precision(self._model(), "float64")
        assert model_precision(model) == "float64"
        assert {p.dtype for p in model.parameters()} == {torch.float64,
                                                         torch.complex128}

    def test_double_is_not_a_substitute(self):
        """
        ``Module.double()`` converts only ``is_floating_point()`` tensors, so
        the complex spectral weights are left at complex64 and the next
        forward pass dies on a dtype mismatch.
        """
        model = self._model().double()
        assert any(p.dtype == torch.complex64 for p in model.parameters())
        assert model_precision(model) == "mixed"

    def test_to_float64_destroys_the_spectral_weights(self):
        """
        The dangerous one. ``model.to(torch.float64)`` casts complex64 to
        *float64*, discarding the imaginary part of every Fourier multiplier,
        and raises nothing.
        """
        model = self._model().to(torch.float64)
        assert not any(p.is_complex() for p in model.parameters()), (
            "if PyTorch ever fixes this, set_precision's warning should be "
            "revisited")

    def test_a_float64_forward_agrees_with_float32(self):
        model = self._model().eval()
        x = torch.randn(1, 1, 16, 16, 16)
        cell = (torch.eye(3) * 5.0).unsqueeze(0)
        with torch.no_grad():
            single = model(x, cell)
            set_precision(model, "float64")
            double = model(x.double(), cell.double())
        assert double.dtype == torch.float64
        assert (double.float() - single).abs().max().item() < 1e-4

    def test_gradients_flow_in_float64(self):
        model = set_precision(self._model(), "float64").train()
        x = torch.randn(1, 1, 16, 16, 16, dtype=torch.float64)
        cell = (torch.eye(3) * 5.0).unsqueeze(0).double()
        model(x, cell).pow(2).mean().backward()
        spectral = [p for p in model.parameters() if p.is_complex()]
        assert spectral
        for parameter in spectral:
            assert parameter.grad is not None
            assert parameter.grad.abs().max().item() > 0

    def test_the_conversion_round_trips(self):
        model = self._model()
        set_precision(model, "float64")
        set_precision(model, "float32")
        assert model_precision(model) == "float32"

    def test_an_unknown_precision_is_refused(self):
        with pytest.raises(ValueError, match="Unknown precision"):
            resolve_precision("bfloat16")

    def test_resolve_accepts_torch_dtypes(self):
        assert resolve_precision(torch.float64) == PRECISIONS["float64"]


class TestOperatorPrecision:
    @staticmethod
    def _operator():
        from poraque.ml import FieldOperator

        return FieldOperator("ext2chg", width=8, modes=4, n_layers=1,
                             projection_channels=16, device="cpu", init_seed=0)

    def test_compute_dtype_is_read_from_the_weights(self):
        operator = self._operator()
        assert operator.compute_dtype() == torch.float32
        operator.set_precision("float64")
        assert operator.compute_dtype() == torch.float64

    def test_predict_works_in_both_precisions(self, material):
        values, grid, structure = material
        potential = ExternalPotential(values, grid, structure)
        operator = self._operator()
        single = operator.predict(potential)
        operator.set_precision("float64")
        double = operator.predict(potential)
        assert np.abs(double.data - single.data).max() < 1e-4

    def test_a_float32_field_feeds_a_float64_operator(self, material):
        """
        The two settings are independent, so the mismatched combination has to
        work rather than raise.
        """
        values, grid, structure = material
        potential = ExternalPotential(values, grid, structure, dtype="float32")
        operator = self._operator().set_precision("float64")
        assert operator.predict(potential) is not None


class TestConfigValidation:
    @staticmethod
    def _validate(data=None, model=None, device="cpu"):
        import sys

        sys.path.insert(0, __import__("os").path.join(
            __import__("os").path.dirname(__file__), "..", "scripts"))
        from poraque_train import validate_precision_settings

        from poraque.ml.config import TrainingConfig

        config = TrainingConfig()
        config.training.device = device
        if data is not None:
            config.data.precision = data
        if model is not None:
            config.model.precision = model
        return validate_precision_settings(config)

    def test_the_defaults_pass(self):
        self._validate()

    def test_an_unknown_field_precision_is_refused(self):
        with pytest.raises(SystemExit, match="data.precision"):
            self._validate(data="f8")

    def test_an_unknown_model_precision_is_refused(self):
        with pytest.raises(SystemExit, match="model.precision"):
            self._validate(model="half")

    def test_float16_data_into_a_float64_model_is_refused(self):
        with pytest.raises(SystemExit, match="three\n?\\s*decimal digits"):
            self._validate(data="float16", model="float64")

    def test_float64_on_cpu_is_allowed(self):
        self._validate(model="float64", device="cpu")

    @pytest.mark.skipif(not torch.backends.mps.is_available(),
                        reason="needs an Apple Metal device")
    def test_float64_on_mps_is_refused_with_a_way_out(self):
        """Metal has no float64 at all -- absent, not slow."""
        with pytest.raises(SystemExit, match="does not implement double"):
            self._validate(model="float64", device="mps")

    def test_precision_is_not_passed_to_the_backbone(self):
        from poraque.ml.config import TrainingConfig

        assert "precision" not in TrainingConfig().model_kwargs(), (
            "FNO3d takes no precision argument; it is applied afterwards")


# ===================================================================== #
# Configuration automation
# ===================================================================== #
class TestDerivedOutputPaths:
    """
    A run should be named once.

    Every artefact already came from ``task.name`` -- ``models/<name>.pfno``,
    ``reports/<name>_report.pdf``, ``results/plots/<name>/``. The log and the
    metrics were the two a user had to repeat by hand, and the only two that
    two different runs could end up sharing while writing separate weights.
    """

    @staticmethod
    def _config(**task):
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig()
        for key, value in task.items():
            setattr(config.task, key, value)
        return config

    def test_the_log_follows_the_name(self):
        import os

        assert self._config(name="au_w16").log_path() == os.path.join(
            "models", "au_w16", "log", "au_w16.log")

    def test_the_metrics_follow_the_name(self):
        import os

        assert self._config(name="au_w16").json_path() == os.path.join(
            "models", "au_w16", "log", "au_w16.json")

    def test_everything_lands_in_one_run_folder(self):
        """
        A trained model is weights plus the numbers that say how good they
        are, the figures behind those numbers, and the config that produced
        them. They arrive and leave together.
        """
        config = self._config(name="au_w16")
        run = config.run_dir()
        for path in (config.checkpoint_path(), config.log_path(),
                     config.plot_dir(), config.report_dir()):
            assert path.startswith(run + "/"), path

    def test_each_toggle_switches_off_its_own_artefact(self):
        config = self._config(name="au_w16")
        config.output.write_pdf_report = False
        config.output.plot_figures = False
        config.output.write_log = False
        assert config.report_dir() is None
        assert config.plot_dir() is None
        assert config.log_path() is None and config.json_path() is None
        assert config.checkpoint_path() is not None, "weights are separate"

    def test_a_null_root_switches_everything_off(self):
        config = self._config(name="au_w16")
        config.output.root = None
        assert config.run_dir() is None
        assert config.checkpoint_path() is None
        assert config.plot_dir() is None
        assert config.report_dir() is None
        assert config.log_path() is None

    def test_an_explicit_path_still_wins(self):
        config = self._config(name="au_w16")
        config.output.log = "somewhere/else.log"
        assert config.log_path() == "somewhere/else.log"

    def test_two_names_cannot_share_a_log(self):
        assert self._config(name="a").log_path() != self._config(
            name="b").log_path()

    def test_an_empty_string_switches_the_log_off(self):
        config = self._config(name="a")
        config.output.log = ""
        assert config.log_path() is None


class TestMinimalConfig:
    """
    Every key is optional, and a config that says so is far shorter.

    Of the 80 keys a full config carries, the shipped example differed in 15.
    ``--minimal`` writes those; the round trip below is what makes dropping
    the other 65 safe rather than merely shorter.
    """

    @staticmethod
    def _configured():
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig()
        config.task.name = "probe"
        config.training.epochs = 123
        config.model.n_layers = 7
        return config

    def test_only_the_differences_are_listed(self):
        trimmed = self._configured().non_default_dict()
        assert trimmed == {"task": {"name": "probe"},
                           "model": {"n_layers": 7},
                           "training": {"epochs": 123}}

    def test_defaults_produce_an_empty_document(self):
        from poraque.ml.config import TrainingConfig

        assert TrainingConfig().non_default_dict() == {}

    def test_a_minimal_config_round_trips_to_the_same_run(self, tmp_path):
        from poraque.ml.config import TrainingConfig

        original = self._configured()
        path = tmp_path / "minimal.yaml"
        path.write_text(original.to_yaml(minimal=True))
        assert TrainingConfig.from_yaml(path).to_dict() == original.to_dict()

    def test_the_minimal_form_is_shorter(self):
        config = self._configured()
        assert len(config.to_yaml(minimal=True)) < len(config.to_yaml()) / 4

    def test_the_full_form_is_still_the_default(self):
        from poraque.ml.config import TrainingConfig

        assert "epochs" in TrainingConfig().to_yaml(), (
            "the archived copy beside a run must record every value in force")


def _shipped_configs():
    """
    Every config in ``configs/``, discovered rather than listed.

    A hard-coded list silently stops covering a file the moment one is renamed
    or added, which is exactly when the coverage is wanted.
    """
    import glob
    import os

    root = os.path.join(os.path.dirname(__file__), "..", "configs")
    return sorted(os.path.basename(p)[:-5]
                  for p in glob.glob(os.path.join(root, "*.yaml")))


class TestShippedConfigs:
    """The committed examples must load, and must stay short."""

    import os as _os

    ROOT = _os.path.join(_os.path.dirname(__file__), "..", "configs")

    @pytest.mark.parametrize("name", _shipped_configs())
    def test_it_loads(self, name):
        import os

        from poraque.ml.config import TrainingConfig

        TrainingConfig.from_yaml(os.path.join(self.ROOT, f"{name}.yaml"))

    #: Keys allowed to restate their default in a shipped config.
    #:
    #: ``task.type`` is spelled out on purpose: it is the first decision a
    #: reader makes, it changes what the resulting model can do -- only
    #: ``all`` gives both halves of the chain, and so a total energy -- and a
    #: config that leaves it implicit reads as if the question had not been
    #: considered.
    #: Each entry carries an explanatory comment in the file, which is the
    #: reason it is written out rather than left to the default:
    #:
    #: ``task.type``      the first decision a reader makes, and the one that
    #:                    decides whether the model can reach a total energy
    #: ``model.width``,   the three settings most often confused with each
    #: ``model.modes``,   other; the config is where that is explained
    #: ``model.projection_channels``
    #: ``model.mode_selection`` the choice to revisit when the cells stop
    #:                    being uniform, and silent when it is wrong
    #: ``symbolic.physics`` shown beside ``training.physics`` so the two are
    #:                    visibly separate objectives on separate objects
    DELIBERATE = {"task.type", "model.width", "model.modes",
                  "model.projection_channels", "model.mode_selection",
                  "symbolic.physics"}

    @pytest.mark.parametrize("name", _shipped_configs())
    def test_it_states_only_what_it_changes(self, name):
        """
        A shipped config that restated defaults taught readers to restate
        them too, and buried the handful of settings that mattered.
        """
        import os

        import yaml

        with open(os.path.join(self.ROOT, f"{name}.yaml")) as handle:
            raw = yaml.safe_load(handle)

        from poraque.ml.config import TrainingConfig

        defaults = TrainingConfig().to_dict()
        redundant = [f"{section}.{key}"
                     for section, values in raw.items()
                     if isinstance(values, dict)
                     for key, value in values.items()
                     if defaults.get(section, {}).get(key, object()) == value
                     and f"{section}.{key}" not in self.DELIBERATE]
        assert not redundant, f"these keys restate the default: {redundant}"

    @pytest.mark.parametrize("name", _shipped_configs())
    def test_the_task_type_is_always_spelled_out(self, name):
        """The first decision a reader makes should not be implicit."""
        import os

        import yaml

        with open(os.path.join(self.ROOT, f"{name}.yaml")) as handle:
            raw = yaml.safe_load(handle)
        assert "type" in raw.get("task", {}), (
            "task.type decides whether the model can reach a total energy")


class TestDocstringsAreNotMangled:
    r"""
    A docstring holding LaTeX must be a raw string.

    In a plain ``"..."`` docstring Python resolves the escapes before Sphinx
    ever sees them: ``\rm`` becomes a carriage return, ``\tau`` a tab, ``\beta``
    a backspace, ``\frac`` a form feed. Nothing raises, the module imports, the
    docs build without complaint -- and the rendered page shows ``:math:`` +
    a tab + ``au`` where it should show a Greek letter.

    ``FeatureTable`` had exactly that for every ``\tau`` in it, and
    ``ModelConfig`` acquired it the moment an indexed equation was written into
    it. The failure is invisible in the source, so it needs a test rather than
    care.
    """

    #: Control characters a resolved LaTeX escape leaves behind, and the
    #: command each one most likely came from.
    SUSPECT = {
        "\r": r"\r  (\rm, \rho, \right, ...)",
        "\t": r"\t  (\tau, \text, \times, ...)",
        "\x08": r"\b  (\beta, \bf, \big, ...)",
        "\x0c": r"\f  (\frac, \forall, ...)",
        "\x07": r"\a  (\alpha, \approx, ...)",
        "\v": r"\v  (\vec, \varepsilon, ...)",
    }

    @staticmethod
    def _sources():
        import os

        root = os.path.join(os.path.dirname(__file__), "..")
        for base, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs
                       if d not in (".git", "__pycache__", "_build", "docbuild")]
            for name in sorted(names):
                if name.endswith(".py"):
                    yield os.path.join(base, name)

    def test_no_docstring_contains_a_resolved_escape(self):
        import ast
        import os

        offenders = []
        for path in self._sources():
            with open(path) as handle:
                tree = ast.parse(handle.read(), filename=path)
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Module, ast.ClassDef,
                                         ast.FunctionDef,
                                         ast.AsyncFunctionDef)):
                    continue
                doc = ast.get_docstring(node, clean=False)
                if not doc:
                    continue
                for char, why in self.SUSPECT.items():
                    if char in doc:
                        offenders.append(
                            f"{os.path.relpath(path)}:"
                            f"{getattr(node, 'lineno', 1)} "
                            f"{getattr(node, 'name', '<module>')}: {why}")

        assert not offenders, (
            "these docstrings hold LaTeX escapes Python already resolved; "
            "make them raw (r\"\"\"...\"\"\"):\n  " + "\n  ".join(offenders))

    def test_the_indexed_equation_survives_into_the_docs(self):
        """The specific content the raw string was needed for."""
        from poraque.ml.config import ModelConfig

        doc = ModelConfig.__doc__
        for fragment in (r"v^{(0)}_o(\mathbf r)", r"\hat v^{(\ell-1)}_i",
                         r"\mathrm{GroupNorm}", r"C_{\rm out}",
                         r"\gamma_o(\text{cell})"):
            assert fragment in doc, fragment

    def test_every_index_range_is_stated(self):
        """
        The point of the indexed form: each config key is tied to a range.
        """
        from poraque.ml.config import ModelConfig

        doc = ModelConfig.__doc__
        assert r"o = 1 \dots C" in doc               # width
        assert r"\ell = 1 \dots L" in doc            # n_layers
        assert r"p = 1 \dots P" in doc               # projection_channels
        assert r"m_1 = \min(\texttt{modes}, N_x/2)" in doc   # modes


class TestModeSelectionReporting:
    r"""
    ``g_max`` can only take modes away, and does it silently.

    The retained count is ``min(modes, floor(g_max*L/2pi), N/2)``. ``modes``
    *allocates* the weight tensor; ``g_max`` only *masks* at run time. A
    ``g_max`` tighter than ``modes`` therefore leaves the unreached weights
    with exactly zero gradient — allocated, checkpointed, never trained — with
    no error and nothing in the log to notice. Hence the report.
    """

    @staticmethod
    def _config(g_max, modes=8, selection="physical"):
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig()
        config.model.modes = modes
        config.model.mode_selection = selection
        config.model.g_max = g_max
        return config

    class _Set(list):
        """Minimal stand-in for a dataset: only ``cell`` is read."""

        def __init__(self, lengths):
            import numpy as np

            super().__init__({"cell": np.eye(3) * length}
                             for length in lengths)

    @staticmethod
    def _report(config, train_set):
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "scripts"))
        from poraque_train import report_mode_selection

        lines = []
        report_mode_selection(train_set, config, lines.append)
        return "\n".join(lines)

    def test_fixed_selection_reports_nothing(self):
        text = self._report(self._config(None, selection="fixed"),
                            self._Set([4.08]))
        assert text == ""

    def test_physical_without_g_max_reports_nothing(self):
        text = self._report(self._config(None), self._Set([4.08]))
        assert text == ""

    def test_a_generous_g_max_reports_no_truncation(self):
        """L = 20 A at g_max 6 supplies 19 modes, so 8 survive untouched."""
        text = self._report(self._config(6.0), self._Set([20.0]))
        assert "8-8 of 8 modes retained" in text
        assert "NOTE" not in text

    def test_a_tight_g_max_says_how_much_is_dead(self):
        """fcc Au at 4.08 A with g_max 6 keeps 3 of 8."""
        text = self._report(self._config(6.0), self._Set([4.08]))
        assert "3-3 of 8 modes retained" in text
        assert "dead parameters" in text
        assert "8.4 Ang" in text, "the threshold has to be actionable"

    def test_a_mixed_dataset_reports_the_range(self):
        text = self._report(self._config(6.0), self._Set([4.08, 6.0, 12.0]))
        assert "3-8 of 8 modes retained" in text
        assert "NOTE" not in text, (
            "the largest cell reaches the ceiling, so nothing is dead")

    def test_the_arithmetic_matches_the_model(self):
        """The report must not drift from what SpectralConv3d actually does."""
        import numpy as np
        import torch

        from poraque.ml.fno import FNO3d

        model = FNO3d(in_channels=1, out_channels=1, width=4, modes=8,
                      n_layers=1, projection_channels=8,
                      cell_conditioning=False, mode_selection="physical",
                      g_max=6.0)
        for length in (4.08, 6.0, 8.4, 20.0):
            cell = (torch.eye(3) * length).unsqueeze(0)
            caps = model.physical_modes(cell, (48, 48, 48))
            effective = model.blocks[0].spectral.effective_modes(
                (48, 48, 48), caps)
            expected = min(8, max(1, int(np.floor(6.0 * length
                                                  / (2.0 * np.pi)))))
            assert effective[0] == expected, (length, effective, expected)

    def test_unreached_weights_receive_no_gradient(self):
        """
        The claim the whole report rests on: outside the retained block the
        spectral weights are not merely small, they are untouched.
        """
        import torch

        from poraque.ml.fno import FNO3d

        torch.manual_seed(0)
        model = FNO3d(in_channels=1, out_channels=1, width=4, modes=8,
                      n_layers=1, projection_channels=8,
                      cell_conditioning=False, mode_selection="physical",
                      g_max=6.0).train()
        cell = (torch.eye(3) * 4.08).unsqueeze(0)      # keeps 3 of 8
        model(torch.randn(1, 1, 32, 32, 32), cell).pow(2).mean().backward()

        gradient = model.blocks[0].spectral.weight.grad.abs()
        assert gradient[:, :, :, :3, :3, :3].sum().item() > 0
        outside = gradient.clone()
        outside[:, :, :, :3, :3, :3] = 0
        assert outside.max().item() == 0.0, (
            "weights beyond the g_max cap must be exactly untrained")
