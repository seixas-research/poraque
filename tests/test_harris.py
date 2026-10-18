"""Analytic and integration checks for the fixed-density Harris functional."""
import numpy as np
import pytest

from poraque.fields import FieldGrid
from poraque.physics.harris import HarrisEvaluator, integrate, lda_terms


def test_uniform_density_cancels_xc_and_hartree():
    """Two free electrons in a uniform Gamma orbital have E = E_xc."""
    grid = FieldGrid((3, 3, 3), np.eye(3) * 5)
    n = np.full(grid.shape, 2 / grid.volume)
    result = HarrisEvaluator(grid, np.zeros(grid.shape), 2, ion_ion=1.2).evaluate(n)
    exc, _ = lda_terms(n, grid)
    assert result.total == pytest.approx(exc + 1.2, abs=1e-10)
    assert result.hartree == pytest.approx(0, abs=1e-12)
    assert integrate(result.output_density, grid) == pytest.approx(2)
    assert np.allclose(result.output_density, n)


def test_double_counting_input_density_and_gauge():
    grid = FieldGrid((5, 5, 5), np.eye(3) * 5)
    x = np.indices(grid.shape)[0]
    n = (2 / grid.volume) * (1 + 0.3 * np.cos(2 * np.pi * x / 5))
    original = n.copy()
    result = HarrisEvaluator(grid, np.zeros(grid.shape), 2).evaluate(n)
    shifted = HarrisEvaluator(grid, np.full(grid.shape, 3.0), 2).evaluate(n)
    assert result.hartree_double_counting == pytest.approx(2 * result.hartree)
    assert shifted.total - result.total == pytest.approx(6.0, abs=1e-9)
    assert result.total == pytest.approx(result.band - result.hartree_double_counting
                                        + result.hartree - result.xc_double_counting + result.xc)
    assert np.array_equal(n, original)
    assert not np.allclose(result.output_density, n)


def test_xc_derivative():
    grid = FieldGrid((3, 3, 3), np.eye(3) * 4)
    n = np.full(grid.shape, 0.05)
    _, potential = lda_terms(n, grid)
    delta = np.random.default_rng(2).normal(size=grid.shape)
    step = 1e-6
    plus, _ = lda_terms(n + step * delta, grid)
    minus, _ = lda_terms(n - step * delta, grid)
    assert (plus - minus) / (2 * step) == pytest.approx(integrate(potential * delta, grid), rel=1e-7)


@pytest.mark.parametrize('kind', ['negative', 'count', 'nan', 'shape'])
def test_invalid_density(kind):
    grid = FieldGrid((3, 3, 3), np.eye(3) * 5)
    n = np.full(grid.shape, 2 / grid.volume)
    if kind == 'negative':
        n.flat[0] = -0.1
    elif kind == 'count':
        n *= 2
    elif kind == 'nan':
        n.flat[0] = np.nan
    else:
        n = n[0]
    with pytest.raises(ValueError):
        HarrisEvaluator(grid, np.zeros(grid.shape), 2).evaluate(n)


@pytest.mark.parametrize('skew', [False, True])
def test_iterative_matches_dense(skew):
    """Exercise the matrix-free branch against a small direct reference."""
    cell = np.eye(3) * 5
    if skew:
        cell[1, 0] = 1.3
    grid = FieldGrid((6, 6, 4), cell)
    n = np.full(grid.shape, 2 / grid.volume)
    external = np.random.default_rng(3).normal(size=grid.shape)
    evaluator = HarrisEvaluator(grid, external, 2)
    result = evaluator.evaluate(n)
    _, vxc = lda_terms(n, grid)
    size = n.size
    def apply(v):
        psi = v.reshape(grid.shape)
        return (np.fft.ifftn(evaluator.kinetic * np.fft.fftn(psi)) +
                (external + vxc) * psi).ravel()
    matrix = np.column_stack([apply(v) for v in np.eye(size)])
    assert np.allclose(matrix, matrix.conj().T)
    assert result.eigenvalues[0] == pytest.approx(np.linalg.eigvalsh(matrix)[0], abs=1e-8)


def test_ext2chg_adapter_uses_only_model_density(monkeypatch):
    """Geometry reaches ext2chg, and repaired input preserves raw prediction."""
    from types import SimpleNamespace
    from ase.build import molecule
    from poraque.fields import ChargeDensity, ExternalPotential
    import poraque.ml.harris as bridge

    calls = []
    def predict(external):
        calls.append(external.structure.formula)
        values = np.full(external.grid.shape, 0.03)
        values.flat[0] = -0.01
        return ChargeDensity(values, external.grid, external.structure)

    def load(bundle, task, device):
        assert task == 'ext2chg'
        return SimpleNamespace(predict=predict)

    monkeypatch.setattr(bridge, 'load_bundle', load)
    monkeypatch.setattr(bridge.Potcar, 'from_library', lambda *args: [
        SimpleNamespace(element='H', zval=1), SimpleNamespace(element='O', zval=6)])
    monkeypatch.setattr(bridge.ExternalPotential, 'from_potcar_tables',
                        lambda structure, grid, potcar: ExternalPotential(
                            np.zeros(grid.shape), grid, structure))
    atoms = molecule('H2O')
    atoms.set_cell([5] * 3)
    atoms.center()
    pipeline = bridge.Ext2ChgHarris('test', 'test', (3, 3, 3))
    with pytest.raises(ValueError, match='nonnegative'):
        pipeline.predict(atoms)
    pipeline.repair_density = True
    prediction = pipeline.predict(atoms)
    assert calls == ['H2O1', 'H2O1']
    assert prediction.raw_density.flat[0] == -0.01
    assert prediction.input_density.flat[0] == 0
    assert prediction.result.electrons == pytest.approx(8)
    assert prediction.clipped_electrons > 0
    assert prediction.result.total == pytest.approx(
        prediction.evaluator.evaluate(prediction.input_density).total)


def test_legacy_chon_buffer_names():
    """Old spelling migration preserves strict, bit-identical model weights."""
    import torch
    from poraque.ml.training import FieldOperator
    operator = FieldOperator('ext2chg', width=4, modes=2, n_layers=1,
                             use_coordinates=False, equivariant=True,
                             n_radial=3, device='cpu', site_layouts={1: [0]},
                             readout_g_max=2.0)
    state = operator.state()
    state['model_state'] = {
        key.replace('centers', 'centres').replace('neighbors', 'neighbours'): value
        for key, value in state['model_state'].items()}
    assert any('centres' in key for key in state['model_state'])
    restored = FieldOperator.from_state(state, device='cpu')
    for key, value in operator.model.state_dict().items():
        assert torch.equal(value, restored.model.state_dict()[key])


def test_scf_reference_accepts_only_convergence(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from poraque_harris import scf_reference
    grid = FieldGrid((3, 3, 3), np.eye(3) * 5)
    evaluator = HarrisEvaluator(grid, np.zeros(grid.shape), 2)
    _, iterations, residual = scf_reference(evaluator, max_steps=1)
    assert iterations == 1
    assert residual < 1e-6
    external = np.random.default_rng(4).normal(size=grid.shape)
    with pytest.raises(RuntimeError, match='SCF failed'):
        scf_reference(HarrisEvaluator(grid, external, 2), max_steps=1)
