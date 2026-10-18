"""Analytic projector/Bloch checks and end-to-end ext2chg input-deck tests."""
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

from poraque.fields import FieldGrid
from poraque.fields.structure import Structure
from poraque.fields.vasp import Incar
from poraque.fields.vasp.kpoints import Kpoints
from poraque.physics.bloch import (BlochHamiltonian, BlochHarrisEvaluator,
                                    ElectronicSettings, occupations)
from poraque.physics.harris import integrate, lda_terms
from poraque.physics.pseudopotentials import NormConservingUPF, NonlocalPotential
from poraque.fields.constants import BOHR_TO_ANGSTROM as A0, HARTREE_TO_EV, HBAR2_OVER_2M_EV_ANGSTROM2


def dataset(coupling=1.0, angular=0):
    r = np.linspace(0, 10, 1001)
    local = -2 * np.divide(erf(r), r, out=np.full_like(r, 2/np.sqrt(np.pi)), where=r > 0)
    # Unit radial norm, so beta(r)Y_00 is a normalized three-dimensional Gaussian.
    beta = 2 / np.pi**0.25 * r * np.exp(-r*r/2)
    return NormConservingUPF('H', 1, r, local, (beta,), (angular,), np.array([[coupling]]))


def system(coupling=1.0):
    structure = Structure(np.eye(3)*5, ['H'], [1], [[0.2, 0.3, 0.4]])
    return structure, NonlocalPotential(structure, {'H': dataset(coupling)})


def test_incar_typed_comments_lists_and_validation():
    incar = Incar.from_string('''SYSTEM="a;!#b"; ENCUT=1D1 ! cutoff
ISMEAR=-1; SIGMA=5d-2; EDIFF=1D-8; NELM=20
MAGMOM=2*0 1; LSORBIT=.FALSE.
''')
    assert incar['SYSTEM'] == '"a;!#b"'
    assert incar.numbers('MAGMOM') == [0, 0, 1]
    assert incar.get_bool('LSORBIT') is False
    settings = ElectronicSettings.from_incar(incar)
    assert settings.encut == 10
    assert settings.ediff == 1e-8
    assert settings.sigma == 0.05
    assert settings.nelm == 20
    for tag in ['ISMEAR=-5', 'ISPIN=3', 'ENCUT=-1', 'NBANDS=2.5', 'GGA=PS', 'SIGMA=bad']:
        with pytest.raises(ValueError):
            ElectronicSettings.from_incar(Incar.from_string('ENCUT=10; ISMEAR=0; ' + tag))


def test_meshes_shifts_and_weights():
    mesh = Kpoints.from_string('mesh\n0\nMonkhorst-Pack\n2 1 1\n0 0 0\n', np.eye(3))
    assert np.allclose(mesh.points, [[-0.25, 0, 0], [0.25, 0, 0]])
    assert np.allclose(mesh.weights, [0.5, 0.5])
    gamma = Kpoints.from_string('mesh\n0\nGamma\n2 1 1\n0.5 0 0', np.eye(3))
    assert np.allclose(gamma.points[:, 0], [0.25, 0.75])
    explicit = Kpoints.from_string('points\n2\nReciprocal\n0 0 0 2\n.5 0 0 1', np.eye(3))
    assert np.allclose(explicit.weights, [2/3, 1/3])


def test_path_cartesian_distance_and_discontinuities():
    text = '''path
3
Line-mode
Cartesian
0 0 0 ! Gamma
.5 0 0 ! X

0 .5 0 ! Y
0 .5 .5 ! Z
'''
    with pytest.raises(ValueError, match='scale'):
        Kpoints.from_string(text, np.eye(3)*5)
    path = Kpoints.from_string(text, np.eye(3)*5, scale=5)
    assert path.weights is None
    assert path.segment_starts == (0, 3)
    assert path.labels[1] == (2, 'X')
    assert np.allclose(path.distances(np.eye(3)*5), np.array([0, .25, .5, .5, .75, 1])*2*np.pi/5)


@pytest.mark.parametrize('text', [
    'bad\n0\nMonkhorst\n0 1 1',
    'bad\n3\nLine\nReciprocal\n0 0 0\n',
    'bad\n2\nReciprocal\n0 0 0 -1\n0 0 0 1',
    'bad\n0\nUnknown\n1 1 1',
])
def test_bad_kpoints(text):
    with pytest.raises(ValueError):
        Kpoints.from_string(text, np.eye(3))


def test_gaussian_projector_transform_normalization_phase():
    data = dataset()
    q = np.array([[0, 0, 0], [1, 0, 0], [0.4, -0.2, 0.7]])
    position, volume = np.array([0.2, 0.3, 0.4]), 125.0
    b, d = data.columns(q, position, volume)
    # Fourier transform of pi^-3/4 exp(-r²/2), using native Bohr coordinates.
    exact = ((2*np.pi)**1.5 / np.pi**0.75 * A0**1.5 / np.sqrt(volume) *
             np.exp(-np.sum(q*q, axis=1)*A0*A0/2) * np.exp(-1j*q@position))
    assert np.allclose(b[:, 0], exact, rtol=1e-10, atol=1e-12)
    assert d[0, 0] == HARTREE_TO_EV/2
    assert np.allclose(b @ d @ b.conj().T, HARTREE_TO_EV/2 * np.outer(exact, exact.conj()))


def test_p_projectors_include_all_m_and_origin_limit():
    data = dataset(angular=1)
    b, d = data.columns(np.array([[0, 0, 0], [1, 0, 0]]), np.zeros(3), 125)
    assert b.shape == (2, 3)
    assert d.shape == (3, 3)
    assert np.allclose(b[0], 0)
    assert np.linalg.norm(b[1]) > 0


def test_nonlocal_changes_bands_and_is_hermitian():
    structure, nonlocal_potential = system(2.0)
    grid = FieldGrid((9, 9, 9), structure.cell)
    h = BlochHamiltonian(grid, np.zeros(grid.shape), 10, nonlocal_potential)
    k = np.array([0.13, 0.1, -0.1])
    matrix, indices = h.matrix(k)
    q = (indices + k) @ grid.reciprocal_cell
    kinetic = HBAR2_OVER_2M_EV_ANGSTROM2*np.sum(q*q, axis=1)
    b, d = dataset(2.0).columns(q, structure.positions[0], grid.volume)
    assert np.allclose(matrix - np.diag(kinetic), b @ d @ b.conj().T)
    assert np.allclose(matrix, matrix.conj().T)
    assert not np.allclose(np.linalg.eigvalsh(matrix), np.sort(kinetic))
    state = h.solve(k, 3)
    assert state.residual < 1e-10
    assert np.allclose(state.eigenvalues, np.linalg.eigvalsh(matrix)[:3])
    # Time reversal and integer reciprocal shifts, with k-dependent cutoff sphere.
    assert np.allclose(h.solve(-k, 3).eigenvalues, state.eigenvalues)
    assert np.allclose(h.solve(k + [1, -2, 0], 3).eigenvalues, state.eigenvalues)


def test_alias_and_memory_limits():
    structure, nl = system()
    grid = FieldGrid((3, 3, 3), structure.cell)
    with pytest.raises(ValueError, match='aliases'):
        BlochHamiltonian(grid, np.zeros(grid.shape), 10, nl).matrix(np.zeros(3))
    grid = FieldGrid((9, 9, 9), structure.cell)
    with pytest.raises(ValueError, match='basis size'):
        BlochHamiltonian(grid, np.zeros(grid.shape), 10, nl, max_basis=2).matrix(np.zeros(3))


@pytest.mark.parametrize('ismear,sigma', [(0, 0), (-1, 0.05), (0, 0.05)])
def test_weighted_occupations(ismear, sigma):
    values = np.array([[0.0, 1, 3], [0.5, 1.5, 4]])
    weights = np.array([0.25, 0.75])
    f, mu, ts = occupations(values, weights, 2, ismear, sigma)
    assert np.sum(weights[:, None]*f) == pytest.approx(2, abs=1e-10)
    assert np.all((f >= 0) & (f <= 2))
    assert ts >= 0
    assert np.isfinite(mu)
    if sigma == 0:
        assert np.allclose(f[:, 0], 2)
        assert np.allclose(f[:, 1:], 0)
    else:
        with pytest.raises(ValueError, match='Highest band'):
            occupations(values[:, :1], weights, 1, ismear, sigma)


def test_degenerate_weighted_filling():
    f, _, _ = occupations(np.array([[0., 1], [0., 1]]), np.array([.2, .8]), 1, 0, 0)
    assert np.allclose(f[:, 0], 1)


def test_harris_k_sum_density_and_path():
    structure, nl = system()
    grid = FieldGrid((9, 9, 9), structure.cell)
    density = np.full(grid.shape, 2/grid.volume)
    evaluator = BlochHarrisEvaluator(grid, np.zeros(grid.shape), 2, 1.2,
                                     ElectronicSettings(10, nbands=3), nl)
    mesh = Kpoints.from_string('mesh\n0\nMonkhorst\n2 1 1', grid.cell)
    result = evaluator.evaluate(density, mesh)
    exc, vxc = lda_terms(density, grid)
    band = np.sum(mesh.weights[:, None] * result.eigenvalues * result.occupations)
    assert result.total == pytest.approx(band - integrate(density*vxc, grid) + exc + 1.2)
    assert integrate(result.output_density, grid) == pytest.approx(2)
    assert result.total == result.free_energy
    assert np.array_equal(density, np.full(grid.shape, 2/grid.volume))
    path = Kpoints.from_string('path\n3\nLine\nReciprocal\n0 0 0 ! G\n.5 0 0 ! X', grid.cell)
    bands = evaluator.bands(density, path, nbands=3)
    assert np.array(bands['eigenvalues_ev']).shape == (3, 3)
    assert bands['labels'][-1] == (2, 'X')
    with pytest.raises(ValueError, match='Line-mode'):
        evaluator.evaluate(density, path)


def write_upf(path: Path, pseudo_type='NC'):
    data = dataset()
    def numbers(array):
        return ' '.join(str(x) for x in array)
    path.write_text(f'''<UPF version="2.0.1">
<PP_HEADER element="H" z_valence="1" pseudo_type="{pseudo_type}" number_of_proj="1" functional="SLA PW" />
<PP_MESH><PP_R>{numbers(data.r)}</PP_R></PP_MESH>
<PP_LOCAL>{numbers(data.local)}</PP_LOCAL>
<PP_NONLOCAL><PP_BETA.1 angular_momentum="0">{numbers(data.beta[0])}</PP_BETA.1>
<PP_DIJ>1.0</PP_DIJ></PP_NONLOCAL></UPF>''')


def test_upf_read_and_reject_paw(tmp_path):
    path = tmp_path/'H.upf'
    write_upf(path)
    parsed = NormConservingUPF.from_file(path)
    assert parsed.element == 'H'
    assert np.array_equal(parsed.beta[0], dataset().beta[0])
    write_upf(path, 'PAW')
    with pytest.raises(ValueError, match='norm-conserving'):
        NormConservingUPF.from_file(path)


def test_gaussian_local_fourier_coulomb_limit():
    data = dataset()
    g = np.array([[0, 0, 0], [1, .2, 0]])
    actual = data.local_fourier(g)
    q2 = np.sum((g*A0)**2, axis=1)
    # Vloc=-2 erf(r)/r Ry; Fourier=-8pi exp(-q²/4)/q², G=0 short range=2pi.
    expected = np.array([2*np.pi, -8*np.pi*np.exp(-q2[1]/4)/q2[1]]) * HARTREE_TO_EV/2 * A0**3
    assert np.allclose(actual, expected, rtol=1e-9)


def test_vasp_rejects_upf_paths(tmp_path):
    from poraque.ml.bloch import VaspBlochHarris
    from poraque.fields.vasp import Poscar
    structure,_=system()
    Poscar.from_structure(structure).write(tmp_path/'POSCAR')
    (tmp_path/'INCAR').write_text('ENCUT=10; ISMEAR=0; GGA=PE')
    upf=tmp_path/'H.upf'
    write_upf(upf)
    with pytest.raises(ValueError):
        VaspBlochHarris(tmp_path,'unused',{'H':str(upf)})


def test_coupled_projector_expectation_and_translation():
    base = dataset()
    data = NormConservingUPF('H', 1, base.r, base.local,
                            (base.beta[0], base.beta[0]*base.r), (0, 0),
                            np.array([[1., .3], [.3, -2.]]))
    q = np.array([[0., 0, 0], [1., 0, 0], [0, -1., .2]])
    b, d = data.columns(q, np.array([.2, .3, .5]), 125)
    c = np.array([1., .4j, .2])
    c /= np.linalg.norm(c)
    overlaps = b.conj().T @ c
    assert np.vdot(c, b @ d @ b.conj().T @ c) == pytest.approx(np.vdot(overlaps, d @ overlaps))
    shift = np.array([.12, .37, -.1])
    moved, _ = data.columns(q, np.array([.2, .3, .5])+shift, 125)
    assert np.allclose(moved, np.exp(-1j*q@shift)[:, None]*b)


def test_finite_width_harris_free_energy():
    structure, nl = system()
    grid = FieldGrid((9, 9, 9), structure.cell)
    density = np.full(grid.shape, 2/grid.volume)
    mesh = Kpoints.from_string('mesh\n0\nMonkhorst\n2 1 1', grid.cell)
    result = BlochHarrisEvaluator(grid, np.zeros(grid.shape), 2, 0,
        ElectronicSettings(10, nbands=5, ismear=-1, sigma=.05), nl).evaluate(density, mesh)
    assert result.free_energy == pytest.approx(result.total-result.entropy_term)
    assert result.entropy_term > 0
    assert integrate(result.output_density, grid) == pytest.approx(2)
