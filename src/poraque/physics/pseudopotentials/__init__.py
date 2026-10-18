"""Scalar norm-conserving UPF v2 pseudopotentials and separable projectors.

PAW/ultrasoft/NLCC datasets are rejected. Fully relativistic norm-conserving
datasets use j-resolved spin-angular projectors, including explicit L.S terms.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.integrate import simpson
from scipy.special import spherical_jn
try:
    from scipy.special import sph_harm_y
except ImportError:  # SciPy < 1.15 uses reversed angle/quantum-number ordering.
    from scipy.special import sph_harm

    def sph_harm_y(ell: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Compatibility with the spherical-harmonic convention of recent SciPy."""
        return sph_harm(m, ell, phi, theta)

from ...fields import ExternalPotential, FieldGrid
from ...fields.structure import Structure
from ...fields.constants import (BOHR_TO_ANGSTROM as A0, HARTREE_TO_EV,
                                  COULOMB_CONSTANT_EV_ANGSTROM as E2)


@dataclass(frozen=True)
class NormConservingUPF:
    """Radial data in native UPF units: Bohr, Ry, r*beta(r).

    The radial projectors need not be normalized individually. Their original
    normalization is retained jointly with D; no arbitrary rescaling is applied.
    """

    element: str
    valence: float
    r: np.ndarray
    local: np.ndarray
    beta: tuple[np.ndarray, ...]
    angular: tuple[int, ...]
    coupling: np.ndarray
    functional: str = ''
    total_j: tuple[float, ...] | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> 'NormConservingUPF':
        """Read XML UPF v2 with dense PP_DIJ; refuse unsupported physics."""
        root = ET.fromstring(Path(path).read_text())
        header = root.find('PP_HEADER')
        if header is None or not root.attrib.get('version', '').startswith('2.'):
            raise ValueError('Only XML UPF v2 is supported')
        attrs = header.attrib
        def truth(key):
            return attrs.get(key, 'F').strip('.').upper() in ('T', 'TRUE')

        if (attrs.get('pseudo_type', '').upper() not in ('NC', 'SL') or
                any(truth(key) for key in ('is_paw', 'is_ultrasoft', 'core_correction'))):
            raise ValueError('Require norm-conserving UPF without NLCC; PAW is unsupported')

        def array(node: ET.Element | None) -> np.ndarray:
            if node is None or not node.text:
                raise ValueError('Missing UPF radial data')
            tokens = node.text.replace('D', 'E').replace('d', 'e').split()
            return np.array([float(token) for token in tokens])

        r, local = array(root.find('PP_MESH/PP_R')), array(root.find('PP_LOCAL'))
        nonlocal_node = root.find('PP_NONLOCAL')
        if nonlocal_node is None:
            raise ValueError('Missing PP_NONLOCAL')
        nodes = sorted((node for node in nonlocal_node if node.tag.startswith('PP_BETA.')),
                       key=lambda node: int(node.tag.split('.')[-1]))
        beta, angular = [], []
        for node in nodes:
            values = array(node)
            if len(values) > len(r):
                raise ValueError('UPF projector longer than radial mesh')
            beta.append(np.pad(values, (0, len(r) - len(values))))
            angular.append(int(node.attrib['angular_momentum']))
        count = int(attrs.get('number_of_proj', len(nodes)))
        if count != len(nodes) or count == 0:
            raise ValueError('Require a nonempty, complete nonlocal projector set')
        coupling = array(nonlocal_node.find('PP_DIJ')).reshape(count, count)
        total_j = None
        if truth('has_so') or attrs.get('relativistic', '').lower() == 'full':
            spin = root.find('PP_SPIN_ORB')
            if spin is None:
                raise ValueError('Fully relativistic UPF requires PP_SPIN_ORB j-resolved projectors')
            rel = sorted((node for node in spin if node.tag.startswith('PP_RELBETA.')),
                         key=lambda node: int(node.tag.split('.')[-1]))
            if len(rel) != count:
                raise ValueError('Incomplete PP_RELBETA metadata')
            total_j = tuple(float(node.attrib['jjj']) for node in rel)
            if any(int(node.attrib['lll']) != ell for node, ell in zip(rel, angular)):
                raise ValueError('PP_RELBETA angular momentum does not match PP_BETA')
        result = cls(attrs['element'].strip(), float(attrs['z_valence']), r, local,
                     tuple(beta), tuple(angular), coupling, attrs.get('functional', ''), total_j)
        result.validate()
        return result

    def validate(self) -> None:
        """Check finite radial tables, Hermitian D and spherical channel selection."""
        if (self.r.ndim != 1 or len(self.r) < 3 or np.any(self.r < 0) or
                np.any(np.diff(self.r) <= 0) or self.local.shape != self.r.shape or
                len(self.beta) != len(self.angular) or not self.beta or
                self.coupling.shape != (len(self.beta), len(self.beta)) or
                not np.isfinite(self.valence) or self.valence <= 0):
            raise ValueError('Invalid norm-conserving radial dataset')
        for values in (self.r, self.local, self.coupling, *self.beta):
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite pseudopotential data')
        if any(b.shape != self.r.shape for b in self.beta) or any(ell < 0 for ell in self.angular):
            raise ValueError('Invalid radial projector shape/angular momentum')
        if not np.allclose(self.coupling, self.coupling.T.conj(), atol=1e-12):
            raise ValueError('Non-Hermitian projector coupling')
        if self.total_j is not None:
            from ..spin_orbit import j_projector
            if len(self.total_j) != len(self.angular):
                raise ValueError('Incomplete j-resolved projector metadata')
            for ell, j in zip(self.angular, self.total_j):
                j_projector(ell, j)
        for i, li in enumerate(self.angular):
            for j, lj in enumerate(self.angular):
                if li != lj and abs(self.coupling[i, j]) > 1e-12:
                    raise ValueError('Scalar spherical D cannot couple different angular momenta')
                if (self.total_j is not None and self.total_j[i] != self.total_j[j]
                        and abs(self.coupling[i, j]) > 1e-12):
                    raise ValueError('Spherical relativistic D cannot couple different j manifolds')

    def columns(self, q: np.ndarray, position: np.ndarray, volume: float
                ) -> tuple[np.ndarray, np.ndarray]:
        r"""Return B_G,im=<G+k|beta_im> and expanded D (eV).

        B = 4*pi*(-i)^l/sqrt(Omega) exp[-i(G+k).R] Y_lm(qhat)
            integral r^2 dr j_l(qr) beta_i(r).
        UPF stores u_i(r)=r*beta_i(r); integrate r*u_i, with Bohr-to-Angstrom
        factors retained. Every m=-l..l is included. Thus V_nl = B D B†.
        """
        self.validate()
        q = np.asarray(q, dtype=float)
        magnitude = np.linalg.norm(q, axis=1)
        theta = np.arccos(np.divide(q[:, 2], magnitude, out=np.ones_like(magnitude),
                                    where=magnitude > 0).clip(-1, 1))
        phi = np.arctan2(q[:, 1], q[:, 0])
        columns, indices = [], []
        for i, (u, ell) in enumerate(zip(self.beta, self.angular)):
            radial = np.empty(len(q))
            for start in range(0, len(q), 256):
                stop = min(start + 256, len(q))
                radial[start:stop] = simpson(self.r[None, :] * u[None, :] *
                    spherical_jn(ell, magnitude[start:stop, None] * A0 * self.r), x=self.r, axis=1)
            radial *= A0**1.5
            for m in range(-ell, ell + 1):
                columns.append(4 * np.pi * (-1j)**ell * radial * sph_harm_y(ell, m, theta, phi)
                               * np.exp(-1j * (q @ position)) / np.sqrt(volume))
                indices.append((i, m))
        coupling = np.array([[self.coupling[i, j] if m == n else 0
                              for j, n in indices] for i, m in indices]) * HARTREE_TO_EV / 2
        return np.column_stack(columns), coupling

    def local_fourier(self, g: np.ndarray) -> np.ndarray:
        """Transform Vloc + Z/r, then restore Coulomb for G!=0 (eV A^3).

        The G=0 coefficient retains the finite short-range mean. The omitted
        Coulomb zero mode matches the ion Ewald neutralizing background.
        Integrate the regular combination r^2*Vloc+2*Z*r to handle r=0.
        """
        magnitude = np.linalg.norm(g, axis=1)
        integrand = self.r**2 * self.local + 2 * self.valence * self.r
        answer = np.empty(len(g))
        for start in range(0, len(g), 256):
            stop = min(start + 256, len(g))
            bessel = np.sinc(magnitude[start:stop, None] * A0 * self.r / np.pi)
            answer[start:stop] = (4 * np.pi * simpson(bessel * integrand, x=self.r, axis=1)
                                   * HARTREE_TO_EV / 2 * A0**3)
        active = magnitude > 0
        answer[active] -= 4 * np.pi * self.valence * E2 / magnitude[active]**2
        return answer


class NonlocalPotential:
    """Sum of atom-centered separable NC projectors, without inter-atom D terms."""

    def __init__(self, structure: Structure, datasets: dict[str, NormConservingUPF]) -> None:
        self.structure, self.datasets = structure, datasets
        self.atoms = [(datasets[element], position) for element, position in zip(
            np.repeat(structure.elements, structure.counts), structure.positions)]
        for element in structure.elements:
            datasets[element].validate()
            if datasets[element].element != element:
                raise ValueError(f'UPF element mismatch for {element}')

    def matrix(self, q: np.ndarray) -> np.ndarray:
        """Assemble V_nl(G,G',k)=sum_a B_a D_a B_a† in eV."""
        result = np.zeros((len(q), len(q)), dtype=complex)
        for dataset, position in self.atoms:
            if dataset.total_j is not None:
                raise ValueError('Fully relativistic UPF requires LSORBIT spinor Hamiltonian')
            b, d = dataset.columns(q, position, self.structure.volume)
            result += b @ d @ b.conj().T
        return result

    def spinor_matrix(self, q: np.ndarray, soc: bool = True) -> np.ndarray:
        """Spin-major 2N x 2N nonlocal matrix from exact j-resolved NC projectors.

        Each radial pair contributes D_ij diag(B_i,B_i) P_lj diag(B_j,B_j)†.
        P_lj is an explicit polynomial in L.S, so its off-diagonal blocks
        perform spin flips. Scalar datasets supply V_nl tensor I_spin only.
        """
        from scipy.linalg import block_diag
        from ..spin_orbit import j_projector
        n = len(q)
        result = np.zeros((2*n, 2*n), dtype=complex)
        for dataset, position in self.atoms:
            b, d = dataset.columns(q, position, self.structure.volume)
            if dataset.total_j is None:
                scalar = b @ d @ b.conj().T
                result += block_diag(scalar, scalar)
                continue
            if not soc:
                raise ValueError('Fully relativistic UPF requires LSORBIT=True')
            offsets = np.r_[0, np.cumsum([2*ell+1 for ell in dataset.angular])]
            blocks = [block_diag(b[:, offsets[i]:offsets[i+1]],
                                 b[:, offsets[i]:offsets[i+1]])
                      for i in range(len(dataset.angular))]
            for i, (ell, j) in enumerate(zip(dataset.angular, dataset.total_j)):
                projector = j_projector(ell, j)
                for k, coefficient in enumerate(dataset.coupling[i]):
                    if abs(coefficient) > 0:
                        result += (coefficient * HARTREE_TO_EV/2 *
                                   blocks[i] @ projector @ blocks[k].conj().T)
        return result

    def external(self, grid: FieldGrid) -> ExternalPotential:
        """Construct the matching NC local potential; never mix it with POTCAR."""
        if not np.allclose(grid.cell, self.structure.cell):
            raise ValueError('Pseudopotential/grid cell mismatch')
        g = np.stack(grid.get_g_vectors(), axis=-1).reshape(-1, 3)
        coefficients = np.zeros(len(g), dtype=complex)
        transforms = {element: dataset.local_fourier(g) for element, dataset in self.datasets.items()}
        for dataset, position in self.atoms:
            coefficients += transforms[dataset.element] * np.exp(-1j * (g @ position)) / grid.volume
        # Real samples imply conjugate Nyquist averaging on even meshes.
        data = np.fft.ifftn(coefficients.reshape(grid.shape)).real * np.prod(grid.shape)
        return ExternalPotential(data, grid, self.structure, metadata={'model': 'norm-conserving-UPF'})
