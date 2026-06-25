# Poraquê DFT Code Modification Plan

This plan documents the necessary modifications and suggestions to address the scientific conceptual errors, implementation bugs, and performance bottlenecks identified in the quality assessment audit of Poraquê.

---

## 1. Reciprocal Grid ($G$-vectors) and $|G|^2$ for Non-Orthogonal Cells

### File to Modify
* `src/poraque/core/grid.py`

### Description of Change
The current reciprocal grid implementation assumes an orthogonal cell and uses cell vector norms. For non-orthogonal lattices, the reciprocal lattice vectors $\mathbf{b}_1, \mathbf{b}_2, \mathbf{b}_3$ must be constructed from the inverse cell matrix and used to transform the integer FFT frequencies.

### Proposed Code Change
In `src/poraque/core/grid.py`, update `get_g_vectors` and `get_g2` as follows:

```python
    def get_g_vectors(self):
        """
        Build the reciprocal-space grid vectors (Gx, Gy, Gz) for any Bravais lattice.
        """
        # Create fractional reciprocal coordinates corresponding to FFT frequencies
        kx = np.fft.fftfreq(self.Nx) * self.Nx
        ky = np.fft.fftfreq(self.Ny) * self.Ny
        kz = np.fft.fftfreq(self.Nz) * self.Nz
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        
        # Project them onto the reciprocal lattice vectors (rows of self.reciprocal_cell)
        b1, b2, b3 = self.reciprocal_cell
        Gx = KX * b1[0] + KY * b2[0] + KZ * b3[0]
        Gy = KX * b1[1] + KY * b2[1] + KZ * b3[1]
        Gz = KX * b1[2] + KY * b2[2] + KZ * b3[2]
        return Gx, Gy, Gz

    def get_g2(self):
        """
        Squared magnitude of the reciprocal grid vectors |G|^2.
        """
        gx, gy, gz = self.get_g_vectors()
        return gx**2 + gy**2 + gz**2
```

---

## 2. Unify the Poisson Solver in NumpyBackend

### File to Modify
* `src/poraque/backends/numpy/operators.py`

### Description of Change
The Poisson solver currently manually rebuilds the $G$-vectors under the orthogonal-cell assumption. It should instead reuse the corrected `grid.get_g2()` method, ensuring both consistency and support for non-orthogonal cells.

### Proposed Code Change
In `src/poraque/backends/numpy/operators.py`, update the `poisson` method:

```python
    def poisson(self, charge_density, grid):
        """
        Solve Poisson equation via FFT: V(G) = 4 * pi * n(G) / G^2
        """
        n_g = np.fft.fftn(charge_density)
        
        # Get the corrected |G|^2 from the grid
        G2 = grid.get_g2().copy()
        G2[0, 0, 0] = 1.0  # Avoid division by zero at Gamma
        
        v_g = 4 * np.pi * n_g / G2
        v_g[0, 0, 0] = 0.0  # Set average potential to zero
        
        return np.real(np.fft.ifftn(v_g))
```

---

## 3. Omission of Ion-Ion (Nuclear-Nuclear) Repulsion Energy

### Files to Modify
* `src/poraque/potentials/external.py`
* `src/poraque/engine.py`

### Description of Change
Poraquê currently omits the classical electrostatic repulsion energy between ions ($E_{\text{ion-ion}}$). This makes the absolute total energy physically wrong and leads to atoms collapsing to $R \to 0$ during geometry optimizations (since numerical forces will lack the repulsive contribution). 

We will implement a general `compute_ion_ion_energy` function in `external.py` (supporting Ewald summation for periodic cells and pairwise Coulomb for non-periodic cells) and add it to the energy accounting of the DFT engines.

### Proposed Code Change

#### A. In `src/poraque/potentials/external.py`
Add the following imports and function:

```python
from scipy.special import erfc

def compute_ion_ion_energy(system, grid, alpha=None, r_cut=None, k_cut=None):
    """
    Compute the classical electrostatic ion-ion repulsion energy (Hartree).
    Uses Ewald summation for periodic grids and pairwise Coulomb for non-periodic grids.
    """
    positions = system.positions
    charges = system.atomic_numbers.astype(float)
    n_atoms = len(charges)
    if n_atoms <= 1:
        return 0.0

    # Non-periodic simple Coulomb repulsion
    if not any(grid.pbc):
        energy = 0.0
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r = np.linalg.norm(positions[i] - positions[j])
                energy += charges[i] * charges[j] / max(r, 1e-12)
        return energy

    # Periodic case: Ewald summation
    cell = grid.cell
    V_box = grid.volume
    H_inv = np.linalg.inv(cell)
    B = 2 * np.pi * H_inv.T
    perpendicular_heights = 2 * np.pi / np.linalg.norm(B, axis=1)

    if alpha is None:
        alpha = 5.0 / (V_box ** (1 / 3.0))
    if r_cut is None:
        r_cut = np.min(perpendicular_heights) / 2.0
    if k_cut is None:
        k_cut = 5.0 * alpha * 2 * np.pi

    # 1. Real space term
    e_real = 0.0
    n_max_real = np.ceil(r_cut / perpendicular_heights).astype(int)
    nx_r = np.arange(-n_max_real[0], n_max_real[0] + 1)
    ny_r = np.arange(-n_max_real[1], n_max_real[1] + 1)
    nz_r = np.arange(-n_max_real[2], n_max_real[2] + 1)
    mesh_n_r = np.array(np.meshgrid(nx_r, ny_r, nz_r, indexing='ij')).reshape(3, -1).T
    shift_vectors = np.dot(mesh_n_r, cell)

    for i in range(n_atoms):
        for j in range(n_atoms):
            for shift in shift_vectors:
                if i == j and np.allclose(shift, 0.0):
                    continue
                dr = positions[i] - (positions[j] + shift)
                r = np.linalg.norm(dr)
                if r < r_cut:
                    e_real += 0.5 * charges[i] * charges[j] * erfc(alpha * r) / max(r, 1e-12)

    # 2. Reciprocal space term
    e_recip = 0.0
    n_max_recip = np.ceil(k_cut / np.linalg.norm(B, axis=1)).astype(int)
    nx_k = np.arange(-n_max_recip[0], n_max_recip[0] + 1)
    ny_k = np.arange(-n_max_recip[1], n_max_recip[1] + 1)
    nz_k = np.arange(-n_max_recip[2], n_max_recip[2] + 1)
    mesh_n_k = np.array(np.meshgrid(nx_k, ny_k, nz_k, indexing='ij')).reshape(3, -1).T
    k_vectors = np.dot(mesh_n_k, B)

    for k in k_vectors:
        k_sq = np.dot(k, k)
        if k_sq == 0 or np.sqrt(k_sq) > k_cut:
            continue
        S_k = np.sum(charges * np.exp(-1j * np.dot(positions, k)))
        S_k_sq = np.abs(S_k)**2
        prefactor = (4 * np.pi / V_box) * np.exp(-k_sq / (4 * alpha**2)) / k_sq
        e_recip += 0.5 * prefactor * S_k_sq

    # 3. Self-interaction correction
    e_self = (alpha / np.sqrt(np.pi)) * np.sum(charges**2)

    # 4. Background term for charged cells
    net_charge = np.sum(charges)
    e_bg = - (np.pi / (2.0 * alpha**2 * V_box)) * net_charge**2

    return e_real + e_recip - e_self + e_bg
```

#### B. In `src/poraque/engine.py`
1. Import `compute_ion_ion_energy` at the top of the file:
   ```python
   from .potentials.external import compute_ion_ion_energy
   ```
2. Update `KSDFTEngine._total_energy` to include the Ion-Ion term:
   ```python
       def _total_energy(self, density, e_band, v_eff):
           e_ext = self.backend.integrate(self.v_ext * density.data, self.grid)
           v_eff_n = self.backend.integrate(v_eff * density.data, self.grid)
           e_kin = e_band - v_eff_n
   
           e_h = 0.0
           if self.hartree is not None:
               e_h = self.hartree.energy(density, self.system, self.grid, self.backend)
   
           e_xc = self.xc.energy(density, self.system, self.grid, self.backend)
           e_nonlocal = 0.0
           
           # ADD THE ION-ION COULOMB ENERGY
           e_ion = compute_ion_ion_energy(self.system, self.grid)
           e_total = e_kin + e_ext + e_h + e_xc + e_nonlocal + e_ion
   
           components = {
               "Kinetic": e_kin,
               "External": e_ext,
               "Hartree": e_h,
               "XC": e_xc,
               "Nonlocal": e_nonlocal,
               "Ion-Ion": e_ion,
           }
           return e_total, components
   ```
3. Update `OFDFTEngine.compute_total_energy` to include the Ion-Ion term:
   ```python
       def compute_total_energy(self, density):
           """Compute the total energy and its per-functional components."""
           total_e = 0.0
           components = {}
           for func in self.functionals:
               e = func.energy(density, self.system, self.grid, self.backend)
               components[func.name] = e
               total_e += e
           
           # ADD THE ION-ION COULOMB ENERGY
           e_ion = compute_ion_ion_energy(self.system, self.grid)
           components["Ion-Ion"] = e_ion
           total_e += e_ion
           return total_e, components
   ```

---

## 4. Respect Grid PBC in Minimum-Image Convention

### Files to Modify
* `src/poraque/potentials/external.py`
* `src/poraque/pseudopotentials/__init__.py`

### Description of Change
Potential builders currently force the minimum-image convention (`mic=True`) by default, wrapping coordinates even when the system is non-periodic (`pbc=False`). They should determine periodicity dynamically from the grid's PBC settings.

### Proposed Code Change

#### A. In `src/poraque/potentials/external.py`
Modify `build_external_potential` to default `mic` to the periodicity of the grid:
```python
def build_external_potential(grid, system, kind="soft", **kwargs):
    ...
    positions = system.positions
    charges = system.atomic_numbers.astype(float)
    
    # Default minimum-image convention to True only if any grid direction is periodic
    kwargs.setdefault("mic", any(grid.pbc))

    if kind == "soft":
        return soft_coulomb_potential(grid, positions, charges, **kwargs)
    ...
```

#### B. In `src/poraque/pseudopotentials/__init__.py`
Modify `build_pseudopotential_potential` to default `mic` based on the grid's PBC settings if not explicitly specified:
```python
def build_pseudopotential_potential(grid, system, pseudopotentials, mic=None,
                                    functional="LDA"):
    ...
    if mic is None:
        mic = any(grid.pbc)
        
    table = resolve_pseudopotentials(system, pseudopotentials, functional=functional)
    v_ext = np.zeros(grid.shape)
    ...
```

---

## 5. PBE Exchange-Correlation Fail-Safe

### File to Modify
* `src/poraque/ase/calculator.py`

### Description of Change
The Poraquê registry loads PBE pseudopotentials, but the actual PBE exchange-correlation functional is not implemented. When `xc="pbe"` is requested, a string `"pbe"` is currently passed down, causing a runtime crash. We should intercept unsupported XC functional strings early and raise a helpful `NotImplementedError`.

### Proposed Code Change
In `src/poraque/ase/calculator.py`, update `_xc_functional`:

```python
    def _xc_functional(self):
        """Resolve the XC functional argument for KS-DFT."""
        if self.xc is None:
            return None
        if isinstance(self.xc, str):
            if self.xc.lower() == "lda":
                return LDA()
            else:
                raise NotImplementedError(
                    f"Exchange-correlation functional {self.xc!r} is not yet implemented. "
                    f"Only 'lda' is currently supported."
                )
        return self.xc
```

---

## 6. Spectral Laplacian in von Weizsäcker KEDF

### File to Modify
* `src/poraque/functionals/kinetic.py`

### Description of Change
The von Weizsäcker kinetic energy functional uses a 2nd-order finite-difference Laplacian, introducing large grid errors in plane-wave grids designed for spectral accuracy. It should use the spectral FFT-based Laplacian `backend.laplacian_fft` already implemented in the backend.

### Proposed Code Change
In `src/poraque/functionals/kinetic.py`, update `VonWeizsaecker`'s `energy` and `potential` methods:

```python
class VonWeizsaecker(Functional):
    ...
    def energy(self, density, system, grid, backend):
        sqrt_n = np.sqrt(np.maximum(density.data, 0.0))
        # Use spectral Laplacian for plane-wave accuracy
        lap_sqrt_n = backend.laplacian_fft(sqrt_n, grid)
        return -0.5 * self.lambda_ * backend.integrate(sqrt_n * lap_sqrt_n, grid)

    def potential(self, density, system, grid, backend):
        sqrt_n = np.sqrt(np.maximum(density.data, 0.0))
        safe_sqrt_n = np.where(sqrt_n > 1e-12, sqrt_n, 1e-12)
        # Use spectral Laplacian for plane-wave accuracy
        lap_sqrt_n = backend.laplacian_fft(sqrt_n, grid)
        return -0.5 * self.lambda_ * lap_sqrt_n / safe_sqrt_n
```

---

## Verification Plan

After applying these changes, run the test suite to verify that no regressions have been introduced:

```bash
pytest
```

To verify the non-orthogonal grid and ion-ion energy fixes specifically:
1. Write a test case for FCC silicon using the primitive non-orthogonal cell. Check that the KS-DFT SCF converges and yields the correct valence electron integration.
2. Write a test case for H2 molecule geometry optimization. Verify that the H2 molecule relaxes to a physical bond length (around 0.74 Å) instead of collapsing to 0.0 Å.
