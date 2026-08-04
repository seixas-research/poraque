# Analysis of the modified VASP source: `EXTCAR` and `TAUCAR`

**Source analysed:** `/Users/leseixas/Codes/vasp/6.2.0/build/std_taucar/`
**Files:** `main.F`, `pot.F`, `pseudo.F`, `metagga.F`, `pseudo_struct.F`, `base.F`
**Status:** discrepancy diagnosed **and fixed** — the Python implementation now reproduces VASP's `EXTCAR` to a relative $L^2$ of $2\times10^{-5}$.

---

## 1. Executive summary

| | Before | After |
|---|---|---|
| `EXTCAR` relative $L^2$ vs VASP | 0.134 | **0.000021** |
| MAE | 3.38 eV | **0.00034 eV** |
| Pearson $r$ | 0.9919 | **1.000000** |

**Root cause.** VASP does *not* use an analytic form factor. It reads a
**tabulated short-ranged local pseudopotential** $v_{\rm short}(q)$ from the
`POTCAR`, spline-interpolates it onto the FFT grid, and adds the
$-4\pi Z_{\rm val}e^2/q^2$ Coulomb tail back analytically. Our Gaussian
pseudo-ion was a model for the sum of *both* terms, and no monotonic form
factor can reproduce the oscillatory short-range part.

**Secondary finding.** The `POTCAR` `local part` block was being misparsed. The
first number after the marker is **`PSGMAX`** (the table's maximum wavevector,
75.589 Å⁻¹ for the Au PAW\_PBE potential), **not `ZVAL`**. This is the q-mesh
convention that was previously flagged as unrecoverable from the `POTCAR`
alone; the source settles it.

Everything else already agreed: units (eV), the absence of volume scaling, the
$\mathbf G=0$ convention, and the single-block layout were all correct.

---

## 2. `EXTCAR` — where it is written

`main.F:4761–4808`, guarded by the new `INCAR` tag `EXTCAR` (declared as
`LEXTCAR` in `base.F:143`, read at `main.F:669`):

```fortran
IF (IO%LEXTCAR) THEN
   ALLOCATE(CEXT(GRIDC%MPLWV),CEXTD(GRIDC%MPLWV))
   CALL POTION(GRIDC,P,LATT_CUR,T_INFO,CEXT,CEXTD,CSTRF,PSCDUM)
   CALL SETUNB_COMPAT(CEXT,GRIDC)
   CALL FFT3D(CEXT,GRIDC,1)                    ! to real space
   CALL OUTPOS(55,...)                         ! POSCAR header
   CALL OUTPOT(GRIDC,55,.TRUE.,CEXT)           ! LOCPOT-style, no volume factor
ENDIF
```

The author's own comment states the intent precisely:

> `V_ext(G) = 1/Omega sum_types S_type(G) v_loc^type(G)` … the tabulated short
> ranged part of the POTCAR local potential **with the `-Z e^2/r` Coulomb tail
> added back on**. It contains no PAW one-centre terms and no non-local
> projector contribution … as in LOCPOT the potential is written in eV (no
> volume scaling, unlike the densities in CHGCAR/TAUCAR) and … the divergent
> G=0 component is set to zero.

Note `OUTPOT` (not `OUTCHG`): that is what makes `EXTCAR` a *potential* file in
the `LOCPOT` sense — **no multiplication by $\Omega$**.

## 3. `EXTCAR` — the formula

`pot.F:1005–1095`, subroutine `POTION`:

```fortran
ARGSC  = NPSPTS/P(NT)%PSGMAX
PSGMA2 = P(NT)%PSGMAX - P(NT)%PSGMAX/NPSPTS
ZZ     = -4*PI*P(NT)%ZVALF*FELECT

G = SQRT(GX**2+GY**2+GZ**2)*2*PI
IF ( (G /= 0) .AND. (G < PSGMA2) ) THEN
   I    = INT(G*ARGSC)+1                       ! uniform-mesh bin
   REM  = G - P(NT)%PSP(I,1)
   VPST = PSP(I,2)+REM*(PSP(I,3)+REM*(PSP(I,4)+REM*PSP(I,5)))   ! cubic spline
   CVPS(N) = CVPS(N) + ( VPST + ZZ/G**2 ) / LATT_CUR%OMEGA * CSTRF(N,NT)
ELSE
   CVPS(N) = 0._q
ENDIF
```

which is

$$
\boxed{\;V_{\rm ext}(\mathbf G)=\frac{1}{\Omega}\sum_{s}S_s(\mathbf G)\left[v^{s}_{\rm short}(G)\;-\;\frac{4\pi Z^{\rm val}_{s}e^{2}}{G^{2}}\right],\qquad V(\mathbf 0)=0\;}
$$

Four details that matter, each of which we now reproduce:

1. **The Coulomb term is exactly our old `model="coulomb"`.** `ZZ/G²/Ω` with
   `FELECT` $= e^2/4\pi\epsilon_0 = 14.3996$ eV·Å is identical to our
   $-4\pi K Z/(\Omega G^2)$. Our long-range physics was never wrong — which is
   why the Pearson correlation was already 0.992.
2. **The short-range term is tabulated, not analytic.** `VPST` is a cubic
   spline through the `POTCAR` values. This is the entire discrepancy.
3. **Hard truncation.** For $G \ge$ `PSGMA2` the *whole* contribution is set to
   zero, Coulomb tail included — VASP does not extrapolate the table. Here
   `PSGMAX` = 75.6 Å⁻¹ while the 128³ grid reaches $|G|_{\max}$ = 106 Å⁻¹, so
   the truncation is genuinely active in this dataset and must be reproduced.
4. **`G` carries the $2\pi$.** `LATT_CUR%B` is the reciprocal lattice *without*
   $2\pi$, so `G = |n·B|·2π` is the ordinary angular wavevector in Å⁻¹ — the
   same convention as `FieldGrid.get_g_vectors`.

## 4. `EXTCAR` — the `POTCAR` table layout

`pseudo.F:294–305` is the decisive fragment:

```fortran
READ(10,'(1X,A1)') CSEL          ! the "local part" marker line
READ(10,*) P(NTYP)%PSGMAX        ! <-- FIRST number is PSGMAX, not ZVAL
ALLOCATE(P(NTYP)%PSP(NPSPTS,5))
READ(10,*) (P(NTYP)%PSP(I,2),I=1,NPSPTS)
DO I=1,NPSPTS
    P(NTYP)%PSP(I,1)=(P(NTYP)%PSGMAX/NPSPTS)*(I-1)
ENDDO
P(NTYP)%PSCORE=P(NTYP)%PSP(1,2)  ! q->0 limit, used for the PSCENC correction
CALL SPLCOF(P(NTYP)%PSP(1,1),NPSPTS,NPSPTS,0._q)   ! cubic spline coefficients
```

with `NPSPTS = 1000` (`pseudo_struct.F:10`). So:

$$q_i = \frac{\mathrm{PSGMAX}}{1000}(i-1),\qquad i=1\ldots1000 .$$

Verified directly against `data/vasp/struct_000/POTCAR`: the value after
`local part` is **75.5890395431569**, and the block contains **1001** numbers
= `PSGMAX` + exactly 1000 samples. `ZVAL` for that potential is 11, so the two
are unmistakably different quantities. Units are eV·Å³ (`VPST` is added to
`ZZ/G²` before the division by $\Omega$).

`PSCORE` = $v_{\rm short}(q\!\to\!0)$ = 105.89 eV·Å³ for Au. It feeds the
`PSCENC` energy correction (`pot.F:1035–1045`) and does **not** enter the
potential, since $V(\mathbf G=0)$ is zeroed.

## 5. What is *not* in `EXTCAR`

The comment is explicit, and it bounds what any learned model can be asked to do:

- **no PAW one-centre terms** — `EXTCAR` is a pseudo quantity, consistent with
  the pseudo `CHGCAR` and `TAUCAR`;
- **no non-local projectors** — "not a local function of $\mathbf r$ at all",
  so they cannot appear in a volumetric file even in principle;
- **no Hartree and no XC** — unlike `LOCPOT`, which is $v_H + v_{\rm ext}$.
  `EXTCAR` is the bare ionic term, which is exactly what the
  $V_{\rm ext}\mapsto\rho$ map should take as input;
- **no Ewald / ion-ion energy** — that is a scalar (`PSCENC`, `TEWEN`), not a
  field. *This answers one of the questions posed: no Ewald contribution is
  missing, because none belongs in this file.*

---

## 6. `TAUCAR` — definition and extraction

`main.F:5048–5106`, guarded by the `TAUCAR` tag (`base.F:142`, read at
`main.F:667`). The comment gives the definition:

> `tau_s(r) = hbar^2/2m sum_nk w_k f_nk |grad psi_nk,s(r)|^2`

so it is the **positive-definite gradient form**, *not* the Laplacian form.
`metagga.F:9795` confirms: "calculates the kinetic energy of the PW part of the
wavefunctions (`0.5*|grad psi|**2`)".

`TAU_PW` (`metagga.F:9806–9986`) evaluates it per Cartesian direction:

```fortran
DO I=1,NPL                                     ! plane-wave coefficients
   G1=WDES%IGX(I,NK)+WDES%VKPT(1,NK)           ! G + k, fractional
   G2=...; G3=...
   GC=(G1*LATT_CUR%B(IDIR,1)+G2*LATT_CUR%B(IDIR,2)+G3*LATT_CUR%B(IDIR,3))
   CFA(I)=GC*W%CPTWFP(...)*CITPI               ! i*2pi*(G+k)_IDIR * c_G
ENDDO
CALL FFTWAV(NPL,WDES%NINDPW(1,NK),CFAR(1),CFA(1),GRID)   ! -> real space
DO I=1,GRID%RL%NP
   CKIN(I)=CKIN(I)+HSQDTM*REAL(CONJG(CFBR(I))*CFAR(I))*WEIGHT
ENDDO
```

i.e. the gradient is applied **spectrally** as $i(\mathbf G+\mathbf k)$ on the
wavefunction coefficients, inverse-FFT'd to real space, and accumulated as
$|\partial_\alpha\psi|^2$ weighted by $w_k f_{nk}$, with
`HSQDTM` $=\hbar^2/2m = 3.80998$ eV·Å².

**Storage convention** (`main.F:5059–5061`):

> the file layout is identical to CHGCAR: TAUCAR stores `tau*Omega`, just as
> CHGCAR stores `rho*Omega`, so that `sum_r TAUCAR(r)/(NGXF*NGYF*NGZF)` is the
> plane wave kinetic energy (in eV) of the cell.

confirmed in code by `CTAUC=CTAUC*LATT_CUR%OMEGA` and the use of `OUTCHG`
(not `OUTPOT`). For `ISPIN=2` the two blocks are (total, magnetization), and
`RC_FLIP` converts from (up, down) — matching the `CHGCAR` layout.

### Verdict on our `TAUCAR` handling

**Our implementation is already correct**, on every point:

| Aspect | VASP | `poraque` | |
|---|---|---|---|
| Definition | $\frac{\hbar^2}{2m}\sum|\nabla\psi|^2$, positive-definite | assumed positive-definite | ✅ |
| Volume factor | stores $\tau\Omega$ | `KineticEnergyDensity.volume_scaled = True` | ✅ |
| Units | eV (so $\tau$ in eV/Å³) | eV/Å³ | ✅ |
| Grid | `GRIDC`, same as `CHGCAR` | shared `FieldGrid` | ✅ |
| Content | plane-wave part only, pseudo | — | ✅ |

That the τ is the **plane-wave/pseudo** part also explains an earlier
observation: the Hoffmann-Ostenhof bound $\tau\ge\tau_{\rm vW}[\rho]$ held at
essentially every grid point, because $\tau$ and $\rho$ are pseudised
*consistently*. The bound is only guaranteed for all-electron quantities, so
this was worth checking rather than assuming — and it holds here.

---

## 7. Fix applied

### 7.1 `POTCAR` parser — `poraque/fields/vasp/potcar.py`

`PotcarSingle` gained `psgmax`, `local_potential`, `local_q_grid` and `pscore`,
replacing the previous "first value is ZVAL" misreading. The module docstring
now records the source fragment the layout was recovered from.

### 7.2 Exact local potential — `poraque/fields/ExternalPotential`

New `from_potcar_tables()` implements `POTION` verbatim: natural cubic spline
of $v_{\rm short}$ on the `PSGMAX` mesh, analytic Coulomb tail, $\mathbf G=0$
zeroed, hard truncation at `PSGMAX - PSGMAX/NPSPTS`, structure factors per
species.

`from_calculation(model="auto")` — now the default — uses this route whenever
the tables can be read and falls back to the Gaussian model otherwise, so the
most accurate available construction is used without being requested.

### 7.3 Measured result

| Structure | grid | Gaussian ($\sigma^*$) | **POTCAR tables** |
|---|---|---|---|
| struct_000 | 128³ | 0.1338 | **0.0000213** |
| struct_001 | 120×128×128 | 0.1303 | **0.0000595** |
| struct_002 | 128×120×128 | 0.1285 | **0.0000575** |

(relative $L^2$; Pearson $r$ = 1.000000 in all three.)

**Residual.** The remaining $\sim6\times10^{-5}$ is the spline boundary
condition: VASP's `SPLCOF` is called as `SPLCOF(PSP(1,1),NPSPTS,NPSPTS,0._q)`,
whereas we use SciPy's *natural* cubic spline. The difference is confined to
the ends of the table and is worth ~0.6 meV MAE. Closing it fully would mean
porting `SPLCOF`'s boundary treatment; it is below any threshold that matters
for training and is documented rather than chased.

---

## 8. Recommendations

1. **Done — use `model="auto"`/`"potcar"`.** The Gaussian model remains useful
   for structures with no `POTCAR` (pure-geometry inference) but should not be
   used when a reference `EXTCAR` is expected to match.
2. **Retrain `ext2chg` on the corrected `EXTCAR`.** Models trained on the
   Gaussian potential learned the Hohenberg-Kohn map *for that model
   potential*. With the exact input the map becomes the physical one, and the
   inference pipeline no longer has a 13 % systematic input error before the
   network is even reached.
3. **Watch `PSGMAX` when changing `ENCUT`.** `pseudo.F:1286` warns
   "PSGMAX for local potential too small" when the grid outruns the table. Our
   implementation mirrors VASP's truncation, so behaviour matches — but a
   `PSGMAX` well below $|G|_{\max}$ means the reference itself is truncated.
4. **Multi-species is untested.** The formula sums over species with per-species
   `PSGMAX`, and the implementation follows that, but this dataset is
   single-species (Au). Validate on a binary before relying on it.
5. **`ISPIN=2` is not handled.** Both `CHGCAR` and `TAUCAR` then carry a second
   (magnetization) block, which our reader currently ignores — it reads the
   first block only. Fine for these non-spin-polarised runs (`ISPIN=1`), a
   silent truncation otherwise.

## 9. Source index

| Concern | File | Lines |
|---|---|---|
| `EXTCAR`/`TAUCAR` INCAR flags | `base.F` | 142–143 |
| flag parsing | `main.F` | 665–670 |
| `EXTCAR` write | `main.F` | 4761–4808 |
| local potential formula | `pot.F` | 1005–1095 (`POTION`) |
| `POTCAR` table layout | `pseudo.F` | 294–305 |
| `NPSPTS = 1000` | `pseudo_struct.F` | 10 |
| `TAUCAR` write | `main.F` | 5048–5106 |
| τ evaluation | `metagga.F` | 9795–9986 (`TAU_PW`) |
