"""Angular spin-orbit algebra in dimensionless hbar=1 projector coordinates."""
import numpy as np


def l_dot_s(ell: int) -> np.ndarray:
    r"""L.S = (L_z sigma_z + L_- sigma_+ + L_+ sigma_-)/2.

    Basis ordering is [up m=-ell..ell, down m=-ell..ell]. The upper-right L_-/2 and
    lower-left L_+/2 blocks implement spin flips with orbital angular momentum
    conservation. Eigenvalues are ell/2 (j=ell+1/2) and -(ell+1)/2 (j=ell-1/2).
    No empirical SOC coefficient is inferred from atomic number.
    """
    if int(ell) != ell or ell < 0:
        raise ValueError('l must be a nonnegative integer')
    m = np.arange(-ell, ell+1)
    raising = np.zeros((2*ell+1, 2*ell+1))
    for i, value in enumerate(m[:-1]):
        raising[i+1, i] = np.sqrt(ell*(ell+1)-value*(value+1))
    z = np.diag(m.astype(float))
    return np.block([[z, raising.T], [raising, -z]])/2


def j_projector(ell: int, j: float) -> np.ndarray:
    """Spectral projector onto a j=ell±1/2 manifold expressed explicitly in L.S.

    P_+=(ell+1+2 L.S)/(2l+1), P_-=(ell-2 L.S)/(2l+1).
    With common radial functions D_+ P_+ + D_- P_- = A I + xi L.S,
    xi=2(D_+-D_-)/(2l+1). Using separate j-resolved radial functions retains
    the full relativistic NC operator rather than assuming a common radial part.
    """
    identity = np.eye(2*(2*ell+1))
    if np.isclose(j, ell+.5):
        return ((ell+1)*identity+2*l_dot_s(ell))/(2*ell+1)
    if ell > 0 and np.isclose(j, ell-.5):
        return (ell*identity-2*l_dot_s(ell))/(2*ell+1)
    raise ValueError(f'Invalid total j={j} for l={ell}')
