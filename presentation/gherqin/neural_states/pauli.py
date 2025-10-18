from math import sqrt
import torch
from ..config import DEVICE

def create_dict(**overrides):
    """
    Build {X,Y,Z} single-qubit unitaries as cdouble.
    Y uses [[1,-i],[1,i]]/√2 so branch orientation matches our measurement convention.
    """
    inv_sqrt2 = 1.0 / sqrt(2.0)

    X = inv_sqrt2 * torch.tensor([[1.0+0.0j,  1.0+0.0j],
                                  [1.0+0.0j, -1.0+0.0j]],
                                 dtype=torch.cdouble, device=DEVICE)

    Y = inv_sqrt2 * torch.tensor([[1.0+0.0j,  0.0-1.0j],
                                  [1.0+0.0j,  0.0+1.0j]],
                                 dtype=torch.cdouble, device=DEVICE)

    Z = torch.tensor([[1.0+0.0j, 0.0+0.0j],
                      [0.0+0.0j, 1.0+0.0j]],
                     dtype=torch.cdouble, device=DEVICE)

    U = {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}
    for name, mat in overrides.items():  # normalize overrides once
        U[name] = as_complex_unitary(mat, DEVICE)
    return U


def as_complex_unitary(U, device: torch.device):
    """Return a (2,2) complex (cdouble) matrix on `device`."""
    if torch.is_tensor(U):
        if U.dim() != 2 or U.shape != (2, 2):
            raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U.shape)}")
        return U.to(device=device, dtype=torch.cdouble).contiguous()

    U_t = torch.tensor(U, device=device)
    if U_t.dim() != 2 or U_t.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U_t.shape)}")
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()