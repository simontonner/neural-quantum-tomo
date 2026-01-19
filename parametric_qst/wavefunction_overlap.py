import itertools
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from data_handling import load_state_npz

def generate_basis_states(num_qubits: int, device: torch.device) -> torch.Tensor:
    """
    Generates all 2^N basis states (00...0 to 11...1) as a float tensor.
    """
    lst = list(itertools.product([0, 1], repeat=num_qubits))
    return torch.tensor(lst, dtype=torch.float32, device=device)

def load_gt_wavefunction(path: Path, device: torch.device) -> Optional[torch.Tensor]:
    """
    Loads the ground truth wavefunction from an NPZ file, takes the real part
    (for TFIM), converts to float32, and moves to the specified device.
    """
    if not path.exists():
        return None

    psi_np, _ = load_state_npz(path)
    psi_true = torch.from_numpy(psi_np).real.float().to(device)
    return psi_true

@torch.no_grad()
def get_normalized_wavefunction(model: nn.Module, cond_batch: torch.Tensor, basis_states: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized wavefunction amplitude for the given basis states
    under specific conditions.
    """
    if cond_batch.dim() == 1:
        cond_batch = cond_batch.unsqueeze(0)

    # expand condition to match the number of basis states
    cond_exp = cond_batch.expand(basis_states.shape[0], -1)

    log_psi = model.log_score(basis_states, cond_exp)

    # compute normalization constant (Z) in log domain
    log_norm_sq = torch.logsumexp(2.0 * log_psi, dim=0)

    return torch.exp(log_psi - 0.5 * log_norm_sq)

@torch.no_grad()
def calculate_exact_overlap(model: nn.Module, cond_val: float,
                            psi_true: torch.Tensor, all_states: torch.Tensor) -> float:
    """
    Calculates the overlap <psi_true | psi_model>.
    Auto-normalizes psi_true and generates psi_model.
    """
    device = all_states.device

    # Ensure psi_true is float/device correct
    if not isinstance(psi_true, torch.Tensor):
        psi_true = torch.tensor(psi_true)
    psi_true = psi_true.real.float().to(device)

    psi_true = psi_true / torch.norm(psi_true)

    cond_batch = torch.tensor([cond_val], device=device, dtype=torch.float32)

    psi_model = get_normalized_wavefunction(model, cond_batch, all_states)

    return torch.abs(torch.dot(psi_true, psi_model)).item()