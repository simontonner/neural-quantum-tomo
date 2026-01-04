import os
import sys
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.operator import GraphOperator, LocalOperator, spin

# === PATH SETUP ===
sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")

# ==========================================
# 1. GROUND TRUTH GENERATION
# ==========================================
def get_exact_ground_truth(L=10):
    """Generates the exact vector of probabilities for the target state."""
    # Build Hamiltonian
    graph = Hypercube(length=L, n_dim=1, pbc=True)
    hilbert = Spin(s=0.5, N=graph.n_nodes)
    sx = spin.sigmax; sy = spin.sigmay; sz = spin.sigmaz
    bond_hilbert = Spin(s=0.5, N=2)
    xy = sx(bond_hilbert, 0)*sx(bond_hilbert, 1) + sy(bond_hilbert, 0)*sy(bond_hilbert, 1)
    zz = sz(bond_hilbert, 0)*sz(bond_hilbert, 1)
    # Marshall sign rule: xy_coeff = -delta (-1.0)
    bond_matrix = (-1.0 * xy + zz).to_dense()
    H = GraphOperator(hilbert, graph=graph, bond_ops=[bond_matrix])

    # Diagonalize
    sp_mat = H.to_sparse()
    evals, evecs = scipy.sparse.linalg.eigsh(sp_mat, k=1, which="SA")
    psi = evecs[:, 0]

    # Fix Phase (Marshall Rule -> All amplitudes should be real positive)
    idx = np.argmax(np.abs(psi))
    psi = psi * np.exp(-1j * np.angle(psi[idx]))
    psi = np.abs(psi) # Take magnitude to compare with RBM probabilities

    # Normalize probabilities
    probs_true = psi**2
    return probs_true / probs_true.sum()

# ==========================================
# 2. MODEL DEFINITION
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha
        self.T = 1.0
        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))
        self.b_scalar = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.c_vector = nn.Parameter(torch.zeros(1, alpha))
        nn.init.normal_(self.kernel, mean=0.0, std=0.02)
        nn.init.constant_(self.c_vector, 0.0)

    def compute_energy_term(self, v: torch.Tensor):
        v_in = v.unsqueeze(1)
        v_padded = F.pad(v_in, (0, self.num_visible - 1), mode='circular')
        weight = self.kernel.view(self.alpha, 1, self.num_visible)
        return F.conv1d(v_padded, weight).view(v.shape[0], -1)

    def _free_energy(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        wv = self.compute_energy_term(v)
        wv_r = wv.view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        term2 = -F.softplus(wv_r + c_r).sum(dim=(1, 2))
        return term2

    def _gibbs_step(self, v: torch.Tensor, rng: torch.Generator):
        wv = self.compute_energy_term(v).view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        p_h = torch.sigmoid((wv + c_r) / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        w_back = torch.flip(self.kernel, dims=[-1]).view(1, self.alpha, self.num_visible)
        h_padded = F.pad(h, (0, self.num_visible - 1), mode='circular')
        wh = F.conv1d(h_padded, w_back).view(v.shape[0], self.num_visible)
        p_v = torch.sigmoid((wh + self.b_scalar) / self.T)
        return torch.bernoulli(p_v, generator=rng)

    def forward(self, batch, aux_vars):
        v_data = batch[0].float()
        v_pos = torch.cat([v_data, 1.0 - v_data], dim=0)
        v_neg = self._gibbs_step(v_pos, aux_vars['rng'])
        v_neg = v_neg.detach()
        return (self._free_energy(v_pos) - self._free_energy(v_neg)).mean()

# ==========================================
# 3. DIAGNOSTIC RUNNER
# ==========================================
if __name__ == "__main__":
    # 1. Get Ground Truth
    print("Calculating Exact Ground Truth...")
    probs_true = get_exact_ground_truth(10)

    # 2. Train Model (Quick Run to reproduce your state)
    print("Training RBM to reproduce the issue...")
    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=256, shuffle=True, drop_last=True, rng=rng_loader)

    torch.manual_seed(42)
    model = SymmetricRBM(10, alpha=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    # Train for 60 epochs (where you saw the issue)
    for epoch in range(60):
        for batch in loader:
            optimizer.zero_grad()
            model(batch, {"rng": torch.Generator()}).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"\rTraining Epoch {epoch+1}/60...", end="")
    print("\nTraining Complete.")

    # 3. THE X-RAY: Compare Probabilities
    print("Analyzing Distribution...")
    N = 10
    device = next(model.parameters()).device
    indices = torch.arange(2**N, device=device).unsqueeze(1)
    powers = 2**torch.arange(N - 1, -1, -1, device=device)
    all_states = (indices.bitwise_and(powers) != 0).float()

    with torch.no_grad():
        fe = model._free_energy(all_states)
        probs_model = torch.exp(-fe - fe.min())
        probs_model = probs_model / probs_model.sum()
        probs_model_np = probs_model.cpu().numpy()

    # 4. PLOT
    plt.figure(figsize=(10, 8))

    # Log-Log Scatter Plot
    # Filter out tiny probabilities to make plot clean
    mask = probs_true > 1e-6

    plt.scatter(probs_true[mask], probs_model_np[mask], alpha=0.6, c='blue', label='States')

    # Reference Line
    min_val = min(probs_true[mask].min(), probs_model_np[mask].min())
    max_val = max(probs_true[mask].max(), probs_model_np[mask].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')

    # Highlight Néel State
    # Construct Néel state 1010101010
    neel_int = int('1010101010', 2)
    plt.scatter(probs_true[neel_int], probs_model_np[neel_int], c='red', s=100, marker='*', label='Néel State (1010...)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Probability (Exact)', fontsize=12)
    plt.ylabel('Model Probability (RBM)', fontsize=12)
    plt.title('The X-Ray: Model vs Truth', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5. PRINT STATISTICS
    print("\n--- DIAGNOSTIC RESULTS ---")

    # Ratio of Neel to Neighbors
    neel_prob_true = probs_true[neel_int]
    neel_prob_model = probs_model_np[neel_int]

    # Find a neighbor (1 spin flip from Neel)
    # 1010101010 -> 0010101010 (Flip bit 0)
    neighbor_int = neel_int ^ (1 << 9)
    neighbor_prob_true = probs_true[neighbor_int]
    neighbor_prob_model = probs_model_np[neighbor_int]

    print(f"Néel State Probability:")
    print(f"  Truth: {neel_prob_true:.4f}")
    print(f"  Model: {neel_prob_model:.4f}")

    print(f"\nNeighbor State Probability (One flip away):")
    print(f"  Truth: {neighbor_prob_true:.4f}")
    print(f"  Model: {neighbor_prob_model:.4f}")

    ratio_true = neel_prob_true / neighbor_prob_true
    ratio_model = neel_prob_model / neighbor_prob_model

    print(f"\nSharpness Ratio (Néel / Neighbor):")
    print(f"  Truth: {ratio_true:.2f}")
    print(f"  Model: {ratio_model:.2f}")

    if ratio_model > ratio_true * 1.5:
        print("\n>>> DIAGNOSIS: SHARPENING DETECTED <<<")
        print("The model makes the Néel state WAY too probable compared to its neighbors.")
        print("This explains why Czz is too strong (too ordered).")
    elif ratio_model < ratio_true * 0.7:
        print("\n>>> DIAGNOSIS: FLATTENING DETECTED <<<")
        print("The model is too blurry.")
    else:
        print("\n>>> DIAGNOSIS: RATIO LOOKS OKAY <<<")
        print("The issue might be in the tails (rare states).")