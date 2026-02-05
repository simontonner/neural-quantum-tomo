import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(str(Path("..").resolve()))
from data_handling import load_measurements_npz, MeasurementDataset, MeasurementLoader

data_dir = Path("measurements")
REF_CZZ = -0.5996
REF_ENTROPY = {1: 0.693, 2: 0.650, 3: 0.860, 4: 0.780, 5: 0.910}

# ==========================================
# 1. EXACT RBM ANALYZER (Because N=10 is small)
# ==========================================
class RBMExactAnalyzer:
    """Helper to calculate Exact Properties of the RBM by summing all 2^N states."""
    def __init__(self, model):
        self.model = model
        self.N = model.num_visible
        self.device = next(model.parameters()).device

        # Generate ALL 2^10 = 1024 states
        indices = torch.arange(2**self.N, device=self.device).unsqueeze(1)
        powers = 2**torch.arange(self.N - 1, -1, -1, device=self.device)
        self.all_states = (indices.bitwise_and(powers) != 0).float()

        # Pre-calculate Sz=0 mask (Physical Sector)
        self.mask_physical = (self.all_states.sum(dim=1) == (self.N // 2))

    def analyze(self):
        """Returns exact Z, Czz, and Entropy."""
        with torch.no_grad():
            # 1. Compute Free Energy for ALL states
            fe = self.model._free_energy(self.all_states)

            # 2. Probability P(v) = exp(-F(v)) / Z
            # Work in log space for stability
            log_probs = -fe
            log_probs -= log_probs.max() # Shift for numerical stability
            probs = torch.exp(log_probs)
            Z = probs.sum()
            probs = probs / Z # Normalize

            # 3. Exact Czz (Physical Sector Only)
            # We weight the Czz of each state by its probability
            # But we only care about the Sz=0 sector contribution?
            # Actually, standard expectation is sum(P(v) * O(v)).
            # Let's verify if the model puts mass on valid states.

            valid_mass = probs[self.mask_physical].sum().item()

            # Calculate Czz Operator for all states
            spins = 2.0 * self.all_states - 1.0
            czz_ops = (spins[:, :-1] * spins[:, 1:]).mean(dim=1)

            # Exact Expectation
            exact_czz = (probs * czz_ops).sum().item()

            # Conditional Czz (normalized over physical sector)
            if valid_mass > 1e-6:
                probs_phys = probs * self.mask_physical.float()
                exact_czz_phys = (probs_phys * czz_ops).sum().item() / valid_mass
            else:
                exact_czz_phys = 0.0

            # 4. Exact Entropy S2 (Using Renyi Formula on the probs)
            # S2 = -log(Sum p_i^2) is simpler for pure states,
            # but for RBM (mixed state representation), we use the wavefunction amplitude
            # psi = sqrt(P).
            psi = torch.sqrt(probs)

            # Renyi S2 for L=5
            dim_A = 2**5
            dim_B = 2**5
            mat = psi.view(dim_A, dim_B)
            S = torch.linalg.svdvals(mat)
            s2_exact = -math.log(torch.sum(S**4).item())

            return exact_czz, exact_czz_phys, valid_mass, s2_exact

# ==========================================
# 2. SYMMETRIC RBM
# ==========================================
class SymmetricRBM(nn.Module):
    def __init__(self, num_visible: int, alpha: int):
        super().__init__()
        self.num_visible = num_visible
        self.alpha = alpha

        self.kernel = nn.Parameter(torch.empty(1, alpha, num_visible))
        self.b_scalar = nn.Parameter(torch.zeros(1), requires_grad=False) # Fixed 0
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
        term1 = -self.b_scalar * v.sum(dim=-1)
        wv = self.compute_energy_term(v)
        wv_r = wv.view(-1, self.alpha, self.num_visible)
        c_r = self.c_vector.view(1, self.alpha, 1)
        term2 = -F.softplus(wv_r + c_r).sum(dim=(1, 2))
        return term1 + term2

# ==========================================
# 3. PARALLEL TEMPERING SAMPLER
# ==========================================
class PTSampler:
    def __init__(self, model, n_chains, n_replicas):
        self.model = model
        self.n_chains = n_chains
        self.n_replicas = n_replicas # Number of temperatures
        self.device = next(model.parameters()).device
        self.N = model.num_visible

        # Temperatures: T=1 to T=10
        # Betas: 1/T
        self.betas = torch.linspace(1.0, 0.1, n_replicas).to(self.device)

        # State: [Replicas, Chains, N]
        self.states = torch.bernoulli(torch.full((n_replicas, n_chains, self.N), 0.5, device=self.device))

    def step(self):
        """Performs Gibbs Sampling + Replica Exchange"""
        # 1. Gibbs Step for ALL replicas in parallel
        # We need to adapt the RBM gibbs step to handle Batched Betas

        # Flatten: [Replicas * Chains, N]
        flat_v = self.states.view(-1, self.N)
        flat_beta = self.betas.view(-1, 1).repeat(1, self.n_chains).view(-1, 1)

        # --- Manual Gibbs Step with Beta ---
        wv = self.model.compute_energy_term(flat_v).view(-1, self.model.alpha, self.N)
        c_r = self.model.c_vector.view(1, self.model.alpha, 1)

        # Effective Field * Beta
        h_input = (wv + c_r) * flat_beta.unsqueeze(-1)
        p_h = torch.sigmoid(h_input)
        h = torch.bernoulli(p_h)

        # Backward
        w_back = torch.flip(self.model.kernel, dims=[-1]).view(1, self.model.alpha, self.N)
        h_padded = F.pad(h, (0, self.N - 1), mode='circular')
        wh = F.conv1d(h_padded, w_back).view(flat_v.shape[0], self.N)

        v_input = (wh + self.model.b_scalar) * flat_beta
        p_v = torch.sigmoid(v_input)
        v_next = torch.bernoulli(p_v)

        # Reshape back
        self.states = v_next.view(self.n_replicas, self.n_chains, self.N)

        # 2. Replica Exchange (Swap)
        # We try to swap replica i and i+1
        # Probability: min(1, exp( (Ei - Ei+1) * (Betai - Betai+1) ))

        # Calculate Energies E(v) (Free Energy is approx Energy for sampling)
        energies = self.model._free_energy(self.states.view(-1, self.N))
        energies = energies.view(self.n_replicas, self.n_chains)

        # Iterate even pairs then odd pairs
        for offset in [0, 1]:
            for i in range(offset, self.n_replicas - 1, 2):
                beta_diff = self.betas[i] - self.betas[i+1]
                energy_diff = energies[i] - energies[i+1]

                log_accept = energy_diff * beta_diff
                accept_prob = torch.exp(log_accept)

                mask = torch.rand(self.n_chains, device=self.device) < accept_prob

                # Perform Swap
                if mask.any():
                    temp = self.states[i, mask].clone()
                    self.states[i, mask] = self.states[i+1, mask]
                    self.states[i+1, mask] = temp

    def get_samples(self):
        # Return samples from the T=1 (Beta=1) chain
        return self.states[0].detach()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    ALPHA = 12
    BATCH_SIZE = 256
    EPOCHS = 60
    LR = 0.002

    # Load Data
    file_name = f"xxz_10_delta1.00_5000000.npz"
    file_path = data_dir / file_name
    ds = MeasurementDataset([file_path], load_fn=load_measurements_npz,
                            system_param_keys=["delta"], samples_per_file=[20_000])
    rng_loader = torch.Generator().manual_seed(42)
    loader = MeasurementLoader(dataset=ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, rng=rng_loader)

    print("Training with PARALLEL TEMPERING (10 Replicas)...")
    torch.manual_seed(42)

    model = SymmetricRBM(10, alpha=ALPHA)
    analyzer = RBMExactAnalyzer(model)

    # Sampler: 10 temperatures, 256 chains each
    sampler = PTSampler(model, n_chains=BATCH_SIZE, n_replicas=10)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            # 1. Data (Positive Phase)
            # Augmentation: v and 1-v
            v_raw = batch[0].float()
            v_pos = torch.cat([v_raw, 1.0 - v_raw], dim=0) # Double batch size

            # 2. Model (Negative Phase) via PT
            # We step the sampler (Gibbs + Swap)
            sampler.step()
            v_neg = sampler.get_samples()

            # Note: v_neg is batch_size, v_pos is 2*batch_size
            # We need to balance the loss terms.
            # Usually we just take a batch from sampler matching data size.
            # Here let's just use what we have, means usually work out.

            optimizer.zero_grad()

            # Loss = FreeEnergy(Data) - FreeEnergy(Model)
            cost = model._free_energy(v_pos).mean() - model._free_energy(v_neg).mean()
            cost.backward()
            optimizer.step()

            epoch_loss += cost.item()

        scheduler.step()

        if (epoch+1) % 5 == 0:
            # === EXACT VERIFICATION ===
            exact_czz, exact_czz_phys, valid_mass, s2_exact = analyzer.analyze()

            print(f"Ep {epoch+1}: Loss={epoch_loss:.2f}")
            print(f"  > EXACT Czz (Sz=0) : {exact_czz_phys:.4f} (Target -0.60)")
            print(f"  > EXACT S2 (L=5)   : {s2_exact:.4f}     (Target 0.91)")
            print(f"  > Valid Mass       : {valid_mass*100:.1f}%")

    print("\nFinal Check complete.")