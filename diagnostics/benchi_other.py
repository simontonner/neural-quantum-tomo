import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List

# ==========================================
# 1. SETUP & MOCK DATA
# ==========================================
device = torch.device("cpu") # Benchmarking on CPU for fairness/stability
print(f"Benchmarking on: {device}")

# Generate Synthetic Data (In-Memory to remove I/O bottleneck)
print("Generating synthetic data...")
N_SAMPLES = 10_000
N_VISIBLE = 16
N_HIDDEN = 32
COND_DIM = 1
BATCH_SIZE = 128 # Smaller batch size to make the loops strictly compute-bound

# Mock Dataset matching your structure
class MockDataset:
    def __init__(self):
        self.num_qubits = N_VISIBLE
        self.values = torch.randint(0, 2, (N_SAMPLES, N_VISIBLE)).float()
        self.system_params = torch.randn(N_SAMPLES, COND_DIM)
        # Mocking basis logic to satisfy your loader
        self.bases = None
        self.implicit_basis = ("Z",) * N_VISIBLE
    def __len__(self): return N_SAMPLES

# Your Exact Loader (Simplified for standalone use)
class MeasurementLoader:
    def __init__(self, dataset, batch_size=128, shuffle=True, drop_last=False, rng=None):
        self.ds = dataset
        self.bs = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.gen = rng or torch.Generator()
        N = len(self.ds)
        self.slice_bounds = [(i, i + self.bs) for i in range(0, N, self.bs) if not self.drop_last or (i + self.bs) <= N]

    def __len__(self) -> int: return len(self.slice_bounds)

    def __iter__(self):
        N = len(self.ds)
        self.order = torch.randperm(N, generator=self.gen) if self.shuffle else torch.arange(N)
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= len(self.slice_bounds): raise StopIteration
        start, end = self.slice_bounds[self._idx]
        self._idx += 1
        idxs = self.order[start:end]

        values = self.ds.values[idxs]
        sys = self.ds.system_params[idxs]
        return values, [], sys # Return empty bases for benchmark

# ==========================================
# 2. RBM MODEL (Dual Mode)
# ==========================================
class Conditioner(nn.Module):
    def __init__(self, num_visible, num_hidden, cond_dim, hidden_width):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 2 * (num_visible + num_hidden))

    def forward(self, cond):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        N, H = self.num_visible, self.num_hidden
        return torch.split(x, [N, N, H, H], dim=-1)

class ConditionalRBM(nn.Module):
    def __init__(self, k, init_from_data=False):
        super().__init__()
        self.k = k
        self.init_from_data = init_from_data # <--- THE SWITCH
        self.T = 1.0

        self.W = nn.Parameter(torch.randn(N_VISIBLE, N_HIDDEN) * 0.01)
        self.b = nn.Parameter(torch.zeros(N_VISIBLE))
        self.c = nn.Parameter(torch.zeros(N_HIDDEN))
        self.conditioner = Conditioner(N_VISIBLE, N_HIDDEN, COND_DIM, 64)

    def _compute_effective_biases(self, cond):
        gamma_b, beta_b, gamma_c, beta_c = self.conditioner(cond)
        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c
        return b_mod, c_mod

    def forward(self, batch):
        v_data, _, cond = batch
        v_data = v_data.to(device)
        cond = cond.to(device)

        b_mod, c_mod = self._compute_effective_biases(cond)

        # --- INITIALIZATION STRATEGY ---
        if self.init_from_data:
            # STRATEGY B: Start at Data (Contrastive Divergence)
            v_model = v_data.clone()
        else:
            # STRATEGY A: Start at Noise
            v_model = torch.bernoulli(torch.full_like(v_data, 0.5))

        # Gibbs Loop
        for _ in range(self.k):
            # Pos
            p_h = torch.sigmoid((v_model @ self.W + c_mod) / self.T)
            h = torch.bernoulli(p_h)
            # Neg
            p_v = torch.sigmoid((h @ self.W.t() + b_mod) / self.T)
            v_model = torch.bernoulli(p_v)

        # Loss (Free Energy Difference)
        def free_energy(v):
            return -(v * b_mod).sum(-1) - F.softplus(v @ self.W + c_mod).sum(-1)

        loss = (free_energy(v_data) - free_energy(v_model.detach())).mean()
        return loss

# ==========================================
# 3. BENCHMARK RUNNER
# ==========================================
def run_benchmark(name, k, init_from_data, epochs=10):
    print(f"\n==========================================")
    print(f"RUNNING: {name}")
    print(f"Config : k={k} | Init From Data: {init_from_data}")
    print(f"==========================================")

    ds = MockDataset()
    loader = MeasurementLoader(ds, batch_size=BATCH_SIZE)
    model = ConditionalRBM(k=k, init_from_data=init_from_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start_total = time.perf_counter()

    for epoch in range(epochs):
        start_epoch = time.perf_counter()
        losses = []

        for batch in loader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        end_epoch = time.perf_counter()
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch+1}: Time={end_epoch-start_epoch:.4f}s | Avg Loss={avg_loss:.4f}")

    end_total = time.perf_counter()
    return end_total - start_total

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # RUN 1: Your original setup (Noise + k=100)
    time_noise = run_benchmark("STRATEGY A: Noise Init (Original)", k=100, init_from_data=False)

    # RUN 2: CD-k setup (Data + k=5)
    time_data = run_benchmark("STRATEGY B: Data Init (Recommended)", k=5, init_from_data=True)

    print("\n\n" + "#"*50)
    print("           FINAL RESULTS TABLE")
    print("#"*50)
    print(f"{'Strategy':<30} | {'Total Time':<10}")
    print("-" * 45)
    print(f"{'Noise Init (k=100)':<30} | {time_noise:.4f} s")
    print(f"{'Data Init  (k=5)':<30} | {time_data:.4f} s")
    print("-" * 45)

    speedup = time_noise / time_data
    print(f"\nSPEEDUP FACTOR: {speedup:.2f}x faster")

    print("\nOBSERVATION:")
    print("Data Initialization allows you to use drastically smaller 'k'.")
    print("Because the chain starts in a 'good' place (the data), it doesn't need 100 steps to converge.")
    print("Noise initialization requires huge 'k' (or persistent chains) to cross the energy barriers.")