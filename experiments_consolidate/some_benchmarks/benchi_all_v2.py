import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cpu")

# --- MODEL (Same as before) ---
class ProfilingRBM(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.W = nn.Parameter(torch.randn(16, 32) * 0.01)
        self.b = nn.Parameter(torch.zeros(16))
        self.c = nn.Parameter(torch.zeros(32))
        self.cond_net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 2 * (16 + 32))
        )

    def forward(self, v_data, cond):
        # Conditioner
        params = self.cond_net(cond)
        gamma_b, beta_b, gamma_c, beta_c = torch.split(params, [16, 16, 32, 32], dim=-1)
        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c

        # Gibbs (CD-k, Init from Data)
        v_model = v_data.clone()
        for _ in range(self.k):
            h = torch.bernoulli(torch.sigmoid(v_model @ self.W + c_mod))
            v_model = torch.bernoulli(torch.sigmoid(h @ self.W.t() + b_mod))

        # Loss
        def free_energy(v):
            return -(v * b_mod).sum(-1) - F.softplus(v @ self.W + c_mod).sum(-1)

        loss = (free_energy(v_data) - free_energy(v_model.detach())).mean()
        return loss

# --- BENCHMARK ---
def run_test(batch_size):
    print(f"\n--- Testing Batch Size: {batch_size} ---")

    # 50,000 Samples (Fixed Dataset Size)
    N_SAMPLES = 50_000
    v_data = torch.randint(0, 2, (N_SAMPLES, 16)).float()
    c_data = torch.randn(N_SAMPLES, 1)

    # Create Loader
    dataset = TensorDataset(v_data, c_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ProfilingRBM(k=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    start = time.perf_counter()

    # Run 1 Full Epoch
    for batch in loader:
        v, c = batch
        loss = model(v, c)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    end = time.perf_counter()
    total_time = end - start

    print(f"Time for 50k samples: {total_time:.4f} s")
    print(f"Samples per second:   {N_SAMPLES/total_time:.0f}")

# --- EXECUTION ---
if __name__ == "__main__":
    print("Goal: Process 50,000 training samples.")

    # Your current setup
    run_test(batch_size=128)

    # The optimized setup
    run_test(batch_size=4096)