import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Explicitly CPU
device = torch.device("cpu")
print(f"Benchmarking on: {device}")

# Settings
BATCH_SIZE = 128
N_VISIBLE = 16
N_HIDDEN = 32
K_STEPS = 5

# ==========================================
# 1. STANDARD PYTHON VERSION
# ==========================================
class PythonRBM(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(N_VISIBLE, N_HIDDEN) * 0.01)
        self.b = nn.Parameter(torch.zeros(N_VISIBLE))
        self.c = nn.Parameter(torch.zeros(N_HIDDEN))
        self.cond_net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 2*(16+32)))

    def forward(self, v_data, cond):
        params = self.cond_net(cond)
        gamma_b, beta_b, gamma_c, beta_c = torch.split(params, [16, 16, 32, 32], dim=-1)
        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c

        v_model = v_data.clone()
        for _ in range(K_STEPS):
            h = torch.bernoulli(torch.sigmoid(v_model @ self.W + c_mod))
            v_model = torch.bernoulli(torch.sigmoid(h @ self.W.t() + b_mod))

        def free_energy(v):
            return -(v * b_mod).sum(-1) - F.softplus(v @ self.W + c_mod).sum(-1)

        return (free_energy(v_data) - free_energy(v_model.detach())).mean()

# ==========================================
# 2. JIT COMPILED VERSION (Fixed Typo)
# ==========================================
@torch.jit.script
def compute_biases_jit(params: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                       n_v: int, n_h: int) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma_b, beta_b, gamma_c, beta_c = torch.split(params, [n_v, n_v, n_h, n_h], dim=-1)
    b_mod = (1.0 + gamma_b) * b.unsqueeze(0) + beta_b
    c_mod = (1.0 + gamma_c) * c.unsqueeze(0) + beta_c
    return b_mod, c_mod

@torch.jit.script
def gibbs_step_jit(v: torch.Tensor, W: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor, k: int) -> torch.Tensor:
    v_next = v
    for _ in range(k):
        # Explicit sigmoid + bernoulli (fused by JIT)
        h_prob = torch.sigmoid(v_next @ W + c_mod)
        h = torch.bernoulli(h_prob)

        v_prob = torch.sigmoid(h @ W.t() + b_mod)
        v_next = torch.bernoulli(v_prob)
    return v_next

@torch.jit.script
def free_energy_jit(v: torch.Tensor, W: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor) -> torch.Tensor:
    # Manual softplus for JIT stability: log(1 + exp(x))
    return -(v * b_mod).sum(-1) - torch.log(1.0 + torch.exp(v @ W + c_mod)).sum(-1)

class JitRBM(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(N_VISIBLE, N_HIDDEN) * 0.01)
        self.b = nn.Parameter(torch.zeros(N_VISIBLE))
        self.c = nn.Parameter(torch.zeros(N_HIDDEN))
        self.cond_net = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 2*(16+32)))
        self.nv = N_VISIBLE
        self.nh = N_HIDDEN
        self.k = K_STEPS

    def forward(self, v_data, cond):
        params = self.cond_net(cond)
        b_mod, c_mod = compute_biases_jit(params, self.b, self.c, self.nv, self.nh)

        v_model = v_data.clone()
        v_model = gibbs_step_jit(v_model, self.W, b_mod, c_mod, self.k)

        fe_data = free_energy_jit(v_data, self.W, b_mod, c_mod)
        fe_model = free_energy_jit(v_model.detach(), self.W, b_mod, c_mod)

        return (fe_data - fe_model).mean()

# ==========================================
# 3. BENCHMARK RUNNER
# ==========================================
def run_benchmark(name, model_cls):
    print(f"\n--- Testing {name} ---")

    # 2000 batches
    N_BATCHES = 2000
    v = torch.randint(0, 2, (BATCH_SIZE, N_VISIBLE)).float()
    c = torch.randn(BATCH_SIZE, 1)

    model = model_cls().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Warmup
    for _ in range(10):
        loss = model(v, c)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    start = time.perf_counter()

    for _ in range(N_BATCHES):
        loss = model(v, c)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    end = time.perf_counter()
    duration = end - start

    print(f"Total Time ({N_BATCHES} batches): {duration:.4f} s")
    print(f"Batches/Sec: {N_BATCHES/duration:.1f}")
    return duration

if __name__ == "__main__":
    t_py = run_benchmark("Standard Python", PythonRBM)
    t_jit = run_benchmark("TorchScript JIT", JitRBM)

    print("\n" + "="*40)
    print(f"Speedup with JIT: {t_py / t_jit:.2f}x")
    if t_py / t_jit > 1.2:
        print("Verdict: USE JIT. It fuses operations and reduces CPU overhead.")
    else:
        print("Verdict: NO GAIN. Just use larger batch sizes.")
    print("="*40)