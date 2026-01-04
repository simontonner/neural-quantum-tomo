import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DEVICE SETUP ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"--> Success: Device set to Apple MPS (GPU)")
else:
    device = torch.device("cpu")
    print(f"--> Warning: Device set to CPU")

def sync_device():
    if device.type == 'mps':
        try:
            torch.mps.synchronize()
        except AttributeError:
            torch.zeros(1, device=device).cpu()

# --- STANDARD PYTHON SAMPLER (NO JIT) ---
def gibbs_sampling_python(v: torch.Tensor, W: torch.Tensor,
                          b_mod: torch.Tensor, c_mod: torch.Tensor,
                          k: int, T: float) -> torch.Tensor:
    curr_v = v
    for _ in range(k):
        # Native MPS Matrix Multiplications
        h_logits = curr_v @ W + c_mod
        p_h = torch.sigmoid(h_logits / T)
        h = torch.bernoulli(p_h)

        v_logits = h @ W.t() + b_mod
        p_v = torch.sigmoid(v_logits / T)
        curr_v = torch.bernoulli(p_v)
    return curr_v

# --- DUMMY MODEL ---
class BenchmarkRBM(nn.Module):
    def __init__(self, n_v, n_h, k):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_v, n_h).to(device) * 0.1)
        self.k = k

    def forward(self, v, b, c):
        return gibbs_sampling_python(v, self.W, b, c, self.k, 1.0)

# --- EXECUTION ---
BATCH_SIZE = 1024
K_STEPS = 5  # Standard training setting
N_V = 16
N_H = 64

print(f"\nRunning Speed Test on {device}...")
print(f"Batch={BATCH_SIZE}, k={K_STEPS}")

v = torch.randint(0, 2, (BATCH_SIZE, N_V)).float().to(device)
b = torch.zeros(1, N_V).to(device)
c = torch.zeros(1, N_H).to(device)
model = BenchmarkRBM(N_V, N_H, K_STEPS)

# Warmup
for _ in range(5):
    _ = model(v, b, c)
sync_device()

# Timing
start = time.perf_counter()
STEPS = 100
for _ in range(STEPS):
    _ = model(v, b, c)
sync_device()
end = time.perf_counter()

avg_time = (end - start) / STEPS
print(f"Time per batch: {avg_time:.5f} seconds")
print(f"Batches per sec: {1.0/avg_time:.2f}")

if avg_time < 0.05:
    print("--> RESULT: GPU IS FAST ENOUGH. You don't need JIT.")
else:
    print("--> RESULT: Still slow. Data transfer might be the issue.")