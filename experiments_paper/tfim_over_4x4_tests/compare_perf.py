import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# --- 1. The "Naive" Implementation (Two MatMuls) ---
class NaiveSymmetrizedRBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.1)
        self.b = nn.Parameter(torch.randn(num_visible) * 0.1)
        self.c = nn.Parameter(torch.randn(num_hidden) * 0.1)
        self.T = 1.0

    def free_energy(self, v):
        # 1. Energy of v (MatMul #1)
        term1_v = -(v * self.b).sum(dim=-1)
        term2_v = F.softplus(v @ self.W + self.c).sum(dim=-1)
        fe_v = term1_v - term2_v

        # 2. Energy of (1-v) (MatMul #2)
        v_flipped = 1.0 - v
        term1_f = -(v_flipped * self.b).sum(dim=-1)
        term2_f = F.softplus(v_flipped @ self.W + self.c).sum(dim=-1)
        fe_flipped = term1_f - term2_f

        stacked = torch.stack([-fe_v, -fe_flipped], dim=-1)
        return -self.T * torch.logsumexp(stacked / self.T, dim=-1)

# --- 2. The "Optimized" Implementation (One MatMul) ---
class OptimizedSymmetrizedRBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.1)
        self.b = nn.Parameter(torch.randn(num_visible) * 0.1)
        self.c = nn.Parameter(torch.randn(num_hidden) * 0.1)
        self.T = 1.0

    def free_energy(self, v):
        # 1. Compute the raw matrix product ONCE
        v_W = v @ self.W

        # 2. Compute the row sums of W ONCE
        W_sum = self.W.sum(dim=0)

        # 3. Construct the two linear terms
        linear_v = v_W + self.c
        linear_flip = W_sum.unsqueeze(0) - v_W + self.c

        # 4. Softplus
        term2_v = F.softplus(linear_v).sum(dim=-1)
        term2_f = F.softplus(linear_flip).sum(dim=-1)

        # 5. Bias Terms
        term1_v = -(v * self.b).sum(dim=-1)
        term1_f = -((1.0 - v) * self.b).sum(dim=-1)

        fe_v = term1_v - term2_v
        fe_flipped = term1_f - term2_f

        stacked = torch.stack([-fe_v, -fe_flipped], dim=-1)
        return -self.T * torch.logsumexp(stacked / self.T, dim=-1)

# --- Benchmarking Logic ---
def benchmark(device_name="cpu"):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)
    print(f"\nRunning Benchmark on: {device_name.upper()}")

    # Standard Config
    BATCH_SIZE = 4096
    N_VISIBLE = 16
    N_HIDDEN = 256
    ITERATIONS = 1000

    # Init Models
    model_naive = NaiveSymmetrizedRBM(N_VISIBLE, N_HIDDEN).to(device)
    model_opt = OptimizedSymmetrizedRBM(N_VISIBLE, N_HIDDEN).to(device)

    # Copy weights
    with torch.no_grad():
        model_opt.W.copy_(model_naive.W)
        model_opt.b.copy_(model_naive.b)
        model_opt.c.copy_(model_naive.c)

    # Fake Data
    v = torch.bernoulli(torch.rand(BATCH_SIZE, N_VISIBLE)).to(device)

    # --- VERIFICATION PHASE (DOUBLE PRECISION) ---
    print("--- Verifying Logic (using float64) ---")

    # Temporarily cast to Double (float64) to eliminate precision noise
    model_naive.double()
    model_opt.double()
    v_double = v.double()

    out1 = model_naive.free_energy(v_double)
    out2 = model_opt.free_energy(v_double)

    # We expect near-perfect equality now (approx 1e-15 error)
    diff = (out1 - out2).abs().max().item()
    print(f"Max Difference (float64): {diff:.2e}")

    if diff > 1e-10:
        print("WARNING: Actual Logic mismatch!")
        return
    else:
        print("Logic Check: PASSED")

    # --- TIMING PHASE (FLOAT32) ---
    # Cast back to Float for realistic speed testing
    model_naive.float()
    model_opt.float()

    print(f"\n--- Timing ({ITERATIONS} iterations, float32) ---")

    # Warmup
    for _ in range(50):
        model_naive.free_energy(v)
        model_opt.free_energy(v)

    # Timing Loop
    if device.type == 'cuda':
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(ITERATIONS):
            model_naive.free_energy(v)
        end_event.record()
        torch.cuda.synchronize()
        time_naive = start_event.elapsed_time(end_event)

        torch.cuda.synchronize()
        start_event.record()
        for _ in range(ITERATIONS):
            model_opt.free_energy(v)
        end_event.record()
        torch.cuda.synchronize()
        time_opt = start_event.elapsed_time(end_event)

    else:
        # CPU Timing
        t0 = time.time()
        for _ in range(ITERATIONS):
            model_naive.free_energy(v)
        time_naive = (time.time() - t0) * 1000

        t0 = time.time()
        for _ in range(ITERATIONS):
            model_opt.free_energy(v)
        time_opt = (time.time() - t0) * 1000

    print(f"Naive Time (Total):     {time_naive:.2f} ms")
    print(f"Optimized Time (Total): {time_opt:.2f} ms")
    print(f"Speedup Factor:         {time_naive / time_opt:.2f}x")

if __name__ == "__main__":
    benchmark("cpu")