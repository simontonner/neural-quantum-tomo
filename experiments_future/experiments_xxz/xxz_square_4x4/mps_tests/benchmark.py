import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. DEVICE SETUP
# ==========================================
#if torch.backends.mps.is_available():
#    device = torch.device("mps")
#    print(f"--> Success: Device set to Apple MPS (GPU)")
#elif torch.cuda.is_available():
#    device = torch.device("cuda")
#    print(f"--> Success: Device set to CUDA (GPU)")
#else:
device = torch.device("cpu")
print(f"--> Warning: GPU not found. Device set to CPU")

def sync_device():
    """Forces the CPU to wait for the GPU to finish calculations."""
    if device.type == 'mps':
        try:
            torch.mps.synchronize()
        except AttributeError:
            # Fallback for older PyTorch versions on Mac
            torch.zeros(1, device=device).cpu()
    elif device.type == 'cuda':
        torch.cuda.synchronize()

# ==========================================
# 2. DEFINING THE SAMPLERS (JIT vs PYTHON)
# ==========================================

# OPTION A: JIT COMPILED
@torch.jit.script
def gibbs_sampling_jit(v: torch.Tensor, W: torch.Tensor,
                       b_mod: torch.Tensor, c_mod: torch.Tensor,
                       k: int, T: float) -> torch.Tensor:
    curr_v = v
    for _ in range(k):
        # Positive Phase
        h_logits = curr_v @ W + c_mod
        p_h = torch.sigmoid(h_logits / T)
        h = torch.bernoulli(p_h)
        # Negative Phase
        v_logits = h @ W.t() + b_mod
        p_v = torch.sigmoid(v_logits / T)
        curr_v = torch.bernoulli(p_v)
    return curr_v

# OPTION B: STANDARD PYTHON
def gibbs_sampling_python(v: torch.Tensor, W: torch.Tensor,
                          b_mod: torch.Tensor, c_mod: torch.Tensor,
                          k: int, T: float) -> torch.Tensor:
    curr_v = v
    for _ in range(k):
        # Positive Phase
        h_logits = curr_v @ W + c_mod
        p_h = torch.sigmoid(h_logits / T)
        h = torch.bernoulli(p_h)
        # Negative Phase
        v_logits = h @ W.t() + b_mod
        p_v = torch.sigmoid(v_logits / T)
        curr_v = torch.bernoulli(p_v)
    return curr_v

# ==========================================
# 3. THE MODEL
# ==========================================
class BenchmarkRBM(nn.Module):
    def __init__(self, num_visible, num_hidden, cond_dim, k=25, use_jit=True):
        super().__init__()
        self.k = k
        self.use_jit = use_jit
        self.T = 1.0
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        # Weights
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.1)
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))

        # Conditioner
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 2 * (num_visible + num_hidden))
        )

    def forward(self, v_data, cond):
        # 1. Compute Biases
        params = self.cond_net(cond)
        gamma_b, beta_b, gamma_c, beta_c = torch.split(
            params,
            [self.num_visible, self.num_visible, self.num_hidden, self.num_hidden],
            dim=-1
        )
        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c

        # 2. Gibbs Sampling
        v_start = torch.bernoulli(torch.full_like(v_data, 0.5))

        if self.use_jit:
            v_model = gibbs_sampling_jit(v_start, self.W, b_mod, c_mod, self.k, self.T)
        else:
            v_model = gibbs_sampling_python(v_start, self.W, b_mod, c_mod, self.k, self.T)

        # 3. Loss (Free Energy)
        def free_energy(v):
            return -(v * b_mod).sum(-1) - F.softplus(v @ self.W + c_mod).sum(-1)

        loss = (free_energy(v_data) - free_energy(v_model.detach())).mean()
        return loss

# ==========================================
# 4. BENCHMARKING LOOP
# ==========================================
def run_benchmark(mode_name, use_jit, batch_size=1024, k=25, steps=50):
    print(f"\n[{mode_name.upper()}] Preparing...")

    # Setup
    N_V = 16
    N_H = 64
    C_DIM = 1

    # Fake Data
    v = torch.randint(0, 2, (batch_size, N_V)).float().to(device)
    c = torch.randn(batch_size, C_DIM).to(device)

    model = BenchmarkRBM(N_V, N_H, C_DIM, k=k, use_jit=use_jit).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Warmup (Compile JIT / Initialize Caches)
    print(f"[{mode_name.upper()}] Warming up (5 steps)...")
    for _ in range(5):
        loss = model(v, c)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    sync_device()

    # Benchmarking
    print(f"[{mode_name.upper()}] Running {steps} steps (Batch={batch_size}, k={k})...")
    start_time = time.perf_counter()

    for _ in range(steps):
        loss = model(v, c)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    sync_device()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / steps

    print(f"[{mode_name.upper()}] RESULT: {avg_time:.4f} sec/batch")
    return avg_time

# ==========================================
# 5. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("========================================")
    print("       RBM JIT vs PYTHON BENCHMARK      ")
    print("========================================")

    # Config matching your setup
    BATCH_SIZE = 1024
    K_STEPS = 25

    try:
        # Run JIT
        jit_time = run_benchmark("JIT", use_jit=True, batch_size=BATCH_SIZE, k=K_STEPS)

        # Run Python
        py_time = run_benchmark("PYTHON", use_jit=False, batch_size=BATCH_SIZE, k=K_STEPS)

        print("\n========================================")
        print("              FINAL REPORT              ")
        print("========================================")
        print(f"Gibbs Steps (k) : {K_STEPS}")
        print(f"JIT Time        : {jit_time:.4f} s/batch")
        print(f"Python Time     : {py_time:.4f} s/batch")

        if jit_time < py_time:
            ratio = py_time / jit_time
            print(f"--> JIT is {ratio:.2f}x FASTER")
        else:
            ratio = jit_time / py_time
            print(f"--> JIT is {ratio:.2f}x SLOWER (or broken)")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("If JIT crashed, it confirms JIT is broken on your specific PyTorch/MPS version.")