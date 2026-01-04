import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ==========================================
# 1. SETUP
# ==========================================
device = torch.device("cpu")
print(f"Profiling on: {device}")

N_SAMPLES = 5000
N_VISIBLE = 16
N_HIDDEN = 32
COND_DIM = 1
BATCH_SIZE = 128

# Synthetic Data in RAM
v_data = torch.randint(0, 2, (N_SAMPLES, N_VISIBLE)).float()
c_data = torch.randn(N_SAMPLES, COND_DIM)
dataset = TensorDataset(v_data, c_data)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 2. INSTRUMENTED MODEL
# ==========================================
class ProfilingRBM(nn.Module):
    def __init__(self, k, init_from_data):
        super().__init__()
        self.k = k
        self.init_from_data = init_from_data

        self.W = nn.Parameter(torch.randn(N_VISIBLE, N_HIDDEN) * 0.01)
        self.b = nn.Parameter(torch.zeros(N_VISIBLE))
        self.c = nn.Parameter(torch.zeros(N_HIDDEN))

        # Conditioner
        self.cond_net = nn.Sequential(
            nn.Linear(COND_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 2 * (N_VISIBLE + N_HIDDEN))
        )

    def forward(self, v_data, cond):
        timers = {}

        # --- TIMER: CONDITIONER ---
        t0 = time.perf_counter()
        params = self.cond_net(cond)
        gamma_b, beta_b, gamma_c, beta_c = torch.split(params, [N_VISIBLE, N_VISIBLE, N_HIDDEN, N_HIDDEN], dim=-1)
        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c
        timers['Conditioner'] = time.perf_counter() - t0

        # --- TIMER: GIBBS SAMPLING ---
        t1 = time.perf_counter()
        if self.init_from_data:
            v_model = v_data.clone()
        else:
            v_model = torch.bernoulli(torch.full_like(v_data, 0.5))

        for _ in range(self.k):
            # Pos
            p_h = torch.sigmoid((v_model @ self.W + c_mod))
            h = torch.bernoulli(p_h)
            # Neg
            p_v = torch.sigmoid((h @ self.W.t() + b_mod))
            v_model = torch.bernoulli(p_v)
        timers['Gibbs_Sampling'] = time.perf_counter() - t1

        # --- LOSS CALC ---
        def free_energy(v):
            return -(v * b_mod).sum(-1) - F.softplus(v @ self.W + c_mod).sum(-1)

        loss = (free_energy(v_data) - free_energy(v_model.detach())).mean()

        return loss, timers

# ==========================================
# 3. PROFILING ENGINE
# ==========================================
def profile_run(name, k, init_from_data):
    print(f"\n--- Profiling: {name} (k={k}) ---")

    model = ProfilingRBM(k=k, init_from_data=init_from_data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Storage
    stats = {
        "Data_Loading": 0.0,
        "Conditioner": 0.0,
        "Gibbs_Sampling": 0.0,
        "Backprop_Opt": 0.0,
        "Total_Loop": 0.0
    }

    start_total = time.perf_counter()
    count = 0

    iter_loader = iter(loader)

    # Fetch first batch manually to start the timer after
    try:
        t_load_start = time.perf_counter()
        batch = next(iter_loader)
        t_load_end = time.perf_counter()
        stats["Data_Loading"] += (t_load_end - t_load_start)
    except StopIteration:
        return

    while True:
        try:
            v_batch, c_batch = batch

            # 1. FORWARD (Includes Conditioner + Gibbs timers)
            loss, internal_timers = model(v_batch, c_batch)

            # Add internal timers
            stats["Conditioner"] += internal_timers['Conditioner']
            stats["Gibbs_Sampling"] += internal_timers['Gibbs_Sampling']

            # 2. BACKPROP & OPTIMIZER
            t_back_start = time.perf_counter()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            t_back_end = time.perf_counter()

            stats["Backprop_Opt"] += (t_back_end - t_back_start)
            count += 1

            # 3. NEXT BATCH LOADING
            t_load_start = time.perf_counter()
            batch = next(iter_loader)
            t_load_end = time.perf_counter()
            stats["Data_Loading"] += (t_load_end - t_load_start)

        except StopIteration:
            break

    total_time = time.perf_counter() - start_total
    stats["Total_Loop"] = total_time

    # PRINT RESULTS
    print(f"Total Time: {total_time:.4f} s")
    print(f"Batches Processed: {count}")
    print("-" * 30)
    print(f"{'Component':<20} | {'Total (s)':<10} | {'% of Time':<10}")
    print("-" * 45)

    sum_tracked = stats['Data_Loading'] + stats['Conditioner'] + stats['Gibbs_Sampling'] + stats['Backprop_Opt']

    for key in ["Data_Loading", "Conditioner", "Gibbs_Sampling", "Backprop_Opt"]:
        val = stats[key]
        pct = (val / sum_tracked) * 100
        print(f"{key:<20} | {val:.4f}     | {pct:.1f}%")
    print("-" * 45)

# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    # Profile the "Slow" case first
    profile_run("Noise Init", k=100, init_from_data=False)

    # Profile the "Fast" case
    profile_run("Data Init", k=5, init_from_data=True)