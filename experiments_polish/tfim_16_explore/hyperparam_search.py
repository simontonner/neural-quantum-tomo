import sys
import math
import itertools
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to find data_handling
sys.path.append(str(Path("..").resolve()))
try:
    from data_handling import load_measurements_npz, load_state_npz, MeasurementDataset, MeasurementLoader
except ImportError:
    print("Error: Could not import data_handling.py")
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
SYSTEM_SIZE = 16
DATA_DIR = Path("measurements")
STATE_DIR = Path("state_vectors")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparams
CD_K = 20
LR_INIT = 1e-2
LR_FINAL = 1e-4
BATCH_SIZE = 1024
TRAIN_SAMPLES = 20_000
FILE_SUFFIX = "5000000"

# 1. Refined Grid Ranges
HIDDEN_CANDIDATES = [64, 80, 96]
NOISE_CANDIDATES = [0.1, 0.2, 0.3]

# Evaluation Points
H_SUPPORT = [0.50, 0.80, 0.95, 1.00, 1.05, 1.20, 1.50]
H_NOVEL   = [0.60, 0.70, 1.30, 1.40]
ALL_EVAL_H = sorted(list(set(H_SUPPORT + H_NOVEL)))

# ==========================================
# MODEL CLASS
# ==========================================
class Conditioner(nn.Module):
    def __init__(self, num_visible, num_hidden, cond_dim, hidden_width):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 2 * (num_visible + num_hidden))
        self.num_visible = num_visible
        self.num_hidden = num_hidden

    def forward(self, cond):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(x, [self.num_visible, self.num_visible, self.num_hidden, self.num_hidden], dim=-1)

class ConditionalRBM(nn.Module):
    def __init__(self, num_visible, num_hidden, cond_dim, conditioner_width, k, T=1.0):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T
        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))
        self.conditioner = Conditioner(num_visible, num_hidden, cond_dim, conditioner_width)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.W, std=0.01)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _free_energy(self, v, b_mod, c_mod):
        v_W = v @ self.W
        W_sum = self.W.sum(dim=0)
        linear_v = v_W + c_mod
        linear_f = W_sum.unsqueeze(0) - v_W + c_mod
        term2_v = F.softplus(linear_v).sum(dim=-1)
        term2_f = F.softplus(linear_f).sum(dim=-1)
        term1_v = -(v * b_mod).sum(dim=-1)
        term1_f = -((1.0 - v) * b_mod).sum(dim=-1)
        F_v = term1_v - term2_v
        F_f = term1_f - term2_f
        stacked = torch.stack([-F_v, -F_f], dim=-1)
        return -self.T * torch.logsumexp(stacked / self.T, dim=-1)

    def _compute_biases(self, cond):
        gamma_b, beta_b, gamma_c, beta_c = self.conditioner(cond)
        if cond.dim() == 1:
            b = (1.0 + gamma_b) * self.b + beta_b
            c = (1.0 + gamma_c) * self.c + beta_c
        else:
            b = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
            c = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c
        return b, c

    def _gibbs_step(self, v, h, s0, b_mod, c_mod, rng):
        v_eff = s0 * v + (1.0 - s0) * (1.0 - v)
        p_h = torch.sigmoid((v_eff @ self.W + c_mod) / self.T)
        h = torch.bernoulli(p_h, generator=rng)

        a = h @ self.W.t()
        vb = (v * b_mod).sum(-1); va = (v * a).sum(-1)
        bsum = b_mod.sum(-1); asum = a.sum(-1)
        dE = -bsum - asum + 2.0*vb + 2.0*va
        p_s0 = torch.sigmoid(dE / self.T)
        s0 = torch.bernoulli(p_s0, generator=rng).unsqueeze(-1)

        p_v = torch.sigmoid((a + b_mod) / self.T)
        v_eff = torch.bernoulli(p_v, generator=rng)
        v_next = s0 * v_eff + (1.0 - s0) * (1.0 - v_eff)
        return v_next, h, s0

    def forward(self, batch, aux):
        v_data, _, cond = batch
        # CAST TO FLOAT
        v_data = v_data.to(device=DEVICE, dtype=torch.float32)
        cond = cond.to(device=DEVICE, dtype=torch.float32)

        b_mod, c_mod = self._compute_biases(cond)
        v_model = v_data.clone()

        # Noise
        if aux["noise"] > 0:
            n = int(v_data.shape[0] * aux["noise"])
            if n > 0: v_model[:n] = torch.bernoulli(torch.full_like(v_model[:n], 0.5), generator=aux["rng"])

        B = v_model.size(0)
        s0 = torch.bernoulli(torch.full((B,1), 0.5, device=DEVICE), generator=aux["rng"])
        h = torch.zeros((B, self.num_hidden), device=DEVICE)

        for _ in range(self.k):
            v_model, h, s0 = self._gibbs_step(v_model, h, s0, b_mod, c_mod, aux["rng"])

        fe_d = self._free_energy(v_data, b_mod, c_mod)
        fe_m = self._free_energy(v_model.detach(), b_mod, c_mod)
        return fe_d.mean() - fe_m.mean()

    @torch.no_grad()
    def get_log_psi(self, cond, states):
        if cond.dim() == 1: cond = cond.unsqueeze(0)
        cond = cond.expand(states.shape[0], -1)
        old_T = self.T; self.T = 1.0
        b_mod, c_mod = self._compute_biases(cond)
        log_prob = -self._free_energy(states, b_mod, c_mod)
        self.T = old_T
        return 0.5 * log_prob

# ==========================================
# HELPERS
# ==========================================
def get_sigmoid_curve(high, low, steps, falloff=0.005):
    center = steps / 2.0
    def fn(step):
        s = min(step, steps)
        return float(low + (high - low) / (1.0 + math.exp(falloff * (s - center))))
    return fn

def generate_all_states(n):
    return torch.tensor(list(itertools.product([0, 1], repeat=n)), dtype=torch.float32, device=DEVICE)

def evaluate(model, all_states):
    overlaps = []
    for h in ALL_EVAL_H:
        gt_path = STATE_DIR / f"tfim_{SYSTEM_SIZE}_h{h:.2f}.npz"
        if not gt_path.exists(): continue
        psi_np, _ = load_state_npz(gt_path)
        psi_true = torch.from_numpy(psi_np).real.float().to(DEVICE)
        psi_true /= psi_true.norm()
        cond = torch.tensor([h], device=DEVICE)
        log_psi = model.get_log_psi(cond, all_states)
        log_norm = torch.logsumexp(2.0 * log_psi, dim=0)
        psi_model = torch.exp(log_psi - 0.5 * log_norm)
        ov = torch.abs(torch.dot(psi_true, psi_model)).item()
        overlaps.append(ov)
    return min(overlaps)

def train_model(config, dataset, all_states, epochs):
    """Generic trainer"""
    seed = 42
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    loader = MeasurementLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, rng=rng)

    nh = config["hidden"]
    nf = config["noise"]

    model = ConditionalRBM(SYSTEM_SIZE, nh, 1, nh, CD_K).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)

    # Scheduler adjusted for the specific epoch count
    schedule = get_sigmoid_curve(LR_INIT, LR_FINAL, epochs * len(loader), 0.005)

    model.train()
    step = 0

    # Simple progress print
    print(f"  Training [{nh} hidden, {nf} noise] for {epochs} epochs...")

    for _ in range(epochs):
        for batch in loader:
            lr = schedule(step)
            for g in optimizer.param_groups: g["lr"] = lr

            optimizer.zero_grad()
            loss = model(batch, {"rng": rng, "noise": nf})
            loss.backward()
            optimizer.step()
            step += 1

    min_ov = evaluate(model, all_states)
    return min_ov

# ==========================================
# MAIN
# ==========================================
def main():
    print(f"--- Final Hyperparam Sweep ---")
    print(f"Device: {DEVICE}")
    print(f"Ranges: Hidden={HIDDEN_CANDIDATES}, Noise={NOISE_CANDIDATES}")
    print("-" * 60)

    # 1. Load Data
    fps = [DATA_DIR/f"tfim_{SYSTEM_SIZE}_h{h:.2f}_{FILE_SUFFIX}.npz" for h in H_SUPPORT]
    fps = [p for p in fps if p.exists()]
    ds = MeasurementDataset(fps, load_measurements_npz, ["h"], [TRAIN_SAMPLES]*len(fps))
    all_states = generate_all_states(SYSTEM_SIZE)

    # 2. Phase 1: Short Grid Search
    print("PHASE 1: Refined Grid Search (10 Epochs)")
    print(f"{'Hidden':<8} | {'Noise':<8} | {'Min Overlap':<12}")
    print("-" * 60)

    results = []

    for nh in HIDDEN_CANDIDATES:
        for nf in NOISE_CANDIDATES:
            cfg = {"hidden": nh, "noise": nf}
            # Train for 10 epochs
            ov = train_model(cfg, ds, all_states, epochs=10)
            print(f"{nh:<8} | {nf:<8.1f} | {ov:.5f}")
            results.append((ov, cfg))

    # Sort by overlap descending
    results.sort(key=lambda x: x[0], reverse=True)
    top_2 = results[:2]

    print("-" * 60)
    print("Top 2 Candidates selected:")
    for i, (score, cfg) in enumerate(top_2):
        print(f"  {i+1}. {cfg} (Score: {score:.5f})")

    # 3. Phase 2: Long Training
    print("-" * 60)
    print("PHASE 2: The Finals (50 Epochs)")
    print("Retraining top candidates from scratch...")

    final_results = []

    for _, cfg in top_2:
        ov = train_model(cfg, ds, all_states, epochs=50)
        final_results.append((ov, cfg))
        print(f"  -> Result: {ov:.5f}")

    final_results.sort(key=lambda x: x[0], reverse=True)
    winner = final_results[0]

    print("=" * 60)
    print(f"WINNER: {winner[1]}")
    print(f"FINAL MIN OVERLAP (50 Epochs): {winner[0]:.5f}")
    print("=" * 60)

if __name__ == "__main__":
    main()