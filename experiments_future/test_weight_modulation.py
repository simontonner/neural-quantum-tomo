# tfim_condW_debug.py
#
# Goal: test whether your "dip near h~3" comes from the fact that your conditional RBM
# only conditions biases (b,c) but keeps W fixed across h.
#
# This script trains TWO models on the same support dataset:
#   (A) Bias-only conditional RBM (your baseline)
#   (B) Conditional RBM that ALSO modulates W columns via FiLM: W_eff[:,h] = W[:,h] * (1+gamma_w) + beta_w
#
# Then it evaluates overlap vs ED states at support+novel h values and plots both.
#
# Run:
#   python tfim_condW_debug.py
#   python tfim_condW_debug.py --epochs 50 --sample_sizes 2000 10000
#
import os
import sys
import math
import itertools
import json
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# your project utils
sys.path.append(str(Path("..").resolve()))
from data_handling import (
    load_measurements_npz,
    load_state_npz,
    MeasurementDataset,
    MeasurementLoader,
)

# -------------------------
# Paths / device
# -------------------------
data_dir = Path("measurements")
state_dir = Path("state_vectors")
exp_dir = Path("experiments")
data_dir.mkdir(parents=True, exist_ok=True)
state_dir.mkdir(parents=True, exist_ok=True)
exp_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# -------------------------
# Experiment config
# -------------------------
GEN_SIDE_LENGTH = 4
J_VAL = -1.00
GEN_SAMPLES = 50_000  # the file name suffix you generated with

h_support = [1.00, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00]
h_novel   = [1.50, 2.80, 3.20, 3.50, 4.50, 5.50]
all_h_eval = sorted(list(set(h_support + h_novel)))

# -------------------------
# Helpers
# -------------------------
def get_sigmoid_curve(high, low, steps, falloff):
    center = steps / 2.0
    def fn(step):
        s = min(step, steps)
        return float(low + (high - low) / (1.0 + math.exp(falloff * (s - center))))
    return fn

def generate_all_states(num_qubits: int, device: torch.device):
    lst = list(itertools.product([0, 1], repeat=num_qubits))
    return torch.tensor(lst, dtype=torch.float32, device=device)

@torch.no_grad()
def compute_overlap(model, h_val: float, gt_path: Path, all_states: torch.Tensor) -> float:
    psi_np, _ = load_state_npz(gt_path)
    psi_true = torch.from_numpy(psi_np).real.float().to(device)
    psi_true = psi_true / torch.norm(psi_true)

    cond = torch.tensor([h_val], device=device, dtype=torch.float32)  # shape (1,)
    psi_model = model.get_normalized_wavefunction(cond, all_states)
    return torch.abs(torch.dot(psi_true, psi_model)).item()

def train(model, optimizer, loader, num_epochs, rng, lr_schedule_fn, noise_frac=0.1):
    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        tot = 0.0
        for batch in loader:
            lr = lr_schedule_fn(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch, {"rng": rng, "noise_frac": noise_frac})
            loss.backward()
            optimizer.step()
            tot += float(loss.item())
            global_step += 1
        print(f"  epoch {epoch+1:03d}/{num_epochs}  loss={tot/len(loader):.6f}")
    return model

# -------------------------
# Conditioner modules
# -------------------------
class ConditionerBiasOnly(nn.Module):
    # outputs gamma_b,beta_b,gamma_c,beta_c
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 2 * (num_visible + num_hidden))
        self.num_visible = num_visible
        self.num_hidden = num_hidden

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(
            x,
            [self.num_visible, self.num_visible, self.num_hidden, self.num_hidden],
            dim=-1
        )

class ConditionerWithW(nn.Module):
    # outputs gamma_b,beta_b,gamma_c,beta_c,gamma_w,beta_w  (gamma_w/beta_w are per-hidden)
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        out_dim = 2 * num_visible + 4 * num_hidden
        self.fc2 = nn.Linear(hidden_width, out_dim)
        self.num_visible = num_visible
        self.num_hidden = num_hidden

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(
            x,
            [self.num_visible, self.num_visible, self.num_hidden, self.num_hidden, self.num_hidden, self.num_hidden],
            dim=-1
        )

# -------------------------
# Conditional RBMs
# -------------------------
class ConditionalRBM_BiasOnly(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int,
                 conditioner_width: int = 64, k: int = 1, T: float = 1.0):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))
        self.conditioner = ConditionerBiasOnly(num_visible, num_hidden, cond_dim, conditioner_width)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.W, std=0.1)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _compute_effective_biases(self, cond: torch.Tensor):
        gamma_b, beta_b, gamma_c, beta_c = self.conditioner(cond)
        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c
        return b_mod, c_mod

    def _free_energy(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=self.W.dtype, device=self.W.device)
        v_W = v @ self.W
        W_sum = self.W.sum(dim=0)

        linear_v = v_W + c_mod
        linear_flip = W_sum.unsqueeze(0) - v_W + c_mod

        term2_v = F.softplus(linear_v).sum(dim=-1)
        term2_f = F.softplus(linear_flip).sum(dim=-1)

        term1_v = -(v * b_mod).sum(dim=-1)
        term1_f = -((1.0 - v) * b_mod).sum(dim=-1)

        fe_v = term1_v - term2_v
        fe_f = term1_f - term2_f

        stacked = torch.stack([-fe_v, -fe_f], dim=-1)
        return -self.T * torch.logsumexp(stacked / self.T, dim=-1)

    def _gibbs_step(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor, rng: torch.Generator):
        p_h = torch.sigmoid((v @ self.W + c_mod) / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        p_v = torch.sigmoid((h @ self.W.t() + b_mod) / self.T)
        return torch.bernoulli(p_v, generator=rng)

    def log_score(self, v: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b_mod, c_mod = self._compute_effective_biases(cond)
        return -0.5 * self._free_energy(v, b_mod, c_mod) / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        v_data, _, cond = batch
        v_data = v_data.to(device=self.W.device, dtype=self.W.dtype)
        cond = cond.to(device=self.W.device, dtype=self.W.dtype)
        rng = aux_vars.get("rng")
        noise_frac = float(aux_vars.get("noise_frac", 0.1))

        b_mod, c_mod = self._compute_effective_biases(cond)
        v_model = v_data.clone()

        n_noise = int(v_data.shape[0] * noise_frac)
        if n_noise > 0:
            v_model[:n_noise] = torch.bernoulli(torch.full_like(v_model[:n_noise], 0.5), generator=rng)

        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, b_mod, c_mod, rng)
        v_model = v_model.detach()

        fe_data = self._free_energy(v_data, b_mod, c_mod)
        fe_model = self._free_energy(v_model, b_mod, c_mod)
        loss = (fe_data - fe_model).mean()
        return loss, {}

    @torch.no_grad()
    def get_normalized_wavefunction(self, cond: torch.Tensor, all_states: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_exp = cond.expand(all_states.shape[0], -1)

        old_T = self.T
        self.T = 1.0
        log_psi = self.log_score(all_states, cond_exp)
        self.T = old_T

        log_norm_sq = torch.logsumexp(2.0 * log_psi, dim=0)
        return torch.exp(log_psi - 0.5 * log_norm_sq)

class ConditionalRBM_WithW(nn.Module):
    """
    Same as BiasOnly, but conditioner ALSO outputs gamma_w,beta_w (per hidden unit),
    and we use:
        W_eff[:,h] = W[:,h]*(1+gamma_w[h]) + beta_w[h]
    implemented without constructing W_eff explicitly.
    """
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int,
                 conditioner_width: int = 64, k: int = 1, T: float = 1.0):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))
        self.conditioner = ConditionerWithW(num_visible, num_hidden, cond_dim, conditioner_width)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.W, std=0.1)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _compute_effective_params(self, cond: torch.Tensor):
        gamma_b, beta_b, gamma_c, beta_c, gamma_w, beta_w = self.conditioner(cond)
        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c
        w_scale = (1.0 + gamma_w)          # (B,H)
        w_shift = beta_w                   # (B,H)
        return b_mod, c_mod, w_scale, w_shift

    def _vW_eff(self, v: torch.Tensor, w_scale: torch.Tensor, w_shift: torch.Tensor) -> torch.Tensor:
        # v@W_eff = (v@W)*scale + (sum_i v_i)*shift
        vW = v @ self.W                      # (B,H)
        vsum = v.sum(dim=1, keepdim=True)    # (B,1)
        return vW * w_scale + vsum * w_shift

    def _Wsum_eff(self, w_scale: torch.Tensor, w_shift: torch.Tensor) -> torch.Tensor:
        # sum_i W_eff[i,h] = (sum_i W[i,h])*scale + V*shift
        Wsum = self.W.sum(dim=0).unsqueeze(0)  # (1,H)
        return Wsum * w_scale + float(self.num_visible) * w_shift  # (B,H)

    def _hWt_eff(self, h: torch.Tensor, w_scale: torch.Tensor, w_shift: torch.Tensor) -> torch.Tensor:
        # h @ W_eff^T = (h*scale)@W^T + (h·shift) * 1_V
        h_scaled = h * w_scale                         # (B,H)
        term = h_scaled @ self.W.t()                   # (B,V)
        shift_term = (h * w_shift).sum(dim=1, keepdim=True)  # (B,1)
        return term + shift_term.expand(-1, self.num_visible)

    def _free_energy(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor,
                     w_scale: torch.Tensor, w_shift: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=self.W.dtype, device=self.W.device)

        vW_eff = self._vW_eff(v, w_scale, w_shift)          # (B,H)
        Wsum_eff = self._Wsum_eff(w_scale, w_shift)         # (B,H)

        linear_v = vW_eff + c_mod
        linear_flip = Wsum_eff - vW_eff + c_mod

        term2_v = F.softplus(linear_v).sum(dim=-1)
        term2_f = F.softplus(linear_flip).sum(dim=-1)

        term1_v = -(v * b_mod).sum(dim=-1)
        term1_f = -((1.0 - v) * b_mod).sum(dim=-1)

        fe_v = term1_v - term2_v
        fe_f = term1_f - term2_f

        stacked = torch.stack([-fe_v, -fe_f], dim=-1)
        return -self.T * torch.logsumexp(stacked / self.T, dim=-1)

    def _gibbs_step(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor,
                    w_scale: torch.Tensor, w_shift: torch.Tensor, rng: torch.Generator):
        pre_h = (self._vW_eff(v, w_scale, w_shift) + c_mod) / self.T
        p_h = torch.sigmoid(pre_h)
        h = torch.bernoulli(p_h, generator=rng)

        pre_v = (self._hWt_eff(h, w_scale, w_shift) + b_mod) / self.T
        p_v = torch.sigmoid(pre_v)
        return torch.bernoulli(p_v, generator=rng)

    def log_score(self, v: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b_mod, c_mod, w_scale, w_shift = self._compute_effective_params(cond)
        return -0.5 * self._free_energy(v, b_mod, c_mod, w_scale, w_shift) / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        v_data, _, cond = batch
        v_data = v_data.to(device=self.W.device, dtype=self.W.dtype)
        cond = cond.to(device=self.W.device, dtype=self.W.dtype)
        rng = aux_vars.get("rng")
        noise_frac = float(aux_vars.get("noise_frac", 0.1))

        b_mod, c_mod, w_scale, w_shift = self._compute_effective_params(cond)
        v_model = v_data.clone()

        n_noise = int(v_data.shape[0] * noise_frac)
        if n_noise > 0:
            v_model[:n_noise] = torch.bernoulli(torch.full_like(v_model[:n_noise], 0.5), generator=rng)

        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, b_mod, c_mod, w_scale, w_shift, rng)
        v_model = v_model.detach()

        fe_data = self._free_energy(v_data, b_mod, c_mod, w_scale, w_shift)
        fe_model = self._free_energy(v_model, b_mod, c_mod, w_scale, w_shift)
        loss = (fe_data - fe_model).mean()
        return loss, {}

    @torch.no_grad()
    def get_normalized_wavefunction(self, cond: torch.Tensor, all_states: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond_exp = cond.expand(all_states.shape[0], -1)

        old_T = self.T
        self.T = 1.0
        log_psi = self.log_score(all_states, cond_exp)
        self.T = old_T

        log_norm_sq = torch.logsumexp(2.0 * log_psi, dim=0)
        return torch.exp(log_psi - 0.5 * log_norm_sq)

# -------------------------
# Main
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lr_low", type=float, default=1e-4)
    ap.add_argument("--noise_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_sizes", type=int, nargs="*", default=[2000, 10000])
    args = ap.parse_args()

    # RNG
    torch.manual_seed(args.seed)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)

    # precompute all basis states
    all_states = generate_all_states(GEN_SIDE_LENGTH**2, device)

    # dataset paths (support only)
    sys_str = f"tfim_{GEN_SIDE_LENGTH}x{GEN_SIDE_LENGTH}"
    file_names = [f"{sys_str}_h{h:.2f}_{GEN_SAMPLES}.npz" for h in h_support]
    file_paths = [data_dir / fn for fn in file_names]
    for p in file_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing measurement file: {p}")

    # plotting setup
    fig, ax = plt.subplots(figsize=(10, 6), dpi=140)
    colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(args.sample_sizes)))

    legend_elems = [
        Line2D([0], [0], marker="o", color="k", linestyle="None", label="Support", markersize=8),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="k",
               color="k", linestyle="None", label="Novel", markersize=8),
    ]

    reports: List[Dict[str, Any]] = []

    for ci, n_samples in enumerate(args.sample_sizes):
        print(f"\n==============================")
        print(f"TRAINING with n_samples={n_samples} per support point")
        print(f"==============================")

        ds = MeasurementDataset(file_paths, load_measurements_npz, ["h"], [n_samples] * len(file_paths))
        loader = MeasurementLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False, rng=rng)

        steps = args.epochs * len(loader)
        scheduler = get_sigmoid_curve(args.lr, args.lr_low, steps, 0.005)

        # --- Model A: bias-only ---
        print("\n[Model A] Bias-only conditional RBM")
        modelA = ConditionalRBM_BiasOnly(ds.num_qubits, num_hidden=args.hidden, cond_dim=1,
                                         conditioner_width=64, k=args.k).to(device)
        optA = torch.optim.Adam(modelA.parameters(), lr=args.lr)
        train(modelA, optA, loader, args.epochs, rng, scheduler, noise_frac=args.noise_frac)

        # --- Model B: modulate W columns too ---
        print("\n[Model B] Conditional RBM with W modulation (per hidden FiLM)")
        modelB = ConditionalRBM_WithW(ds.num_qubits, num_hidden=args.hidden, cond_dim=1,
                                      conditioner_width=64, k=args.k).to(device)
        optB = torch.optim.Adam(modelB.parameters(), lr=args.lr)
        train(modelB, optB, loader, args.epochs, rng, scheduler, noise_frac=args.noise_frac)

        # --- Evaluate overlap ---
        resultsA = []
        resultsB = []
        for h_val in all_h_eval:
            gt_path = state_dir / f"{sys_str}_h{h_val:.2f}.npz"
            if not gt_path.exists():
                continue

            ovA = compute_overlap(modelA, h_val, gt_path, all_states)
            ovB = compute_overlap(modelB, h_val, gt_path, all_states)

            rtype = "support" if h_val in h_support else "novel"
            resultsA.append({"h": h_val, "overlap": ovA, "type": rtype})
            resultsB.append({"h": h_val, "overlap": ovB, "type": rtype})

        reports.append({"n_samples": n_samples, "modelA_bias_only": resultsA, "modelB_withW": resultsB})

        # plot both models for this sample size
        col = colors[ci]
        # Model A
        hsA = [r["h"] for r in resultsA]
        ovA = [r["overlap"] for r in resultsA]
        ax.plot(hsA, ovA, "-", color=col, alpha=0.45, linewidth=2)

        for r in resultsA:
            mfc = col if r["type"] == "support" else "white"
            ax.plot(r["h"], r["overlap"], "o", markersize=7, markeredgewidth=2,
                    markerfacecolor=mfc, markeredgecolor=col)

        # Model B (dashed)
        hsB = [r["h"] for r in resultsB]
        ovB = [r["overlap"] for r in resultsB]
        ax.plot(hsB, ovB, "--", color=col, alpha=0.95, linewidth=2)

        legend_elems.append(Line2D([0], [0], color=col, lw=3, label=f"n={n_samples} (A solid, B dashed)"))

        # save report
        out_json = exp_dir / f"condW_debug_{n_samples}.json"
        with open(out_json, "w") as f:
            json.dump({"config": vars(args), "n_samples": n_samples, "results": {"A": resultsA, "B": resultsB}}, f, indent=2)
        print(f"Saved: {out_json}")

    ax.set_title("TFIM 4x4 - Overlap | Model A (bias-only) vs Model B (W-modulated)")
    ax.set_xlabel(r"Transverse field $h$")
    ax.set_ylabel(r"Overlap $|\langle \psi_{\rm ED} | \psi_{\rm RBM} \rangle|$")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.95, 1.001)
    ax.legend(handles=legend_elems, loc="lower left", frameon=True)
    plt.tight_layout()
    plt.show()

    out_all = exp_dir / "condW_debug_all.json"
    with open(out_all, "w") as f:
        json.dump({"config": vars(args), "reports": reports}, f, indent=2)
    print(f"Saved: {out_all}")


if __name__ == "__main__":
    main()
