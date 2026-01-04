# tfim_condW_debug_optimized.py
#
# FIXED: imports argparse
# Optimized / stabilized version of the A-vs-B test.
#
# Run:
#   python tfim_condW_debug_optimized.py
#   python tfim_condW_debug_optimized.py --epochs 50 --sample_sizes 2000 10000 --no_beta_w
#
import os
import sys
import math
import itertools
import json
import argparse
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
# Experiment config (matches your setup)
# -------------------------
GEN_SIDE_LENGTH = 4
GEN_SAMPLES = 50_000  # file suffix you generated
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

    cond = torch.tensor([h_val], device=device, dtype=torch.float32)
    psi_model = model.get_normalized_wavefunction(cond, all_states)
    return torch.abs(torch.dot(psi_true, psi_model)).item()

def train(model, optimizer, loader, num_epochs, rng, lr_schedule_fn, noise_frac=0.1):
    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        tot = 0.0
        for batch in loader:
            lr = lr_schedule_fn(global_step)

            # scale both param groups proportionally
            base_lr0 = optimizer.param_groups[0]["_base_lr"]
            base_lr1 = optimizer.param_groups[1]["_base_lr"]
            optimizer.param_groups[0]["lr"] = lr * base_lr0
            optimizer.param_groups[1]["lr"] = lr * base_lr1

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch, {"rng": rng, "noise_frac": noise_frac})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tot += float(loss.item())
            global_step += 1
        print(f"  epoch {epoch+1:03d}/{num_epochs}  loss={tot/len(loader):.6f}")
    return model


# -------------------------
# Conditioners
# -------------------------
class ConditionerBiasOnly(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, 2 * (num_visible + num_hidden))
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self._init_last_layer_zero()

    def _init_last_layer_zero(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(
            x,
            [self.num_visible, self.num_visible, self.num_hidden, self.num_hidden],
            dim=-1
        )

class ConditionerWithW(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int, hidden_width: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hidden_width)
        out_dim = 2 * num_visible + 4 * num_hidden  # b,c plus gamma_w,beta_w
        self.fc2 = nn.Linear(hidden_width, out_dim)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self._init_last_layer_zero()

    def _init_last_layer_zero(self):
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

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

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.W, std=0.1)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.c)

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

class ConditionalRBM_WithW_Stable(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, cond_dim: int,
                 conditioner_width: int = 64, k: int = 1, T: float = 1.0,
                 gamma_w_max: float = 0.05, beta_w_max: float = 0.01, use_beta_w: bool = True):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        self.T = T

        self.gamma_w_max = float(gamma_w_max)
        self.beta_w_max = float(beta_w_max)
        self.use_beta_w = bool(use_beta_w)

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden))
        self.b = nn.Parameter(torch.zeros(num_visible))
        self.c = nn.Parameter(torch.zeros(num_hidden))
        self.conditioner = ConditionerWithW(num_visible, num_hidden, cond_dim, conditioner_width)

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.W, std=0.1)
        nn.init.zeros_(self.b)
        nn.init.zeros_(self.c)

    def _compute_effective_params(self, cond: torch.Tensor):
        gamma_b, beta_b, gamma_c, beta_c, gamma_w, beta_w = self.conditioner(cond)

        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c

        # clamp W modulation
        w_scale = 1.0 + self.gamma_w_max * torch.tanh(gamma_w)  # (B,H)
        if self.use_beta_w:
            w_shift = self.beta_w_max * torch.tanh(beta_w)      # (B,H)
        else:
            w_shift = torch.zeros_like(beta_w)

        return b_mod, c_mod, w_scale, w_shift

    def _vW_eff(self, v: torch.Tensor, w_scale: torch.Tensor, w_shift: torch.Tensor) -> torch.Tensor:
        vW = v @ self.W                      # (B,H)
        if self.use_beta_w:
            vsum = v.sum(dim=1, keepdim=True)  # (B,1)
            return vW * w_scale + vsum * w_shift
        return vW * w_scale

    def _Wsum_eff(self, w_scale: torch.Tensor, w_shift: torch.Tensor) -> torch.Tensor:
        Wsum = self.W.sum(dim=0).unsqueeze(0)  # (1,H)
        if self.use_beta_w:
            return Wsum * w_scale + float(self.num_visible) * w_shift
        return Wsum * w_scale

    def _hWt_eff(self, h: torch.Tensor, w_scale: torch.Tensor, w_shift: torch.Tensor) -> torch.Tensor:
        h_scaled = h * w_scale
        term = h_scaled @ self.W.t()  # (B,V)
        if self.use_beta_w:
            shift_term = (h * w_shift).sum(dim=1, keepdim=True)  # (B,1)
            term = term + shift_term.expand(-1, self.num_visible)
        return term

    def _free_energy(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor,
                     w_scale: torch.Tensor, w_shift: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=self.W.dtype, device=self.W.device)

        vW_eff = self._vW_eff(v, w_scale, w_shift)
        Wsum_eff = self._Wsum_eff(w_scale, w_shift)

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
        v = torch.bernoulli(p_v, generator=rng)
        return v

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


def main():
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

    ap.add_argument("--cond_lr_mult", type=float, default=0.2)
    ap.add_argument("--gamma_w_max", type=float, default=0.05)
    ap.add_argument("--beta_w_max", type=float, default=0.01)
    ap.add_argument("--no_beta_w", action="store_true", default=False)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)

    sys_str = f"tfim_{GEN_SIDE_LENGTH}x{GEN_SIDE_LENGTH}"
    file_names = [f"{sys_str}_h{h:.2f}_{GEN_SAMPLES}.npz" for h in h_support]
    file_paths = [data_dir / fn for fn in file_names]
    for p in file_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing measurement file: {p}")

    all_states = generate_all_states(GEN_SIDE_LENGTH**2, device)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
    colors = plt.cm.viridis(np.linspace(0.25, 0.9, len(args.sample_sizes)))

    legend_elems = [
        Line2D([0], [0], marker="o", color="k", linestyle="None", label="Support", markersize=8),
        Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="k",
               color="k", linestyle="None", label="Novel", markersize=8),
    ]

    for ci, n_samples in enumerate(args.sample_sizes):
        print(f"\n==============================")
        print(f"TRAINING with n_samples={n_samples} per support point")
        print(f"==============================")

        ds = MeasurementDataset(file_paths, load_measurements_npz, ["h"], [n_samples] * len(file_paths))
        loader = MeasurementLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False, rng=rng)

        steps = args.epochs * len(loader)
        sched = get_sigmoid_curve(args.lr, args.lr_low, steps, 0.005)
        col = colors[ci]

        # Model A
        print("\n[Model A] Bias-only conditional RBM")
        modelA = ConditionalRBM_BiasOnly(ds.num_qubits, num_hidden=args.hidden, cond_dim=1,
                                         conditioner_width=64, k=args.k).to(device)
        optA = torch.optim.Adam([
            {"params": [modelA.W, modelA.b, modelA.c], "lr": 1.0, "_base_lr": 1.0},
            {"params": modelA.conditioner.parameters(), "lr": args.cond_lr_mult, "_base_lr": args.cond_lr_mult},
        ], lr=args.lr, weight_decay=1e-5)
        train(modelA, optA, loader, args.epochs, rng, sched, noise_frac=args.noise_frac)

        # Model B
        print("\n[Model B] W-modulated conditional RBM (stabilized)")
        modelB = ConditionalRBM_WithW_Stable(
            ds.num_qubits,
            num_hidden=args.hidden,
            cond_dim=1,
            conditioner_width=64,
            k=args.k,
            gamma_w_max=args.gamma_w_max,
            beta_w_max=args.beta_w_max,
            use_beta_w=(not args.no_beta_w),
        ).to(device)
        optB = torch.optim.Adam([
            {"params": [modelB.W, modelB.b, modelB.c], "lr": 1.0, "_base_lr": 1.0},
            {"params": modelB.conditioner.parameters(), "lr": args.cond_lr_mult, "_base_lr": args.cond_lr_mult},
        ], lr=args.lr, weight_decay=1e-5)
        train(modelB, optB, loader, args.epochs, rng, sched, noise_frac=args.noise_frac)

        # Evaluate
        resA, resB = [], []
        for h_val in all_h_eval:
            gt_path = state_dir / f"{sys_str}_h{h_val:.2f}.npz"
            if not gt_path.exists():
                continue
            ovA = compute_overlap(modelA, h_val, gt_path, all_states)
            ovB = compute_overlap(modelB, h_val, gt_path, all_states)
            typ = "support" if h_val in h_support else "novel"
            resA.append({"h": h_val, "overlap": ovA, "type": typ})
            resB.append({"h": h_val, "overlap": ovB, "type": typ})

        # Plot A (solid + markers)
        ax.plot([r["h"] for r in resA], [r["overlap"] for r in resA], "-", color=col, alpha=0.45, lw=2)
        for r in resA:
            mfc = col if r["type"] == "support" else "white"
            ax.plot(r["h"], r["overlap"], "o", ms=7, mew=2, mfc=mfc, mec=col)

        # Plot B (dashed)
        ax.plot([r["h"] for r in resB], [r["overlap"] for r in resB], "--", color=col, alpha=0.95, lw=2)

        # Save report
        out_json = exp_dir / f"condW_debug_optimized_{n_samples}.json"
        with open(out_json, "w") as f:
            json.dump({"config": vars(args), "n_samples": n_samples, "results": {"A": resA, "B": resB}}, f, indent=2)
        print(f"Saved: {out_json}")

        legend_elems.append(Line2D([0], [0], color=col, lw=3, label=f"n={n_samples} (A solid, B dashed)"))

    ax.set_title("TFIM 4x4 - Overlap | Model A (bias-only) vs Model B (W-modulated, stabilized)")
    ax.set_xlabel(r"Transverse field $h$")
    ax.set_ylabel(r"Overlap $|\langle \psi_{\rm ED} | \psi_{\rm RBM} \rangle|$")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0.95, 1.001)
    ax.legend(handles=legend_elems, loc="lower left", frameon=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
