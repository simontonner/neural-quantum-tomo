#!/usr/bin/env python3
# tfim_4x4_everything_debug.py
#
# One self-contained script that:
#   1) Computes ED ground states for TFIM 4x4 (or any L^dim) with NetKet + SciPy
#   2) Computes fidelity susceptibility chi_F(h) and finds its peak (pseudo-critical h*)
#   3) (Option A) Loads your overlap reports (JSON) and computes the implied field error:
#          delta_eff(h) = sqrt( 2*(1-F(h)) / chi_F(h) )
#      This tells you if a dip is "just sensitivity amplification" (delta_eff flat)
#      or "real model failure" (delta_eff spikes).
#   4) (Option B) Can generate training samples from ED |psi|^2 and train
#      two conditional RBMs (bias-only and stabilized W-modulated) to reproduce
#      the low-shot vs high-shot behavior in a fully self-contained way.
#
# Key message this script will let you VERIFY numerically:
#   - On 4x4, chi_F peaks near h ~ 2.5-2.6 (pseudo-critical).
#   - In LOW-SHOT regimes, overlap dips tend to appear near that peak because noise is amplified:
#         1 - F(h) ~ 0.5 * chi_F(h) * delta^2
#     so if your *effective* parameter error delta is roughly constant, the dip sits at chi_F peak.
#   - A dip around h ~ 3 that persists in higher-shot regimes is usually *bias/training mismatch*,
#     which shows up as a spike in delta_eff(h).
#
# Typical usage (recommended):
#   (A) Analyze your existing overlaps:
#       python tfim_4x4_everything_debug.py \
#           --hmin 1 --hmax 4 --chi_npts 100 --chi_dh 1e-3 \
#           --reports experiments/overlap_2000_report.json experiments/overlap_10000_report.json
#
#   (B) Fully self-contained demo (generate samples + train RBMs + evaluate overlaps):
#       python tfim_4x4_everything_debug.py --do_train \
#           --hmin 1 --hmax 7 --chi_npts 121 --chi_dh 1e-3 \
#           --epochs 50 --sample_sizes 2000 10000 --pool_samples 50000
#
# Outputs:
#   - tfim_chiF_scan.csv
#   - overlap_deltaeff_diagnostic.csv
#   - (if training) overlap_demo_report_*.json
#   - plots: chi_F(h), overlaps, delta_eff
#
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from netket.graph import Hypercube
from netket.hilbert import Spin
from netket.operator import Ising


# ----------------------------
# ED: TFIM build + groundstate
# ----------------------------
def build_tfim(L: int, dim: int, pbc: bool, J: float):
    graph = Hypercube(length=L, n_dim=dim, pbc=pbc)
    hilbert = Spin(s=0.5, N=graph.n_nodes)

    def H_sparse(h: float) -> sp.csr_matrix:
        return Ising(hilbert, graph, h=h, J=J).to_sparse().tocsr()

    return graph, hilbert, H_sparse


def sign_fix_real(psi: np.ndarray) -> np.ndarray:
    """Fix global sign so the largest-amplitude component is positive."""
    psi = np.asarray(psi)
    idx = int(np.argmax(np.abs(psi)))
    val = psi[idx]
    if np.iscomplexobj(psi):
        # align complex phase
        phase = np.angle(val) if val != 0 else 0.0
        psi = psi * np.exp(-1j * phase)
        psi = np.real_if_close(psi, tol=1e6)
    # now real
    if psi[idx] < 0:
        psi = -psi
    return psi


def ground_state_only(Hs: sp.csr_matrix) -> Tuple[float, np.ndarray]:
    vals, vecs = eigsh(Hs, k=1, which="SA", tol=1e-10, maxiter=20000)
    E0 = float(vals[0])
    psi = vecs[:, 0]
    psi = sign_fix_real(psi)
    psi = np.asarray(psi, dtype=np.float64)
    # normalize defensively
    psi = psi / np.linalg.norm(psi)
    return E0, psi


# ----------------------------
# Basis states in correct order
# ----------------------------
def hilbert_all_states_01(hilbert: Spin) -> np.ndarray:
    """
    Returns all basis states in the same ordering as the sparse matrix acts on.
    Converts local states to {0,1} encoding.
    """
    st = np.asarray(hilbert.all_states())  # shape (dimH, N)
    # local values could be {-1, +1} or {-0.5, +0.5} etc.
    uniq = np.unique(st)
    if len(uniq) != 2:
        raise ValueError(f"Expected 2 local states for Spin-1/2, got uniq={uniq}")

    lo, hi = float(np.min(uniq)), float(np.max(uniq))
    # map lo -> 0, hi -> 1
    v01 = (st == hi).astype(np.float32)
    return v01


# ----------------------------
# chi_F scan
# ----------------------------
def compute_chiF_scan(
        H_sparse_fn,
        hs: np.ndarray,
        dh: float,
        cache: Dict[float, np.ndarray],
) -> pd.DataFrame:
    """
    Compute chi_F(h) = 2*(1 - |<psi(h)|psi(h+dh)>|)/dh^2 using ED.
    Uses a simple cache for psi(h).
    """
    chi = np.zeros_like(hs, dtype=np.float64)
    overlaps = np.zeros_like(hs, dtype=np.float64)

    def get_psi(h: float) -> np.ndarray:
        # round key for stable caching
        hk = float(np.round(h, 12))
        if hk in cache:
            return cache[hk]
        _, psi = ground_state_only(H_sparse_fn(hk))
        cache[hk] = psi
        return psi

    for i, h in enumerate(tqdm(hs, desc="ED chi_F scan")):
        psi0 = get_psi(float(h))
        psi1 = get_psi(float(h + dh))
        Fh = float(abs(np.vdot(psi0, psi1)))
        overlaps[i] = Fh
        chi[i] = 2.0 * (1.0 - Fh) / (dh * dh)

    df = pd.DataFrame({"h": hs, "chiF_overlap": chi, "F_neighbor": overlaps, "dh": dh})
    return df


# ----------------------------
# Load overlap reports (your JSON format)
# ----------------------------
def load_overlap_reports(paths: List[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p, "r") as f:
            obj = json.load(f)
        if "results" not in obj:
            raise ValueError(f"{p} missing key 'results'")
        df = pd.DataFrame(obj["results"])
        if not {"h", "overlap"}.issubset(df.columns):
            raise ValueError(f"{p} results must include 'h' and 'overlap'")
        if "type" not in df.columns:
            df["type"] = "unknown"
        df["report"] = p.stem
        rows.append(df[["report", "h", "overlap", "type"]])
    return pd.concat(rows, ignore_index=True).sort_values(["report", "h"]).reset_index(drop=True)


def add_delta_eff(overlap_df: pd.DataFrame, chi_df: pd.DataFrame) -> pd.DataFrame:
    chi_h = chi_df["h"].to_numpy()
    chi = chi_df["chiF_overlap"].to_numpy()

    out = overlap_df.copy()
    out["chiF_interp"] = np.interp(out["h"].to_numpy(), chi_h, chi)
    out["chiF_interp"] = np.clip(out["chiF_interp"].to_numpy(), 1e-12, None)
    one_minus_F = np.clip(1.0 - out["overlap"].to_numpy(), 0.0, None)
    out["delta_eff"] = np.sqrt(2.0 * one_minus_F / out["chiF_interp"].to_numpy())
    return out


# ----------------------------
# Conditional RBMs (self-contained)
# ----------------------------
class ConditionerBiasOnly(nn.Module):
    def __init__(self, V: int, H: int, cond_dim: int, width: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, width)
        self.fc2 = nn.Linear(width, 2 * (V + H))
        self.V, self.H = V, H
        # start as "no conditioning"
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(x, [self.V, self.V, self.H, self.H], dim=-1)


class ConditionerWithW(nn.Module):
    def __init__(self, V: int, H: int, cond_dim: int, width: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, width)
        self.fc2 = nn.Linear(width, 2 * V + 4 * H)
        self.V, self.H = V, H
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(x, [self.V, self.V, self.H, self.H, self.H, self.H], dim=-1)


class ConditionalRBM_BiasOnly(nn.Module):
    def __init__(self, V: int, H: int, cond_dim: int, width: int = 64, k: int = 5, T: float = 1.0):
        super().__init__()
        self.V, self.H, self.k, self.T = V, H, k, T
        self.W = nn.Parameter(torch.empty(V, H))
        self.b = nn.Parameter(torch.zeros(V))
        self.c = nn.Parameter(torch.zeros(H))
        self.conditioner = ConditionerBiasOnly(V, H, cond_dim, width)
        nn.init.normal_(self.W, std=0.1)

    def _compute_biases(self, cond: torch.Tensor):
        gb, bb, gc, bc = self.conditioner(cond)
        b_mod = (1.0 + gb) * self.b.unsqueeze(0) + bb
        c_mod = (1.0 + gc) * self.c.unsqueeze(0) + bc
        return b_mod, c_mod

    def _free_energy_sym(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor):
        vW = v @ self.W
        Wsum = self.W.sum(dim=0)
        lin_v = vW + c_mod
        lin_f = Wsum.unsqueeze(0) - vW + c_mod
        t2v = F.softplus(lin_v).sum(dim=-1)
        t2f = F.softplus(lin_f).sum(dim=-1)
        t1v = -(v * b_mod).sum(dim=-1)
        t1f = -((1.0 - v) * b_mod).sum(dim=-1)
        fe_v = t1v - t2v
        fe_f = t1f - t2f
        stacked = torch.stack([-fe_v, -fe_f], dim=-1)
        return -self.T * torch.logsumexp(stacked / self.T, dim=-1)

    def _gibbs_step(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor, rng: torch.Generator):
        p_h = torch.sigmoid((v @ self.W + c_mod) / self.T)
        h = torch.bernoulli(p_h, generator=rng)
        p_v = torch.sigmoid((h @ self.W.t() + b_mod) / self.T)
        return torch.bernoulli(p_v, generator=rng)

    def log_score(self, v: torch.Tensor, cond: torch.Tensor):
        b_mod, c_mod = self._compute_biases(cond)
        return -0.5 * self._free_energy_sym(v, b_mod, c_mod) / self.T

    def forward(self, v_data: torch.Tensor, cond: torch.Tensor, rng: torch.Generator, noise_frac: float = 0.1):
        b_mod, c_mod = self._compute_biases(cond)
        v_model = v_data.clone()
        n_noise = int(v_data.shape[0] * noise_frac)
        if n_noise > 0:
            v_model[:n_noise] = torch.bernoulli(torch.full_like(v_model[:n_noise], 0.5), generator=rng)
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, b_mod, c_mod, rng)
        v_model = v_model.detach()
        fe_data = self._free_energy_sym(v_data, b_mod, c_mod)
        fe_model = self._free_energy_sym(v_model, b_mod, c_mod)
        return (fe_data - fe_model).mean()

    @torch.no_grad()
    def get_normalized_wavefunction(self, cond_1d: torch.Tensor, all_states: torch.Tensor):
        # cond_1d: shape (1,) or (1,cond_dim)
        if cond_1d.dim() == 1:
            cond_1d = cond_1d.unsqueeze(0)
        cond = cond_1d.expand(all_states.shape[0], -1)
        oldT = self.T
        self.T = 1.0
        logpsi = self.log_score(all_states, cond)
        self.T = oldT
        logZ = torch.logsumexp(2.0 * logpsi, dim=0)
        return torch.exp(logpsi - 0.5 * logZ)


class ConditionalRBM_WithW_Stable(nn.Module):
    def __init__(
            self, V: int, H: int, cond_dim: int, width: int = 64, k: int = 5, T: float = 1.0,
            gamma_w_max: float = 0.05, beta_w_max: float = 0.0  # default: NO beta_w (recommended)
    ):
        super().__init__()
        self.V, self.H, self.k, self.T = V, H, k, T
        self.gamma_w_max = float(gamma_w_max)
        self.beta_w_max = float(beta_w_max)

        self.W = nn.Parameter(torch.empty(V, H))
        self.b = nn.Parameter(torch.zeros(V))
        self.c = nn.Parameter(torch.zeros(H))
        self.conditioner = ConditionerWithW(V, H, cond_dim, width)
        nn.init.normal_(self.W, std=0.1)

    def _params(self, cond: torch.Tensor):
        gb, bb, gc, bc, gw, bw = self.conditioner(cond)
        b_mod = (1.0 + gb) * self.b.unsqueeze(0) + bb
        c_mod = (1.0 + gc) * self.c.unsqueeze(0) + bc
        w_scale = 1.0 + self.gamma_w_max * torch.tanh(gw)     # (B,H)
        w_shift = self.beta_w_max * torch.tanh(bw)            # (B,H)  (often 0)
        return b_mod, c_mod, w_scale, w_shift

    def _vW_eff(self, v: torch.Tensor, w_scale: torch.Tensor, w_shift: torch.Tensor):
        vW = v @ self.W
        if self.beta_w_max != 0.0:
            vsum = v.sum(dim=1, keepdim=True)
            return vW * w_scale + vsum * w_shift
        return vW * w_scale

    def _Wsum_eff(self, w_scale: torch.Tensor, w_shift: torch.Tensor):
        Wsum = self.W.sum(dim=0).unsqueeze(0)
        if self.beta_w_max != 0.0:
            return Wsum * w_scale + float(self.V) * w_shift
        return Wsum * w_scale

    def _hWt_eff(self, h: torch.Tensor, w_scale: torch.Tensor, w_shift: torch.Tensor):
        h_scaled = h * w_scale
        term = h_scaled @ self.W.t()
        if self.beta_w_max != 0.0:
            shift_term = (h * w_shift).sum(dim=1, keepdim=True)
            term = term + shift_term.expand(-1, self.V)
        return term

    def _free_energy_sym(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor,
                         w_scale: torch.Tensor, w_shift: torch.Tensor):
        vW = self._vW_eff(v, w_scale, w_shift)
        Wsum = self._Wsum_eff(w_scale, w_shift)
        lin_v = vW + c_mod
        lin_f = Wsum - vW + c_mod
        t2v = F.softplus(lin_v).sum(dim=-1)
        t2f = F.softplus(lin_f).sum(dim=-1)
        t1v = -(v * b_mod).sum(dim=-1)
        t1f = -((1.0 - v) * b_mod).sum(dim=-1)
        fe_v = t1v - t2v
        fe_f = t1f - t2f
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

    def log_score(self, v: torch.Tensor, cond: torch.Tensor):
        b_mod, c_mod, w_scale, w_shift = self._params(cond)
        return -0.5 * self._free_energy_sym(v, b_mod, c_mod, w_scale, w_shift) / self.T

    def forward(self, v_data: torch.Tensor, cond: torch.Tensor, rng: torch.Generator, noise_frac: float = 0.1):
        b_mod, c_mod, w_scale, w_shift = self._params(cond)
        v_model = v_data.clone()
        n_noise = int(v_data.shape[0] * noise_frac)
        if n_noise > 0:
            v_model[:n_noise] = torch.bernoulli(torch.full_like(v_model[:n_noise], 0.5), generator=rng)
        for _ in range(self.k):
            v_model = self._gibbs_step(v_model, b_mod, c_mod, w_scale, w_shift, rng)
        v_model = v_model.detach()
        fe_data = self._free_energy_sym(v_data, b_mod, c_mod, w_scale, w_shift)
        fe_model = self._free_energy_sym(v_model, b_mod, c_mod, w_scale, w_shift)
        return (fe_data - fe_model).mean()

    @torch.no_grad()
    def get_normalized_wavefunction(self, cond_1d: torch.Tensor, all_states: torch.Tensor):
        if cond_1d.dim() == 1:
            cond_1d = cond_1d.unsqueeze(0)
        cond = cond_1d.expand(all_states.shape[0], -1)
        oldT = self.T
        self.T = 1.0
        logpsi = self.log_score(all_states, cond)
        self.T = oldT
        logZ = torch.logsumexp(2.0 * logpsi, dim=0)
        return torch.exp(logpsi - 0.5 * logZ)


def train_rbm(
        model: nn.Module,
        loader: DataLoader,
        epochs: int,
        lr: float,
        cond_lr_mult: float,
        seed: int,
        noise_frac: float,
        lr_low: float = 1e-4,
):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)

    # two param groups: base params vs conditioner
    base_params = []
    cond_params = []
    for n, p in model.named_parameters():
        if "conditioner" in n:
            cond_params.append(p)
        else:
            base_params.append(p)

    opt = torch.optim.Adam(
        [{"params": base_params, "lr": lr},
         {"params": cond_params, "lr": lr * cond_lr_mult}],
        weight_decay=1e-5
    )

    steps = epochs * len(loader)
    sched = lambda t: float(lr_low + (lr - lr_low) / (1.0 + math.exp(0.005 * (t - steps / 2.0))))

    global_step = 0
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for v, cond in loader:
            v = v.to(dtype=torch.float32)
            cond = cond.to(dtype=torch.float32)

            cur = sched(global_step)
            opt.param_groups[0]["lr"] = cur
            opt.param_groups[1]["lr"] = cur * cond_lr_mult

            opt.zero_grad(set_to_none=True)
            loss = model(v, cond, rng=rng, noise_frac=noise_frac)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot += float(loss.item())
            global_step += 1
        print(f"  epoch {ep+1:03d}/{epochs} loss={tot/len(loader):.6f}")


def sample_from_psi(states01: np.ndarray, psi: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    prob = psi * psi
    prob = prob / prob.sum()
    idx = rng.choice(len(prob), size=num_samples, replace=True, p=prob)
    return states01[idx]  # (num_samples, N), float32 {0,1}


def make_training_loader(
        samples_by_h: Dict[float, np.ndarray],
        batch_size: int,
        shuffle: bool,
        seed: int,
):
    xs = []
    cs = []
    for h, x in samples_by_h.items():
        xs.append(x)
        cs.append(np.full((x.shape[0], 1), float(h), dtype=np.float32))
    X = np.concatenate(xs, axis=0)
    C = np.concatenate(cs, axis=0)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(C))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, generator=g)


@torch.no_grad()
def eval_overlaps(
        model: nn.Module,
        H_sparse_fn,
        states01_torch: torch.Tensor,
        hs_eval: List[float],
        cache: Dict[float, np.ndarray],
) -> List[Dict]:
    out = []
    for h in hs_eval:
        hk = float(np.round(h, 12))
        if hk not in cache:
            _, psi = ground_state_only(H_sparse_fn(hk))
            cache[hk] = psi
        psi_true = torch.from_numpy(cache[hk]).float()
        psi_true = psi_true / torch.linalg.norm(psi_true)

        cond = torch.tensor([h], dtype=torch.float32)
        psi_model = model.get_normalized_wavefunction(cond, states01_torch).float()
        ov = float(torch.abs(torch.dot(psi_true, psi_model)).item())
        out.append({"h": float(h), "overlap": ov})
    return out


def summarize_dips(delta_df: pd.DataFrame, chi_df: pd.DataFrame):
    chi_peak_h = float(chi_df.loc[chi_df["chiF_overlap"].idxmax(), "h"])
    chi_peak_val = float(chi_df["chiF_overlap"].max())
    print("\n================ SUMMARY ================")
    print(f"chi_F peak (pseudo-critical) at h ≈ {chi_peak_h:.4f} with chi_F ≈ {chi_peak_val:.6f}")

    for rep, g in delta_df.groupby("report"):
        g = g.sort_values("h")
        i_min = int(g["overlap"].idxmin())
        i_maxd = int(g["delta_eff"].idxmax())
        h_min = float(delta_df.loc[i_min, "h"])
        F_min = float(delta_df.loc[i_min, "overlap"])
        h_maxd = float(delta_df.loc[i_maxd, "h"])
        d_max = float(delta_df.loc[i_maxd, "delta_eff"])
        print(f"\nReport: {rep}")
        print(f"  min overlap at h ≈ {h_min:.4f}, overlap ≈ {F_min:.6f}")
        print(f"  max delta_eff at h ≈ {h_maxd:.4f}, delta_eff ≈ {d_max:.6e}")

    print("\nInterpretation rule of thumb:")
    print("  - If overlap dips near chi_F peak but delta_eff stays relatively flat -> sensitivity amplification (low-shot effect).")
    print("  - If delta_eff spikes around some h (often ~3 in your case) -> real model/training mismatch there.")
    print("=========================================\n")


def plot_all(chi_df: pd.DataFrame, delta_df: Optional[pd.DataFrame], title_suffix: str = ""):
    # chi_F plot
    plt.figure(figsize=(9, 5), dpi=140)
    plt.plot(chi_df["h"], chi_df["chiF_overlap"], marker="o", markersize=3, linestyle="-")
    plt.xlabel(r"Transverse field $h$")
    plt.ylabel(r"Fidelity susceptibility $\chi_F$")
    plt.title("TFIM ED chi_F scan" + title_suffix)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    if delta_df is None:
        return

    # overlaps
    plt.figure(figsize=(10, 6), dpi=140)
    for rep, g in delta_df.groupby("report"):
        g = g.sort_values("h")
        plt.plot(g["h"], g["overlap"], marker="o", markersize=4, linestyle="-", label=f"{rep}: overlap")
    plt.xlabel(r"Transverse field $h$")
    plt.ylabel(r"Overlap $|\langle \psi_{\rm ED}|\psi_{\rm model}\rangle|$")
    plt.title("Overlap curves" + title_suffix)
    plt.grid(alpha=0.3)
    plt.ylim(0.90, 1.001)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # delta_eff
    plt.figure(figsize=(10, 6), dpi=140)
    for rep, g in delta_df.groupby("report"):
        g = g.sort_values("h")
        plt.plot(g["h"], g["delta_eff"], marker="o", markersize=4, linestyle="--", label=f"{rep}: delta_eff")
    plt.xlabel(r"Transverse field $h$")
    plt.ylabel(r"Implied field error $\delta_{\rm eff}(h)$")
    plt.title(r"Implied error $\delta_{\rm eff}(h)=\sqrt{2(1-F)/\chi_F}$" + title_suffix)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--pbc", type=bool, default=True)
    ap.add_argument("--J", type=float, default=-1.0)

    ap.add_argument("--hmin", type=float, default=1.0)
    ap.add_argument("--hmax", type=float, default=4.0)
    ap.add_argument("--chi_npts", type=int, default=100)
    ap.add_argument("--chi_dh", type=float, default=1e-3)

    ap.add_argument("--reports", type=str, nargs="*", default=[],
                    help="Your existing overlap JSON report files to analyze")

    # self-contained training demo
    ap.add_argument("--do_train", action="store_true", default=False)
    ap.add_argument("--pool_samples", type=int, default=50_000,
                    help="samples drawn from ED |psi|^2 per support h to build a pool")
    ap.add_argument("--sample_sizes", type=int, nargs="*", default=[2000, 10000],
                    help="subsample size per support h used for training")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--cond_lr_mult", type=float, default=0.2)
    ap.add_argument("--noise_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    # W-modulation stabilization
    ap.add_argument("--gamma_w_max", type=float, default=0.05)
    ap.add_argument("--beta_w_max", type=float, default=0.0)  # keep 0 unless you know you need shift

    args = ap.parse_args()

    # support/novel points (matches your setup)
    h_support = [1.00, 2.00, 2.50, 3.00, 4.00, 5.00, 6.00, 7.00]
    h_novel   = [1.50, 2.80, 3.20, 3.50, 4.50, 5.50]
    all_eval  = sorted(list(set(h_support + h_novel)))

    # build TFIM
    graph, hilbert, H_sparse = build_tfim(args.L, args.dim, args.pbc, args.J)
    N = graph.n_nodes
    dimH = 1 << N
    print(f"TFIM: L={args.L} dim={args.dim} pbc={args.pbc} N={N} dimH={dimH} J={args.J}")

    # basis states in correct order
    states01 = hilbert_all_states_01(hilbert)  # (dimH, N)
    states01_torch = torch.from_numpy(states01).float()

    # cache for ED states
    cache: Dict[float, np.ndarray] = {}

    # chiF scan range
    hs = np.linspace(args.hmin, args.hmax, args.chi_npts, dtype=np.float64)
    chi_df = compute_chiF_scan(H_sparse, hs, args.chi_dh, cache)
    chi_df.to_csv("tfim_chiF_scan.csv", index=False)
    print("Saved: tfim_chiF_scan.csv")

    # Option A: analyze your report files
    delta_df = None
    report_paths = [Path(p) for p in args.reports] if args.reports else []

    # Option B: self-contained training demo
    if args.do_train:
        print("\n=== Self-contained demo: generate samples from ED and train RBMs ===")
        rng_np = np.random.default_rng(args.seed)

        # generate a big pool for each support h
        samples_pool: Dict[float, np.ndarray] = {}
        for h in h_support:
            hk = float(np.round(h, 12))
            if hk not in cache:
                _, psi = ground_state_only(H_sparse(hk))
                cache[hk] = psi
            print(f"Sampling pool at h={h:.2f} ...")
            samples_pool[h] = sample_from_psi(states01, cache[hk], args.pool_samples, rng_np)

        # run for each subsample size
        demo_reports = []
        for n_samp in args.sample_sizes:
            print(f"\n--- Training with n_samples={n_samp} per support h ---")
            subsampled = {h: samples_pool[h][:n_samp].copy() for h in h_support}
            loader = make_training_loader(subsampled, args.batch_size, shuffle=True, seed=args.seed)

            # Model A: bias-only
            print("[Model A] bias-only")
            modelA = ConditionalRBM_BiasOnly(V=N, H=args.hidden, cond_dim=1, width=64, k=args.k).cpu()
            train_rbm(modelA, loader, args.epochs, args.lr, args.cond_lr_mult, args.seed, args.noise_frac)

            # Model B: stabilized W-modulated
            print("[Model B] W-modulated (stabilized)")
            modelB = ConditionalRBM_WithW_Stable(
                V=N, H=args.hidden, cond_dim=1, width=64, k=args.k,
                gamma_w_max=args.gamma_w_max, beta_w_max=args.beta_w_max
            ).cpu()
            train_rbm(modelB, loader, args.epochs, args.lr, args.cond_lr_mult, args.seed, args.noise_frac)

            # evaluate overlaps
            resA = eval_overlaps(modelA, H_sparse, states01_torch, all_eval, cache)
            resB = eval_overlaps(modelB, H_sparse, states01_torch, all_eval, cache)

            # tag + save
            def tag(res, name):
                out = []
                for r in res:
                    hv = r["h"]
                    typ = "support" if hv in h_support else "novel"
                    out.append({"h": hv, "overlap": r["overlap"], "type": typ})
                return {"report": name, "results": out}

            repA = tag(resA, f"demo_n{n_samp}_A_biasonly")
            repB = tag(resB, f"demo_n{n_samp}_B_wmod")
            demo_reports.append(repA)
            demo_reports.append(repB)

            outA = Path(f"overlap_demo_report_n{n_samp}_A.json")
            outB = Path(f"overlap_demo_report_n{n_samp}_B.json")
            outA.write_text(json.dumps({"results": repA["results"]}, indent=2))
            outB.write_text(json.dumps({"results": repB["results"]}, indent=2))
            print(f"Saved: {outA}")
            print(f"Saved: {outB}")

            # also include in analysis set
            report_paths += [outA, outB]

    if report_paths:
        ov_df = load_overlap_reports(report_paths)
        delta_df = add_delta_eff(ov_df, chi_df)
        delta_df.to_csv("overlap_deltaeff_diagnostic.csv", index=False)
        print("Saved: overlap_deltaeff_diagnostic.csv")

        summarize_dips(delta_df, chi_df)

    plot_all(chi_df, delta_df, title_suffix=f" (L={args.L}, dim={args.dim}, pbc={args.pbc})")


if __name__ == "__main__":
    main()
