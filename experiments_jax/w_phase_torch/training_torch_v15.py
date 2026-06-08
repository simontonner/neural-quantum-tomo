#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run 10-seed pooled-CD ComplexRBM tomography with constant learning rate 0.03
up to epoch 20, then plot the multi-seed phase comparison.

Outputs:
- mean model phase with std error bars over the top-99%-mass target support
- mean wrapped phase error with std error bars
- CSV files with per-state phase statistics
"""

from __future__ import annotations

from math import prod, sqrt
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


CONFIG = {
    "data": {
        "train_path": "w_phase_meas_values.txt",
        "train_bases_path": "w_phase_meas_bases.txt",
        "psi_path": "w_phase_state.txt",
        "eval_bases_path": "w_phase_unique_bases.txt",
    },
    "model": {
        "num_hidden": 4,
        "k_steps": 10,
        "init_std": None,
    },
    "training": {
        "epochs": 20,
        "batch_size": 128,
        "neg_batch_size": 128,
        "log_every": 1,
        "shuffle": True,
        "drop_last": False,
        "grad_clip_norm": None,
        "seed": 1234,
    },
    "optimizer": {
        "am_cls": torch.optim.Adam,
        "ph_cls": torch.optim.Adam,
        "am_kwargs": {},
        "ph_kwargs": {},
    },
    "schedule": {
        "am_init_lr": 3e-2,
        "am_final_lr": 3e-2,
        "ph_init_lr": 3e-2,
        "ph_final_lr": 3e-2,
        "falloff": 0.005,
        "mode": "constant",
    },
}


##### LINEAR ALGEBRA / BASIS HELPERS #####

def create_unitary_dict():
    inv_sqrt2 = 1.0 / sqrt(2.0)

    X = inv_sqrt2 * torch.tensor(
        [[1.0 + 0.0j, 1.0 + 0.0j],
         [1.0 + 0.0j, -1.0 + 0.0j]],
        dtype=torch.cdouble,
        device=DEVICE,
    )
    Y = inv_sqrt2 * torch.tensor(
        [[1.0 + 0.0j, 0.0 - 1.0j],
         [1.0 + 0.0j, 0.0 + 1.0j]],
        dtype=torch.cdouble,
        device=DEVICE,
    )
    Z = torch.tensor(
        [[1.0 + 0.0j, 0.0 + 0.0j],
         [0.0 + 0.0j, 1.0 + 0.0j]],
        dtype=torch.cdouble,
        device=DEVICE,
    )
    return {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}


def kron_apply(matrices, x):
    if not all(torch.is_complex(m) for m in matrices):
        raise TypeError("All local operators must be complex.")
    if not torch.is_complex(x):
        raise TypeError("State must be complex.")

    x_cd = x.to(torch.cdouble)
    leading_dim = x_cd.shape[0]
    batch = int(x_cd.numel() // leading_dim)
    y = x_cd.reshape(leading_dim, batch)

    dims = [m.size(-1) for m in matrices]
    if prod(dims) != leading_dim:
        raise ValueError(f"Incompatible sizes: expected leading dim {prod(dims)}, got {leading_dim}")

    left = leading_dim
    for U in reversed(matrices):
        local_dim = U.shape[-1]
        left //= local_dim
        y = y.reshape(left, local_dim, -1)
        y = torch.einsum("ij,ljm->lim", U, y).reshape(left * local_dim, -1)

    return y.reshape(*x_cd.shape)


def enumerate_rotated_branches(model, basis, states):
    rotated_sites = [i for i, b in enumerate(basis) if b != "Z"]
    if len(rotated_sites) == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=model.device)
        return Ut, v

    local_unitaries = [model.U[basis[i]].reshape(2, 2).contiguous() for i in rotated_sites]
    Uc = torch.stack(local_unitaries, dim=0)

    num_rotated = len(rotated_sites)
    batch_size = states.shape[0]
    num_branches = 2 ** num_rotated

    combos = model.generate_basis_states(num_rotated, device=model.device)

    v = states.unsqueeze(0).repeat(num_branches, 1, 1)
    v[:, :, rotated_sites] = combos.unsqueeze(1)
    v = v.contiguous()

    inp_sb = states[:, rotated_sites].round().long().T
    outp_csb = v[:, :, rotated_sites].round().long().permute(0, 2, 1)
    inp_csb = inp_sb.unsqueeze(0).expand(num_branches, -1, -1)

    s_idx = torch.arange(num_rotated, device=model.device).view(1, num_rotated, 1).expand(
        num_branches, num_rotated, batch_size
    )
    sel = Uc[s_idx, inp_csb, outp_csb]
    Ut = sel.prod(dim=1)

    return Ut.to(torch.cdouble), v


##### DATA #####

class TomographyDataset:
    def __init__(self, cfg):
        paths = cfg["data"]

        train_samples_np = np.loadtxt(paths["train_path"], dtype="float32")
        psi_np = np.loadtxt(paths["psi_path"], dtype="float64")
        train_bases_np = np.loadtxt(paths["train_bases_path"], dtype=str)
        eval_bases_np = np.loadtxt(paths["eval_bases_path"], dtype=str, ndmin=1)

        self.train_samples = torch.tensor(train_samples_np, dtype=DTYPE, device=DEVICE)
        self.target_state = torch.tensor(
            psi_np[:, 0] + 1j * psi_np[:, 1],
            dtype=torch.cdouble,
            device=DEVICE,
        )

        self.train_bases = [tuple(row) for row in np.asarray(train_bases_np, dtype=object)]
        self.eval_basis_rows = [tuple(row) for row in np.asarray(eval_bases_np, dtype=object)]

        widths = {len(row) for row in self.train_bases}
        self.num_qubits = next(iter(widths))

        z_mask_np = np.array([all(ch == "Z" for ch in row) for row in self.train_bases], dtype=bool)
        self.z_mask = torch.as_tensor(z_mask_np, dtype=torch.bool)
        self._z_indices = self.z_mask.nonzero(as_tuple=False).view(-1)

        counts_by_basis = {}
        for row in self.train_bases:
            counts_by_basis[row] = counts_by_basis.get(row, 0) + 1
        self.equal_shot_counts = len(set(counts_by_basis.values())) == 1

    def __len__(self):
        return int(self.train_samples.shape[0])

    def z_indices(self):
        return self._z_indices.clone()

    def eval_bases(self):
        return list(self.eval_basis_rows)

    def target(self):
        return self.target_state


class RBMTomographyLoader:
    def __init__(self, dataset, cfg, rng=None):
        train_cfg = cfg["training"]
        self.ds = dataset
        self.bs = int(train_cfg["batch_size"])
        self.neg_bs = int(train_cfg["neg_batch_size"])
        self.shuffle = bool(train_cfg["shuffle"])
        self.drop_last = bool(train_cfg["drop_last"])
        self.rng = rng

        num_samples = len(self.ds)
        self.slice_bounds = [
            (i, i + self.bs)
            for i in range(0, num_samples, self.bs)
            if (not self.drop_last) or ((i + self.bs) <= num_samples)
        ]

    def __len__(self):
        return len(self.slice_bounds)

    def _randperm(self, n):
        if not self.shuffle:
            return torch.arange(n)
        if self.rng is None:
            return torch.randperm(n)
        return torch.randperm(n, generator=self.rng)

    def _randint(self, high, size):
        if self.rng is None:
            return torch.randint(high, size=size)
        return torch.randint(high, size=size, generator=self.rng)

    def iter_epoch(self):
        num_samples = len(self.ds)
        num_batches = len(self)

        perm = self._randperm(num_samples)
        pos_samples_all = self.ds.train_samples[perm].to(DEVICE, dtype=DTYPE)
        pos_bases_all = [self.ds.train_bases[i] for i in perm.detach().cpu().tolist()]

        z_pool = self.ds.z_indices()
        neg_choices = self._randint(z_pool.numel(), size=(num_batches * self.neg_bs,))
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].to(DEVICE, dtype=DTYPE)

        for batch_idx, (start, end) in enumerate(self.slice_bounds):
            pos_batch = pos_samples_all[start:end]
            bases_batch = pos_bases_all[start:end]

            neg_start = batch_idx * self.neg_bs
            neg_end = neg_start + self.neg_bs
            neg_batch = neg_samples_all[neg_start:neg_end]

            yield pos_batch, neg_batch, bases_batch


##### MODEL #####

class RBM(nn.Module):
    def __init__(self, num_v, num_h=None):
        super().__init__()
        self.num_v = int(num_v)
        self.num_h = int(num_h) if num_h else self.num_v
        self.initialize_weights()

    def initialize_weights(self, std=None):
        scale = (1.0 / np.sqrt(self.num_v)) if std is None else float(std)
        self.W = nn.Parameter(torch.randn(self.num_h, self.num_v, device=DEVICE, dtype=DTYPE) * scale)
        self.b = nn.Parameter(torch.zeros(self.num_v, device=DEVICE, dtype=DTYPE))
        self.c = nn.Parameter(torch.zeros(self.num_h, device=DEVICE, dtype=DTYPE))

    def effective_energy(self, v):
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0)
            unsq = True

        v = v.to(self.W)
        visible_bias_term = torch.matmul(v, self.b)
        hidden_bias_term = F.softplus(F.linear(v, self.W, self.c)).sum(-1)
        out = -(visible_bias_term + hidden_bias_term)
        return out.squeeze(0) if unsq else out

    @torch.no_grad()
    def gibbs_steps(self, k, initial_state, overwrite=False):
        v = (initial_state if overwrite else initial_state.clone()).to(self.W)
        h = torch.empty(*v.shape[:-1], self.num_h, device=DEVICE, dtype=DTYPE)

        for _ in range(k):
            h_prob = torch.sigmoid(F.linear(v, self.W, self.c))
            h_prob = torch.nan_to_num(h_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(h_prob, out=h)

            v_prob = torch.sigmoid(F.linear(h, self.W.t(), self.b))
            v_prob = torch.nan_to_num(v_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(v_prob, out=v)

        return v


class ComplexRBM(nn.Module):
    def __init__(self, cfg, num_v):
        super().__init__()
        self.device = DEVICE
        self.k = int(cfg["model"]["k_steps"])
        self.rbm_am = RBM(num_v, cfg["model"]["num_hidden"])
        self.rbm_ph = RBM(num_v, cfg["model"]["num_hidden"])
        self.U = create_unitary_dict()
        self.num_v = self.rbm_am.num_v
        self._max_size = 20
        self.initialize_weights(std=cfg["model"]["init_std"])

    def initialize_weights(self, std=None):
        self.rbm_am.initialize_weights(std=std)
        self.rbm_ph.initialize_weights(std=std)

    def generate_basis_states(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.num_v if size is None else int(size)
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def psi_complex(self, v):
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_complex_normalized(self, v):
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    def stable_log_overlap_amp2(self, basis, states, eps_rot=1e-6):
        Ut, v = enumerate_rotated_branches(self, basis, states)
        F_am = self.rbm_am.effective_energy(v)
        F_ph = self.rbm_ph.effective_energy(v)

        logmag_total = (-0.5 * F_am) + torch.log(Ut.abs().to(DTYPE).clamp_min(1e-300))
        phase_total = (-0.5 * F_ph).to(torch.cdouble) + torch.angle(Ut).to(torch.cdouble)

        M, _ = torch.max(logmag_total, dim=0, keepdim=True)
        scaled_mag = torch.exp((logmag_total - M).to(DTYPE))
        contrib = scaled_mag.to(torch.cdouble) * torch.exp(1j * phase_total)
        S_prime = contrib.sum(dim=0)
        S_abs2 = (S_prime.conj() * S_prime).real.to(DTYPE)
        return (2.0 * M.squeeze(0)).to(DTYPE) + torch.log(S_abs2 + eps_rot)

    def positive_phase_loss(self, samples, bases_batch):
        buckets = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)

        loss_rot = samples.new_tensor(0.0, dtype=DTYPE)
        loss_z = samples.new_tensor(0.0, dtype=DTYPE)

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            if any(ch != "Z" for ch in basis_t):
                log_amp2 = self.stable_log_overlap_amp2(basis_t, samples[idxs_t])
                loss_rot = loss_rot - log_amp2.sum().to(DTYPE)
            else:
                Epos = self.rbm_am.effective_energy(samples[idxs_t])
                loss_z = loss_z + Epos.sum()

        return loss_rot + loss_z

    def negative_phase_loss(self, neg_init):
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(self.k, neg_init, overwrite=True)
        Eneg = self.rbm_am.effective_energy(vk)
        return Eneg.sum(), vk.shape[0]

    def forward(self, batch):
        pos_batch, neg_batch, bases_batch = batch
        L_pos = self.positive_phase_loss(pos_batch, bases_batch)
        B_pos = float(pos_batch.shape[0])
        L_neg, B_neg = self.negative_phase_loss(neg_batch)
        pos_term = L_pos / B_pos
        neg_term = L_neg / B_neg
        return pos_term - neg_term


def train_loop(model, optimizer_am, optimizer_ph, loader, dataset, cfg):
    num_epochs = cfg["training"]["epochs"]
    basis_states = model.generate_basis_states()
    history = {"epoch": [], "Fidelity": []}

    for epoch in range(num_epochs):
        for batch in loader.iter_epoch():
            optimizer_am.zero_grad(set_to_none=True)
            optimizer_ph.zero_grad(set_to_none=True)
            loss = model(batch)
            loss.backward()
            optimizer_am.step()
            optimizer_ph.step()

        with torch.no_grad():
            psi = model.psi_complex_normalized(basis_states).reshape(-1).contiguous()
            tgt = dataset.target().to(device=model.device, dtype=torch.cdouble).reshape(-1).contiguous()
            npsi = torch.linalg.vector_norm(psi)
            nt = torch.linalg.vector_norm(tgt)
            psi_n = psi / npsi
            tgt_n = tgt / nt
            inner = (tgt_n.conj() * psi_n).sum()
            fid = float(inner.abs().pow(2).real)

        history["epoch"].append(epoch + 1)
        history["Fidelity"].append(fid)

    return model, history, dataset, basis_states


def train_experiment_model(cfg):
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    dataset = TomographyDataset(cfg)
    loader = RBMTomographyLoader(dataset, cfg, rng=None)
    model = ComplexRBM(cfg, num_v=dataset.num_qubits).to(DEVICE)

    optimizer_am = cfg["optimizer"]["am_cls"](
        model.rbm_am.parameters(),
        lr=cfg["schedule"]["am_init_lr"],
        **cfg["optimizer"]["am_kwargs"],
    )
    optimizer_ph = cfg["optimizer"]["ph_cls"](
        model.rbm_ph.parameters(),
        lr=cfg["schedule"]["ph_init_lr"],
        **cfg["optimizer"]["ph_kwargs"],
    )

    return train_loop(model, optimizer_am, optimizer_ph, loader, dataset, cfg)


def phase_stats_after_training(model, dataset, basis_states, mass_cut=0.99, k_cap=512):
    with torch.no_grad():
        psi_m = model.psi_complex_normalized(basis_states).reshape(-1).to(torch.cdouble)

    psi_t = dataset.target().reshape(-1).to(torch.cdouble)
    psi_m = psi_m / torch.linalg.vector_norm(psi_m)
    psi_t = psi_t / torch.linalg.vector_norm(psi_t)

    probs = psi_t.abs().pow(2)
    order = torch.argsort(probs, descending=True)
    cum = torch.cumsum(probs[order], dim=0)
    idx = torch.searchsorted(cum, torch.tensor(mass_cut, device=cum.device)).item()
    k_sel = min(idx + 1, k_cap, probs.numel())
    sel = order[:k_sel]

    phi_t_sel = torch.angle(psi_t[sel])
    phi_m_sel = torch.angle(psi_m[sel])

    phi_diff = torch.remainder(phi_m_sel - phi_t_sel + torch.pi, 2 * torch.pi) - torch.pi
    shift = 0.5 * (phi_diff.min() + phi_diff.max())
    phi_m_shift = phi_m_sel - shift
    phi_diff_shift = torch.remainder(phi_m_shift - phi_t_sel + torch.pi, 2 * torch.pi) - torch.pi

    return {
        "sel": sel.detach().cpu().numpy(),
        "target_phase": phi_t_sel.detach().cpu().numpy(),
        "model_phase": phi_m_shift.detach().cpu().numpy(),
        "phase_error": phi_diff_shift.detach().cpu().numpy(),
    }


def run_multiseed_phase_experiment(base_cfg, seeds):
    phase_tables = []
    seed_summaries = []

    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["seed"] = int(seed)

        print(f"Running seed {seed} ...")
        model, history, dataset, basis_states = train_experiment_model(cfg)
        stats = phase_stats_after_training(model, dataset, basis_states)

        epochs = np.array(history["epoch"], dtype=int)
        fidelities = np.array(history["Fidelity"], dtype=float)
        best_idx = int(np.argmax(fidelities))

        seed_summaries.append({
            "seed": seed,
            "best_fidelity": float(fidelities[best_idx]),
            "best_epoch": int(epochs[best_idx]),
            "final_fidelity": float(fidelities[-1]),
        })

        phase_tables.append(pd.DataFrame({
            "seed": seed,
            "rank": np.arange(len(stats["sel"])),
            "state_index": stats["sel"],
            "target_phase": stats["target_phase"],
            "model_phase": stats["model_phase"],
            "phase_error": stats["phase_error"],
        }))

        del model, dataset, basis_states

    return pd.concat(phase_tables, ignore_index=True), pd.DataFrame(seed_summaries)


def plot_phase_aggregate(phase_df, summary_df, out_prefix="multiseed_phase_lr003_bs512_epoch20"):
    grouped = phase_df.groupby(["rank", "state_index"], as_index=False).agg(
        target_phase=("target_phase", "first"),
        mean_model_phase=("model_phase", "mean"),
        std_model_phase=("model_phase", "std"),
        mean_phase_error=("phase_error", "mean"),
        std_phase_error=("phase_error", "std"),
    )
    grouped["std_model_phase"] = grouped["std_model_phase"].fillna(0.0)
    grouped["std_phase_error"] = grouped["std_phase_error"].fillna(0.0)

    x = grouped["rank"].to_numpy()
    target_phase = grouped["target_phase"].to_numpy()
    mean_model_phase = grouped["mean_model_phase"].to_numpy()
    std_model_phase = grouped["std_model_phase"].to_numpy()
    mean_phase_error = grouped["mean_phase_error"].to_numpy()
    std_phase_error = grouped["std_phase_error"].to_numpy()

    fig1, ax1 = plt.subplots(figsize=(8.0, 4.6), dpi=150)
    ax1.plot(x, target_phase, marker=".", linestyle="", label="target phase")
    ax1.errorbar(
        x,
        mean_model_phase,
        yerr=std_model_phase,
        fmt="x",
        capsize=3,
        linestyle="",
        label="model phase mean ± std",
    )
    ax1.set_xlabel("Basis states (sorted by target mass)")
    ax1.set_ylabel("Phase [rad]")
    ax1.set_title("10-seed phase comparison - bs=512, lr=0.03, epoch 20")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(f"{out_prefix}_phase_comparison.png", dpi=200)

    fig2, ax2 = plt.subplots(figsize=(8.0, 4.6), dpi=150)
    ax2.errorbar(
        x,
        mean_phase_error,
        yerr=std_phase_error,
        fmt="o",
        capsize=3,
        linestyle="",
        label="wrapped phase error mean ± std",
    )
    ax2.axhline(0.0, linewidth=1.0)
    ax2.set_xlabel("Basis states (sorted by target mass)")
    ax2.set_ylabel("Δphase [rad] in [-π, π]")
    ax2.set_title("10-seed wrapped phase error - bs=512, lr=0.03, epoch 20")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{out_prefix}_phase_error.png", dpi=200)

    grouped.to_csv(f"{out_prefix}_phase_summary.csv", index=False)
    summary_df.to_csv(f"{out_prefix}_seed_summary.csv", index=False)

    return fig1, fig2


if __name__ == "__main__":
    seeds = list(range(5))
    phase_df, summary_df = run_multiseed_phase_experiment(CONFIG, seeds)

    print("\nPer-seed summary:")
    print(summary_df.to_string(index=False))

    print("\nAggregate:")
    print(f"mean(best fidelity) = {summary_df['best_fidelity'].mean():.6f}")
    print(f"std(best fidelity)  = {summary_df['best_fidelity'].std(ddof=1):.6f}")
    print(f"mean(best epoch)    = {summary_df['best_epoch'].mean():.3f}")
    print(f"mean(final fidelity)= {summary_df['final_fidelity'].mean():.6f}")

    plot_phase_aggregate(phase_df, summary_df)
    plt.show()
