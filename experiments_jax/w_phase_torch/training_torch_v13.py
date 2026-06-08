#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean project-style pooled-CD complex RBM tomography.

Design goals:
- keep the working tomography semantics
- use a structure that feels close to the newer project code
- avoid overly argument-heavy helper functions
"""

from __future__ import annotations

from math import ceil, prod, sqrt
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
        "epochs": 30,
        "batch_size": 128,
        "neg_batch_size": 128,
        "log_every": 5,
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
        "am_init_lr": 1e-2,
        "am_final_lr": 1e-4,
        "ph_init_lr": 1e-2,
        "ph_final_lr": 1e-2,
        "falloff": 0.005,
        "mode": "sigmoid",   # "sigmoid" or "constant"
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


def rotate_wavefunction(model, basis, basis_states, psi=None):
    local_ops = [model.U[b].to(device=model.device, dtype=torch.cdouble) for b in basis]
    state = model.psi_complex(basis_states) if psi is None else psi.to(model.device, dtype=torch.cdouble)
    return kron_apply(local_ops, state)


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

        if self.train_samples.shape[0] != len(self.train_bases):
            raise ValueError("Sample count does not match number of basis rows.")

        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("Inconsistent basis widths.")
        self.num_qubits = next(iter(widths))

        if self.num_qubits != self.train_samples.shape[1]:
            raise ValueError("Basis width does not match sample width.")

        z_mask_np = np.array([all(ch == "Z" for ch in row) for row in self.train_bases], dtype=bool)
        self.z_mask = torch.as_tensor(z_mask_np, dtype=torch.bool)
        self._z_indices = self.z_mask.nonzero(as_tuple=False).view(-1)

        if self._z_indices.numel() == 0:
            raise ValueError("No Z-only rows available for negative sampling.")

        counts_by_basis = {}
        for row in self.train_bases:
            counts_by_basis[row] = counts_by_basis.get(row, 0) + 1
        self.counts_by_basis = counts_by_basis
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
    """
    Working reference semantics:
    - pooled positive minibatches
    - Z-only negative minibatches
    """

    def __init__(self, dataset, cfg, rng=None):
        train_cfg = cfg["training"]

        self.ds = dataset
        self.bs = int(train_cfg["batch_size"])
        self.neg_bs = int(train_cfg["neg_batch_size"])
        self.shuffle = bool(train_cfg["shuffle"])
        self.drop_last = bool(train_cfg["drop_last"])
        self.rng = rng

        if self.bs <= 0 or self.neg_bs <= 0:
            raise ValueError("Batch sizes must be positive.")
        if not self.ds.equal_shot_counts:
            raise ValueError("Pooled positive batching assumes equal shot counts per basis.")
        if self.ds.z_indices().numel() == 0:
            raise ValueError("Z-only pool is empty.")

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
        model_cfg = cfg["model"]

        self.device = DEVICE
        self.k = int(model_cfg["k_steps"])
        self.rbm_am = RBM(num_v, model_cfg["num_hidden"])
        self.rbm_ph = RBM(num_v, model_cfg["num_hidden"])
        self.U = create_unitary_dict()

        self.num_v = self.rbm_am.num_v
        self.num_h = self.rbm_am.num_h
        self.num_visible = self.num_v
        self._max_size = 20

        init_std = model_cfg["init_std"]
        self.initialize_weights(std=init_std)

    def initialize_weights(self, std=None):
        self.rbm_am.initialize_weights(std=std)
        self.rbm_ph.initialize_weights(std=std)

    def generate_basis_states(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.num_v if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def amplitude(self, v):
        return (-self.rbm_am.effective_energy(v.to(self.device, dtype=DTYPE))).exp().sqrt()

    def phase(self, v):
        return -0.5 * self.rbm_ph.effective_energy(v.to(self.device, dtype=DTYPE))

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
        cnt_z = 0
        cnt_rot = 0

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            if any(ch != "Z" for ch in basis_t):
                log_amp2 = self.stable_log_overlap_amp2(basis_t, samples[idxs_t])
                loss_rot = loss_rot - log_amp2.sum().to(DTYPE)
                cnt_rot += len(idxs)
            else:
                Epos = self.rbm_am.effective_energy(samples[idxs_t])
                loss_z = loss_z + Epos.sum()
                cnt_z += len(idxs)

        return loss_rot + loss_z, loss_z, loss_rot, cnt_z, cnt_rot

    def negative_phase_loss(self, neg_init):
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(self.k, neg_init, overwrite=True)
        Eneg = self.rbm_am.effective_energy(vk)
        return Eneg.sum(), vk.shape[0]

    def forward(self, batch):
        pos_batch, neg_batch, bases_batch = batch

        L_pos, L_z_only, L_rot_only, cnt_z, cnt_rot = self.positive_phase_loss(pos_batch, bases_batch)
        B_pos = float(pos_batch.shape[0])

        L_neg, B_neg = self.negative_phase_loss(neg_batch)

        pos_term = L_pos / B_pos
        neg_term = L_neg / B_neg
        loss = pos_term - neg_term

        aux = {
            "pos": float(pos_term.detach().item()),
            "neg": float(neg_term.detach().item()),
            "zpos": float((L_z_only / cnt_z).detach().item()) if cnt_z > 0 else float("nan"),
            "rotpos": float((L_rot_only / cnt_rot).detach().item()) if cnt_rot > 0 else float("nan"),
        }
        return loss, aux


##### METRICS #####

@torch.no_grad()
def fidelity(model, target, basis_states):
    psi = model.psi_complex_normalized(basis_states).reshape(-1).contiguous()
    tgt = target.to(device=model.device, dtype=torch.cdouble).reshape(-1).contiguous()

    npsi = torch.linalg.vector_norm(psi)
    nt = torch.linalg.vector_norm(tgt)
    if npsi == 0 or nt == 0:
        return 0.0

    psi_n = psi / npsi
    tgt_n = tgt / nt
    inner = (tgt_n.conj() * psi_n).sum()
    return float(inner.abs().pow(2).real)


@torch.no_grad()
def average_basis_kl(model, target, eval_bases, basis_states):
    tgt = target.to(device=model.device, dtype=torch.cdouble).reshape(-1)
    nt = torch.linalg.vector_norm(tgt)
    if nt == 0:
        return 0.0
    tgt_norm = tgt / nt

    psi_norm = model.psi_complex_normalized(basis_states).reshape(-1)

    eps = 1e-12
    KL_val = 0.0
    for basis in eval_bases:
        tgt_r = rotate_wavefunction(model, basis, basis_states, psi=tgt_norm)
        psi_r = rotate_wavefunction(model, basis, basis_states, psi=psi_norm)

        p = (tgt_r.abs().to(DTYPE) ** 2)
        q = (psi_r.abs().to(DTYPE) ** 2)
        p = (p / p.sum().clamp_min(eps)).clamp_min(eps)
        q = (q / q.sum().clamp_min(eps)).clamp_min(eps)

        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))

    return float((KL_val / len(eval_bases)).item())


##### TRAINING #####

def get_constant_curve(value):
    def fn(step):
        return float(value)
    return fn


def get_sigmoid_curve(high, low, steps, falloff, center=None):
    import math

    if center is None:
        center = steps / 2.0

    def fn(step):
        s = min(step, steps)
        return float(low + (high - low) / (1.0 + math.exp(falloff * (s - center))))
    return fn


def build_schedules(cfg, num_steps):
    sched_cfg = cfg["schedule"]

    if sched_cfg["mode"] == "constant":
        am_schedule = get_constant_curve(sched_cfg["am_init_lr"])
        ph_schedule = get_constant_curve(sched_cfg["ph_init_lr"])
    else:
        am_schedule = get_sigmoid_curve(
            sched_cfg["am_init_lr"],
            sched_cfg["am_final_lr"],
            num_steps,
            sched_cfg["falloff"],
        )
        ph_schedule = get_sigmoid_curve(
            sched_cfg["ph_init_lr"],
            sched_cfg["ph_final_lr"],
            num_steps,
            sched_cfg["falloff"],
        )

    return am_schedule, ph_schedule


def train_loop(model, optimizer_am, optimizer_ph, loader, dataset, cfg):
    train_cfg = cfg["training"]

    num_epochs = train_cfg["epochs"]
    log_every = train_cfg["log_every"]
    grad_clip_norm = train_cfg["grad_clip_norm"]

    basis_states = model.generate_basis_states()
    am_schedule, ph_schedule = build_schedules(cfg, num_epochs * len(loader))

    history = {"epoch": [], "Fidelity": [], "KL": [], "LR_AM": [], "LR_PH": []}
    global_step = 0

    model.train()

    for epoch in range(num_epochs):
        for batch in loader.iter_epoch():
            lr_am = am_schedule(global_step)
            lr_ph = ph_schedule(global_step)

            optimizer_am.param_groups[0]["lr"] = lr_am
            optimizer_ph.param_groups[0]["lr"] = lr_ph

            optimizer_am.zero_grad(set_to_none=True)
            optimizer_ph.zero_grad(set_to_none=True)

            loss, _ = model(batch)
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer_am.step()
            optimizer_ph.step()
            global_step += 1

        if (epoch + 1) % log_every == 0:
            fid = fidelity(model, dataset.target(), basis_states)
            kl = average_basis_kl(model, dataset.target(), dataset.eval_bases(), basis_states)

            history["epoch"].append(epoch + 1)
            history["Fidelity"].append(fid)
            history["KL"].append(kl)
            history["LR_AM"].append(float(lr_am))
            history["LR_PH"].append(float(lr_ph))

            print(
                f"Epoch {epoch + 1}: "
                f"Fidelity = {fid:.6f} | "
                f"KL = {kl:.6f} | "
                f"lr_am = {lr_am:.6f} | "
                f"lr_ph = {lr_ph:.6f}"
            )

    return model, history, basis_states


##### PROJECT-STYLE ENTRYPOINT #####

def train_experiment_model(cfg):
    train_cfg = cfg["training"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["schedule"]

    torch.manual_seed(train_cfg["seed"])

    # Leave rng=None to preserve the original global torch RNG behaviour
    rng = None

    dataset = TomographyDataset(cfg)
    loader = RBMTomographyLoader(dataset, cfg, rng=rng)

    model = ComplexRBM(cfg, num_v=dataset.num_qubits).to(DEVICE)

    optimizer_am = opt_cfg["am_cls"](
        model.rbm_am.parameters(),
        lr=sched_cfg["am_init_lr"],
        **opt_cfg["am_kwargs"],
    )
    optimizer_ph = opt_cfg["ph_cls"](
        model.rbm_ph.parameters(),
        lr=sched_cfg["ph_init_lr"],
        **opt_cfg["ph_kwargs"],
    )

    model, history, basis_states = train_loop(
        model,
        optimizer_am,
        optimizer_ph,
        loader,
        dataset,
        cfg,
    )

    return model, history, dataset, basis_states


##### PLOTTING #####

def plot_phase_comparison(model, target_state, basis_states):
    with torch.no_grad():
        psi_m = model.psi_complex_normalized(basis_states).reshape(-1).to(torch.cdouble)

    psi_t = target_state.reshape(-1).to(torch.cdouble)
    psi_m = psi_m / torch.linalg.vector_norm(psi_m)
    psi_t = psi_t / torch.linalg.vector_norm(psi_t)

    phi_m = torch.angle(psi_m)
    phi_t = torch.angle(psi_t)

    mass_cut = 0.99
    k_cap = 512
    probs = psi_t.abs().pow(2)
    order = torch.argsort(probs, descending=True)
    cum = torch.cumsum(probs[order], dim=0)
    idx = torch.searchsorted(cum, torch.tensor(mass_cut, device=cum.device)).item()
    k_sel = min(idx + 1, k_cap, probs.numel())
    sel = order[:k_sel]

    phi_m_sel = phi_m[sel]
    phi_t_sel = phi_t[sel]
    phi_diff_sel = torch.remainder(phi_m_sel - phi_t_sel + torch.pi, 2 * torch.pi) - torch.pi
    phi_m_sel_shift = phi_m_sel - 0.5 * (phi_diff_sel.min() + phi_diff_sel.max())
    phi_diff_sel = torch.remainder(phi_m_sel_shift - phi_t_sel + torch.pi, 2 * torch.pi) - torch.pi

    fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axp.plot(range(sel.numel()), phi_t_sel.cpu().numpy(), marker=".", linestyle="", label="target phase")
    axp.plot(range(sel.numel()), phi_m_sel_shift.cpu().numpy(), marker="x", linestyle="", label="model phase (shifted)")
    axp.set_xlabel("basis states (sorted by target mass)")
    axp.set_ylabel("phase [rad]")
    axp.set_title("Phase comparison - top 99% mass")
    axp.grid(True, alpha=0.3)
    axp.legend()
    fig_p.tight_layout()

    fig_e, axe = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axe.plot(range(sel.numel()), phi_diff_sel.cpu().numpy(), marker=".", linestyle="", label="Δphase (wrapped)")
    axe.axhline(0.0, linewidth=1.0)
    axe.set_xlabel("basis states (sorted by target mass)")
    axe.set_ylabel("Δphase [rad] in [-π, π]")
    axe.set_title("Phase error after global shift")
    axe.grid(True, alpha=0.3)
    axe.legend()
    fig_e.tight_layout()


def plot_training_curves(history):
    epochs = history.get("epoch", [])
    if not epochs:
        return

    fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=140)
    ax2 = ax1.twinx()

    ax1.plot(epochs, history["Fidelity"], marker="o", label="Fidelity")
    ax2.plot(epochs, history["KL"], marker="s", linestyle="--", label="KL")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
    ax2.set_ylabel(r"KL$(p\,\|\,q)$")
    ax1.set_title("RBM Tomography - pooled autodiff CD")
    ax1.grid(True, alpha=0.3)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()


##### MAIN #####

if __name__ == "__main__":
    model, history, dataset, basis_states = train_experiment_model(CONFIG)
    plot_phase_comparison(model, dataset.target(), basis_states)
    plot_training_curves(history)
    plt.show()
