#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import load_measurements_txt, load_state_txt, MeasurementDataset, MeasurementLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


##### LINEAR ALGEBRA / BASIS HELPERS #####

def create_unitary_dict() -> Dict[str, torch.Tensor]:
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
    Z = torch.eye(2, dtype=torch.cdouble, device=DEVICE)

    return {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}


def as_complex_unitary(U, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(U):
        if U.shape != (2, 2):
            raise ValueError(f"Expected (2,2) unitary, got {tuple(U.shape)}")
        return U.to(device=device, dtype=torch.cdouble).contiguous()

    U_t = torch.tensor(U, device=device)
    if U_t.shape != (2, 2):
        raise ValueError(f"Expected (2,2) unitary, got {tuple(U_t.shape)}")
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()


def enumerate_rotated_branches(model, basis, states, unitaries=None):
    device = model.device
    basis_seq = list(basis)

    if len(basis_seq) != model.num_visible:
        raise ValueError(f"enumerate_rotated_branches: basis length {len(basis_seq)} != num_visible {model.num_visible}")
    if states.shape[-1] != model.num_visible:
        raise ValueError(f"enumerate_rotated_branches: states width {states.shape[-1]} != num_visible {model.num_visible}")

    rotated_sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(rotated_sites) == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)
        return Ut, v

    src = model.U if unitaries is None else unitaries
    local_unitaries = [as_complex_unitary(src[basis_seq[i]], device).reshape(2, 2).contiguous() for i in rotated_sites]
    Uc = torch.stack(local_unitaries, dim=0)

    num_rotated = len(rotated_sites)
    batch_size = states.shape[0]
    num_branches = 2 ** num_rotated

    combos = model.generate_basis_states(size=num_rotated, device=device)

    v = states.unsqueeze(0).repeat(num_branches, 1, 1)
    v[:, :, rotated_sites] = combos.unsqueeze(1)
    v = v.contiguous()

    inp_sb = states[:, rotated_sites].round().long().T
    outp_csb = v[:, :, rotated_sites].round().long().permute(0, 2, 1)
    inp_csb = inp_sb.unsqueeze(0).expand(num_branches, -1, -1)

    s_idx = torch.arange(num_rotated, device=device).view(1, num_rotated, 1).expand(
        num_branches, num_rotated, batch_size
    )
    sel = Uc[s_idx, inp_csb, outp_csb]
    Ut = sel.prod(dim=1)

    return Ut.to(torch.cdouble), v


##### DATA #####

def load_target_state(psi_path: Path) -> torch.Tensor:
    amps_np, _ = load_state_txt(psi_path)
    return torch.tensor(amps_np, dtype=torch.cdouble, device=DEVICE)


def build_measurement_dataset(measurement_paths: List[Path]) -> MeasurementDataset:
    return MeasurementDataset(
        file_paths=measurement_paths,
        load_fn=load_measurements_txt,
        system_param_keys=None,
    )


def build_measurement_loader(dataset: MeasurementDataset, batch_size: int, shuffle: bool = True):
    return MeasurementLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_training_tensors(measurement_paths: List[Path]):
    pos_chunks = []
    bases_all: List[Tuple[str, ...]] = []
    neg_chunks = []

    for path in measurement_paths:
        values_np, bases_list, _ = load_measurements_txt(path)
        values_t = torch.tensor(values_np, dtype=DTYPE, device=DEVICE)
        basis_t = tuple(bases_list)

        pos_chunks.append(values_t)
        bases_all.extend([basis_t] * values_t.shape[0])

        if all(ch == "Z" for ch in basis_t):
            neg_chunks.append(values_t)

    full_samples = torch.vstack(pos_chunks)
    if not neg_chunks:
        raise ValueError("No Z-only pool found for PCD negative sampling.")
    neg_pool = torch.vstack(neg_chunks)

    return full_samples, bases_all, neg_pool


##### MODEL #####

class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden=None, zero_weights=False, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.device = device
        self.initialize_weights(zero_weights=zero_weights)

    def initialize_weights(self, zero_weights=False):
        gen_tensor = torch.zeros if zero_weights else torch.randn
        scale = 1.0 / np.sqrt(self.num_visible)

        self.weights = nn.Parameter(
            gen_tensor(self.num_hidden, self.num_visible, device=self.device, dtype=DTYPE) * scale,
            requires_grad=True,
        )
        self.visible_bias = nn.Parameter(
            torch.zeros(self.num_visible, device=self.device, dtype=DTYPE),
            requires_grad=True,
        )
        self.hidden_bias = nn.Parameter(
            torch.zeros(self.num_hidden, device=self.device, dtype=DTYPE),
            requires_grad=True,
        )

    def effective_energy(self, v):
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0)
            unsq = True

        v = v.to(self.weights)
        visible_bias_term = torch.matmul(v, self.visible_bias)
        hidden_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(visible_bias_term + hidden_bias_term)
        return out.squeeze(0) if unsq else out

    @torch.no_grad()
    def gibbs_steps(self, k, initial_state, overwrite=False):
        v = initial_state if overwrite else initial_state.clone()
        v = v.to(self.weights)
        h = torch.empty(*v.shape[:-1], self.num_hidden, device=self.device, dtype=DTYPE)

        for _ in range(k):
            h_prob = torch.sigmoid(F.linear(v, self.weights, self.hidden_bias))
            h_prob = torch.nan_to_num(h_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(h_prob, out=h)

            v_prob = torch.sigmoid(F.linear(h, self.weights.t(), self.visible_bias))
            v_prob = torch.nan_to_num(v_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(v_prob, out=v)

        return v


class ComplexRBM(nn.Module):
    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        self.rbm_am = RBM(num_visible, num_hidden, device=self.device)
        self.rbm_ph = RBM(num_visible, num_hidden, device=self.device)

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden

        raw = unitary_dict if unitary_dict is not None else create_unitary_dict()
        self.U = {k: as_complex_unitary(v, self.device) for k, v in raw.items()}

        self._stop_training = False
        self._max_size = 20

    @property
    def stop_training(self):
        return self._stop_training

    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool):
            self._stop_training = new_val
        else:
            raise ValueError("stop_training must be bool")

    def reinitialize_weights(self):
        self.rbm_am.initialize_weights()
        self.rbm_ph.initialize_weights()

    def amplitude(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

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

    def generate_basis_states(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)

        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")

        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def stable_log_overlap_amp2(self, basis: Tuple[str, ...], states: torch.Tensor, eps_rot: float = 1e-12):
        Ut, v = enumerate_rotated_branches(self, basis, states)
        F_am = self.rbm_am.effective_energy(v)
        F_ph = self.rbm_ph.effective_energy(v)

        logmag_total = (-0.5 * F_am) + torch.log(Ut.abs().to(DTYPE).clamp_min(1e-300))
        phase_total = (-0.5 * F_ph).to(torch.cdouble) + torch.angle(Ut).to(torch.cdouble)

        M, _ = torch.max(logmag_total, dim=0, keepdim=True)
        scaled_mag = torch.exp(logmag_total - M)
        contrib = scaled_mag.to(torch.cdouble) * torch.exp(1j * phase_total)
        S_prime = contrib.sum(dim=0)

        S_abs2 = (S_prime.conj() * S_prime).real.to(DTYPE)
        return 2.0 * M.squeeze(0) + torch.log(S_abs2 + eps_rot)

    def exact_positive_batch_loss(self, samples: torch.Tensor, bases_batch: List[Tuple[str, ...]]):
        buckets = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)

        data_term = samples.new_tensor(0.0, dtype=DTYPE)

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            batch_s = samples[idxs_t]

            if all(ch == "Z" for ch in basis_t):
                Epos = self.rbm_am.effective_energy(batch_s)
                data_term = data_term + Epos.sum()
            else:
                log_amp2 = self.stable_log_overlap_amp2(basis_t, batch_s)
                data_term = data_term - log_amp2.sum()

        return data_term / samples.shape[0]

    def exact_log_partition(self, basis_states: torch.Tensor):
        E = self.rbm_am.effective_energy(basis_states)
        return torch.logsumexp(-E, dim=0)

    def exact_dataset_nll(self, full_samples: torch.Tensor, full_bases: List[Tuple[str, ...]], basis_states: torch.Tensor):
        return self.exact_positive_batch_loss(full_samples, full_bases) + self.exact_log_partition(basis_states)


##### METRICS #####

def fidelity(model: ComplexRBM, target: torch.Tensor, basis_states=None):
    if not torch.is_complex(target):
        raise TypeError("fidelity: target must be complex (cdouble)")

    basis_states = model.generate_basis_states() if basis_states is None else basis_states

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
def phase_error_stats(model: ComplexRBM, target: torch.Tensor, basis_states=None, eps=1e-12):
    if basis_states is None:
        basis_states = model.generate_basis_states()

    psi_m = model.psi_complex_normalized(basis_states).reshape(-1).to(torch.cdouble)
    psi_t = target.reshape(-1).to(torch.cdouble)

    psi_m = psi_m / torch.linalg.vector_norm(psi_m)
    psi_t = psi_t / torch.linalg.vector_norm(psi_t)

    ip = torch.sum(psi_t.conj() * psi_m)
    if ip.abs() > 1e-12:
        theta = torch.angle(ip)
    else:
        j = int(torch.argmax(psi_t.abs()))
        theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])

    psi_m_aligned = psi_m * torch.exp(-1j * theta)

    bits = basis_states.round().long()
    one_hot_mask = (bits.sum(dim=1) == 1)
    support_mask = one_hot_mask & (psi_t.abs() > eps)

    phi_t = torch.angle(psi_t[support_mask])
    phi_m = torch.angle(psi_m_aligned[support_mask])

    dphi = torch.remainder(phi_m - phi_t + torch.pi, 2 * torch.pi) - torch.pi
    abs_dphi = dphi.abs()

    return {
        "max_abs_phase_err": float(abs_dphi.max().item()),
        "mean_abs_phase_err": float(abs_dphi.mean().item()),
    }


##### TRAINING #####

def train_complex_rbm_pcd(
    model: ComplexRBM,
    loader,
    neg_pool: torch.Tensor,
    full_samples: torch.Tensor,
    full_bases: List[Tuple[str, ...]],
    *,
    epochs: int,
    lr: float,
    pcd_k: int,
    num_chains: int,
    reset_frac: float,
    log_every: int,
    optimizer_cls=torch.optim.SGD,
    optimizer_kwargs=None,
    target=None,
    basis_states=None,
    grad_clip_norm: float | None = 10.0,
    print_metrics: bool = True,
):
    if model.stop_training:
        return {"epoch": []}

    optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
    params = list(model.rbm_am.parameters()) + list(model.rbm_ph.parameters())
    optimizer = optimizer_cls(params, lr=lr, **optimizer_kwargs)

    history = {
        "epoch": [],
        "Fidelity": [],
        "ExactNLL": [],
        "MaxAbsPhaseErr": [],
        "MeanAbsPhaseErr": [],
        "PosTerm": [],
        "NegTerm": [],
        "GradNormAM": [],
        "GradNormPH": [],
        "NumResets": [],
    }

    if basis_states is None:
        basis_states = model.generate_basis_states()

    neg_pool = neg_pool.to(model.device, dtype=DTYPE)
    full_samples = full_samples.to(model.device, dtype=DTYPE)

    if num_chains <= 0:
        raise ValueError("num_chains must be positive")

    with torch.no_grad():
        init_idx = torch.randint(neg_pool.shape[0], (num_chains,), device=model.device)
        fantasy_particles = neg_pool[init_idx].clone()

    num_resets = int(reset_frac * num_chains)
    if reset_frac > 0.0 and num_resets == 0:
        num_resets = 1

    best_fid = -1.0
    best_state = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        pos_terms_epoch = []
        neg_terms_epoch = []
        grad_am_epoch = []
        grad_ph_epoch = []

        for pos_batch, bases_batch, _ in loader:
            pos_batch = pos_batch.to(model.device, dtype=DTYPE)
            bases_batch = list(bases_batch)

            pos_loss = model.exact_positive_batch_loss(pos_batch, bases_batch)

            with torch.no_grad():
                if num_resets > 0:
                    reset_rows = torch.randperm(num_chains, device=model.device)[:num_resets]
                    reset_idx = torch.randint(neg_pool.shape[0], (num_resets,), device=model.device)
                    fantasy_particles[reset_rows] = neg_pool[reset_idx]

                fantasy_particles = model.rbm_am.gibbs_steps(pcd_k, fantasy_particles, overwrite=True)

            neg_loss = model.rbm_am.effective_energy(fantasy_particles).mean()
            loss = pos_loss - neg_loss

            optimizer.zero_grad()
            loss.backward()

            am_sq = 0.0
            for p in model.rbm_am.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    am_sq += float(torch.sum(g * g).item())

            ph_sq = 0.0
            for p in model.rbm_ph.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    ph_sq += float(torch.sum(g * g).item())

            grad_am_epoch.append(am_sq ** 0.5)
            grad_ph_epoch.append(ph_sq ** 0.5)
            pos_terms_epoch.append(float(pos_loss.detach().item()))
            neg_terms_epoch.append(float(neg_loss.detach().item()))

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)

            optimizer.step()

            if model.stop_training:
                break

        if target is not None and (epoch % log_every == 0):
            with torch.no_grad():
                fid_val = fidelity(model, target, basis_states=basis_states)
                ph_stats = phase_error_stats(model, target, basis_states=basis_states)
                exact_nll = model.exact_dataset_nll(full_samples, full_bases, basis_states)

            history["epoch"].append(epoch)
            history["Fidelity"].append(fid_val)
            history["ExactNLL"].append(float(exact_nll.item()))
            history["MaxAbsPhaseErr"].append(ph_stats["max_abs_phase_err"])
            history["MeanAbsPhaseErr"].append(ph_stats["mean_abs_phase_err"])
            history["PosTerm"].append(float(np.mean(pos_terms_epoch)))
            history["NegTerm"].append(float(np.mean(neg_terms_epoch)))
            history["GradNormAM"].append(float(np.mean(grad_am_epoch)))
            history["GradNormPH"].append(float(np.mean(grad_ph_epoch)))
            history["NumResets"].append(int(num_resets))

            if fid_val > best_fid:
                best_fid = fid_val
                best_epoch = epoch
                best_state = {
                    "rbm_am": {k: v.detach().clone() for k, v in model.rbm_am.state_dict().items()},
                    "rbm_ph": {k: v.detach().clone() for k, v in model.rbm_ph.state_dict().items()},
                }

            if print_metrics:
                print(
                    f"[PCD-{pcd_k:>2}] Epoch {epoch}: "
                    f"Fidelity = {fid_val:.6f} | "
                    f"exactNLL = {float(exact_nll.item()):.6f} | "
                    f"max |Δφ| = {ph_stats['max_abs_phase_err']:.6f} | "
                    f"mean |Δφ| = {ph_stats['mean_abs_phase_err']:.6f} | "
                    f"pos = {history['PosTerm'][-1]:.6f} | "
                    f"neg = {history['NegTerm'][-1]:.6f} | "
                    f"g_am = {history['GradNormAM'][-1]:.6f} | "
                    f"g_ph = {history['GradNormPH'][-1]:.6f} | "
                    f"resets = {num_resets}"
                )

        if model.stop_training:
            break

    if best_state is not None:
        model.rbm_am.load_state_dict(best_state["rbm_am"])
        model.rbm_ph.load_state_dict(best_state["rbm_ph"])
        print(f"[PCD-{pcd_k:>2}] Restored best checkpoint from epoch {best_epoch} with fidelity {best_fid:.6f}")

    return history


##### PLOTTING #####

def plot_training_curves(history, pcd_k: int):
    epochs = history.get("epoch", [])
    if not epochs:
        return

    fig_f, axf = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    axf.plot(epochs, history["Fidelity"], marker="o", label=f"PCD-{pcd_k}")
    axf.set_xlabel("Epoch")
    axf.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
    axf.set_title("PCD tomography - fidelity")
    axf.grid(True, alpha=0.3)
    axf.legend(loc="best")
    fig_f.tight_layout()

    fig_nll, axn = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    axn.plot(epochs, history["ExactNLL"], marker="o", label=f"PCD-{pcd_k}")
    axn.set_xlabel("Epoch")
    axn.set_ylabel("Exact dataset NLL")
    axn.set_title("PCD tomography - exact NLL diagnostic")
    axn.grid(True, alpha=0.3)
    axn.legend(loc="best")
    fig_nll.tight_layout()

    fig_g, axg = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    axg.plot(epochs, history["GradNormAM"], marker="o", label="g_am")
    axg.plot(epochs, history["GradNormPH"], marker="x", linestyle="--", label="g_ph")
    axg.set_xlabel("Epoch")
    axg.set_ylabel("Gradient norm")
    axg.set_title("PCD tomography - gradient diagnostics")
    axg.grid(True, alpha=0.3)
    axg.legend(loc="best", fontsize=8)
    fig_g.tight_layout()


def plot_phase_comparison(model: ComplexRBM, target_state: torch.Tensor, basis_states: torch.Tensor):
    with torch.no_grad():
        psi_m = model.psi_complex_normalized(basis_states).reshape(-1).to(torch.cdouble)
        psi_t = target_state.reshape(-1).to(torch.cdouble)

        psi_m = psi_m / torch.linalg.vector_norm(psi_m)
        psi_t = psi_t / torch.linalg.vector_norm(psi_t)

        ip = torch.sum(psi_t.conj() * psi_m)
        if ip.abs() > 1e-12:
            theta = torch.angle(ip)
        else:
            j = int(torch.argmax(psi_t.abs()))
            theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])

        psi_m_aligned = psi_m * torch.exp(-1j * theta)

        phi_m = torch.angle(psi_m_aligned)
        phi_t = torch.angle(psi_t)

        probs = psi_t.abs().pow(2)
        order = torch.argsort(probs, descending=True)
        cum = torch.cumsum(probs[order], dim=0)

        mass_cut = 0.99
        k_cap = 512
        idx = torch.searchsorted(cum, torch.tensor(mass_cut, device=cum.device)).item()
        k_sel = min(idx + 1, k_cap, probs.numel())
        sel = order[:k_sel]

        phi_m_sel = phi_m[sel]
        phi_t_sel = phi_t[sel]
        phi_diff_sel = torch.remainder(phi_m_sel - phi_t_sel + torch.pi, 2 * torch.pi) - torch.pi

    fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axp.plot(range(sel.numel()), phi_t_sel.cpu().numpy(), marker=".", linestyle="", label="target phase")
    axp.plot(range(sel.numel()), phi_m_sel.cpu().numpy(), marker="x", linestyle="", label="model phase")
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
    axe.set_title("Phase error after global-phase alignment")
    axe.grid(True, alpha=0.3)
    axe.legend()
    fig_e.tight_layout()


##### PROJECT-STYLE ENTRYPOINT #####

def train_experiment_model(
    *,
    measurement_paths: List[Path],
    psi_path: Path,
    num_hidden: int,
    batch_size: int,
    epochs: int,
    lr: float,
    pcd_k: int,
    num_chains: int,
    reset_frac: float,
    log_every: int,
    grad_clip_norm: float | None,
    momentum: float,
    seed: int,
):
    dataset = build_measurement_dataset(measurement_paths)
    loader = build_measurement_loader(dataset, batch_size=batch_size, shuffle=True)
    full_samples, full_bases, neg_pool = build_training_tensors(measurement_paths)
    target_state = load_target_state(psi_path)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ComplexRBM(
        num_visible=dataset.num_qubits,
        num_hidden=num_hidden,
        unitary_dict=create_unitary_dict(),
        device=DEVICE,
    )
    basis_states = model.generate_basis_states()

    history = train_complex_rbm_pcd(
        model,
        loader,
        neg_pool,
        full_samples,
        full_bases,
        epochs=epochs,
        lr=lr,
        pcd_k=pcd_k,
        num_chains=num_chains,
        reset_frac=reset_frac,
        log_every=log_every,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"momentum": momentum},
        target=target_state,
        basis_states=basis_states,
        grad_clip_norm=grad_clip_norm,
        print_metrics=True,
    )

    return model, history, target_state, basis_states


##### MAIN #####

if __name__ == "__main__":
    seed = 1234

    batch_size = 128
    num_chains = 128
    epochs = 150
    lr = 5e-2
    log_every = 5
    grad_clip_norm = 10.0
    momentum = 0.0
    pcd_k = 10
    reset_frac = 0.1

    psi_path = Path("state_vectors/w_phase_state.txt")

    meas_directory = Path("measurements")
    measurement_paths = [
        meas_directory / "w_phase_ZZZZ_5000.txt",
        meas_directory / "w_phase_XXZZ_5000.txt",
        meas_directory / "w_phase_XYZZ_5000.txt",
        meas_directory / "w_phase_ZXXZ_5000.txt",
        meas_directory / "w_phase_ZXYZ_5000.txt",
        meas_directory / "w_phase_ZZXX_5000.txt",
        meas_directory / "w_phase_ZZXY_5000.txt",
    ]

    model, history, target_state, basis_states = train_experiment_model(
        measurement_paths=measurement_paths,
        psi_path=psi_path,
        num_hidden=4,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        pcd_k=pcd_k,
        num_chains=num_chains,
        reset_frac=reset_frac,
        log_every=log_every,
        grad_clip_norm=grad_clip_norm,
        momentum=momentum,
        seed=seed,
    )

    plot_training_curves(history, pcd_k=pcd_k)
    plot_phase_comparison(model, target_state, basis_states)
    plt.show()
