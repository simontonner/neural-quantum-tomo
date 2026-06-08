#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage PCD tomography with amplitude-first warm start.

Stage 1:
- phase RBM initialized to zero and frozen
- train amplitude only on Z-basis data with PCD

Stage 2:
- unfreeze phase RBM
- continue full multi-basis PCD training

This is derived from the current PCD script structure and keeps:
- exact positive term
- PCD estimate only for the amplitude model expectation
- SGD
- exact full-dataset NLL as a diagnostic on the small benchmark
"""

from __future__ import annotations

import sys
from pathlib import Path
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import load_measurements_txt, load_state_txt, MeasurementDataset, MeasurementLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


#### UNITARIES AND BASIS ROTATION HELPERS ####


def create_dict():
    norm = 1.0 / sqrt(2.0)

    X = norm * torch.tensor(
        [[1 + 0j, 1 + 0j],
         [1 + 0j, -1 + 0j]],
        dtype=torch.cdouble,
        device=DEVICE,
    )

    Y = norm * torch.tensor(
        [[1 + 0j, -1j],
         [1 + 0j, 1j]],
        dtype=torch.cdouble,
        device=DEVICE,
    )

    Z = torch.eye(2, dtype=torch.cdouble, device=DEVICE)

    return {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}


def as_complex_unitary(U, device: torch.device):
    if torch.is_tensor(U):
        if U.shape != (2, 2):
            raise ValueError(f"Expected (2,2) unitary, got {tuple(U.shape)}")
        return U.to(device=device, dtype=torch.cdouble).contiguous()

    U_t = torch.tensor(U, device=device)
    if U_t.shape != (2, 2):
        raise ValueError(f"Expected (2,2) unitary, got {tuple(U_t.shape)}")
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()


def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    device = nn_state.device
    n_vis = nn_state.num_visible
    basis_seq = list(basis)

    if len(basis_seq) != n_vis:
        raise ValueError(f"_rotate_basis_state: basis length {len(basis_seq)} != num_visible {n_vis}")
    if states.shape[-1] != n_vis:
        raise ValueError(f"_rotate_basis_state: states width {states.shape[1]} != num_visible {n_vis}")

    sites = [i for i, b in enumerate(basis_seq) if b != "Z"]

    if len(sites) == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)
        return Ut, v

    src = nn_state.U if unitaries is None else unitaries
    Ulist = [as_complex_unitary(src[basis_seq[i]], device).reshape(2, 2).contiguous() for i in sites]
    Uc = torch.stack(Ulist, dim=0)

    S = len(sites)
    B = states.shape[0]
    C = 2 ** S

    combos = nn_state.generate_hilbert_space(size=S, device=device)

    v = states.unsqueeze(0).repeat(C, 1, 1)
    v[:, :, sites] = combos.unsqueeze(1)
    v = v.contiguous()

    inp_sb = states[:, sites].round().long().T
    outp_csb = v[:, :, sites].round().long().permute(0, 2, 1)
    inp_csb = inp_sb.unsqueeze(0).expand(C, -1, -1)

    s_idx = torch.arange(S, device=device).view(1, S, 1).expand(C, S, B)
    sel = Uc[s_idx, inp_csb, outp_csb]
    Ut = sel.prod(dim=1)

    return Ut.to(torch.cdouble), v


#### BINARY RESTRICTED BOLTZMANN MACHINE ####


class BinaryRBM(nn.Module):
    """Bernoulli/Bernoulli RBM with free energy F(v)."""

    def __init__(self, num_visible, num_hidden=None, zero_weights=False, init_std=None, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.device = device
        self.initialize_parameters(zero_weights=zero_weights, init_std=init_std)

    def initialize_parameters(self, zero_weights=False, init_std=None):
        scale = (1.0 / np.sqrt(self.num_visible)) if init_std is None else float(init_std)
        if zero_weights:
            weight_init = torch.zeros(self.num_hidden, self.num_visible, device=self.device, dtype=DTYPE)
        else:
            weight_init = torch.randn(self.num_hidden, self.num_visible, device=self.device, dtype=DTYPE) * scale

        self.weights = nn.Parameter(weight_init, requires_grad=True)
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
        hid_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(visible_bias_term + hid_bias_term)
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


#### COMPLEX WAVE FUNCTION ####


class ComplexWaveFunction:
    """psi(sigma) = exp(-F_lambda/2) * exp(-i F_mu/2)."""

    def __init__(
        self,
        num_visible,
        num_hidden=None,
        unitary_dict=None,
        init_std=1e-2,
        phase_zero_init=True,
        device: torch.device = DEVICE,
    ):
        self.device = device
        self.rbm_am = BinaryRBM(num_visible, num_hidden, zero_weights=False, init_std=init_std, device=self.device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, zero_weights=phase_zero_init, init_std=init_std, device=self.device)

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden

        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v, self.device) for k, v in raw.items()}

        self._max_size = 20

    def set_phase_trainable(self, is_trainable: bool):
        for p in self.rbm_ph.parameters():
            p.requires_grad_(is_trainable)

    def amplitude_parameters(self):
        return list(self.rbm_am.parameters())

    def full_parameters(self):
        return list(self.rbm_am.parameters()) + list(self.rbm_ph.parameters())

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

    def generate_hilbert_space(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)

        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")

        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def _stable_log_overlap_amp2(self, basis: Tuple[str, ...], states: torch.Tensor, eps_rot: float = 1e-12):
        Ut, v = _rotate_basis_state(self, basis, states)
        F_am = self.rbm_am.effective_energy(v)
        F_ph = self.rbm_ph.effective_energy(v)

        logmag_total = (-0.5 * F_am) + torch.log(Ut.abs().to(DTYPE).clamp_min(1e-300))
        phase_total = (-0.5 * F_ph).to(torch.cdouble) + torch.angle(Ut).to(torch.cdouble)

        M, _ = torch.max(logmag_total, dim=0, keepdim=True)
        scaled_mag = torch.exp(logmag_total - M)
        contrib = scaled_mag.to(torch.cdouble) * torch.exp(1j * phase_total)
        S_prime = contrib.sum(dim=0)

        S_abs2 = (S_prime.conj() * S_prime).real.to(DTYPE)
        log_amp2 = 2.0 * M.squeeze(0) + torch.log(S_abs2 + eps_rot)
        return log_amp2

    def exact_logZ(self, space: torch.Tensor):
        E = self.rbm_am.effective_energy(space)
        return torch.logsumexp(-E, dim=0)

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
                log_amp2 = self._stable_log_overlap_amp2(basis_t, batch_s)
                data_term = data_term - log_amp2.sum()

        B = samples.shape[0]
        return data_term / B

    def exact_dataset_nll(self, samples: torch.Tensor, bases_batch: List[Tuple[str, ...]], space: torch.Tensor):
        return self.exact_positive_batch_loss(samples, bases_batch) + self.exact_logZ(space)

    def _train_stage(
        self,
        stage_name: str,
        loader,
        neg_pool: torch.Tensor,
        fantasy_particles: torch.Tensor,
        full_samples: torch.Tensor,
        full_bases: List[Tuple[str, ...]],
        epochs: int,
        lr: float,
        pcd_k: int,
        reset_frac: float,
        optimizer_cls,
        optimizer_args: Dict,
        target: torch.Tensor,
        space: torch.Tensor,
        grad_clip_norm: float,
        log_every: int,
        params_to_optimize,
        epoch_offset: int,
        history: Dict[str, List[float]],
        best_state: Dict,
        best_fid: float,
        best_epoch: int,
        print_metrics: bool,
    ):
        opt = optimizer_cls(params_to_optimize, lr=lr, **optimizer_args)

        num_chains = fantasy_particles.shape[0]
        n_reset = int(reset_frac * num_chains)
        if reset_frac > 0.0 and n_reset == 0:
            n_reset = 1

        for local_ep in range(1, epochs + 1):
            pos_terms_epoch = []
            neg_terms_epoch = []
            grad_am_epoch = []
            grad_ph_epoch = []

            for pos_batch, bases_batch, _ in loader:
                pos_batch = pos_batch.to(self.device, dtype=DTYPE)
                bases_batch = list(bases_batch)

                pos_loss = self.exact_positive_batch_loss(pos_batch, bases_batch)

                with torch.no_grad():
                    if n_reset > 0:
                        reset_rows = torch.randperm(num_chains, device=self.device)[:n_reset]
                        reset_idx = torch.randint(neg_pool.shape[0], (n_reset,), device=self.device)
                        fantasy_particles[reset_rows] = neg_pool[reset_idx]

                    fantasy_particles = self.rbm_am.gibbs_steps(pcd_k, fantasy_particles, overwrite=True)

                neg_loss = self.rbm_am.effective_energy(fantasy_particles).mean()
                loss = pos_loss - neg_loss

                opt.zero_grad()
                loss.backward()

                am_sq = 0.0
                for p in self.rbm_am.parameters():
                    if p.grad is not None:
                        g = p.grad.detach()
                        am_sq += float(torch.sum(g * g).item())

                ph_sq = 0.0
                for p in self.rbm_ph.parameters():
                    if p.grad is not None:
                        g = p.grad.detach()
                        ph_sq += float(torch.sum(g * g).item())

                grad_am_epoch.append(am_sq ** 0.5)
                grad_ph_epoch.append(ph_sq ** 0.5)
                pos_terms_epoch.append(float(pos_loss.detach().item()))
                neg_terms_epoch.append(float(neg_loss.detach().item()))

                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(list(params_to_optimize), grad_clip_norm)

                opt.step()

            global_ep = epoch_offset + local_ep
            if global_ep % log_every == 0:
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space)
                    ph_stats = phase_error_stats(self, target, space=space)
                    exact_nll = self.exact_dataset_nll(full_samples, full_bases, space=space)

                history["epoch"].append(global_ep)
                history["stage"].append(stage_name)
                history["Fidelity"].append(fid_val)
                history["ExactNLL"].append(float(exact_nll.item()))
                history["MaxAbsPhaseErr"].append(ph_stats["max_abs_phase_err"])
                history["MeanAbsPhaseErr"].append(ph_stats["mean_abs_phase_err"])
                history["PosTerm"].append(float(np.mean(pos_terms_epoch)))
                history["NegTerm"].append(float(np.mean(neg_terms_epoch)))
                history["GradNormAM"].append(float(np.mean(grad_am_epoch)))
                history["GradNormPH"].append(float(np.mean(grad_ph_epoch)))
                history["NumResets"].append(int(n_reset))

                if fid_val > best_fid:
                    best_fid = fid_val
                    best_epoch = global_ep
                    best_state = {
                        "rbm_am": {k: v.detach().clone() for k, v in self.rbm_am.state_dict().items()},
                        "rbm_ph": {k: v.detach().clone() for k, v in self.rbm_ph.state_dict().items()},
                    }

                if print_metrics:
                    print(
                        f"[{stage_name}] Epoch {global_ep}: "
                        f"Fidelity = {fid_val:.6f} | "
                        f"exactNLL = {float(exact_nll.item()):.6f} | "
                        f"max |Δφ| = {ph_stats['max_abs_phase_err']:.6f} | "
                        f"mean |Δφ| = {ph_stats['mean_abs_phase_err']:.6f} | "
                        f"pos = {history['PosTerm'][-1]:.6f} | "
                        f"neg = {history['NegTerm'][-1]:.6f} | "
                        f"g_am = {history['GradNormAM'][-1]:.6f} | "
                        f"g_ph = {history['GradNormPH'][-1]:.6f} | "
                        f"resets = {n_reset}"
                    )

        return fantasy_particles, history, best_state, best_fid, best_epoch

    def fit_two_stage_pcd(
        self,
        loader_stage1,
        loader_stage2,
        neg_pool: torch.Tensor,
        full_samples: torch.Tensor,
        full_bases: List[Tuple[str, ...]],
        stage1_epochs=30,
        stage2_epochs=120,
        lr_stage1=2e-2,
        lr_stage2=1e-2,
        pcd_k=10,
        num_chains=512,
        reset_frac=0.01,
        log_every=5,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=None,
        space=None,
        grad_clip_norm=10.0,
        print_metrics=True,
    ):
        optimizer_args = {} if optimizer_args is None else optimizer_args

        if space is None:
            space = self.generate_hilbert_space()

        neg_pool = neg_pool.to(self.device, dtype=DTYPE)
        full_samples = full_samples.to(self.device, dtype=DTYPE)

        with torch.no_grad():
            init_idx = torch.randint(neg_pool.shape[0], (num_chains,), device=self.device)
            fantasy_particles = neg_pool[init_idx].clone()

        history = {
            "epoch": [],
            "stage": [],
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

        best_fid = -1.0
        best_state = None
        best_epoch = None

        # Stage 1: amplitude only
        self.set_phase_trainable(False)
        fantasy_particles, history, best_state, best_fid, best_epoch = self._train_stage(
            stage_name="stage1_amp_only",
            loader=loader_stage1,
            neg_pool=neg_pool,
            fantasy_particles=fantasy_particles,
            full_samples=full_samples,
            full_bases=full_bases,
            epochs=stage1_epochs,
            lr=lr_stage1,
            pcd_k=pcd_k,
            reset_frac=reset_frac,
            optimizer_cls=optimizer,
            optimizer_args=optimizer_args,
            target=target,
            space=space,
            grad_clip_norm=grad_clip_norm,
            log_every=log_every,
            params_to_optimize=self.amplitude_parameters(),
            epoch_offset=0,
            history=history,
            best_state=best_state,
            best_fid=best_fid,
            best_epoch=best_epoch,
            print_metrics=print_metrics,
        )

        # Stage 2: full training
        self.set_phase_trainable(True)
        fantasy_particles, history, best_state, best_fid, best_epoch = self._train_stage(
            stage_name="stage2_full",
            loader=loader_stage2,
            neg_pool=neg_pool,
            fantasy_particles=fantasy_particles,
            full_samples=full_samples,
            full_bases=full_bases,
            epochs=stage2_epochs,
            lr=lr_stage2,
            pcd_k=pcd_k,
            reset_frac=reset_frac,
            optimizer_cls=optimizer,
            optimizer_args=optimizer_args,
            target=target,
            space=space,
            grad_clip_norm=grad_clip_norm,
            log_every=log_every,
            params_to_optimize=self.full_parameters(),
            epoch_offset=stage1_epochs,
            history=history,
            best_state=best_state,
            best_fid=best_fid,
            best_epoch=best_epoch,
            print_metrics=print_metrics,
        )

        if best_state is not None:
            self.rbm_am.load_state_dict(best_state["rbm_am"])
            self.rbm_ph.load_state_dict(best_state["rbm_ph"])
            print(f"Restored best checkpoint from epoch {best_epoch} with fidelity {best_fid:.6f}")

        return history


#### METRICS ####


@torch.no_grad()
def fidelity(nn_state, target, space=None):
    if not torch.is_complex(target):
        raise TypeError("fidelity: target must be complex (cdouble)")

    space = nn_state.generate_hilbert_space() if space is None else space

    psi = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1).contiguous()

    npsi = torch.linalg.vector_norm(psi)
    nt = torch.linalg.vector_norm(tgt)
    if npsi == 0 or nt == 0:
        return 0.0

    psi_n = psi / npsi
    tgt_n = tgt / nt

    inner = (tgt_n.conj() * psi_n).sum()
    return float(inner.abs().pow(2).real)


@torch.no_grad()
def phase_error_stats(nn_state, target, space=None, eps=1e-12):
    if space is None:
        space = nn_state.generate_hilbert_space()

    psi_m = nn_state.psi_complex_normalized(space).reshape(-1).to(torch.cdouble)
    psi_t = target.reshape(-1).to(torch.cdouble)

    psi_m = psi_m / torch.linalg.vector_norm(psi_m)
    psi_t = psi_t / torch.linalg.vector_norm(psi_t)

    ip = torch.sum(psi_t.conj() * psi_m)
    if ip.abs() > 1e-12:
        theta = torch.angle(ip)
    else:
        j = int(torch.argmax(psi_t.abs()))
        theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])

    psi_m_al = psi_m * torch.exp(-1j * theta)

    bits = space.round().long()
    one_hot_mask = (bits.sum(dim=1) == 1)
    support_mask = one_hot_mask & (psi_t.abs() > eps)

    phi_t = torch.angle(psi_t[support_mask])
    phi_m = torch.angle(psi_m_al[support_mask])

    dphi = torch.remainder(phi_m - phi_t + torch.pi, 2 * torch.pi) - torch.pi
    abs_dphi = dphi.abs()

    return {
        "max_abs_phase_err": float(abs_dphi.max().item()),
        "mean_abs_phase_err": float(abs_dphi.mean().item()),
    }


#### DATA PREP HELPERS ####


def build_full_measurement_tensors(file_paths: List[Path]):
    pos_chunks = []
    bases_all: List[Tuple[str, ...]] = []
    neg_chunks = []

    for path in file_paths:
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


#### RUN SCRIPT ####


if __name__ == "__main__":
    BASE_SEED = 0

    P_BATCH_SIZE_STAGE1 = 128
    P_BATCH_SIZE_STAGE2 = 128
    NUM_CHAINS = 128

    STAGE1_EPOCHS = 30
    STAGE2_EPOCHS = 120

    LR_STAGE1 = 2e-2
    LR_STAGE2 = 1e-2

    LOG_EVERY = 5
    GRAD_CLIP = 10.0
    MOMENTUM = 0.0

    PCD_K = 10
    RESET_FRAC = 0.01

    INIT_STD = 1e-2
    PHASE_ZERO_INIT = True

    psi_path = Path("state_vectors/w_phase_state.txt")
    amps_np, _ = load_state_txt(psi_path)
    target_state = torch.tensor(amps_np, dtype=torch.cdouble, device=DEVICE)

    meas_directory = Path("measurements")
    z_path = meas_directory / "w_phase_ZZZZ_5000.txt"
    all_paths = [
        z_path,
        meas_directory / "w_phase_XXZZ_5000.txt",
        meas_directory / "w_phase_XYZZ_5000.txt",
        meas_directory / "w_phase_ZXXZ_5000.txt",
        meas_directory / "w_phase_ZXYZ_5000.txt",
        meas_directory / "w_phase_ZZXX_5000.txt",
        meas_directory / "w_phase_ZZXY_5000.txt",
    ]

    ds_z = MeasurementDataset(
        file_paths=[z_path],
        load_fn=load_measurements_txt,
        system_param_keys=None,
    )
    loader_z = MeasurementLoader(ds_z, batch_size=P_BATCH_SIZE_STAGE1, shuffle=True)

    ds_all = MeasurementDataset(
        file_paths=all_paths,
        load_fn=load_measurements_txt,
        system_param_keys=None,
    )
    loader_all = MeasurementLoader(ds_all, batch_size=P_BATCH_SIZE_STAGE2, shuffle=True)

    full_samples, full_bases, neg_pool = build_full_measurement_tensors(all_paths)

    U = create_dict()
    nv = ds_all.num_qubits
    nh = nv

    print("\n" + "=" * 84)
    print("Running two-stage PCD experiment")
    print("=" * 84)
    print(f"Stage 1: amplitude only on Z data for {STAGE1_EPOCHS} epochs")
    print(f"Stage 2: full multi-basis training for {STAGE2_EPOCHS} epochs")
    print(f"PCD-k = {PCD_K}, reset_frac = {RESET_FRAC}, init_std = {INIT_STD}, phase_zero_init = {PHASE_ZERO_INIT}")

    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)

    nn_state = ComplexWaveFunction(
        num_visible=nv,
        num_hidden=nh,
        unitary_dict=U,
        init_std=INIT_STD,
        phase_zero_init=PHASE_ZERO_INIT,
        device=DEVICE,
    )
    space = nn_state.generate_hilbert_space()

    history = nn_state.fit_two_stage_pcd(
        loader_stage1=loader_z,
        loader_stage2=loader_all,
        neg_pool=neg_pool,
        full_samples=full_samples,
        full_bases=full_bases,
        stage1_epochs=STAGE1_EPOCHS,
        stage2_epochs=STAGE2_EPOCHS,
        lr_stage1=LR_STAGE1,
        lr_stage2=LR_STAGE2,
        pcd_k=PCD_K,
        num_chains=NUM_CHAINS,
        reset_frac=RESET_FRAC,
        log_every=LOG_EVERY,
        optimizer=torch.optim.SGD,
        optimizer_args={"momentum": MOMENTUM},
        target=target_state,
        space=space,
        grad_clip_norm=GRAD_CLIP,
        print_metrics=True,
    )

    #### FINAL PHASE COMPARISON ####

    with torch.no_grad():
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).to(torch.cdouble)
        psi_t = target_state.reshape(-1).to(torch.cdouble)

        psi_m = psi_m / torch.linalg.vector_norm(psi_m)
        psi_t = psi_t / torch.linalg.vector_norm(psi_t)

        ip = torch.sum(psi_t.conj() * psi_m)
        if ip.abs() > 1e-12:
            theta = torch.angle(ip)
        else:
            j = int(torch.argmax(psi_t.abs()))
            theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])

        psi_m_al = psi_m * torch.exp(-1j * theta)

        phi_m = torch.angle(psi_m_al)
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

    #### PLOTS ####

    fig_f, axf = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    stage1_mask = [s == "stage1_amp_only" for s in history["stage"]]
    stage2_mask = [s == "stage2_full" for s in history["stage"]]

    epoch_arr = np.array(history["epoch"])
    fid_arr = np.array(history["Fidelity"])

    axf.plot(epoch_arr[stage1_mask], fid_arr[stage1_mask], marker="o", label="stage 1")
    axf.plot(epoch_arr[stage2_mask], fid_arr[stage2_mask], marker="o", label="stage 2")
    axf.axvline(STAGE1_EPOCHS, linestyle="--", linewidth=1.0)
    axf.set_xlabel("Epoch")
    axf.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
    axf.set_title("Two-stage PCD tomography - fidelity")
    axf.grid(True, alpha=0.3)
    axf.legend(loc="best")
    fig_f.tight_layout()

    fig_nll, axn = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    nll_arr = np.array(history["ExactNLL"])
    axn.plot(epoch_arr[stage1_mask], nll_arr[stage1_mask], marker="o", label="stage 1")
    axn.plot(epoch_arr[stage2_mask], nll_arr[stage2_mask], marker="o", label="stage 2")
    axn.axvline(STAGE1_EPOCHS, linestyle="--", linewidth=1.0)
    axn.set_xlabel("Epoch")
    axn.set_ylabel("Exact dataset NLL")
    axn.set_title("Two-stage PCD tomography - exact NLL")
    axn.grid(True, alpha=0.3)
    axn.legend(loc="best")
    fig_nll.tight_layout()

    fig_g, axg = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    axg.plot(history["epoch"], history["GradNormAM"], marker="o", label="g_am")
    axg.plot(history["epoch"], history["GradNormPH"], marker="x", linestyle="--", label="g_ph")
    axg.axvline(STAGE1_EPOCHS, linestyle="--", linewidth=1.0)
    axg.set_xlabel("Epoch")
    axg.set_ylabel("Gradient norm")
    axg.set_title("Two-stage PCD tomography - gradient diagnostics")
    axg.grid(True, alpha=0.3)
    axg.legend(loc="best", fontsize=8)
    fig_g.tight_layout()

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

    plt.show()
