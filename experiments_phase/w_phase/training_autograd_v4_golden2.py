#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CD-k tomography script derived as closely as possible from the exact-SGD benchmark.

Philosophy:
- keep the exact positive term
- replace only the amplitude model expectation by a CD-k estimate
- stay with SGD, like the exact version
- expose strong diagnostics so results can be inspected and tuned

This script can run several CD-k values in one go for direct comparison.
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
    """Enumerate coherent branches for measured batch of states under given basis."""
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

    def __init__(self, num_visible, num_hidden=None, zero_weights=False, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.device = device
        self.initialize_parameters(zero_weights=zero_weights)

    def initialize_parameters(self, zero_weights=False):
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


#### COMPLEX WAVE FUNCTION (AMPLITUDE AND PHASE RBM) ####


class ComplexWaveFunction:
    """psi(sigma) = exp(-F_lambda/2) * exp(-i F_mu/2)."""

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, device: torch.device = DEVICE):
        self.device = device
        self.rbm_am = BinaryRBM(num_visible, num_hidden, device=self.device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=self.device)

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden

        raw = unitary_dict if unitary_dict is not None else create_dict()
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

    def reinitialize_parameters(self):
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

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
        """
        Exact positive-data term only.

        For Z basis:
            contributes F_lambda(s)

        For rotated basis:
            contributes -log |A_r(s^[r])|^2
        """
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
        """Exact full-dataset NLL for diagnostics on the tiny benchmark."""
        return self.exact_positive_batch_loss(samples, bases_batch) + self.exact_logZ(space)

    def fit_cd(
        self,
        loader,
        neg_pool: torch.Tensor,
        full_samples: torch.Tensor,
        full_bases: List[Tuple[str, ...]],
        epochs=80,
        lr=5e-2,
        cd_k=10,
        neg_batch_size=400,
        log_every=5,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=None,
        space=None,
        grad_clip_norm=10.0,
        print_metrics=True,
    ):
        if self.stop_training:
            return {"epoch": []}

        optimizer_args = {} if optimizer_args is None else optimizer_args
        params = list(self.rbm_am.parameters()) + list(self.rbm_ph.parameters())
        opt = optimizer(params, lr=lr, **optimizer_args)

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
        }

        if space is None:
            space = self.generate_hilbert_space()

        neg_pool = neg_pool.to(self.device, dtype=DTYPE)
        full_samples = full_samples.to(self.device, dtype=DTYPE)

        best_fid = -1.0
        best_state = None
        best_epoch = None

        for ep in range(1, epochs + 1):
            pos_terms_epoch = []
            neg_terms_epoch = []
            grad_am_epoch = []
            grad_ph_epoch = []

            for pos_batch, bases_batch, _ in loader:
                pos_batch = pos_batch.to(self.device, dtype=DTYPE)
                bases_batch = list(bases_batch)

                pos_loss = self.exact_positive_batch_loss(pos_batch, bases_batch)

                draw_idx = torch.randint(neg_pool.shape[0], (neg_batch_size,), device=self.device)
                neg_init = neg_pool[draw_idx]

                with torch.no_grad():
                    vk = self.rbm_am.gibbs_steps(cd_k, neg_init, overwrite=False)

                neg_loss = self.rbm_am.effective_energy(vk).mean()

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
                    torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)

                opt.step()

                if self.stop_training:
                    break

            if target is not None and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space)
                    ph_stats = phase_error_stats(self, target, space=space)
                    exact_nll = self.exact_dataset_nll(full_samples, full_bases, space=space)

                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["ExactNLL"].append(float(exact_nll.item()))
                history["MaxAbsPhaseErr"].append(ph_stats["max_abs_phase_err"])
                history["MeanAbsPhaseErr"].append(ph_stats["mean_abs_phase_err"])
                history["PosTerm"].append(float(np.mean(pos_terms_epoch)))
                history["NegTerm"].append(float(np.mean(neg_terms_epoch)))
                history["GradNormAM"].append(float(np.mean(grad_am_epoch)))
                history["GradNormPH"].append(float(np.mean(grad_ph_epoch)))

                if fid_val > best_fid:
                    best_fid = fid_val
                    best_epoch = ep
                    best_state = {
                        "rbm_am": {k: v.detach().clone() for k, v in self.rbm_am.state_dict().items()},
                        "rbm_ph": {k: v.detach().clone() for k, v in self.rbm_ph.state_dict().items()},
                    }

                if print_metrics:
                    print(
                        f"[k={cd_k:>2}] Epoch {ep}: "
                        f"Fidelity = {fid_val:.6f} | "
                        f"exactNLL = {float(exact_nll.item()):.6f} | "
                        f"max |Δφ| = {ph_stats['max_abs_phase_err']:.6f} | "
                        f"mean |Δφ| = {ph_stats['mean_abs_phase_err']:.6f} | "
                        f"pos = {history['PosTerm'][-1]:.6f} | "
                        f"neg = {history['NegTerm'][-1]:.6f} | "
                        f"g_am = {history['GradNormAM'][-1]:.6f} | "
                        f"g_ph = {history['GradNormPH'][-1]:.6f}"
                    )

            if self.stop_training:
                break

        if best_state is not None:
            self.rbm_am.load_state_dict(best_state["rbm_am"])
            self.rbm_ph.load_state_dict(best_state["rbm_ph"])
            print(f"[k={cd_k:>2}] Restored best checkpoint from epoch {best_epoch} with fidelity {best_fid:.6f}")

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
    """
    Build full positive dataset tensors and a Z-only negative pool directly from files.

    This avoids depending on internal attributes of the MeasurementDataset implementation
    for diagnostics and negative sampling.
    """
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
        raise ValueError("No Z-only pool found for CD negative sampling.")
    neg_pool = torch.vstack(neg_chunks)

    return full_samples, bases_all, neg_pool


#### RUN SCRIPT ####


if __name__ == "__main__":
    BASE_SEED = 1234

    P_BATCH_SIZE = 128 #100
    NEG_BATCH_SIZE = 512 #400
    EPOCHS = 80
    LR = 5e-2
    LOG_EVERY = 5
    GRAD_CLIP = 10.0
    MOMENTUM = 0.0
    CD_K_VALUES = [30]

    psi_path = Path("state_vectors/w_phase_state.txt")
    amps_np, _ = load_state_txt(psi_path)
    target_state = torch.tensor(amps_np, dtype=torch.cdouble, device=DEVICE)

    meas_directory = Path("measurements")
    all_paths = [
        meas_directory / "w_phase_ZZZZ_5000.txt",
        meas_directory / "w_phase_XXZZ_5000.txt",
        meas_directory / "w_phase_XYZZ_5000.txt",
        meas_directory / "w_phase_ZXXZ_5000.txt",
        meas_directory / "w_phase_ZXYZ_5000.txt",
        meas_directory / "w_phase_ZZXX_5000.txt",
        meas_directory / "w_phase_ZZXY_5000.txt",
    ]

    ds_all = MeasurementDataset(
        file_paths=all_paths,
        load_fn=load_measurements_txt,
        system_param_keys=None,
    )
    loader_all = MeasurementLoader(ds_all, batch_size=P_BATCH_SIZE, shuffle=True)

    full_samples, full_bases, neg_pool = build_full_measurement_tensors(all_paths)

    U = create_dict()
    nv = ds_all.num_qubits
    nh = nv

    histories: Dict[int, Dict[str, List[float]]] = {}
    models: Dict[int, ComplexWaveFunction] = {}

    for cd_k in CD_K_VALUES:
        print("\n" + "=" * 80)
        print(f"Running CD-k experiment with k = {cd_k}")
        print("=" * 80)

        torch.manual_seed(BASE_SEED)
        np.random.seed(BASE_SEED)

        nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, device=DEVICE)
        space = nn_state.generate_hilbert_space()

        history = nn_state.fit_cd(
            loader=loader_all,
            neg_pool=neg_pool,
            full_samples=full_samples,
            full_bases=full_bases,
            epochs=EPOCHS,
            lr=LR,
            cd_k=cd_k,
            neg_batch_size=NEG_BATCH_SIZE,
            log_every=LOG_EVERY,
            optimizer=torch.optim.SGD,
            optimizer_args={"momentum": MOMENTUM},
            target=target_state,
            space=space,
            grad_clip_norm=GRAD_CLIP,
            print_metrics=True,
        )

        histories[cd_k] = history
        models[cd_k] = nn_state

    #### SELECT BEST K BY PEAK FIDELITY ####

    best_k = None
    best_peak = -1.0
    for k, hist in histories.items():
        if hist["Fidelity"]:
            peak = max(hist["Fidelity"])
            if peak > best_peak:
                best_peak = peak
                best_k = k

    print("\n" + "#" * 80)
    print(f"Best run by peak fidelity: k = {best_k}, peak fidelity = {best_peak:.6f}")
    print("#" * 80)

    #### FINAL PHASE COMPARISON FOR BEST RUN ####

    best_model = models[best_k]
    space = best_model.generate_hilbert_space()

    with torch.no_grad():
        psi_m = best_model.psi_complex_normalized(space).reshape(-1).to(torch.cdouble)
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
    for k, hist in histories.items():
        axf.plot(hist.get("epoch", []), hist["Fidelity"], marker="o", label=f"k = {k}")
    axf.set_xlabel("Epoch")
    axf.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
    axf.set_title("CD-k tomography - fidelity")
    axf.grid(True, alpha=0.3)
    axf.legend(loc="best")
    fig_f.tight_layout()

    fig_nll, axn = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    for k, hist in histories.items():
        axn.plot(hist.get("epoch", []), hist["ExactNLL"], marker="o", label=f"k = {k}")
    axn.set_xlabel("Epoch")
    axn.set_ylabel("Exact dataset NLL")
    axn.set_title("CD-k tomography - exact NLL diagnostic")
    axn.grid(True, alpha=0.3)
    axn.legend(loc="best")
    fig_nll.tight_layout()

    fig_g, axg = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    for k, hist in histories.items():
        axg.plot(hist.get("epoch", []), hist["GradNormAM"], marker="o", label=f"g_am, k={k}")
        axg.plot(hist.get("epoch", []), hist["GradNormPH"], marker="x", linestyle="--", label=f"g_ph, k={k}")
    axg.set_xlabel("Epoch")
    axg.set_ylabel("Gradient norm")
    axg.set_title("CD-k tomography - gradient diagnostics")
    axg.grid(True, alpha=0.3)
    axg.legend(loc="best", fontsize=8)
    fig_g.tight_layout()

    fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axp.plot(range(sel.numel()), phi_t_sel.cpu().numpy(), marker=".", linestyle="", label="target phase")
    axp.plot(range(sel.numel()), phi_m_sel.cpu().numpy(), marker="x", linestyle="", label=f"model phase (best k={best_k})")
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
    axe.set_title(f"Phase error after global-phase alignment (best k={best_k})")
    axe.grid(True, alpha=0.3)
    axe.legend()
    fig_e.tight_layout()

    plt.show()
