#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


##### DEVICE AND DTYPES #####
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


##### SINGLE-QUBIT UNITARIES #####
def create_dict(**overrides):
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

    U = {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}

    for name, mat in overrides.items():
        U[name] = as_complex_unitary(mat, DEVICE)

    return U


def as_complex_unitary(U, device: torch.device):
    if torch.is_tensor(U):
        if U.dim() != 2 or U.shape != (2, 2):
            raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U.shape)}")
        return U.to(device=device, dtype=torch.cdouble).contiguous()

    U_t = torch.tensor(U, device=device)
    if U_t.dim() != 2 or U_t.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U_t.shape)}")
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()


##### LINEAR-ALGEBRA HELPERS #####
def inverse(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    zz = z.to(torch.cdouble)
    return zz.conj() / (zz.abs().pow(2).clamp_min(eps))


def _kron_mult(matrices: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    assert all(torch.is_complex(m) for m in matrices), "unitaries must be complex"
    if not torch.is_complex(x):
        raise TypeError("x must be complex (cdouble)")

    x_cd = x.to(torch.cdouble)
    L = x_cd.shape[0]
    batch = int(x_cd.numel() // L)
    y = x_cd.reshape(L, batch)

    n = [m.size(-1) for m in matrices]
    if prod(n) != L:
        raise ValueError(f"Incompatible sizes: expected leading dim {prod(n)}, got {L}")

    left = L
    for U in reversed(matrices):
        ns = U.shape[-1]
        left //= ns
        y = y.reshape(left, ns, -1)
        y = torch.einsum("ij,ljm->lim", U, y).reshape(left * ns, -1)

    return y.reshape(*x_cd.shape)


##### ROTATIONS #####
def rotate_psi(nn_state, basis: Iterable[str], space: torch.Tensor,
               unitaries: Optional[dict] = None, psi: Optional[torch.Tensor] = None):
    n_vis = nn_state.num_visible
    basis = list(basis)
    if len(basis) != n_vis:
        raise ValueError(f"rotate_psi: basis length {len(basis)} != num_visible {n_vis}")

    if unitaries is None:
        us = [nn_state.U[b].to(device=nn_state.device, dtype=torch.cdouble) for b in basis]
    else:
        Udict = {k: as_complex_unitary(v, nn_state.device) for k, v in unitaries.items()}
        us = [Udict[b] for b in basis]

    if psi is None:
        x = nn_state.psi_complex(space)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_psi: psi must be complex (cdouble).")
        x = psi.to(device=nn_state.device, dtype=torch.cdouble)

    return _kron_mult(us, x)


def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    device = nn_state.device
    n_vis = nn_state.num_visible
    basis_seq = list(basis)

    if len(basis_seq) != n_vis:
        raise ValueError(f"_rotate_basis_state: basis length {len(basis_seq)} != num_visible {n_vis}")
    if states.shape[-1] != n_vis:
        raise ValueError(f"_rotate_basis_state: states width {states.shape[-1]} != num_visible {n_vis}")

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


##### BINARY RBM #####
class BinaryRBM(nn.Module):
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
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.empty(*v.shape[:-1], self.num_hidden, device=self.device, dtype=DTYPE)

        for _ in range(k):
            h_lin = F.linear(v, self.weights, self.hidden_bias)
            h_prob = torch.sigmoid(h_lin)
            h_prob = torch.nan_to_num(h_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(h_prob, out=h)

            v_lin = F.linear(h, self.weights.t(), self.visible_bias)
            v_prob = torch.sigmoid(v_lin)
            v_prob = torch.nan_to_num(v_prob, nan=0.5, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
            torch.bernoulli(v_prob, out=v)

        return v


##### COMPLEX WAVE FUNCTION #####
class ComplexWaveFunction:
    """
    psi_{lambda,mu}(s) =
      Z_lambda^{-1/2} exp[-F_lambda(s)/2] exp[-i F_mu(s)/2]
    """

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

    @property
    def max_size(self):
        return self._max_size

    def reinitialize_parameters(self):
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

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

    def generate_hilbert_space(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def _stable_log_overlap_amp2(self, basis: Tuple[str, ...], states: torch.Tensor,
                                 eps_rot: float = 1e-6, unitaries=None):
        Ut, v = _rotate_basis_state(self, basis, states, unitaries=unitaries)
        F_am = self.rbm_am.effective_energy(v)
        F_ph = self.rbm_ph.effective_energy(v)

        logmag_total = (-0.5 * F_am) + torch.log(Ut.abs().to(DTYPE).clamp_min(1e-300))
        phase_total = (-0.5 * F_ph).to(torch.cdouble) + torch.angle(Ut).to(torch.cdouble)

        M, _ = torch.max(logmag_total, dim=0, keepdim=True)
        scaled_mag = torch.exp((logmag_total - M).to(DTYPE))
        contrib = scaled_mag.to(torch.cdouble) * torch.exp(1j * phase_total)
        S_prime = contrib.sum(dim=0)
        S_abs2 = (S_prime.conj() * S_prime).real.to(DTYPE)
        log_amp2 = (2.0 * M.squeeze(0)).to(DTYPE) + torch.log(S_abs2 + eps_rot)
        return log_amp2

    def _positive_phase_loss_components(self, samples: torch.Tensor, bases_batch: List[Tuple[str, ...]], eps_rot: float = 1e-6):
        """
        Return per-basis *mean* positive terms.
        This mirrors the thesis objective: average within each basis, then sum over bases.
        """
        buckets: Dict[Tuple[str, ...], List[int]] = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)

        basis_means = {}
        z_terms = []
        rot_terms = []

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            batch = samples[idxs_t]

            if any(ch != "Z" for ch in basis_t):
                log_amp2 = self._stable_log_overlap_amp2(basis_t, batch, eps_rot=eps_rot)
                mean_term = (-log_amp2.mean()).to(DTYPE)
                basis_means[basis_t] = mean_term
                rot_terms.append(float(mean_term.item()))
            else:
                Epos = self.rbm_am.effective_energy(batch)
                mean_term = Epos.mean()
                basis_means[basis_t] = mean_term
                z_terms.append(float(mean_term.item()))

        return basis_means, z_terms, rot_terms

    def _negative_phase_loss(self, k: int, neg_init: torch.Tensor):
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(k, neg_init, overwrite=True)
        Eneg = self.rbm_am.effective_energy(vk)
        return Eneg.sum(), vk.shape[0]

    def fit(
        self,
        loader,
        epochs=150,
        k=10,
        lr=1e-1,
        log_every=5,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=None,
        bases=None,
        space=None,
        print_metrics=True,
    ):
        if self.stop_training:
            return {"epoch": []}

        optimizer_args = {} if optimizer_args is None else optimizer_args
        opt = optimizer(
            [
                {"params": list(self.rbm_am.parameters()), "lr": lr},
                {"params": list(self.rbm_ph.parameters()), "lr": lr},
            ],
            **optimizer_args,
        )

        params = list(self.rbm_am.parameters()) + list(self.rbm_ph.parameters())

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"] = []
            history["KL"] = []
            history["ExactNLL"] = []
            history["ExactNLL_Z"] = []
            history["ExactNLL_Rot"] = []
            history["Z_KL"] = []
            history["RotKLMean"] = []
            history["RotKLMax"] = []
            history["SupportMass"] = []
            history["OffSupportMass"] = []
            history["MaxOffSupportProb"] = []
            history["GradNormAM"] = []
            history["GradNormPH"] = []
            history["PosBasisAvg"] = []
            history["Neg"] = []
            history["ZPos"] = []
            history["RotPos"] = []
            history["LR_AM"] = []
            history["LR_PH"] = []

        if space is None:
            space = self.generate_hilbert_space()

        num_bases = len(loader.ds.eval_bases())

        for ep in range(1, epochs + 1):
            # Practical stabilization used in the final run.
            if ep < 55:
                lr_am = 1e-1
                lr_ph = 1e-1
            elif ep < 75:
                lr_am = 2e-2
                lr_ph = 8e-2
            else:
                lr_am = 5e-3
                lr_ph = 5e-2

            opt.param_groups[0]["lr"] = lr_am
            opt.param_groups[1]["lr"] = lr_ph

            grad_am_epoch = []
            grad_ph_epoch = []
            pos_epoch = []
            neg_epoch = []
            zpos_epoch = []
            rotpos_epoch = []

            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                pos_batch = pos_batch.to(self.device, dtype=DTYPE)
                neg_batch = neg_batch.to(self.device, dtype=DTYPE)

                basis_means, z_terms, rot_terms = self._positive_phase_loss_components(pos_batch, bases_batch)

                if len(basis_means) != num_bases:
                    raise RuntimeError(
                        f"Expected all {num_bases} bases in every positive batch, got {len(basis_means)}."
                    )

                # Thesis-consistent minibatch objective:
                #   L_batch = (1/(R+1)) * sum_r mean_{B_r}[data term]_r - mean_{B~}[model term]
                #
                # This equals the practical objective in the thesis up to a harmless global factor.
                pos_term = torch.stack(list(basis_means.values())).mean()

                L_neg, B_neg = self._negative_phase_loss(k, neg_batch)
                neg_term = L_neg / B_neg

                loss = pos_term - neg_term

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

                grad_am = am_sq ** 0.5
                grad_ph = ph_sq ** 0.5

                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()

                grad_am_epoch.append(grad_am)
                grad_ph_epoch.append(grad_ph)
                pos_epoch.append(float(pos_term.item()))
                neg_epoch.append(float(neg_term.item()))
                if z_terms:
                    zpos_epoch.append(float(np.mean(z_terms)))
                if rot_terms:
                    rotpos_epoch.append(float(np.mean(rot_terms)))

                if self.stop_training:
                    break

            if target is not None and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space)
                    kl_val = KL(self, target, space=space, bases=bases)

                    exact_diag = exact_nll_breakdown(
                        self,
                        loader.ds.train_samples,
                        loader.ds.train_bases_as_tuples(),
                        space=space,
                    )

                    kl_diag = KL_breakdown(
                        self,
                        target,
                        space=space,
                        bases=bases,
                    )

                    support_diag = support_mass_stats(
                        self,
                        target,
                        space=space,
                    )

                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                history["ExactNLL"].append(exact_diag["exact_nll_total"])
                history["ExactNLL_Z"].append(exact_diag["exact_nll_z"])
                history["ExactNLL_Rot"].append(exact_diag["exact_nll_rot"])
                history["Z_KL"].append(kl_diag["z_kl"])
                history["RotKLMean"].append(kl_diag["rot_kl_mean"])
                history["RotKLMax"].append(kl_diag["rot_kl_max"])
                history["SupportMass"].append(support_diag["support_mass"])
                history["OffSupportMass"].append(support_diag["off_support_mass"])
                history["MaxOffSupportProb"].append(support_diag["max_off_support_prob"])
                history["GradNormAM"].append(float(np.mean(grad_am_epoch)) if grad_am_epoch else float("nan"))
                history["GradNormPH"].append(float(np.mean(grad_ph_epoch)) if grad_ph_epoch else float("nan"))
                history["PosBasisAvg"].append(float(np.mean(pos_epoch)) if pos_epoch else float("nan"))
                history["Neg"].append(float(np.mean(neg_epoch)) if neg_epoch else float("nan"))
                history["ZPos"].append(float(np.mean(zpos_epoch)) if zpos_epoch else float("nan"))
                history["RotPos"].append(float(np.mean(rotpos_epoch)) if rotpos_epoch else float("nan"))
                history["LR_AM"].append(float(lr_am))
                history["LR_PH"].append(float(lr_ph))

                if print_metrics:
                    print(
                        f"Epoch {ep}: "
                        f"Fidelity = {fid_val:.6f} | "
                        f"KL = {kl_val:.6f} | "
                        f"exactNLL = {exact_diag['exact_nll_total']:.6f} | "
                        f"zKL = {kl_diag['z_kl']:.6f} | "
                        f"rotKL = {kl_diag['rot_kl_mean']:.6f} | "
                        f"support = {support_diag['support_mass']:.6f} | "
                        f"off = {support_diag['off_support_mass']:.6f} | "
                        f"maxOff = {support_diag['max_off_support_prob']:.6f} | "
                        f"g_am = {history['GradNormAM'][-1]:.6f} | "
                        f"g_ph = {history['GradNormPH'][-1]:.6f} | "
                        f"pos = {history['PosBasisAvg'][-1]:.6f} | "
                        f"neg = {history['Neg'][-1]:.6f} | "
                        f"zpos = {history['ZPos'][-1]:.6f} | "
                        f"rotpos = {history['RotPos'][-1]:.6f} | "
                        f"lr_am = {history['LR_AM'][-1]:.6f} | "
                        f"lr_ph = {history['LR_PH'][-1]:.6f}"
                    )

            if self.stop_training:
                break

        return history


##### METRICS #####
@torch.no_grad()
def fidelity(nn_state, target, space=None, **kwargs):
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
def KL(nn_state, target, space=None, bases=None, **kwargs):
    if bases is None:
        raise ValueError("KL needs bases")
    if not torch.is_complex(target):
        raise TypeError("KL: target must be complex (cdouble)")

    space = nn_state.generate_hilbert_space() if space is None else space

    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1)
    nt = torch.linalg.vector_norm(tgt)
    if nt == 0:
        return 0.0
    tgt_norm = tgt / nt

    psi_norm_cd = nn_state.psi_complex_normalized(space).reshape(-1)

    KL_val = 0.0
    eps = 1e-12

    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=tgt_norm)
        psi_r = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)

        nn_probs_r = (psi_r.abs().to(DTYPE) ** 2)
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE) ** 2)

        p_sum = tgt_probs_r.sum().clamp_min(eps)
        q_sum = nn_probs_r.sum().clamp_min(eps)
        p = (tgt_probs_r / p_sum).clamp_min(eps)
        q = (nn_probs_r / q_sum).clamp_min(eps)

        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))

    return (KL_val / len(bases)).item()


@torch.no_grad()
def exact_nll_breakdown(nn_state, samples, bases_batch, space, eps_rot=1e-12):
    samples = samples.to(nn_state.device, dtype=DTYPE)
    logZ = torch.logsumexp(-nn_state.rbm_am.effective_energy(space), dim=0)

    buckets = {}
    for i, row in enumerate(bases_batch):
        buckets.setdefault(tuple(row), []).append(i)

    total_sum = samples.new_tensor(0.0, dtype=DTYPE)
    z_sum = samples.new_tensor(0.0, dtype=DTYPE)
    rot_sum = samples.new_tensor(0.0, dtype=DTYPE)
    n_total = 0
    n_z = 0
    n_rot = 0

    for basis_t, idxs in buckets.items():
        idxs_t = torch.tensor(idxs, device=samples.device)
        batch = samples[idxs_t]

        if all(ch == "Z" for ch in basis_t):
            Epos = nn_state.rbm_am.effective_energy(batch)
            nll = Epos + logZ
            z_sum += nll.sum()
            total_sum += nll.sum()
            n_z += len(idxs)
            n_total += len(idxs)
        else:
            log_amp2 = nn_state._stable_log_overlap_amp2(basis_t, batch, eps_rot=eps_rot)
            nll = logZ - log_amp2
            rot_sum += nll.sum()
            total_sum += nll.sum()
            n_rot += len(idxs)
            n_total += len(idxs)

    return {
        "exact_nll_total": float((total_sum / max(n_total, 1)).item()),
        "exact_nll_z": float((z_sum / max(n_z, 1)).item()) if n_z > 0 else float("nan"),
        "exact_nll_rot": float((rot_sum / max(n_rot, 1)).item()) if n_rot > 0 else float("nan"),
    }


@torch.no_grad()
def KL_breakdown(nn_state, target, space, bases):
    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1)
    tgt = tgt / torch.linalg.vector_norm(tgt)
    psi = nn_state.psi_complex_normalized(space).reshape(-1)

    eps = 1e-12
    z_kl = float("nan")
    rot_kls = []

    for basis in bases:
        tgt_r = rotate_psi(nn_state, basis, space, psi=tgt)
        psi_r = rotate_psi(nn_state, basis, space, psi=psi)

        p = (tgt_r.abs().to(DTYPE) ** 2)
        q = (psi_r.abs().to(DTYPE) ** 2)

        p = (p / p.sum().clamp_min(eps)).clamp_min(eps)
        q = (q / q.sum().clamp_min(eps)).clamp_min(eps)

        kl = float(torch.sum(p * (torch.log(p) - torch.log(q))).item())

        if all(ch == "Z" for ch in basis):
            z_kl = kl
        else:
            rot_kls.append(kl)

    return {
        "z_kl": z_kl,
        "rot_kl_mean": float(np.mean(rot_kls)) if rot_kls else float("nan"),
        "rot_kl_max": float(np.max(rot_kls)) if rot_kls else float("nan"),
    }


@torch.no_grad()
def support_mass_stats(nn_state, target, space, eps=1e-12):
    psi = nn_state.psi_complex_normalized(space).reshape(-1)
    probs = psi.abs().pow(2)

    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1)
    support = tgt.abs() > eps
    off_support = ~support

    support_mass = float(probs[support].sum().item())
    off_support_mass = float(probs[off_support].sum().item())
    max_off_support_prob = float(probs[off_support].max().item()) if off_support.any() else 0.0

    return {
        "support_mass": support_mass,
        "off_support_mass": off_support_mass,
        "max_off_support_prob": max_off_support_prob,
    }


##### DATASET #####
class TomographyDataset:
    """
    Container for flattened samples, per-sample bases, and target psi.

    Expected files:
      train_path       : shape (N, V) binary 0/1 samples
      train_bases_path : shape (N, V) string basis labels
      bases_path       : shape (R+1, V) unique basis rows
      psi_path         : shape (2^V, 2) real/imag target coefficients
    """

    def __init__(self, train_path, psi_path, train_bases_path, bases_path, device: torch.device = DEVICE):
        self.device = device

        self.train_samples = torch.tensor(np.loadtxt(train_path, dtype="float32"), dtype=DTYPE, device=device)

        psi_np = np.loadtxt(psi_path, dtype="float64")
        self.target_state = torch.tensor(psi_np[:, 0] + 1j * psi_np[:, 1], dtype=torch.cdouble, device=device)

        self.train_bases = np.loadtxt(train_bases_path, dtype=str)
        self.bases = np.loadtxt(bases_path, dtype=str, ndmin=1)

        tb = np.asarray(self.train_bases)
        z_mask_np = (tb == "Z").all(axis=1)
        self._z_mask = torch.as_tensor(z_mask_np, dtype=torch.bool)
        self._z_indices = self._z_mask.nonzero(as_tuple=False).view(-1)

        if self.train_samples.shape[0] != len(self.train_bases):
            raise ValueError("TomographyDataset: sample count != basis row count")
        if self._z_indices.numel() == 0:
            raise ValueError("TomographyDataset: no Z-only rows for negative sampling")

        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("TomographyDataset: inconsistent basis widths")
        n = next(iter(widths))
        if n != self.train_samples.shape[1]:
            raise ValueError("TomographyDataset: basis width != sample width")

    def __len__(self):
        return int(self.train_samples.shape[0])

    def num_visible(self) -> int:
        return int(self.train_samples.shape[1])

    def z_indices(self) -> torch.Tensor:
        return self._z_indices.clone()

    def train_bases_as_tuples(self) -> List[Tuple[str, ...]]:
        return [tuple(row) for row in np.asarray(self.train_bases, dtype=object)]

    def eval_bases(self) -> List[Tuple[str, ...]]:
        return [tuple(row) for row in np.asarray(self.bases, dtype=object)]

    def target(self) -> torch.Tensor:
        return self.target_state


##### THESIS-CONSISTENT LOADER #####
class EqualBasisTomographyLoader:
    """
    Each positive minibatch contains an explicit equal-weight mixture of all basis settings.
    This mirrors the theory directly: average within each basis, then average over bases.

    The loader keeps update frequency similar to the old pooled implementation by using
    roughly `ceil(N / pos_batch_size)` steps per epoch, while cycling independently
    through each basis bucket.
    """

    def __init__(
        self,
        dataset: TomographyDataset,
        pos_batch_size: int = 100,
        neg_batch_size: Optional[int] = None,
        device: torch.device = DEVICE,
        dtype: torch.dtype = DTYPE,
        strict: bool = True,
    ):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self.device = device
        self.dtype = dtype
        self.strict = strict

        self.basis_list = self.ds.eval_bases()
        self.num_bases = len(self.basis_list)
        if self.num_bases <= 0:
            raise ValueError("EqualBasisTomographyLoader: no bases found")

        base = self.pos_bs // self.num_bases
        rem = self.pos_bs % self.num_bases
        self.counts_per_basis = [base + (1 if i < rem else 0) for i in range(self.num_bases)]
        if min(self.counts_per_basis) <= 0:
            raise ValueError("Increase pos_batch_size so every basis gets at least one sample per batch.")

        train_bases = self.ds.train_bases_as_tuples()
        self.indices_by_basis: Dict[Tuple[str, ...], torch.Tensor] = {}
        for b in self.basis_list:
            idxs = [i for i, row in enumerate(train_bases) if row == b]
            if len(idxs) == 0:
                raise ValueError(f"No samples found for basis {b}.")
            self.indices_by_basis[b] = torch.tensor(idxs, dtype=torch.long)

        self.n_steps = ceil(len(self.ds) / self.pos_bs)

    def __len__(self):
        return self.n_steps

    def iter_epoch(self):
        # independent shuffled pools per basis, with wrap-around
        pools = {}
        ptrs = {}
        for b in self.basis_list:
            idxs = self.indices_by_basis[b]
            pools[b] = idxs[torch.randperm(idxs.numel())]
            ptrs[b] = 0

        z_pool = self.ds.z_indices()
        pool_len = z_pool.numel()

        def draw_from_basis(basis, count):
            chunks = []
            drawn = 0

            while drawn < count:
                pool = pools[basis]
                ptr = ptrs[basis]
                remaining = pool.numel() - ptr
                take = min(count - drawn, remaining)

                chunk = pool[ptr:ptr + take]
                chunks.append(chunk)

                ptrs[basis] = ptr + take
                drawn += take

                if ptrs[basis] >= pool.numel():
                    pools[basis] = self.indices_by_basis[basis][
                        torch.randperm(self.indices_by_basis[basis].numel())
                    ]
                    ptrs[basis] = 0

            return torch.cat(chunks, dim=0)

        for _ in range(self.n_steps):
            pos_idx_parts = []
            bases_batch = []

            for b, c in zip(self.basis_list, self.counts_per_basis):
                idxs = draw_from_basis(b, c)
                pos_idx_parts.append(idxs)
                bases_batch.extend([b] * c)

            pos_idxs = torch.cat(pos_idx_parts, dim=0)

            # shuffle within the positive minibatch so bases are mixed in order
            perm = torch.randperm(pos_idxs.numel())
            pos_idxs = pos_idxs[perm]
            bases_batch = [bases_batch[i] for i in perm.tolist()]

            pos_batch = self.ds.train_samples[pos_idxs].to(self.device, dtype=self.dtype)

            neg_choices = torch.randint(pool_len, size=(self.neg_bs,))
            neg_rows = z_pool[neg_choices]
            neg_batch = self.ds.train_samples[neg_rows].to(self.device, dtype=self.dtype)

            if self.strict:
                if len(bases_batch) != pos_batch.shape[0]:
                    raise RuntimeError("Loader invariant broken: bases_batch length mismatch")
                if pos_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: pos_batch width != num_visible")
                if neg_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: neg_batch width != num_visible")

            yield pos_batch, neg_batch, bases_batch


##### MAIN #####
if __name__ == "__main__":
    train_path = "w_phase_meas_values.txt"
    train_bases_path = "w_phase_meas_bases.txt"
    psi_path = "w_phase_state.txt"
    bases_path = "w_phase_unique_bases.txt"

    torch.manual_seed(1234)
    np.random.seed(1234)

    data = TomographyDataset(train_path, psi_path, train_bases_path, bases_path, device=DEVICE)
    U = create_dict()

    nv = data.num_visible()
    nh = nv
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, device=DEVICE)

    epochs = 150
    pbs = 100
    nbs = 100
    lr = 1e-1
    k_cd = 10
    log_every = 5

    loader = EqualBasisTomographyLoader(
        data,
        pos_batch_size=pbs,
        neg_batch_size=nbs,
        device=DEVICE,
        dtype=DTYPE,
    )

    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(
        loader,
        epochs=epochs,
        k=k_cd,
        lr=lr,
        log_every=log_every,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=data.target(),
        bases=data.eval_bases(),
        space=space,
        print_metrics=True,
    )

    with torch.no_grad():
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
        psi_t = data.target().to(device=DEVICE, dtype=torch.cdouble).reshape(-1).contiguous()

        nm = torch.linalg.vector_norm(psi_m)
        nt = torch.linalg.vector_norm(psi_t)
        if nm > 0:
            psi_m = psi_m / nm
        if nt > 0:
            psi_t = psi_t / nt

        ip = torch.sum(psi_t.conj() * psi_m)
        if ip.abs() > 1e-12:
            theta = torch.angle(ip)
        else:
            j = int(torch.argmax(psi_t.abs()))
            theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])
        psi_m_al = psi_m * torch.exp(-1j * theta)

        phi_t = torch.angle(psi_t).cpu().numpy()
        phi_m = torch.angle(psi_m_al).cpu().numpy()
        dphi = np.remainder((phi_m - phi_t) + np.pi, 2.0 * np.pi) - np.pi

        probs = (psi_t.abs() ** 2).cpu().numpy()
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        mass_cut = 0.99
        k_cap = 512
        k_sel = int(min(np.searchsorted(cum, mass_cut) + 1, k_cap, len(order)))
        sel = order[:k_sel]

        fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        axp.plot(range(k_sel), phi_t[sel], marker=".", linestyle="", label="target phase")
        axp.plot(range(k_sel), phi_m[sel], marker="x", linestyle="", label="model phase (aligned)")
        axp.set_xlabel("basis states (sorted by target mass)")
        axp.set_ylabel("phase [rad]")
        axp.set_title("Phase comparison - top 99% mass")
        axp.grid(True, alpha=0.3)
        axp.legend()
        fig_p.tight_layout()

        fig_e, axe = plt.subplots(figsize=(7.2, 3.8), dpi=150)
        axe.plot(range(k_sel), dphi[sel], marker=".", linestyle="", label="Δphase (wrapped)")
        axe.axhline(0.0, linewidth=1.0)
        axe.set_xlabel("basis states (sorted by target mass)")
        axe.set_ylabel("Δphase [rad] in [-π, π]")
        axe.set_title("Phase error (global phase aligned)")
        axe.grid(True, alpha=0.3)
        axe.legend()
        fig_e.tight_layout()

    ep_hist = history.get("epoch", [])
    if ep_hist and ("Fidelity" in history) and ("KL" in history):
        fig, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=140)
        ax2 = ax1.twinx()

        ax1.plot(ep_hist, history["Fidelity"], marker="o", label="Fidelity")
        ax2.plot(ep_hist, history["KL"], marker="s", linestyle="--", label="KL")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
        ax2.set_ylabel(r"KL$(p\,\|\,q)$")
        ax1.set_title("RBM tomography - thesis-consistent basis-balanced training")
        ax1.grid(True, alpha=0.3)

        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")

        fig.tight_layout()

    plt.show()
