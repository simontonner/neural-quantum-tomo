#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_handling import load_measurements_txt, load_state_txt, MeasurementDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


##### SINGLE-QUBIT UNITARIES (as cdouble) #####
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


##### KRON-APPLY WITHOUT EXPLICIT TENSOR PRODUCT #####
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
        y = torch.einsum('ij,ljm->lim', U, y).reshape(left * ns, -1)

    return y.reshape(*x_cd.shape)


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


##### BASIS-BRANCH ENUMERATION #####
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

    def _positive_phase_loss(self, samples: torch.Tensor, bases_batch: List[Tuple[str, ...]], eps_rot: float = 1e-6):
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
                log_amp2 = self._stable_log_overlap_amp2(basis_t, samples[idxs_t], eps_rot=eps_rot)
                loss_rot = loss_rot - log_amp2.sum().to(DTYPE)
                cnt_rot += len(idxs)
            else:
                Epos = self.rbm_am.effective_energy(samples[idxs_t])
                loss_z = loss_z + Epos.sum()
                cnt_z += len(idxs)

        return loss_rot + loss_z, loss_z, loss_rot, cnt_z, cnt_rot

    def _negative_phase_loss(self, k: int, neg_init: torch.Tensor):
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(k, neg_init, overwrite=True)
        Eneg = self.rbm_am.effective_energy(vk)
        return Eneg.sum(), vk.shape[0]

    def fit(self, loader, epochs=95, k=10, lr=1e-1, log_every=5,
            optimizer=torch.optim.SGD, optimizer_args=None,
            target=None, bases=None, space=None, print_metrics=True):
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

        if space is None:
            space = self.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                pos_batch = pos_batch.to(self.device, dtype=DTYPE)
                neg_batch = neg_batch.to(self.device, dtype=DTYPE)

                L_pos, _, _, _, _ = self._positive_phase_loss(pos_batch, bases_batch)
                B_pos = float(pos_batch.shape[0])

                L_neg, B_neg = self._negative_phase_loss(k, neg_batch)

                pos_term = L_pos / B_pos
                neg_term = L_neg / B_neg
                loss = pos_term - neg_term

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()

                if self.stop_training:
                    break

            if (target is not None) and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space)
                    kl_val = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                if print_metrics:
                    print(f"Epoch {ep}: Fidelity = {fid_val:.6f} | KL = {kl_val:.6f}")

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


##### ADAPTER AROUND MeasurementDataset #####
class MeasurementTomographyDataset:
    """
    Adapter that turns MeasurementDataset-loaded files into the interface
    expected by the pooled tomography trainer.
    """

    def __init__(
        self,
        measurement_paths: Iterable[Path],
        psi_path: Path,
        *,
        samples_per_file: Optional[Iterable[int]] = None,
        device: torch.device = DEVICE,
    ):
        self.device = device

        self.measurement_ds = MeasurementDataset(
            file_paths=measurement_paths,
            load_fn=load_measurements_txt,
            system_param_keys=None,
            samples_per_file=samples_per_file,
        )

        self.train_samples = self.measurement_ds.values.to(torch.float64)

        if self.measurement_ds.bases is not None:
            self._train_bases = list(self.measurement_ds.bases)
        else:
            if self.measurement_ds.implicit_basis is None:
                raise RuntimeError("Dataset has neither per-sample bases nor implicit basis.")
            self._train_bases = [self.measurement_ds.implicit_basis for _ in range(len(self.measurement_ds))]

        self._z_indices = self.measurement_ds.z_mask.nonzero(as_tuple=False).view(-1).to(torch.long)

        amps_np, _ = load_state_txt(psi_path)
        self.target_state = torch.tensor(amps_np, dtype=torch.cdouble, device=device)

        if self.target_state.numel() != (1 << self.measurement_ds.num_qubits):
            raise ValueError("Target state dimension does not match measurement qubit count.")

        counts_by_basis: Dict[Tuple[str, ...], int] = {}
        eval_bases = []
        seen = set()
        for row in self._train_bases:
            counts_by_basis[row] = counts_by_basis.get(row, 0) + 1
            if row not in seen:
                seen.add(row)
                eval_bases.append(row)

        self.counts_by_basis = counts_by_basis
        self.equal_shot_counts = len(set(counts_by_basis.values())) == 1
        self._eval_bases = eval_bases

    def __len__(self):
        return int(self.train_samples.shape[0])

    def num_visible(self) -> int:
        return int(self.measurement_ds.num_qubits)

    def z_indices(self) -> torch.Tensor:
        return self._z_indices.clone()

    def train_bases_as_tuples(self) -> List[Tuple[str, ...]]:
        return list(self._train_bases)

    def eval_bases(self) -> List[Tuple[str, ...]]:
        return list(self._eval_bases)

    def target(self) -> torch.Tensor:
        return self.target_state


class RBMTomographyLoader:
    """
    Pooled positive-batch trainer loader with separate Z-only negative batches.
    """

    def __init__(
        self,
        dataset: MeasurementTomographyDataset,
        pos_batch_size: int = 140,
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
        self._gen: Optional[torch.Generator] = None

        n = self.ds.num_visible()
        bases_seq = self.ds.train_bases_as_tuples()
        if any(len(row) != n for row in bases_seq):
            raise ValueError("RBMTomographyLoader: inconsistent basis widths in dataset")
        if self.ds.z_indices().numel() == 0:
            raise ValueError("RBMTomographyLoader: Z-only pool is empty (need negatives)")
        if not getattr(self.ds, "equal_shot_counts", False):
            raise ValueError(
                "RBMTomographyLoader: pooled-batch objective equivalence assumes equal shot counts per basis."
            )

    def set_seed(self, seed: Optional[int]):
        if seed is None:
            self._gen = None
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            self._gen = g

    def __len__(self):
        return ceil(len(self.ds) / self.pos_bs)

    def iter_epoch(self):
        N = len(self.ds)
        n_batches = ceil(N / self.pos_bs)

        perm = torch.randperm(N, generator=self._gen) if self._gen is not None else torch.randperm(N)
        pos_samples = self.ds.train_samples[perm].to(self.device, dtype=self.dtype)

        bases_list = self.ds.train_bases_as_tuples()
        perm_idx = perm.detach().cpu().tolist()
        pos_bases_perm = [bases_list[i] for i in perm_idx]

        z_pool = self.ds.z_indices()
        pool_len = z_pool.numel()
        if self._gen is None:
            neg_choices = torch.randint(pool_len, size=(n_batches * self.neg_bs,))
        else:
            neg_choices = torch.randint(pool_len, size=(n_batches * self.neg_bs,), generator=self._gen)
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].to(self.device, dtype=self.dtype)

        for b in range(n_batches):
            start = b * self.pos_bs
            end = min(start + self.pos_bs, N)
            pos_batch = pos_samples[start:end]

            nb_start = b * self.neg_bs
            nb_end = nb_start + self.neg_bs
            neg_batch = neg_samples_all[nb_start:nb_end]

            bases_batch = pos_bases_perm[start:end]

            if self.strict:
                if len(bases_batch) != pos_batch.shape[0]:
                    raise RuntimeError("Loader invariant broken: bases_batch length mismatch")
                if pos_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: pos_batch width != num_visible")
                if neg_batch.shape[1] != self.ds.num_visible():
                    raise RuntimeError("Loader invariant broken: neg_batch width != num_visible")

            yield pos_batch, neg_batch, bases_batch


##### RUN SCRIPT #####
if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)

    epochs = 95
    pbs =
    nbs = 512
    lr = 1e-1
    k_cd = 30
    log_every = 5

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

    data = MeasurementTomographyDataset(
        measurement_paths=measurement_paths,
        psi_path=psi_path,
        samples_per_file=None,
        device=DEVICE,
    )

    U = create_dict()
    nv = data.num_visible()
    nh = nv
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, device=DEVICE)
    space = nn_state.generate_hilbert_space()

    loader = RBMTomographyLoader(
        data,
        pos_batch_size=pbs,
        neg_batch_size=nbs,
        device=DEVICE,
        dtype=DTYPE,
    )

    history = nn_state.fit(
        loader=loader,
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

    #### PHASE COMPARISON ####
    with torch.no_grad():
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).to(torch.cdouble)
    psi_t = data.target().reshape(-1).to(torch.cdouble)
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

    #### PHASE AND FIDELITY PLOTS ####
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

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=140)
    ax.plot(history.get("epoch", []), history["Fidelity"], marker="o", label="Fidelity")
    if "KL" in history:
        ax.plot(history.get("epoch", []), history["KL"], marker="s", linestyle="--", label="KL")
    ax.set_xlabel("Epoch")
    ax.set_title("RBM Tomography - adapted pooled CD training")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    plt.show()
