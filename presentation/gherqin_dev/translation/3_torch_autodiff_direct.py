# RBM wavefunction — complex-first with autodiff variant
# -----------------------------------------------------------------------------
# - Same guarantees: one device, cdouble, numerically safe.
# - Adds fit_autodiff(): pure PyTorch autodiff path (NLL or fidelity loss).
# - Ideal for small n (exact Hilbert enumeration).
# -----------------------------------------------------------------------------

from math import ceil, sqrt, prod
from itertools import chain
from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


# -------------------------------
# Device & dtypes
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


# -------------------------------
# Standard unitaries
# -------------------------------
def create_dict(**overrides):
    inv_sqrt2 = 1.0 / sqrt(2.0)
    X = inv_sqrt2 * torch.tensor([[1.0+0.0j, 1.0+0.0j],
                                  [1.0+0.0j, -1.0+0.0j]],
                                 dtype=torch.cdouble, device=DEVICE)
    Y = inv_sqrt2 * torch.tensor([[1.0+0.0j, 0.0-1.0j],
                                  [1.0+0.0j, 0.0+1.0j]],
                                 dtype=torch.cdouble, device=DEVICE)
    Z = torch.tensor([[1.0+0.0j, 0.0+0.0j],
                      [0.0+0.0j, 1.0+0.0j]],
                     dtype=torch.cdouble, device=DEVICE)
    U = {"X": X, "Y": Y, "Z": Z}
    for k, v in overrides.items():
        U[k] = as_complex_unitary(v, DEVICE)
    return U


def as_complex_unitary(U, device):
    if torch.is_tensor(U):
        return U.to(device=device, dtype=torch.cdouble).contiguous()
    return torch.tensor(U, device=device, dtype=torch.cdouble).contiguous()


def inverse(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    zz = z.to(torch.cdouble)
    return zz.conj() / (zz.abs().pow(2).clamp_min(eps))


# -------------------------------
# Kronecker apply
# -------------------------------
def _kron_mult(matrices: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    assert all(torch.is_complex(m) for m in matrices)
    x_cd = x.to(torch.cdouble)
    L = x_cd.shape[0]
    y = x_cd.reshape(L, -1)
    left = L
    for U in reversed(matrices):
        ns = U.shape[-1]
        left //= ns
        y = y.reshape(left, ns, -1)
        y = torch.einsum('ij,ljm->lim', U, y).reshape(left * ns, -1)
    return y.reshape(*x_cd.shape)


def rotate_psi(nn_state, basis, space, unitaries=None, psi=None):
    n_vis = nn_state.num_visible
    basis = list(basis)
    if len(basis) != n_vis:
        raise ValueError("basis len mismatch")
    Udict = nn_state.U if unitaries is None else {
        k: as_complex_unitary(v, nn_state.device) for k, v in unitaries.items()
    }
    us = [Udict[b] for b in basis]
    x = psi.to(nn_state.device, torch.cdouble) if psi is not None else nn_state.psi_complex(space)
    return _kron_mult(us, x)


# -------------------------------
# RBM
# -------------------------------
class BinaryRBM(nn.Module):
    def __init__(self, num_visible, num_hidden=None, device=DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.device = device
        self.initialize_parameters()

    def initialize_parameters(self):
        scale = 1.0 / np.sqrt(self.num_visible)
        self.weights = nn.Parameter(scale * torch.randn(self.num_hidden, self.num_visible,
                                                        device=self.device, dtype=DTYPE))
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible, device=self.device, dtype=DTYPE))
        self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden, device=self.device, dtype=DTYPE))

    def effective_energy(self, v):
        if v.dim() < 2: v = v.unsqueeze(0)
        v = v.to(self.device, dtype=DTYPE)
        return -(torch.matmul(v, self.visible_bias)
                 + F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1))


# -------------------------------
# Complex wavefunction
# -------------------------------
class ComplexWaveFunction:
    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, device=DEVICE):
        self.device = device
        self.rbm_am = BinaryRBM(num_visible, num_hidden, device=device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=device)
        self.num_visible = num_visible
        raw = unitary_dict if unitary_dict else create_dict()
        self.U = {k: as_complex_unitary(v, device) for k, v in raw.items()}

    # ------------------
    # Core ψ functions
    # ------------------
    def amplitude(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi_complex(self, v):
        amp = self.amplitude(v)
        ph = self.phase(v)
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
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    # ------------------
    # Autodiff utilities
    # ------------------
    def psi_vector(self, space=None, normalized=True):
        space = self.generate_hilbert_space() if space is None else space
        return (self.psi_complex_normalized(space) if normalized else self.psi_complex(space)
                ).reshape(-1).contiguous()

    def _bitrows_to_index(self, states):
        s = states.round().to(torch.long)
        n = s.shape[-1]
        shifts = torch.arange(n - 1, -1, -1, device=s.device)
        return (s << shifts).sum(dim=-1)

    def _log_probs_for_basis(self, basis, space, psi_vec_norm):
        psi_r = rotate_psi(self, basis, space, psi=psi_vec_norm)
        probs = (psi_r.abs().to(DTYPE)).pow(2)
        probs = probs / probs.sum()
        return probs.clamp_min(1e-18).log()

    def nll_batch(self, samples_batch, bases_batch, space=None):
        space = self.generate_hilbert_space() if space is None else space
        psi_vec = self.psi_vector(space, normalized=True)
        buckets = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)
        nll, total = samples_batch.new_tensor(0.0), 0
        for basis_t, idxs in buckets.items():
            logp_full = self._log_probs_for_basis(basis_t, space, psi_vec)
            idx_t = self._bitrows_to_index(samples_batch[idxs].to(space.device))
            nll += -logp_full[idx_t].sum()
            total += len(idxs)
        return nll / max(total, 1)

    # ------------------
    # Autodiff training
    # ------------------
    def fit_autodiff(self, loader, epochs=100, lr=1e-3, loss_type="nll",
                     target=None, bases=None, space=None, log_every=5,
                     optimizer=torch.optim.Adam, optimizer_args=None,
                     print_metrics=True,
                     metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        assert loss_type in {"nll", "fid"}
        optimizer_args = {} if optimizer_args is None else optimizer_args
        params = list(chain(self.rbm_am.parameters(), self.rbm_ph.parameters()))
        opt = optimizer(params, lr=lr, **optimizer_args)
        history = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []
        space = self.generate_hilbert_space() if space is None else space

        for ep in range(1, epochs + 1):
            for pos_batch, _, bases_batch in loader.iter_epoch():
                opt.zero_grad()
                if loss_type == "nll":
                    L = self.nll_batch(pos_batch.to(self.device, dtype=DTYPE),
                                       bases_batch, space=space)
                else:
                    psi_vec = self.psi_vector(space, normalized=True)
                    tgt = target.to(self.device, torch.cdouble).reshape(-1)
                    inner = (tgt.conj() * psi_vec).sum()
                    L = (1.0 - (inner.abs().pow(2).real))
                L.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()

            if (target is not None) and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space, bases=bases)
                    kl_val = KL(self, target, space=space, bases=bases) if bases is not None else float("nan")
                history["epoch"].append(ep)
                history.setdefault("Fidelity", []).append(fid_val)
                history.setdefault("KL", []).append(kl_val)
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))
        return history


# -------------------------------
# Metrics
# -------------------------------
@torch.no_grad()
def fidelity(nn_state, target, space=None, **kwargs):
    space = nn_state.generate_hilbert_space() if space is None else space
    psi = nn_state.psi_complex_normalized(space).reshape(-1)
    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1)
    npsi, nt = psi.norm(), tgt.norm()
    if npsi == 0 or nt == 0: return 0.0
    psi_n, tgt_n = psi / npsi, tgt / nt
    inner = (tgt_n.conj() * psi_n).sum()
    return float(inner.abs().pow(2).real)


@torch.no_grad()
def KL(nn_state, target, space=None, bases=None, **kwargs):
    if bases is None: raise ValueError("KL needs bases")
    space = nn_state.generate_hilbert_space() if space is None else space
    tgt = target.to(nn_state.device, torch.cdouble).reshape(-1)
    tgt_norm = tgt / tgt.norm()
    psi_norm_cd = nn_state.psi_complex_normalized(space).reshape(-1)
    KL_val, eps = 0.0, 1e-12
    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=tgt_norm)
        psi_r = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)
        nn_probs_r = (psi_r.abs().to(DTYPE))**2
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE))**2
        p_sum, q_sum = tgt_probs_r.sum().clamp_min(eps), nn_probs_r.sum().clamp_min(eps)
        p = (tgt_probs_r / p_sum).clamp_min(eps)
        q = (nn_probs_r / q_sum).clamp_min(eps)
        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))
    return (KL_val / len(bases)).item()


# -------------------------------
# Dataset + Loader (same as before)
# -------------------------------
class TomographyDataset:
    def __init__(self, train_path, psi_path, train_bases_path, bases_path, device=DEVICE):
        self.device = device
        self.train_samples = torch.tensor(np.loadtxt(train_path, dtype="float32"),
                                          dtype=DTYPE, device=device)
        psi_np = np.loadtxt(psi_path, dtype="float64")
        self.target_state = torch.tensor(psi_np[:, 0] + 1j * psi_np[:, 1],
                                         dtype=torch.cdouble, device=device)
        self.train_bases = np.loadtxt(train_bases_path, dtype=str)
        self.bases = np.loadtxt(bases_path, dtype=str, ndmin=1)
        tb = np.asarray(self.train_bases)
        z_mask_np = (tb == "Z").all(axis=1)
        self._z_mask = torch.as_tensor(z_mask_np, dtype=torch.bool)
        self._z_indices = self._z_mask.nonzero(as_tuple=False).view(-1)
        if self.train_samples.shape[0] != len(self.train_bases):
            raise ValueError("TomographyDataset: sample count != basis row count.")
        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("TomographyDataset: inconsistent basis widths.")
        n = next(iter(widths))
        if n != self.train_samples.shape[1]:
            raise ValueError("TomographyDataset: basis width != sample width.")

    def __len__(self): return int(self.train_samples.shape[0])
    def num_visible(self): return int(self.train_samples.shape[1])
    def z_indices(self): return self._z_indices.clone()
    def train_bases_as_tuples(self): return [tuple(r) for r in np.asarray(self.train_bases)]
    def eval_bases(self): return [tuple(r) for r in np.asarray(self.bases)]
    def target(self): return self.target_state


class RBMTomographyLoader:
    def __init__(self, dataset, pos_batch_size=100, neg_batch_size=None,
                 device=DEVICE, dtype=DTYPE):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self.device, self.dtype = device, dtype
        self._gen = None

    def set_seed(self, seed):
        g = torch.Generator(device="cpu"); g.manual_seed(int(seed))
        self._gen = g

    def __len__(self): return ceil(len(self.ds) / self.pos_bs)

    def iter_epoch(self):
        N = len(self.ds)
        n_batches = ceil(N / self.pos_bs)
        perm = torch.randperm(N, generator=self._gen) if self._gen else torch.randperm(N)
        pos_samples = self.ds.train_samples[perm].to(self.device, dtype=self.dtype)
        bases_list = self.ds.train_bases_as_tuples()
        perm_idx = perm.cpu().tolist()
        pos_bases_perm = [bases_list[i] for i in perm_idx]
        z_pool = self.ds.z_indices()
        pool_len = z_pool.numel()
        neg_choices = torch.randint(pool_len, (n_batches * self.neg_bs,),
                                    generator=self._gen)
        neg_rows = z_pool[neg_choices]
        neg_samples_all = self.ds.train_samples[neg_rows].to(self.device, dtype=self.dtype)
        for b in range(n_batches):
            s, e = b*self.pos_bs, min((b+1)*self.pos_bs, N)
            nb_s, nb_e = b*self.neg_bs, (b+1)*self.neg_bs
            yield pos_samples[s:e], neg_samples_all[nb_s:nb_e], pos_bases_perm[s:e]


# -------------------------------
# Main (autodiff + plots)
# -------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    torch.manual_seed(1234)

    # --- data paths ---
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    # --- dataset & model ---
    data = TomographyDataset(train_path, psi_path, train_bases_path, bases_path, device=DEVICE)
    U = create_dict()
    nv = data.num_visible()
    nn_state = ComplexWaveFunction(nv, nv, unitary_dict=U, device=DEVICE)
    loader = RBMTomographyLoader(data, pos_batch_size=100, neg_batch_size=100, device=DEVICE, dtype=DTYPE)

    # --- training ---
    epochs = 70
    lr = 1e-2
    log_every = 5  # collect metrics every 5 epochs
    hist = nn_state.fit_autodiff(
        loader,
        epochs=epochs,
        lr=lr,
        loss_type="nll",          # or "fid"
        target=data.target(),     # enables Fidelity / KL logging
        bases=data.eval_bases(),  # required for KL
        print_metrics=True,
        log_every=log_every,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
    )

    # -------------------------------
    # Plots (phase comparison + metrics)
    # -------------------------------
    # one-hot computational-basis states (exactly one '1')
    full_hs = nn_state.generate_hilbert_space()
    one_hot_mask = (full_hs.sum(dim=1) == 1)
    one_hot_indices = one_hot_mask.nonzero(as_tuple=False).view(-1)
    one_hot_hs = full_hs[one_hot_indices, :]

    # labels
    bitstrings = ["".join(str(int(b)) for b in row) for row in one_hot_hs.cpu().numpy()]
    idx = np.arange(len(one_hot_indices))
    width = 0.35

    # true & predicted phases (wrapped relative to the first)
    tgt_vec = data.target()  # complex vector of length 2**n
    true_phases_raw = torch.angle(tgt_vec[one_hot_indices])
    pred_phases_raw = nn_state.phase(one_hot_hs)

    true_phases_wrapped = (true_phases_raw - true_phases_raw[0]) % (2 * np.pi)
    pred_phases_wrapped = (pred_phases_raw - pred_phases_raw[0]) % (2 * np.pi)

    # --- Phase bar chart (like your example) ---
    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(idx - width/2, true_phases_wrapped.cpu().numpy(), width, alpha=0.7, label=r'$\phi_{\mathrm{true}}$')
    ax.bar(idx + width/2, pred_phases_wrapped.detach().cpu().numpy(), width, alpha=0.7, label=r'$\phi_{\mathrm{predicted}}$')
    ax.set_xlabel("Basis State")
    ax.set_ylabel("Phase (radians)")
    ax.set_title("Phase Comparison: Phase-Augmented $W$ State")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"${b}$" for b in bitstrings], rotation=45)
    ax.set_ylim(0, 2 * np.pi + 0.2)
    ax.legend(frameon=True, framealpha=1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    # --- Training metrics (Fidelity & KL over epochs) ---
    ep = np.array(hist.get("epoch", []))
    fid = np.array(hist.get("Fidelity", []))
    kl = np.array(hist.get("KL", []))

    if ep.size > 0:
        plt.figure(figsize=(8, 4.5))
        if fid.size > 0:
            plt.plot(ep, fid, marker='o', label="Fidelity")
        if kl.size > 0 and not np.isnan(kl).all():
            plt.plot(ep, kl, marker='s', label="KL")
        plt.xlabel("Epoch")
        plt.title("Training Metrics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
