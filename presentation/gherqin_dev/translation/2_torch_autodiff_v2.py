# RBM wavefunction — complex-first with improved autodiff
# -----------------------------------------------------------------------------
# - One device end-to-end; all unitaries/ψ in torch.cdouble.
# - Autodiff training via NLL or (1 - fidelity) with target normalization.
# - Numerics: safe logs, clip grads, contiguous complex ops, no CPU<->CUDA hops.
# - Plotting: global-phase alignment via overlap (not per-bar anchoring).
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
# Standard unitaries (as cdouble)
# -------------------------------
def create_dict(**overrides):
    """Build {X,Y,Z} single-qubit unitaries as cdouble; Y matches measurement convention."""
    inv_sqrt2 = 1.0 / sqrt(2.0)
    X = inv_sqrt2 * torch.tensor([[1.0+0.0j,  1.0+0.0j],
                                  [1.0+0.0j, -1.0+0.0j]],
                                 dtype=torch.cdouble, device=DEVICE).contiguous()
    Y = inv_sqrt2 * torch.tensor([[1.0+0.0j,  0.0-1.0j],
                                  [1.0+0.0j,  0.0+1.0j]],
                                 dtype=torch.cdouble, device=DEVICE).contiguous()
    Z = torch.tensor([[1.0+0.0j, 0.0+0.0j],
                      [0.0+0.0j, 1.0+0.0j]],
                     dtype=torch.cdouble, device=DEVICE).contiguous()
    U = {"X": X, "Y": Y, "Z": Z}
    for name, mat in overrides.items():
        U[name] = as_complex_unitary(mat, DEVICE)
    return U


def as_complex_unitary(U, device: torch.device):
    """Return a (2,2) complex (cdouble) matrix on `device`, contiguous."""
    if torch.is_tensor(U):
        if U.dim() != 2 or U.shape != (2, 2):
            raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U.shape)}")
        return U.to(device=device, dtype=torch.cdouble).contiguous()
    U_t = torch.tensor(U, device=device)
    if U_t.dim() != 2 or U_t.shape != (2, 2):
        raise ValueError(f"as_complex_unitary expects (2,2), got {tuple(U_t.shape)}")
    return U_t.to(dtype=torch.cdouble, device=device).contiguous()


def inverse(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Safe complex inverse: conj(z) / max(|z|^2, eps)."""
    zz = z.to(torch.cdouble)
    return zz.conj() / (zz.abs().pow(2).clamp_min(eps))


# -------------------------------
# Kronecker apply without forming the full kron
# -------------------------------
def _kron_mult(matrices: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Apply (⊗_s U_s)·x without materializing the Kronecker; supports batched x.
    `matrices` must be complex; `x` must be complex.
    """
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
    """Rotate ψ into `basis`. If `psi` is given (vector on Hilbert space), use it."""
    n_vis = nn_state.num_visible
    basis = list(basis)
    if len(basis) != n_vis:
        raise ValueError(f"rotate_psi: basis length {len(basis)} != num_visible {n_vis}")

    if unitaries is None:
        us = [nn_state.U[b].to(device=nn_state.device, dtype=torch.cdouble) for b in basis]
    else:
        Udict = {k: as_complex_unitary(v, nn_state.device) for k, v in unitaries.items()}
        us = [Udict[b] for b in basis]

    x = psi.to(device=nn_state.device, dtype=torch.cdouble) if psi is not None else nn_state.psi_complex(space)
    return _kron_mult(us, x)


# -------------------------------
# RBM (Bernoulli/Bernoulli)
# -------------------------------
class BinaryRBM(nn.Module):
    """Minimal Bernoulli/Bernoulli RBM; energies in float64 for stability."""
    def __init__(self, num_visible, num_hidden=None, device: torch.device = DEVICE):
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
        """E(v) = -v·a - Σ_j softplus(b_j + W_j·v). Returns (...,)."""
        if v.dim() < 2:
            v = v.unsqueeze(0)
        v = v.to(self.device, dtype=DTYPE)
        return -(torch.matmul(v, self.visible_bias) +
                 F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1))


# -------------------------------
# Complex wavefunction = amplitude RBM + phase RBM
# -------------------------------
class ComplexWaveFunction:
    """Two real RBMs define magnitude and phase over bitstrings."""
    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, device: torch.device = DEVICE):
        self.device = device
        self.rbm_am = BinaryRBM(num_visible, num_hidden, device=device)
        self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=device)
        self.num_visible = int(num_visible)
        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v, device) for k, v in raw.items()}

    # --- ψ core ---
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

    # --- Utilities ---
    def generate_hilbert_space(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def psi_vector(self, space=None, normalized=True):
        space = self.generate_hilbert_space() if space is None else space
        vec = self.psi_complex_normalized(space) if normalized else self.psi_complex(space)
        return vec.reshape(-1).contiguous()

    def _bitrows_to_index(self, states):
        s = states.round().to(torch.long)
        n = s.shape[-1]
        shifts = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.long)
        return (s << shifts).sum(dim=-1)

    def _log_probs_for_basis(self, basis, space, psi_vec_norm):
        psi_r = rotate_psi(self, basis, space, psi=psi_vec_norm)
        probs = (psi_r.abs().to(DTYPE)).pow(2)
        probs = probs / probs.sum().clamp_min(1e-18)
        return probs.clamp_min(1e-18).log()

    # --- Losses ---
    def nll_batch(self, samples_batch, bases_batch, space=None):
        space = self.generate_hilbert_space() if space is None else space
        psi_vec = self.psi_vector(space, normalized=True)  # fixed across bucket
        # Bucket identical bases for efficiency
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
                    tgt = tgt / tgt.norm().clamp_min(1e-12)  # ensure exact optimum at F=1
                    inner = (tgt.conj() * psi_vec).sum()
                    L = (1.0 - inner.abs().pow(2).real)
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
    if not torch.is_complex(target):
        raise TypeError("fidelity: `target` must be complex (cdouble).")
    space = nn_state.generate_hilbert_space() if space is None else space
    psi = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1).contiguous()
    npsi, nt = psi.norm(), tgt.norm()
    if npsi == 0 or nt == 0:
        return 0.0
    psi_n, tgt_n = psi / npsi, tgt / nt
    inner = (tgt_n.conj() * psi_n).sum()
    return float(inner.abs().pow(2).real)


@torch.no_grad()
def KL(nn_state, target, space=None, bases=None, **kwargs):
    if bases is None:
        raise ValueError("KL needs `bases`")
    if not torch.is_complex(target):
        raise TypeError("KL: `target` must be complex (cdouble).")

    space = nn_state.generate_hilbert_space() if space is None else space
    tgt = target.to(nn_state.device, torch.cdouble).reshape(-1)
    nt = tgt.norm()
    if nt == 0:
        return 0.0
    tgt_norm = tgt / nt

    psi_norm_cd = nn_state.psi_complex_normalized(space).reshape(-1)

    KL_val = 0.0
    eps = 1e-12
    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=tgt_norm)
        psi_r     = rotate_psi(nn_state, basis, space, psi=psi_norm_cd)
        nn_probs_r  = (psi_r.abs().to(DTYPE)) ** 2
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE)) ** 2
        p_sum = tgt_probs_r.sum().clamp_min(eps)
        q_sum = nn_probs_r.sum().clamp_min(eps)
        p = (tgt_probs_r / p_sum).clamp_min(eps)
        q = (nn_probs_r / q_sum).clamp_min(eps)
        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))
    return (KL_val / len(bases)).item()


# -------------------------------
# Dataset + Loader (guardrailed)
# -------------------------------
class TomographyDataset:
    """Minimal dataset for tomography with guardrails & consistent shapes."""
    def __init__(self, train_path, psi_path, train_bases_path, bases_path, device: torch.device = DEVICE):
        self.device = device

        # training samples (N, n)
        self.train_samples = torch.tensor(
            np.loadtxt(train_path, dtype="float32"),
            dtype=DTYPE, device=device
        )

        # target wavefunction (Re, Im) -> complex
        psi_np = np.loadtxt(psi_path, dtype="float64")
        self.target_state = torch.tensor(
            psi_np[:, 0] + 1j * psi_np[:, 1],
            dtype=torch.cdouble, device=device
        )

        # basis metadata (remain on CPU)
        self.train_bases = np.loadtxt(train_bases_path, dtype=str)
        self.bases = np.loadtxt(bases_path, dtype=str, ndmin=1)

        # Invariants
        if self.train_samples.shape[0] != len(self.train_bases):
            raise ValueError("TomographyDataset: sample count != basis row count.")
        widths = {len(row) for row in self.train_bases}
        if len(widths) != 1:
            raise ValueError("TomographyDataset: inconsistent basis widths.")
        n = next(iter(widths))
        if n != self.train_samples.shape[1]:
            raise ValueError("TomographyDataset: basis width != sample width.")

    def __len__(self):
        return int(self.train_samples.shape[0])

    def num_visible(self) -> int:
        return int(self.train_samples.shape[1])

    def train_bases_as_tuples(self) -> List[Tuple[str, ...]]:
        return [tuple(row) for row in np.asarray(self.train_bases)]

    def eval_bases(self) -> List[Tuple[str, ...]]:
        return [tuple(row) for row in np.asarray(self.bases)]

    def target(self) -> torch.Tensor:
        return self.target_state


class RBMTomographyLoader:
    """Yields (pos_batch, neg_batch, bases_batch) per epoch; negatives unused in autodiff but kept for API symmetry."""
    def __init__(self, dataset: TomographyDataset,
                 pos_batch_size: int = 100, neg_batch_size: Optional[int] = None,
                 device: torch.device = DEVICE, dtype: torch.dtype = DTYPE):
        self.ds = dataset
        self.pos_bs = int(pos_batch_size)
        self.neg_bs = int(neg_batch_size or pos_batch_size)
        self.device, self.dtype = device, dtype
        self._gen: Optional[torch.Generator] = None

    def set_seed(self, seed: Optional[int]):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))
        self._gen = g

    def __len__(self):
        return ceil(len(self.ds) / self.pos_bs)

    def iter_epoch(self):
        N = len(self.ds)
        n_batches = ceil(N / self.pos_bs)
        perm = torch.randperm(N, generator=self._gen) if self._gen else torch.randperm(N)
        pos_samples = self.ds.train_samples[perm].to(self.device, dtype=self.dtype)

        bases_list = self.ds.train_bases_as_tuples()
        perm_idx = perm.detach().cpu().tolist()
        pos_bases_perm = [bases_list[i] for i in perm_idx]

        # Keep negatives to mirror explicit API; not used by autodiff losses.
        neg_samples_all = torch.empty(n_batches * self.neg_bs, self.ds.num_visible(),
                                      device=self.device, dtype=self.dtype)

        for b in range(n_batches):
            s, e = b * self.pos_bs, min((b + 1) * self.pos_bs, N)
            nb_s, nb_e = b * self.neg_bs, (b + 1) * self.neg_bs
            yield pos_samples[s:e], neg_samples_all[nb_s:nb_e], pos_bases_perm[s:e]


# -------------------------------
# Main (autodiff + aligned plots)
# -------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = False  # ensure mathtext works without LaTeX

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
    loader = RBMTomographyLoader(
        data, pos_batch_size=100, neg_batch_size=100, device=DEVICE, dtype=DTYPE
    )

    # --- training ---
    epochs = 100
    lr = 1e-2
    log_every = 5
    hist = nn_state.fit_autodiff(
        loader,
        epochs=epochs,
        lr=lr,
        loss_type="nll",              # or "fid"
        target=data.target(),         # enables Fidelity / KL logging
        bases=data.eval_bases(),      # required for KL
        print_metrics=True,
        log_every=log_every,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
    )

    # -------------------------------
    # Phase comparison — global-phase aligned
    # -------------------------------
    with torch.no_grad():
        space = nn_state.generate_hilbert_space()
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).contiguous()
        psi_t = data.target().to(device=DEVICE, dtype=torch.cdouble).reshape(-1).contiguous()

        # normalize both
        nm, nt = torch.linalg.vector_norm(psi_m), torch.linalg.vector_norm(psi_t)
        if nm > 0: psi_m = psi_m / nm
        if nt > 0: psi_t = psi_t / nt

        # global phase via overlap
        ip = torch.sum(psi_t.conj() * psi_m)
        if ip.abs() > 1e-12:
            theta = torch.angle(ip)
        else:
            j = int(torch.argmax(psi_t.abs()))
            theta = torch.angle(psi_m[j]) - torch.angle(psi_t[j])
        psi_m_al = psi_m * torch.exp(-1j * theta)

        # one-hot computational basis (like W-state diagnostics)
        full_hs = nn_state.generate_hilbert_space()
        one_hot_mask = (full_hs.sum(dim=1) == 1)
        one_hot_indices = one_hot_mask.nonzero(as_tuple=False).view(-1)

        phi_t = torch.angle(psi_t[one_hot_indices]).cpu().numpy()
        phi_m = torch.angle(psi_m_al[one_hot_indices]).cpu().numpy()
        dphi = np.remainder((phi_m - phi_t) + np.pi, 2.0 * np.pi) - np.pi

        # bar plot (aligned)
        bitstrings = ["".join(str(int(b)) for b in row) for row in full_hs[one_hot_indices, :].cpu().numpy()]
        idx = np.arange(len(one_hot_indices))
        width = 0.35

        plt.rcParams.update({"font.family": "serif"})
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(idx - width/2, phi_t, width, alpha=0.7, label=r'$\phi_{\mathrm{true}}$')
        ax.bar(idx + width/2, phi_m, width, alpha=0.7, label=r'$\phi_{\mathrm{pred}}$')
        ax.set_xlabel("Basis State")
        ax.set_ylabel("Phase (radians)")
        ax.set_title("Phase Comparison (global-phase aligned)")
        ax.set_xticks(idx)
        ax.set_xticklabels([f"${b}$" for b in bitstrings], rotation=45)
        ax.set_ylim(-np.pi, np.pi)
        ax.legend(frameon=True, framealpha=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()

        # wrapped phase error plot
        fig2, ax2 = plt.subplots(figsize=(7.2, 3.8))
        ax2.plot(idx, dphi, marker='.', linestyle='', label=r'$\Delta\phi$ (wrapped)')
        ax2.axhline(0.0, linewidth=1.0)
        ax2.set_xlabel("Basis State")
        ax2.set_ylabel(r'$\Delta$phase [rad] in [$-\pi,\pi$]')
        ax2.set_title("Phase error (global phase aligned)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()

    # -------------------------------
    # Training metrics: Fidelity & KL
    # -------------------------------
    ep = np.array(hist.get("epoch", []))
    fid = np.array(hist.get("Fidelity", []))
    kl = np.array(hist.get("KL", []))

    if ep.size > 0:
        fig3, ax1 = plt.subplots(figsize=(6.0, 4.0), dpi=140)
        ax2 = ax1.twinx()
        if fid.size > 0:
            ax1.plot(ep, fid, marker='o', label="Fidelity")
        if kl.size > 0 and not np.isnan(kl).all():
            ax2.plot(ep, kl, marker='s', linestyle='--', label="KL")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(r'$|\langle \psi_t \mid \psi \rangle|^2$')
        ax2.set_ylabel(r'$\mathrm{KL}(p\,\|\,q)$')
        ax1.set_title("RBM Tomography — training metrics")
        ax1.grid(True, alpha=0.3)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="best")
        fig3.tight_layout()

    plt.show()

