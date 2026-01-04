from math import ceil, sqrt, prod
from typing import Iterable, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

##### DEVICE AND DTYPES #####
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double  # RBM energies in float64 for stability


##### SINGLE-QUBIT UNITARIES (as cdouble) #####
def create_dict(**overrides):
    """Return {X,Y,Z} single-qubit unitaries as torch.cdouble."""
    inv_sqrt2 = 1.0 / sqrt(2.0)

    X = inv_sqrt2 * torch.tensor([[1.0 + 0.0j, 1.0 + 0.0j],
                                  [1.0 + 0.0j, -1.0 + 0.0j]], dtype=torch.cdouble, device=DEVICE)

    Y = inv_sqrt2 * torch.tensor([[1.0 + 0.0j, 0.0 - 1.0j],
                                  [1.0 + 0.0j, 0.0 + 1.0j]], dtype=torch.cdouble, device=DEVICE)

    Z = torch.tensor([[1.0 + 0.0j, 0.0 + 0.0j],
                      [0.0 + 0.0j, 1.0 + 0.0j]], dtype=torch.cdouble, device=DEVICE)

    U = {"X": X.contiguous(), "Y": Y.contiguous(), "Z": Z.contiguous()}

    for name, mat in overrides.items():
        U[name] = as_complex_unitary(mat, DEVICE)

    return U


def as_complex_unitary(U, device: torch.device):
    """Ensure a (2,2) complex matrix on device."""
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


##### KRON-APPLY WITHOUT EXPLICIT TENSOR PRODUCT #####
def _kron_mult(matrices: List[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Apply tensor product over sites of matrices to state/batch x without building the big matrix."""
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
    """Rotate psi into a product basis specified by a tuple of 'X','Y','Z'."""
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


##### BASIS-BRANCH ENUMERATION FOR LOCAL PRODUCT MEASUREMENTS #####
def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    """Enumerate coherent branches for a measured batch of states under the given basis."""
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

    combos = nn_state.generate_hilbert_space(size=S, device=device)  # (C, S)

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


def _convert_basis_element_to_index(states):
    """Map rows in {0,1}^n to flat indices [0, 2^n - 1] (MSB-first)."""
    s = states.round().to(torch.long)
    n = s.shape[-1]
    shifts = torch.arange(n - 1, -1, -1, device=s.device, dtype=torch.long)
    return (s << shifts).sum(dim=-1)


def rotate_psi_inner_prod(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    """Compute overlap for measured outcomes in the given basis: sum_c Ut[c] * psi(branch_state[c])."""
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)

    if psi is None:
        psi_sel = nn_state.psi_complex(v)
    else:
        if not torch.is_complex(psi):
            raise TypeError("rotate_psi_inner_prod: psi must be complex.")
        idx = _convert_basis_element_to_index(v).long()
        psi_c = psi.to(dtype=torch.cdouble, device=nn_state.device)
        psi_sel = psi_c[idx]

    Upsi_v_c = Ut * psi_sel
    Upsi_c = Upsi_v_c.sum(dim=0)

    if include_extras:
        return Upsi_c, Upsi_v_c, v
    return Upsi_c


##### BINARY RBM #####
class BinaryRBM(nn.Module):
    """Bernoulli/Bernoulli RBM with free energy F(v)."""

    def __init__(self, num_visible, num_hidden=None, zero_weights=False, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.device = device
        self.initialize_parameters(zero_weights=zero_weights)

    def __repr__(self):
        return f"BinaryRBM(num_visible={self.num_visible}, num_hidden={self.num_hidden}, device='{self.device}')"

    def initialize_parameters(self, zero_weights=False):
        """Init parameters; optionally all-zero for debugging."""
        gen_tensor = torch.zeros if zero_weights else torch.randn
        scale = 1.0 / np.sqrt(self.num_visible)

        self.weights = nn.Parameter(
            gen_tensor(self.num_hidden, self.num_visible, device=self.device, dtype=DTYPE) * scale,
            requires_grad=True,
            )
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible, device=self.device, dtype=DTYPE), requires_grad=True)
        self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden, device=self.device, dtype=DTYPE), requires_grad=True)

    def effective_energy(self, v):
        """Free energy F(v). Accepts (..., n). Returns (...,)."""
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
        """k-step block Gibbs from initial_state in {0,1}."""
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


##### COMPLEX WAVE FUNCTION (ampl + phase RBM combined into ansatz) #####
class ComplexWaveFunction:
    """psi(sigma) = exp(-F_lambda(sigma)/2) * exp(-i F_mu(sigma)/2)."""

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

    # control
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
        """Re-init both RBMs."""
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

    # psi accessors
    def amplitude(self, v):
        """abs(psi(v)) = exp(-F_lambda(v)/2)."""
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        """phase(v) = -F_mu(v)/2 (real)."""
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi_complex(self, v):
        """Complex psi(v)."""
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_complex_normalized(self, v):
        """Normalized psi(v) using exact log Z_lambda (safe only for small n)."""
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    # aliases
    def psi(self, v):
        return self.psi_complex(v)

    def psi_normalized(self, v):
        return self.psi_complex_normalized(v)

    def phase_angle(self, v):
        return self.phase(v)

    # utilities
    def generate_hilbert_space(self, size=None, device=None):
        """Enumerate computational basis as (2^size, size) bit-matrix in {0,1}."""
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    # stable overlap for rotated bases
    def _stable_log_overlap_amp2(self, basis: Tuple[str, ...], states: torch.Tensor, eps_rot: float = 1e-6, unitaries=None):
        """Stable log of squared overlap for rotated-basis outcomes via complex log-sum-exp."""
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

    # loss pieces
    def _positive_phase_loss(self, samples: torch.Tensor, bases_batch: List[Tuple[str, ...]], eps_rot: float = 1e-6):
        """Data term: Z-basis NLL plus rotated-basis likelihood."""
        buckets = {}
        for i, row in enumerate(bases_batch):
            buckets.setdefault(tuple(row), []).append(i)

        loss_rot = samples.new_tensor(0.0, dtype=DTYPE)
        loss_z = samples.new_tensor(0.0, dtype=DTYPE)

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            if any(ch != "Z" for ch in basis_t):
                log_amp2 = self._stable_log_overlap_amp2(basis_t, samples[idxs_t], eps_rot=eps_rot)
                loss_rot = loss_rot - log_amp2.sum().to(DTYPE)
            else:
                Epos = self.rbm_am.effective_energy(samples[idxs_t])
                loss_z = loss_z + Epos.sum()

        return loss_rot + loss_z

    def _negative_phase_loss(self, k: int, neg_init: torch.Tensor):
        """CD-k negative phase for amplitude RBM."""
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(k, neg_init, overwrite=True)
        Eneg = self.rbm_am.effective_energy(vk)
        return Eneg.sum(), vk.shape[0]

    # training loop
    def fit(self, loader, epochs=70, k=10, lr=1e-1, log_every=5, optimizer=torch.optim.SGD, optimizer_args=None,
            target=None, bases=None, space=None, print_metrics=True,
            metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        """Autodiff CD training with rotated-basis likelihood and CD-k negative phase."""
        if self.stop_training:
            return {"epoch": []}

        optimizer_args = {} if optimizer_args is None else optimizer_args
        params = list(self.rbm_am.parameters()) + list(self.rbm_ph.parameters())
        opt = optimizer(params, lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []

        if space is None:
            space = self.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                L_pos = self._positive_phase_loss(pos_batch, bases_batch)
                B_pos = float(pos_batch.shape[0])

                L_neg, B_neg = self._negative_phase_loss(k, neg_batch)

                loss = (L_pos / B_pos) - (L_neg / B_neg)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()

                if self.stop_training:
                    break

            if (target is not None) and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space, bases=bases)
                    kl_val = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))

            if self.stop_training:
                break

        return history


##### METRICS #####
@torch.no_grad()
def fidelity(nn_state, target, space=None, **kwargs):
    """Return squared magnitude of overlap(target, psi) with both normalized (diagnostic only)."""
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
    """Average KL divergence across bases (diagnostic only)."""
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

        nn_probs_r = (psi_r.abs().to(DTYPE)) ** 2
        tgt_probs_r = (tgt_psi_r.abs().to(DTYPE)) ** 2

        p_sum = tgt_probs_r.sum().clamp_min(eps)
        q_sum = nn_probs_r.sum().clamp_min(eps)
        p = (tgt_probs_r / p_sum).clamp_min(eps)
        q = (nn_probs_r / q_sum).clamp_min(eps)

        KL_val += torch.sum(p * (torch.log(p) - torch.log(q)))

    return (KL_val / len(bases)).item()


##### DATASET AND LOADER #####
class TomographyDataset:
    """Container for measurement samples, bases, and target psi."""

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


class RBMTomographyLoader:
    """Yield (pos_batch, neg_batch, bases_batch) minibatches for training."""

    def __init__(self, dataset: TomographyDataset, pos_batch_size: int = 100, neg_batch_size: Optional[int] = None,
                 device: torch.device = DEVICE, dtype: torch.dtype = DTYPE, strict: bool = True):
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

    def set_seed(self, seed: Optional[int]):
        """Optional deterministic shuffling."""
        if seed is None:
            self._gen = None
        else:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            self._gen = g

    def __len__(self):
        return ceil(len(self.ds) / self.pos_bs)

    def iter_epoch(self):
        """Iterate once over the dataset, yielding aligned batches."""
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


##### STANDALONE TRAINING SCRIPT (EXAMPLE, NO PLOTTING) #####
if __name__ == "__main__":
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    torch.manual_seed(1234)

    data = TomographyDataset(train_path, psi_path, train_bases_path, bases_path, device=DEVICE)
    U = create_dict()

    nv = data.num_visible()
    nh = nv
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, device=DEVICE)

    epochs = 70
    pbs = 100
    nbs = 100
    lr = 1e-1
    k_cd = 10
    log_every = 5

    loader = RBMTomographyLoader(data, pos_batch_size=pbs, neg_batch_size=nbs, device=DEVICE, dtype=DTYPE)

    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(loader, epochs=epochs, k=k_cd, lr=lr, log_every=log_every,
                           optimizer=torch.optim.SGD, optimizer_args=None,
                           target=data.target(), bases=data.eval_bases(), space=space,
                           print_metrics=True,
                           metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}")

