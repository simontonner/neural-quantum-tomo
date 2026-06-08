import sys
from pathlib import Path
from math import sqrt, ceil
from typing import List, Tuple

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
    """
    Enumerate coherent branches for measured batch of states under given basis.

    Returns
    -------
    Ut : complex tensor, shape (2**S, batch)
        Product of local basis-change matrix elements.
    v : real tensor, shape (2**S, batch, n_vis)
        Computational-basis branch states contributing coherently.
    """
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
        """
        Free energy F(v).

        Parameters
        ----------
        v : tensor, shape (..., n_vis)

        Returns
        -------
        tensor, shape (...)
        """
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


#### COMPLEX WAVE FUNCTION (AMPLITUDE AND PHASE RBM) ####


class ComplexWaveFunction:
    """
    psi(s) = Z_lambda^(-1/2) exp(-F_lambda(s)/2) exp(-i F_mu(s)/2)
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

    def reinitialize_parameters(self):
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

    def phase(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi_complex(self, v):
        v = v.to(self.device, dtype=DTYPE)
        amp = torch.exp(-0.5 * self.rbm_am.effective_energy(v))
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

    def _stable_log_overlap_amp2(self, basis: Tuple[str, ...], states: torch.Tensor, eps_rot: float = 1e-6):
        """
        Compute log | sum_s U_r(s_obs, s) psi(s) |^2 stably, without explicit normalization.
        """
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

    def z_data_loss(self, samples: torch.Tensor):
        """
        Computational-basis data term:
        mean_s F_lambda(s)
        """
        Epos = self.rbm_am.effective_energy(samples)
        return Epos.mean()

    def rotated_data_loss(self, samples: torch.Tensor, bases_batch: List[Tuple[str, ...]], eps_rot: float = 1e-6):
        """
        Rotated-basis data term:
        mean_{samples in this basis} [ -log |A_r(s^[r])|^2 ]
        """
        buckets = {}
        for i, row in enumerate(bases_batch):
            basis_t = tuple(row)
            if all(ch == "Z" for ch in basis_t):
                raise ValueError("rotated_data_loss received a Z-basis batch")
            buckets.setdefault(basis_t, []).append(i)

        loss_rot = samples.new_tensor(0.0, dtype=DTYPE)

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            log_amp2 = self._stable_log_overlap_amp2(basis_t, samples[idxs_t], eps_rot=eps_rot)
            loss_rot = loss_rot - log_amp2.mean()

        return loss_rot

    def model_loss(self, k: int, neg_init: torch.Tensor):
        """
        CD-k model term:
        mean_{neg samples} F_lambda(s_tilde)
        """
        with torch.no_grad():
            vk = self.rbm_am.gibbs_steps(k, neg_init, overwrite=False)
        Eneg = self.rbm_am.effective_energy(vk)
        return Eneg.mean()

    def fit(
        self,
        loader_z,
        loader_rot_list,
        steps_per_epoch,
        epochs=150,
        k=10,
        lr=1e-1,
        log_every=5,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=None,
        space=None,
        print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f}",
    ):
        if self.stop_training:
            return {"epoch": []}

        optimizer_args = {} if optimizer_args is None else optimizer_args
        params = list(self.rbm_am.parameters()) + list(self.rbm_ph.parameters())
        opt = optimizer(params, lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"] = []

        if space is None:
            space = self.generate_hilbert_space()

        # One model term per basis contribution: 1 Z + len(rotated bases)
        model_term_weight = 1.0 + len(loader_rot_list)

        for ep in range(1, epochs + 1):
            it_z = iter(loader_z)
            it_rot = [iter(loader) for loader in loader_rot_list]

            for _ in range(steps_per_epoch):
                try:
                    pos_z, bases_z, _ = next(it_z)
                except StopIteration:
                    it_z = iter(loader_z)
                    pos_z, bases_z, _ = next(it_z)

                pos_z = pos_z.to(self.device, dtype=DTYPE)
                z_loss = self.z_data_loss(pos_z)

                rot_total = pos_z.new_tensor(0.0, dtype=DTYPE)
                for j in range(len(loader_rot_list)):
                    try:
                        pos_rot, bases_rot, _ = next(it_rot[j])
                    except StopIteration:
                        it_rot[j] = iter(loader_rot_list[j])
                        pos_rot, bases_rot, _ = next(it_rot[j])

                    pos_rot = pos_rot.to(self.device, dtype=DTYPE)
                    rot_total = rot_total + self.rotated_data_loss(pos_rot, list(bases_rot))

                neg_loss = self.model_loss(k, pos_z)

                loss = z_loss + rot_total - model_term_weight * neg_loss

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 10.0)
                opt.step()

                if self.stop_training:
                    break

            if target is not None and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val))

            if self.stop_training:
                break

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


#### RUN SCRIPT ####


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)

    pbs = 100
    epochs = 150
    lr = 1e-1
    k_cd = 10
    log_every = 5

    psi_path = Path("state_vectors/w_phase_state.txt")
    amps_np, state_headers = load_state_txt(psi_path)
    target_state = torch.tensor(amps_np, dtype=torch.cdouble, device=DEVICE)

    meas_directory = Path("measurements")

    comp_path = meas_directory / "w_phase_ZZZZ_5000.txt"
    rot_paths = [
        meas_directory / "w_phase_XXZZ_5000.txt",
        meas_directory / "w_phase_XYZZ_5000.txt",
        meas_directory / "w_phase_ZXXZ_5000.txt",
        meas_directory / "w_phase_ZXYZ_5000.txt",
        meas_directory / "w_phase_ZZXX_5000.txt",
        meas_directory / "w_phase_ZZXY_5000.txt",
    ]

    ds_comp = MeasurementDataset(
        file_paths=[comp_path],
        load_fn=load_measurements_txt,
        system_param_keys=None,
    )
    loader_comp = MeasurementLoader(ds_comp, batch_size=pbs, shuffle=True)

    ds_rot_list = []
    loader_rot_list = []
    for p in rot_paths:
        ds_r = MeasurementDataset(
            file_paths=[p],
            load_fn=load_measurements_txt,
            system_param_keys=None,
        )
        ds_rot_list.append(ds_r)
        loader_rot_list.append(MeasurementLoader(ds_r, batch_size=pbs, shuffle=True))

    nv = ds_comp.num_qubits
    nh = nv
    U = create_dict()
    nn_state = ComplexWaveFunction(num_visible=nv, num_hidden=nh, unitary_dict=U, device=DEVICE)
    space = nn_state.generate_hilbert_space()

    steps_per_epoch = ceil(len(ds_comp) / pbs)

    history = nn_state.fit(
        loader_z=loader_comp,
        loader_rot_list=loader_rot_list,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        k=k_cd,
        lr=lr,
        log_every=log_every,
        optimizer=torch.optim.SGD,
        optimizer_args=None,
        target=target_state,
        space=space,
        print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f}",
    )

    #### PHASE COMPARISON WITH PROPER GLOBAL-PHASE ALIGNMENT ####

    with torch.no_grad():
        psi_m = nn_state.psi_complex_normalized(space).reshape(-1).to(torch.cdouble)
        psi_t = target_state.reshape(-1).to(torch.cdouble)

        psi_m = psi_m / torch.linalg.vector_norm(psi_m)
        psi_t = psi_t / torch.linalg.vector_norm(psi_t)

        # Proper global-phase alignment via overlap
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

    fig_p, axp = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    axp.plot(range(sel.numel()), phi_t_sel.cpu().numpy(), marker=".", linestyle="", label="target phase")
    axp.plot(range(sel.numel()), phi_m_sel.cpu().numpy(), marker="x", linestyle="", label="model phase (aligned)")
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

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=140)
    ax.plot(history.get("epoch", []), history["Fidelity"], marker="o", label="Fidelity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
    ax.set_title("RBM Tomography - training fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    plt.show()