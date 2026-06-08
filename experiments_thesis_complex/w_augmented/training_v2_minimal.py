from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from data_handling import load_measurements_txt, load_state_txt, MeasurementDataset, MeasurementLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


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


def enumerate_rotated_branches(model, basis, states, unitaries=None):
    # Theory-facing implementation detail: we build the local rotation kernel
    # qubit-by-qubit and sum only over the qubits measured outside Z.
    # This realizes the reduced 2**num_rotated coherent sum from the thesis,
    # rather than a full 2**V basis transformation.
    device = model.device
    basis_seq = list(basis)

    if len(basis_seq) != model.num_visible:
        raise ValueError(
            f"basis length {len(basis_seq)} != num_visible {model.num_visible}"
        )
    if states.shape[-1] != model.num_visible:
        raise ValueError(
            f"states width {states.shape[-1]} != num_visible {model.num_visible}"
        )

    rotated_sites = [i for i, b in enumerate(basis_seq) if b != "Z"]
    if len(rotated_sites) == 0:
        v = states.unsqueeze(0)
        Ut = torch.ones(v.shape[:-1], dtype=torch.cdouble, device=device)
        return Ut, v

    src = model.U if unitaries is None else unitaries
    local_unitaries = [
        as_complex_unitary(src[basis_seq[i]], device).reshape(2, 2).contiguous()
        for i in rotated_sites
    ]
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


##### MODEL #####


class RBM(nn.Module):
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
        # This is the RBM free energy F(v) used in the thesis notation.
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
        self._max_size = 20

    def reinitialize_parameters(self):
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

    def generate_basis_states(self, size=None, device=None):
        """
        Full enumeration of the Hilbert space is only used for diagnostics.

        Training never needs this full 2**V enumeration. It is only used for
        small-system post-training diagnostics such as exact fidelity and the
        final phase comparison plot.
        """
        device = self.device if device is None else device
        size = self.num_visible if size is None else int(size)

        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")

        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def phase(self, v):
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi_complex(self, v):
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_complex_normalized(self, basis_states):
        """
        Diagnostic-only normalized wavefunction evaluation.

        This explicitly evaluates the amplitude normalization by enumerating the
        full computational basis. It is used for exact fidelity and final phase
        plotting, not for training.
        """
        basis_states = basis_states.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(basis_states)
        ph = -0.5 * self.rbm_ph.effective_energy(basis_states)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    def stable_log_overlap_amp2(self, basis: Tuple[str, ...], states: torch.Tensor, eps_rot: float = 1e-12):
        # Exact rotated-basis data term on the current minibatch. The expensive
        # full-basis sum is avoided: only the rotated sites are enumerated.
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
        log_amp2 = 2.0 * M.squeeze(0) + torch.log(S_abs2 + eps_rot)
        return log_amp2

    def exact_positive_batch_loss(self, samples: torch.Tensor, bases_batch: List[Tuple[str, ...]]):
        # The positive term matches the theory directly: Z-basis samples use
        # F_lambda(s), rotated-basis samples use -log|A_r(s^[r])|^2.
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

    @torch.no_grad()
    def sample_random_binary_states(self, num_samples: int):
        probs = torch.full(
            (num_samples, self.num_visible),
            0.5,
            device=self.device,
            dtype=DTYPE,
        )
        return torch.bernoulli(probs)

    def forward(self, pos_batch, bases_batch, fantasy_particles, pcd_k):
        # Training uses the exact positive minibatch term plus a persistent-CD
        # estimate of the amplitude-model term. No full partition sum is needed.
        pos_loss = self.exact_positive_batch_loss(pos_batch, bases_batch)
        fantasy_particles = self.rbm_am.gibbs_steps(pcd_k, fantasy_particles, overwrite=True)
        neg_loss = self.rbm_am.effective_energy(fantasy_particles).mean()
        loss = pos_loss - neg_loss
        return loss, fantasy_particles


##### METRICS #####


@torch.no_grad()
def fidelity(model, target, basis_states=None):
    # Exact fidelity is a post-training small-system diagnostic and therefore
    # uses full Hilbert-space enumeration through psi_complex_normalized(...).
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


##### TRAINING #####


def train_pcd(model, loader, target, basis_states, cfg):
    params = list(model.rbm_am.parameters()) + list(model.rbm_ph.parameters())
    opt = torch.optim.SGD(params, lr=cfg["lr"])

    history = {
        "epoch": [],
        "Fidelity": [],
    }

    num_chains = int(cfg["num_chains"])
    reset_frac = float(cfg["reset_frac"])
    pcd_k = int(cfg["pcd_k"])

    if num_chains <= 0:
        raise ValueError("num_chains must be positive")

    with torch.no_grad():
        fantasy_particles = model.sample_random_binary_states(num_chains)

    num_resets = int(reset_frac * num_chains)
    if reset_frac > 0.0 and num_resets == 0:
        num_resets = 1

    for epoch in range(1, cfg["epochs"] + 1):
        for pos_batch, bases_batch, _ in loader:
            pos_batch = pos_batch.to(model.device, dtype=DTYPE)
            bases_batch = list(bases_batch)

            with torch.no_grad():
                if num_resets > 0:
                    reset_rows = torch.randperm(num_chains, device=model.device)[:num_resets]
                    fantasy_particles[reset_rows] = model.sample_random_binary_states(num_resets)

            loss, fantasy_particles = model(
                pos_batch,
                bases_batch,
                fantasy_particles,
                pcd_k,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % cfg["log_every"] == 0:
            with torch.no_grad():
                fid_val = fidelity(model, target, basis_states=basis_states)

            history["epoch"].append(epoch)
            history["Fidelity"].append(fid_val)

            print(
                f"[PCD-{pcd_k:>2}] Epoch {epoch}: "
                f"Fidelity = {fid_val:.6f}"
            )

    return history


##### PLOTTING #####


def phase_comparison_arrays(model, target_state, basis_states):
    # Final phase readout is also diagnostic-only and uses the exact normalized
    # wavefunction on the fully enumerated computational basis.
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

    return sel, phi_t_sel, phi_m_sel, phi_diff_sel


def plot_training_curves(history, pcd_k):
    fig_f, axf = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    axf.plot(history.get("epoch", []), history["Fidelity"], marker="o", label=f"PCD-{pcd_k}")
    axf.set_xlabel("Epoch")
    axf.set_ylabel(r"$|\langle \psi_t \mid \psi \rangle|^2$")
    axf.set_title("PCD tomography - fidelity")
    axf.grid(True, alpha=0.3)
    axf.legend(loc="best")
    fig_f.tight_layout()


def plot_phase_comparison(model, target_state, basis_states):
    sel, phi_t_sel, phi_m_sel, phi_diff_sel = phase_comparison_arrays(model, target_state, basis_states)

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


##### ENTRYPOINT #####


def run_experiment(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    psi_path = Path("state_vectors/w_phase_state.txt")
    amps_np, _ = load_state_txt(psi_path)
    target_state = torch.tensor(amps_np, dtype=torch.cdouble, device=DEVICE)

    measurement_dir = Path("measurements")
    measurement_paths = [
        measurement_dir / "w_phase_ZZZZ_5000.txt",
        measurement_dir / "w_phase_XXZZ_5000.txt",
        measurement_dir / "w_phase_XYZZ_5000.txt",
        measurement_dir / "w_phase_ZXXZ_5000.txt",
        measurement_dir / "w_phase_ZXYZ_5000.txt",
        measurement_dir / "w_phase_ZZXX_5000.txt",
        measurement_dir / "w_phase_ZZXY_5000.txt",
    ]

    dataset = MeasurementDataset(
        file_paths=measurement_paths,
        load_fn=load_measurements_txt,
        system_param_keys=None,
    )
    loader = MeasurementLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

    model = ComplexRBM(
        num_visible=dataset.num_qubits,
        num_hidden=dataset.num_qubits,
        unitary_dict=create_unitary_dict(),
        device=DEVICE,
    ).to(DEVICE)

    basis_states = model.generate_basis_states()

    history = train_pcd(
        model=model,
        loader=loader,
        target=target_state,
        basis_states=basis_states,
        cfg=cfg,
    )

    return model, history, target_state, basis_states


if __name__ == "__main__":
    CFG = {
        "seed": 1234,
        "batch_size": 128,
        "num_chains": 128,
        "epochs": 150,
        "lr": 5e-2,
        "log_every": 5,
        "pcd_k": 10,
        "reset_frac": 0.1,
    }

    model, history, target_state, basis_states = run_experiment(CFG)
    plot_training_curves(history, pcd_k=CFG["pcd_k"])
    plot_phase_comparison(model, target_state, basis_states)
    plt.show()
