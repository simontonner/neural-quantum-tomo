from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_handling import load_measurements_txt, load_state_txt, MeasurementDataset, MeasurementLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


##### MODEL #####


class BinaryRBM(nn.Module):
    def __init__(self, num_visible: int, num_hidden: int, device: torch.device = DEVICE):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.device = device

        self.W = nn.Parameter(torch.empty(num_visible, num_hidden, device=device, dtype=DTYPE))
        self.b = nn.Parameter(torch.zeros(num_visible, device=device, dtype=DTYPE))
        self.c = nn.Parameter(torch.zeros(num_hidden, device=device, dtype=DTYPE))

        self.initialize_parameters()

    def initialize_parameters(self, std: float | None = None):
        if std is None:
            std = 1.0 / np.sqrt(self.num_visible)
        nn.init.normal_(self.W, std=std)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def effective_energy(self, v: torch.Tensor) -> torch.Tensor:
        unsq = False
        if v.dim() == 1:
            v = v.unsqueeze(0)
            unsq = True

        v = v.to(device=self.W.device, dtype=self.W.dtype)
        hidden_term = F.softplus(v @ self.W + self.c).sum(dim=-1)
        visible_term = (v * self.b).sum(dim=-1)
        out = -visible_term - hidden_term
        return out.squeeze(0) if unsq else out

    @torch.no_grad()
    def gibbs_steps(self, k: int, initial_state: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        v = initial_state.clone().to(device=self.W.device, dtype=self.W.dtype)

        for _ in range(k):
            p_h = torch.sigmoid(v @ self.W + self.c)
            h = torch.bernoulli(p_h, generator=rng)

            p_v = torch.sigmoid(h @ self.W.t() + self.b)
            v = torch.bernoulli(p_v, generator=rng)

        return v


class PositiveWaveFunction(nn.Module):
    """
    Phase-free RBM wavefunction:
        psi(v) = exp(-E(v)/2) >= 0
    """

    def __init__(self, num_visible: int, num_hidden: int, k: int = 10, device: torch.device = DEVICE):
        super().__init__()
        self.device = device
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden)
        self.k = int(k)
        self.rbm = BinaryRBM(num_visible, num_hidden, device=device)
        self._max_size = 20

    def generate_hilbert_space(self, size: int | None = None):
        size = self.num_visible if size is None else int(size)
        if size > self._max_size:
            raise ValueError(f"Hilbert space too large (n={size} > max={self._max_size}).")

        n = 1 << size
        ar = torch.arange(n, device=self.device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=self.device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    def psi_complex_normalized(self, v: torch.Tensor):
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble))

    def forward(self, batch, aux_vars):
        rng = aux_vars.get("rng", None)

        v_data, _, _ = batch
        v_data = v_data.to(device=self.device, dtype=DTYPE)

        v_model = self.rbm.gibbs_steps(self.k, v_data, rng=rng).detach()

        loss = self.rbm.effective_energy(v_data).mean() - self.rbm.effective_energy(v_model).mean()
        return loss, {}

    def fit(
        self,
        loader: MeasurementLoader,
        *,
        epochs: int,
        lr: float,
        log_every: int,
        target: torch.Tensor | None = None,
        print_metrics: bool = True,
        metric_fmt: str = "Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
        seed: int = 0,
    ):
        opt = torch.optim.SGD(self.parameters(), lr=lr)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"] = []
            history["KL"] = []

        space = self.generate_hilbert_space()
        z_basis = [tuple("Z" for _ in range(self.num_visible))]
        rng = torch.Generator(device=self.device).manual_seed(seed)

        for ep in range(1, epochs + 1):
            for pos_batch, bases_batch, _ in loader:
                loss, _ = self((pos_batch, None, bases_batch), {"rng": rng})

                opt.zero_grad()
                loss.backward()
                opt.step()

            if (target is not None) and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space)
                    kl_val = KL(self, target, space=space, bases=z_basis)

                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)

                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))

        return history


##### METRICS #####


@torch.no_grad()
def fidelity(nn_state: PositiveWaveFunction, target: torch.Tensor, space: torch.Tensor | None = None, **kwargs):
    if not torch.is_complex(target):
        raise TypeError("fidelity: `target` must be complex (cdouble).")

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
def KL(
    nn_state: PositiveWaveFunction,
    target: torch.Tensor,
    space: torch.Tensor | None = None,
    bases=None,
    **kwargs,
):
    if bases is None:
        raise ValueError("KL needs `bases`.")
    if not torch.is_complex(target):
        raise TypeError("KL: `target` must be complex (cdouble).")

    space = nn_state.generate_hilbert_space() if space is None else space
    tgt = target.to(device=nn_state.device, dtype=torch.cdouble).reshape(-1)
    nt = torch.linalg.vector_norm(tgt)
    if nt == 0:
        return 0.0
    tgt_norm = tgt / nt

    psi_norm_cd = nn_state.psi_complex_normalized(space).reshape(-1)

    eps = 1e-12
    tgt_probs = (tgt_norm.abs().to(DTYPE)) ** 2
    nn_probs = (psi_norm_cd.abs().to(DTYPE)) ** 2

    p_sum = tgt_probs.sum().clamp_min(eps)
    q_sum = nn_probs.sum().clamp_min(eps)
    p = (tgt_probs / p_sum).clamp_min(eps)
    q = (nn_probs / q_sum).clamp_min(eps)

    return float(torch.sum(p * (torch.log(p) - torch.log(q))).item())


##### W OVERLAP PROXY #####


@torch.no_grad()
def w_overlap_positive(model: PositiveWaveFunction, M: int = 10_000, k: int = 100) -> float:
    N = model.num_visible
    rbm = model.rbm
    device = model.device

    initial = torch.bernoulli(torch.full((M, N), 0.5, device=device, dtype=DTYPE))
    samples = rbm.gibbs_steps(k=k, initial_state=initial)

    onehot_mask = samples.sum(dim=1) == 1
    onehot_samples = samples[onehot_mask]
    if onehot_samples.shape[0] == 0:
        return 0.0

    logp_sample = -rbm.effective_energy(onehot_samples)
    p_sample = torch.exp(logp_sample)
    term1 = torch.sum(1.0 / torch.sqrt(p_sample)).item() / M

    onehot_basis = torch.eye(N, dtype=DTYPE, device=device)
    logp_basis = -rbm.effective_energy(onehot_basis)
    p_basis = torch.exp(logp_basis)
    term2 = torch.sum(torch.sqrt(p_basis)).item() / N

    return float(np.sqrt(term1 * term2))


##### EXPERIMENT #####


def run_single_experiment(
    train_path: Path,
    psi_path: Path,
    N_use: int,
    cfg: dict,
):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    amps_np, _ = load_state_txt(psi_path)

    dataset = MeasurementDataset(
        file_paths=[train_path],
        load_fn=load_measurements_txt,
        system_param_keys=None,
        samples_per_file=[int(N_use)],
    )
    loader = MeasurementLoader(
        dataset,
        batch_size=cfg["pos_batch_size"],
        shuffle=True,
    )

    if dataset.implicit_basis is None or any(b != "Z" for b in dataset.implicit_basis):
        raise ValueError("This script only supports computational-basis Z data.")

    target_state = torch.tensor(amps_np, dtype=torch.cdouble, device=DEVICE)
    if target_state.numel() != (1 << dataset.num_qubits):
        raise ValueError(
            f"State dimension {target_state.numel()} does not match {dataset.num_qubits} qubits."
        )

    nv = dataset.num_qubits
    nh = nv * cfg["hidden_factor"]

    print(f"Using N={len(dataset)} samples, n={nv} qubits, device={DEVICE}")

    nn_state = PositiveWaveFunction(
        num_visible=nv,
        num_hidden=nh,
        k=cfg["k"],
        device=DEVICE,
    ).to(DEVICE)

    history = nn_state.fit(
        loader,
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        log_every=cfg["log_every"],
        target=target_state,
        print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}",
        seed=cfg["seed"],
    )

    overlap = w_overlap_positive(
        nn_state,
        M=cfg["overlap_M"],
        k=cfg["overlap_k"],
    )
    print(f"Overlap: {overlap:.4f}")

    return history, overlap, nv


def append_overlap_csv(csv_path: Path, samples: int, overlap: float):
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "samples", "overlap"])

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), samples, overlap])

    print(f"Logged overlap to {csv_path}")


def plot_overlap_curve(sample_sizes, overlaps, num_qubits: int):
    plt.rcParams.update({"font.family": "serif"})

    x_values = np.asarray(sample_sizes, dtype=int)
    y_values = np.asarray(overlaps, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(x_values, y_values, "s-", color="#a24a55", label=rf"RBM - $N = {num_qubits}$", zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Training Samples", fontsize=14)
    ax.set_ylabel(r"$O_W$", fontsize=14)
    ax.set_title("Sample Requirements for $W$ State", fontsize=16)

    ax.axhline(y=1.0, linestyle="--", color="gray", linewidth=1.5, zorder=1)

    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend(frameon=True, framealpha=1, loc="best", fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(left=50, right=1e5)

    plt.tight_layout()
    plt.savefig(f"training_performance_vanilla_w_state_{num_qubits}.png", dpi=300, bbox_inches="tight")
    plt.show()


##### ENTRYPOINT #####


if __name__ == "__main__":
    NUM_QUBITS = 10

    CFG = {
        "seed": 1234,
        "hidden_factor": 3,
        "epochs": 500,
        "pos_batch_size": 100,
        "lr": 0.1,
        "k": 10,
        "log_every": 10,
        "overlap_M": 10_000,
        "overlap_k": 100,
        "sample_sizes": [50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000],
    }

    train_path = Path(f"measurements/w_vanilla_{NUM_QUBITS}_meas_50000.txt")
    psi_path = Path(f"state_vectors/w_vanilla_{NUM_QUBITS}_state.txt")

    overlaps = []

    for N_target in CFG["sample_sizes"]:
        history, overlap, nv = run_single_experiment(train_path, psi_path, N_target, CFG)
        overlaps.append(overlap)
        append_overlap_csv(Path(f"overlap_{NUM_QUBITS}.csv"), N_target, overlap)

    print("\nSummary:")
    for n, ov in zip(CFG["sample_sizes"], overlaps):
        print(f"N={n:>6d} -> O_W = {ov:.6f}")

    plot_overlap_curve(CFG["sample_sizes"], overlaps, num_qubits=NUM_QUBITS)