import torch

from ..config import DEVICE, DTYPE
from ..models.rbm import BinaryRBM
from ..neural_states.pauli import create_dict, as_complex_unitary
from ..neural_states.measurement import rotate_psi_inner_prod
from ..utils.linalg import inverse
from ..utils.optim import vector_to_grads
from ..training.metrics import fidelity, KL  # used inside .fit()


class PositiveWaveFunction:
    """
    Phase-free RBM wavefunction:
      psi(v) = exp(-E_am(v)/2) ∈ ℝ_{≥0}
    """

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, module=None, device: torch.device = DEVICE):
        self.device = device
        self.rbm = (module.to(self.device) if module is not None
                    else BinaryRBM(num_visible, num_hidden, device=self.device))
        self.rbm.device = self.device

        self.num_visible = self.rbm.num_visible
        self.num_hidden = self.rbm.num_hidden

        raw = unitary_dict if unitary_dict is not None else create_dict()
        self.U = {k: as_complex_unitary(v, self.device) for k, v in raw.items()}

        self._stop_training = False
        self._max_size = 20

    # -------------------------------
    # control
    # -------------------------------
    @property
    def stop_training(self):
        return self._stop_training

    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool):
            self._stop_training = new_val
        else:
            raise ValueError("stop_training must be bool!")

    @property
    def max_size(self):
        return self._max_size

    def reinitialize_parameters(self):
        # match QuCumber-style small init if your BinaryRBM supports it
        # self.rbm.initialize_parameters(std=1e-2, bias=0.0)
        self.rbm.initialize_parameters()

    # -------------------------------
    # amplitudes / psi
    # -------------------------------
    def amplitude(self, v):
        """|psi(v)| = exp(-E_am/2) as real DTYPE."""
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm.effective_energy(v)).exp().sqrt()

    def psi_complex(self, v):
        """psi(v) as complex tensor with zero phase (cdouble)."""
        return self.amplitude(v).to(torch.cdouble)

    def psi_complex_normalized(self, v):
        """Normalized psi via exact partition on the amplitude RBM."""
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble))

    # user-facing aliases
    def psi(self, v): return self.psi_complex(v)
    def psi_normalized(self, v): return self.psi_complex_normalized(v)

    # -------------------------------
    # basis space utils
    # -------------------------------
    def generate_hilbert_space(self, size=None, device=None):
        """Enumerate computational basis as a (2^size, size) bit-matrix."""
        device = self.device if device is None else device
        size = self.rbm.num_visible if size is None else int(size)
        if size > self.max_size:
            raise ValueError("Hilbert space too large!")
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    # -------------------------------
    # explicit positive-phase gradients
    # -------------------------------
    def am_grads(self, v):
        """Per-sample grads of E_am(v) (complex-cast for rotation algebra)."""
        g = self.rbm.effective_energy_gradient(v, reduce=False)
        return g.to(torch.cdouble)

    def rotated_gradient(self, basis, sample):
        """
        Positive-phase grads under a rotated measurement basis (no phase net).
        Present for completeness; not used in training to match QuCumber PositiveWF.
        """
        Upsi, Upsi_v, v = rotate_psi_inner_prod(self, basis, sample, include_extras=True)
        inv_Upsi = inverse(Upsi)                           # (B,)
        g = self.am_grads(v)                               # (C,B,G) complex
        rg = torch.einsum("cb,cbg->bg", Upsi_v, g)         # (B,G)
        grad = torch.einsum("b,bg->g", inv_Upsi, rg).real.to(DTYPE)
        return grad

    def gradient(self, samples, bases=None):
        """
        Positive-phase gradients. For QuCumber-Positive parity, treat as Z-only:
        return SUM over samples; division happens in positive_phase_gradients().
        """
        G = torch.zeros(self.rbm.num_pars, dtype=DTYPE, device=self.device)

        if bases is None:
            # explicit sum to mirror QuCumber "sum then divide"
            return self.rbm.effective_energy_gradient(samples, reduce=False).sum(0)

        # If someone passes bases, accumulate like QuCumber (Z-only contributes via RBM grad)
        try:
            bases_seq = [tuple(row) for row in bases]
        except Exception as e:
            raise ValueError("gradient: `bases` must be an iterable of string rows.") from e

        if len(bases_seq) == 0:
            return G
        n = len(bases_seq[0])
        if any(len(row) != n for row in bases_seq):
            raise ValueError("gradient: inconsistent basis widths.")
        if n != self.num_visible:
            raise ValueError(f"gradient: basis width {n} != num_visible {self.num_visible}.")
        if samples.shape[0] != len(bases_seq):
            raise ValueError("gradient: samples batch != bases rows.")

        if samples.dim() < 2:
            samples = samples.unsqueeze(0)

        # Bucketize identical basis rows
        buckets = {}
        for i, row in enumerate(bases_seq):
            buckets.setdefault(row, []).append(i)

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            if any(ch != "Z" for ch in basis_t):
                G += self.rotated_gradient(basis_t, samples[idxs_t, :])
            else:
                G += self.rbm.effective_energy_gradient(samples[idxs_t, :], reduce=False).sum(0)

        return G

    def positive_phase_gradients(self, samples_batch, bases_batch=None):
        # divide by batch size here (sum -> mean)
        g = self.gradient(samples_batch, bases=bases_batch)
        return g / float(samples_batch.shape[0])

    def compute_batch_gradients(self, k, samples_batch, neg_batch=None, bases_batch=None):
        """
        QuCumber PositiveWF parity:
        - Positive phase from Z-only (bases ignored here).
        - Negative phase: CD-k with data-initialized negatives (neg starts at samples).
        - Both terms averaged explicitly.
        """
        # Z-only signal for PositiveWF parity
        g_pos = self.positive_phase_gradients(samples_batch, bases_batch=None)

        # CD-k with data-seeded negatives
        start = samples_batch if (neg_batch is None) else neg_batch
        vk = self.rbm.gibbs_steps(k, start)

        # mean over model samples (explicit)
        g_model = self.rbm.effective_energy_gradient(vk, reduce=False).mean(0)
        return g_pos - g_model

    # -------------------------------
    # training loop (CD-k), QuCumber PositiveWF parity
    # -------------------------------
    def fit(self, loader, epochs=100, k=1, lr=1e-3, log_every=5,
            optimizer=torch.optim.SGD, optimizer_args=None,
            target=None, bases=None, space=None,
            print_metrics=True, metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        """
        CD training using RBMTomographyLoader, but force:
          - Z-only learning signal (ignore bases during updates)
          - data-initialized negatives (neg_batch := pos_batch)
        """
        if self.stop_training:
            return {"epoch": []}

        optimizer_args = {} if optimizer_args is None else optimizer_args
        opt = optimizer(list(self.rbm.parameters()), lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []
        if space is None:
            space = self.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            for pos_batch, _, _bases_batch in loader.iter_epoch():
                # enforce QuCumber PositiveWF schema:
                # - negatives seeded from the same data batch
                # - bases ignored for updates (Z-only)
                g = self.compute_batch_gradients(k, pos_batch, neg_batch=pos_batch, bases_batch=None)
                opt.zero_grad()
                vector_to_grads(g, self.rbm.parameters())
                opt.step()
                if self.stop_training:
                    break

            if (target is not None) and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space, bases=bases)
                    kl_val = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history.setdefault("Fidelity", []).append(fid_val)
                history.setdefault("KL", []).append(kl_val)
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))

            if self.stop_training:
                break

        return history