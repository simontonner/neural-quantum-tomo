from itertools import chain
import torch

from ..config import DEVICE, DTYPE
from ..models.rbm import BinaryRBM
from ..neural_states.pauli import create_dict, as_complex_unitary
from ..neural_states.measurement import rotate_psi_inner_prod
from ..utils.linalg import inverse
from ..utils.optim import vector_to_grads
from ..training.metrics import fidelity, KL  # used inside .fit()

class ComplexWaveFunction:
    """
    Two real RBMs define magnitude and phase over bitstrings:
      psi(v) = exp(-E_am(v)/2) * exp(i * (-E_ph(v)/2))
    """

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, module=None, device: torch.device = DEVICE):
        self.device = device
        if module is None:
            self.rbm_am = BinaryRBM(num_visible, num_hidden, device=self.device)
            self.rbm_ph = BinaryRBM(num_visible, num_hidden, device=self.device)
        else:
            self.rbm_am = module.to(self.device)
            self.rbm_am.device = self.device
            self.rbm_ph = module.to(self.device).clone()
            self.rbm_ph.device = self.device

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
            raise ValueError("stop_training must be bool!")

    @property
    def max_size(self):
        return self._max_size

    def reinitialize_parameters(self):
        self.rbm_am.initialize_parameters()
        self.rbm_ph.initialize_parameters()

    # amplitudes/phases
    def amplitude(self, v):
        """|psi(v)| as exp(-E_am/2)."""
        v = v.to(self.device, dtype=DTYPE)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        """Phase angle from the phase RBM: -E_ph/2."""
        v = v.to(self.device, dtype=DTYPE)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi_complex(self, v):
        """psi(v) as complex tensor (cdouble)."""
        v = v.to(self.device, dtype=DTYPE)
        amp = (-self.rbm_am.effective_energy(v)).exp().sqrt()
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        return amp.to(torch.cdouble) * torch.exp(1j * ph.to(torch.cdouble))

    def psi_complex_normalized(self, v):
        """Normalized psi via exact partition on the amplitude RBM."""
        v = v.to(self.device, dtype=DTYPE)
        E = self.rbm_am.effective_energy(v)
        ph = -0.5 * self.rbm_ph.effective_energy(v)
        logZ = torch.logsumexp(-E, dim=0)
        return torch.exp(((-0.5 * E) - 0.5 * logZ).to(torch.cdouble) + 1j * ph.to(torch.cdouble))

    # user-facing aliases
    def psi(self, v): return self.psi_complex(v)
    def psi_normalized(self, v): return self.psi_complex_normalized(v)
    def phase_angle(self, v): return self.phase(v)

    def generate_hilbert_space(self, size=None, device=None):
        """Enumerate computational basis as a (2^size, size) bit-matrix."""
        device = self.device if device is None else device
        size = self.rbm_am.num_visible if size is None else int(size)
        if size > self.max_size:
            raise ValueError("Hilbert space too large!")
        n = 1 << size
        ar = torch.arange(n, device=device, dtype=torch.long)
        shifts = torch.arange(size - 1, -1, -1, device=device, dtype=torch.long)
        return ((ar.unsqueeze(1) >> shifts) & 1).to(DTYPE)

    # gradients (complex mapping; final grads are real DTYPE)
    def am_grads(self, v):
        g = self.rbm_am.effective_energy_gradient(v, reduce=False)
        return g.to(torch.cdouble)

    def ph_grads(self, v):
        g = self.rbm_ph.effective_energy_gradient(v, reduce=False)
        return (1j * g.to(torch.cdouble))

    def rotated_gradient(self, basis, sample):
        """Positive-phase grads under a rotated measurement basis."""
        Upsi, Upsi_v, v = rotate_psi_inner_prod(self, basis, sample, include_extras=True)
        inv_Upsi = inverse(Upsi)  # (B,)
        raw_grads = [self.am_grads(v), self.ph_grads(v)]  # complex
        rotated_grad = [torch.einsum("cb,cbg->bg", Upsi_v, g) for g in raw_grads]
        grad = [torch.einsum("b,bg->g", inv_Upsi, rg).real.to(DTYPE) for rg in rotated_grad]
        return grad

    def gradient(self, samples, bases=None):
        """
        Positive-phase gradients. If `bases` is None, only amplitude grads (Z).
        Otherwise group identical basis rows and accumulate rotated grads.
        """
        G_am = torch.zeros(self.rbm_am.num_pars, dtype=DTYPE, device=self.device)
        G_ph = torch.zeros(self.rbm_ph.num_pars, dtype=DTYPE, device=self.device)

        if bases is None:
            G_am = self.rbm_am.effective_energy_gradient(samples)
            return [G_am, G_ph]

        try:
            bases_seq = [tuple(row) for row in bases]
        except Exception as e:
            raise ValueError("gradient: `bases` must be an iterable of string rows.") from e

        B = len(bases_seq)
        if B == 0:
            return [G_am, G_ph]
        n = len(bases_seq[0])
        if any(len(row) != n for row in bases_seq):
            raise ValueError("gradient: inconsistent basis widths.")
        if n != self.num_visible:
            raise ValueError(f"gradient: basis width {n} != num_visible {self.num_visible}.")
        if samples.shape[0] != B:
            raise ValueError(f"gradient: samples batch {samples.shape[0]} != bases rows {B}.")

        if samples.dim() < 2:
            samples = samples.unsqueeze(0)  # if B==1

        # Bucketize identical basis rows
        buckets = {}
        for i, row in enumerate(bases_seq):
            buckets.setdefault(row, []).append(i)

        for basis_t, idxs in buckets.items():
            idxs_t = torch.tensor(idxs, device=samples.device)
            has_non_z = any(ch != "Z" for ch in basis_t)
            if has_non_z:
                g_am, g_ph = self.rotated_gradient(basis_t, samples[idxs_t, :])
                G_am += g_am
                G_ph += g_ph
            else:
                G_am += self.rbm_am.effective_energy_gradient(samples[idxs_t, :])

        return [G_am, G_ph]

    def positive_phase_gradients(self, samples_batch, bases_batch=None):
        grad = self.gradient(samples_batch, bases=bases_batch)
        return [g / float(samples_batch.shape[0]) for g in grad]

    def compute_batch_gradients(self, k, samples_batch, neg_batch, bases_batch=None):
        grad = self.positive_phase_gradients(samples_batch, bases_batch=bases_batch)
        vk = self.rbm_am.gibbs_steps(k, neg_batch)
        grad_model = self.rbm_am.effective_energy_gradient(vk)
        grad[0] -= grad_model / float(neg_batch.shape[0])
        return grad

    # -------- Simplified training loop: the loader does all batching/guardrails --------
    def fit(self, loader, epochs=100, k=1, lr=1e-3, log_every=5,
            optimizer=torch.optim.SGD, optimizer_args=None,
            target=None, bases=None, space=None,
            print_metrics=True, metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):
        """
        Contrastive Divergence (CD) training using RBMTomographyLoader.
        The loader yields (pos_batch, neg_batch, bases_batch) with all checks done.
        """
        if self.stop_training:
            return {"epoch": []}
        optimizer_args = {} if optimizer_args is None else optimizer_args
        all_params = list(chain.from_iterable(getattr(self, n).parameters() for n in ["rbm_am", "rbm_ph"]))
        opt = optimizer(all_params, lr=lr, **optimizer_args)

        history = {"epoch": []}
        if target is not None:
            history["Fidelity"], history["KL"] = [], []
        if space is None:
            space = self.generate_hilbert_space()

        for ep in range(1, epochs + 1):
            for pos_batch, neg_batch, bases_batch in loader.iter_epoch():
                grads = self.compute_batch_gradients(k, pos_batch, neg_batch, bases_batch)
                opt.zero_grad()
                for i, net in enumerate(["rbm_am", "rbm_ph"]):
                    rbm = getattr(self, net)
                    vector_to_grads(grads[i], rbm.parameters())
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