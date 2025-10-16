import time
import warnings
from math import ceil
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.convert_parameters import _check_param_device
from torch.nn.utils import parameters_to_vector
from torch.distributions.utils import probs_to_logits
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# Minimal complex arithmetic
# -------------------------------
I = torch.Tensor([0, 1])  # noqa: E741

def make_complex(x, y=None):
    if isinstance(x, np.ndarray):
        return make_complex(torch.tensor(x.real), torch.tensor(x.imag)).contiguous()
    if y is None:
        y = torch.zeros_like(x)
    return torch.cat((x.unsqueeze(0), y.unsqueeze(0)), dim=0)

def real(x): return x[0, ...]
def imag(x): return x[1, ...]
def conj(x): return make_complex(real(x), -imag(x))

def numpy(x):
    return real(x).detach().cpu().numpy() + 1j * imag(x).detach().cpu().numpy()

def scalar_mult(x, y, out=None):
    y = y.to(x)
    if out is None:
        out = torch.zeros(2, *((real(x) * real(y)).shape)).to(x)
    else:
        if out is x or out is y:
            raise RuntimeError("Can't overwrite an argument!")
    torch.mul(real(x), real(y), out=real(out)).sub_(torch.mul(imag(x), imag(y)))
    torch.mul(real(x), imag(y), out=imag(out)).add_(torch.mul(imag(x), real(y)))
    return out

def matmul(x, y):
    y = y.to(x)
    re = torch.matmul(real(x), real(y)).sub_(torch.matmul(imag(x), imag(y)))
    im = torch.matmul(real(x), imag(y)).add_(torch.matmul(imag(x), real(y)))
    return make_complex(re, im)

def inner_prod(x, y):
    y = y.to(x)
    if x.dim() == 2 and y.dim() == 2:
        return make_complex(
            torch.dot(real(x), real(y)) + torch.dot(imag(x), imag(y)),
            torch.dot(real(x), imag(y)) - torch.dot(imag(x), real(y)),
            )
    elif x.dim() == 1 and y.dim() == 1:
        return make_complex(
            (real(x) * real(y)) + (imag(x) * imag(y)),
            (real(x) * imag(y)) - (imag(x) * real(y)),
            )
    else:
        raise ValueError("Unsupported input shapes!")

def einsum(equation, a, b, real_part=True, imag_part=True):
    if real_part:
        r = torch.einsum(equation, real(a), real(b)).sub_(
            torch.einsum(equation, imag(a), imag(b))
        )
    if imag_part:
        i = torch.einsum(equation, real(a), imag(b)).add_(
            torch.einsum(equation, imag(a), real(b))
        )
    if real_part and imag_part: return make_complex(r, i)
    elif real_part: return r
    elif imag_part: return i
    else: return None

def absolute_value(x):
    return real(scalar_mult(x, conj(x))).sqrt_()

def inverse(z):
    z_star = conj(z)
    denominator = real(scalar_mult(z, z_star))
    return z_star / denominator

# -------------------------------
# Unitaries & rotations
# -------------------------------
def create_dict(**kwargs):
    dictionary = {
        "X": torch.tensor(
            [[[1.0, 1.0], [1.0, -1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.double,
        ) / np.sqrt(2),
        "Y": torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0]], [[0.0, -1.0], [0.0, 1.0]]],
            dtype=torch.double,
        ) / np.sqrt(2),
        "Z": torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]],
            dtype=torch.double,
        ),
    }
    dictionary.update(
        {
            name: (
                matrix.clone().detach()
                if isinstance(matrix, torch.Tensor)
                else torch.tensor(matrix)
            ).to(dtype=torch.double)
            for name, matrix in kwargs.items()
        }
    )
    return dictionary

def _kron_mult(matrices, x):
    n = [m.size()[0] for m in matrices]
    l, r = np.prod(n), 1  # noqa: E741
    if l != x.shape[1]:
        raise ValueError("Incompatible sizes!")
    y = x.clone()
    for s in reversed(range(len(n))):
        l //= n[s]  # noqa: E741
        m = matrices[s]
        for k in range(l):
            for i in range(r):
                slc = slice(k * n[s] * r + i, (k + 1) * n[s] * r + i, r)
                temp = y[:, slc, ...]
                y[:, slc, ...] = matmul(m, temp)
        r *= n[s]
    return y

def rotate_psi(nn_state, basis, space, unitaries=None, psi=None):
    psi = nn_state.psi(space) if psi is None else psi.to(dtype=torch.double, device=nn_state.device)
    unitaries = unitaries if unitaries else nn_state.unitary_dict
    unitaries = {k: v.to(device=nn_state.device) for k, v in unitaries.items()}
    us = [unitaries[b] for b in basis]
    return _kron_mult(us, psi)

def _rotate_basis_state(nn_state, basis, states, unitaries=None):
    unitaries = unitaries if unitaries else nn_state.unitary_dict
    unitaries = {k: v.to(device="cpu") for k, v in unitaries.items()}
    basis = np.array(list(basis))
    sites = np.where(basis != "Z")[0]
    if sites.size != 0:
        Us = torch.stack([unitaries[b] for b in basis[sites]]).cpu().numpy()
        reps = [1 for _ in states.shape]
        v = states.unsqueeze(0).repeat(2 ** sites.size, *reps)
        v[..., sites] = nn_state.generate_hilbert_space(size=sites.size).unsqueeze(1)
        v = v.contiguous()
        int_sample = states[..., sites].round().int().cpu().numpy()
        ints_size = np.arange(sites.size)
        int_vp = v[..., sites].long().cpu().numpy()
        all_Us = Us[ints_size, :, int_sample, int_vp]
        Ut = np.prod(all_Us[..., 0] + (1j * all_Us[..., 1]), axis=-1)
    else:
        v = states.unsqueeze(0)
        Ut = np.ones(v.shape[:-1], dtype=complex)
    return Ut, v

def _convert_basis_element_to_index(states):
    powers = (2 ** (torch.arange(states.shape[-1], 0, -1) - 1)).to(states)
    return torch.matmul(states, powers)

def rotate_psi_inner_prod(nn_state, basis, states, unitaries=None, psi=None, include_extras=False):
    Ut, v = _rotate_basis_state(nn_state, basis, states, unitaries=unitaries)
    if psi is None:
        psi = nn_state.psi(v).detach()
    else:
        idx = _convert_basis_element_to_index(v).long()
        psi = psi[:, idx]
    psi = numpy(psi.cpu())
    Ut *= psi
    Upsi_v = make_complex(Ut).to(dtype=torch.double, device=nn_state.device)
    Upsi = torch.sum(Upsi_v, dim=1)
    return (Upsi, Upsi_v, v) if include_extras else Upsi

# -------------------------------
# Data utils
# -------------------------------
def load_data(tr_samples_path, tr_psi_path=None, tr_bases_path=None, bases_path=None):
    data = []
    data.append(torch.tensor(np.loadtxt(tr_samples_path, dtype="float32"), dtype=torch.double))
    if tr_psi_path is not None:
        target_psi_data = np.loadtxt(tr_psi_path, dtype="float32")
        target_psi = torch.zeros(2, len(target_psi_data), dtype=torch.double)
        target_psi[0] = torch.tensor(target_psi_data[:, 0], dtype=torch.double)
        target_psi[1] = torch.tensor(target_psi_data[:, 1], dtype=torch.double)
        data.append(target_psi)
    if tr_bases_path is not None:
        data.append(np.loadtxt(tr_bases_path, dtype=str))
    if bases_path is not None:
        data.append(np.loadtxt(bases_path, dtype=str, ndmin=1))
    return data

def extract_refbasis_samples(train_samples, train_bases):
    torch_ver = int(torch.__version__[:3].replace(".", ""))
    dtype = torch.bool if torch_ver >= 12 else torch.uint8
    idx = (torch.tensor((train_bases == "Z").astype(np.uint8)).all(dim=1)
           .to(device=train_samples.device, dtype=dtype))
    return train_samples[idx]

# -------------------------------
# Explicit gradient utils
# -------------------------------
def vector_to_grads(vec, parameters):
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, but got: {torch.typename(vec)}")
    param_device = None
    pointer = 0
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        num_param = param.numel()
        param.grad = vec[pointer: pointer + num_param].view(param.size()).data
        pointer += num_param

# -------------------------------
# RBM (Bernoulli/Bernoulli) — NO DECORATORS
# -------------------------------
class BinaryRBM(nn.Module):
    def __init__(self, num_visible, num_hidden=None, zero_weights=False, gpu=True):
        super().__init__()
        self.num_visible = int(num_visible)
        self.num_hidden = int(num_hidden) if num_hidden else self.num_visible
        self.num_pars = (self.num_visible * self.num_hidden) + self.num_visible + self.num_hidden
        self.gpu = gpu and torch.cuda.is_available()
        self.device = torch.device("cuda") if self.gpu else torch.device("cpu")
        self.initialize_parameters(zero_weights=zero_weights)

    def __repr__(self):
        return f"BinaryRBM(num_visible={self.num_visible}, num_hidden={self.num_hidden}, gpu={self.gpu})"

    def initialize_parameters(self, zero_weights=False):
        gen_tensor = torch.zeros if zero_weights else torch.randn
        self.weights = nn.Parameter(
            (gen_tensor(self.num_hidden, self.num_visible, device=self.device, dtype=torch.double)
             / np.sqrt(self.num_visible)),
            requires_grad=False,
        )
        self.visible_bias = nn.Parameter(torch.zeros(self.num_visible, device=self.device, dtype=torch.double),
                                         requires_grad=False)
        self.hidden_bias = nn.Parameter(torch.zeros(self.num_hidden, device=self.device, dtype=torch.double),
                                        requires_grad=False)

    def effective_energy(self, v):
        """Inline version of @auto_unsqueeze_args behavior."""
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0); unsq = True
        v = v.to(self.weights)
        visible_bias_term = torch.matmul(v, self.visible_bias)
        hid_bias_term = F.softplus(F.linear(v, self.weights, self.hidden_bias)).sum(-1)
        out = -(visible_bias_term + hid_bias_term)
        return out.squeeze(0) if unsq else out

    def effective_energy_gradient(self, v, reduce=True):
        v = (v.unsqueeze(0) if v.dim() < 2 else v).to(self.weights)
        prob = self.prob_h_given_v(v)  # returns same batch rank as v
        if reduce:
            W_grad = -torch.matmul(prob.transpose(0, -1), v)
            vb_grad = -torch.sum(v, 0)
            hb_grad = -torch.sum(prob, 0)
            return parameters_to_vector([W_grad, vb_grad, hb_grad])
        else:
            W_grad = -torch.einsum("...j,...k->...jk", prob, v)
            vb_grad = -v
            hb_grad = -prob
            vec = [W_grad.view(*v.shape[:-1], -1), vb_grad, hb_grad]
            return torch.cat(vec, dim=-1)

    def prob_v_given_h(self, h, out=None):
        """Inline unsqueeze; respects optional out."""
        unsq = False
        if h.dim() < 2:
            h = h.unsqueeze(0); unsq = True
        res = torch.matmul(h, self.weights.data).add_(self.visible_bias.data).sigmoid_().clamp_(0, 1)
        if out is not None:
            if unsq and out.dim() == 1:
                out.copy_(res.squeeze(0))
            else:
                out.copy_(res)
            return out
        return res.squeeze(0) if unsq else res

    def prob_h_given_v(self, v, out=None):
        """Inline unsqueeze; respects optional out."""
        unsq = False
        if v.dim() < 2:
            v = v.unsqueeze(0); unsq = True
        res = torch.matmul(v, self.weights.data.t()).add_(self.hidden_bias.data).sigmoid_().clamp_(0, 1)
        if out is not None:
            if unsq and out.dim() == 1:
                out.copy_(res.squeeze(0))
            else:
                out.copy_(res)
            return out
        return res.squeeze(0) if unsq else res

    def sample_v_given_h(self, h, out=None):
        probs = self.prob_v_given_h(h)  # compute without out to avoid shape constraints
        return torch.bernoulli(probs, out=out) if out is not None else torch.bernoulli(probs)

    def sample_h_given_v(self, v, out=None):
        probs = self.prob_h_given_v(v)
        return torch.bernoulli(probs, out=out) if out is not None else torch.bernoulli(probs)

    def gibbs_steps(self, k, initial_state, overwrite=False):
        v = (initial_state if overwrite else initial_state.clone()).to(self.weights)
        h = torch.zeros(*v.shape[:-1], self.num_hidden).to(self.weights)
        for _ in range(k):
            self.sample_h_given_v(v, out=h)
            self.sample_v_given_h(h, out=v)
        return v

    def partition(self, space):
        return (-self.effective_energy(space)).logsumexp(0).exp()

# -------------------------------
# ComplexWaveFunction (amp+phase RBMs)
# -------------------------------
class ComplexWaveFunction:
    _rbm_am = None
    _rbm_ph = None
    _device = None
    _stop_training = False

    def __init__(self, num_visible, num_hidden=None, unitary_dict=None, gpu=False, module=None):
        if gpu and torch.cuda.is_available():
            warnings.warn("ComplexWaveFunction on GPU is often slower than CPU.", ResourceWarning, 2)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if module is None:
            self.rbm_am = BinaryRBM(num_visible, num_hidden, gpu=gpu)
            self.rbm_ph = BinaryRBM(num_visible, num_hidden, gpu=gpu)
        else:
            self.rbm_am = module.to(self.device); self.rbm_am.device = self.device
            self.rbm_ph = module.to(self.device).clone(); self.rbm_ph.device = self.device

        self.num_visible = self.rbm_am.num_visible
        self.num_hidden = self.rbm_am.num_hidden
        self.device = self.rbm_am.device

        self.unitary_dict = unitary_dict if unitary_dict is not None else create_dict()
        self.unitary_dict = {k: v.to(self.device) for k, v in self.unitary_dict.items()}

    # basic props
    @property
    def stop_training(self): return self._stop_training
    @stop_training.setter
    def stop_training(self, new_val):
        if isinstance(new_val, bool): self._stop_training = new_val
        else: raise ValueError("stop_training must be bool!")
    @property
    def max_size(self): return 20
    @property
    def networks(self): return ["rbm_am", "rbm_ph"]
    @property
    def rbm_am(self): return self._rbm_am
    @rbm_am.setter
    def rbm_am(self, new_val): self._rbm_am = new_val
    @property
    def rbm_ph(self): return self._rbm_ph
    @rbm_ph.setter
    def rbm_ph(self, new_val): self._rbm_ph = new_val
    @property
    def device(self): return self._device
    @device.setter
    def device(self, new_val): self._device = new_val

    def __getattr__(self, attr):
        return getattr(self.rbm_am, attr)

    # core ops
    def reinitialize_parameters(self):
        for net in self.networks:
            getattr(self, net).initialize_parameters()

    def amplitude(self, v):
        v = v.to(self.device, dtype=torch.double)
        return (-self.rbm_am.effective_energy(v)).exp().sqrt()

    def phase(self, v):
        v = v.to(self.device, dtype=torch.double)
        return -0.5 * self.rbm_ph.effective_energy(v)

    def psi(self, v):
        amp, ph = self.amplitude(v), self.phase(v)
        return make_complex(amp * ph.cos(), amp * ph.sin())

    def probability(self, v, Z=1.0):
        v = v.to(device=self.device, dtype=torch.double)
        return (-self.rbm_am.effective_energy(v)).exp() / Z

    def normalization(self, space):
        return self.rbm_am.partition(space)

    def generate_hilbert_space(self, size=None, device=None):
        device = self.device if device is None else device
        size = self.rbm_am.num_visible if size is None else size
        if size > self.max_size: raise ValueError("Hilbert space too large!")
        dim = np.arange(2 ** size)
        space = ((dim[:, None] & (1 << np.arange(size))) > 0)[:, ::-1].astype(int)
        return torch.tensor(space, dtype=torch.double, device=device)

    def sample(self, k, num_samples=1, initial_state=None, overwrite=False):
        if initial_state is None:
            dist = torch.distributions.Bernoulli(probs=0.5)
            shape = torch.Size((num_samples, self.num_visible))
            initial_state = dist.sample(shape).to(self.device, dtype=torch.double)
        return self.rbm_am.gibbs_steps(k, initial_state, overwrite=overwrite)

    # gradients
    def am_grads(self, v):
        return make_complex(self.rbm_am.effective_energy_gradient(v, reduce=False))

    def ph_grads(self, v):
        return scalar_mult(make_complex(self.rbm_ph.effective_energy_gradient(v, reduce=False)), I)

    def rotated_gradient(self, basis, sample):
        Upsi, Upsi_v, v = rotate_psi_inner_prod(self, basis, sample, include_extras=True)
        inv_Upsi = inverse(Upsi)
        raw_grads = [self.am_grads(v), self.ph_grads(v)]
        rotated_grad = [einsum("ib,ibg->bg", Upsi_v, g) for g in raw_grads]
        grad = [einsum("b,bg->g", inv_Upsi, rg, imag_part=False) for rg in rotated_grad]
        return grad

    def gradient(self, samples, bases=None):
        grad = [torch.zeros(getattr(self, net).num_pars, dtype=torch.double, device=self.device)
                for net in self.networks]
        if bases is None:
            grad[0] = self.rbm_am.effective_energy_gradient(samples)
            return grad
        if samples.dim() < 2:
            samples = samples.unsqueeze(0)
            bases = np.array(list(bases)).reshape(1, -1)
        unique_bases, indices = np.unique(bases, axis=0, return_inverse=True)
        indices = torch.tensor(indices, device=samples.device)
        for i in range(unique_bases.shape[0]):
            basis = unique_bases[i, :]
            if np.any(basis != "Z"):
                g_am, g_ph = self.rotated_gradient(basis, samples[indices == i, :])
                grad[0] += g_am; grad[1] += g_ph
            else:
                grad[0] += self.rbm_am.effective_energy_gradient(samples[indices == i, :])
        return grad

    def positive_phase_gradients(self, samples_batch, bases_batch=None):
        grad = self.gradient(samples_batch, bases=bases_batch)
        return [g / float(samples_batch.shape[0]) for g in grad]

    def compute_batch_gradients(self, k, samples_batch, neg_batch, bases_batch=None):
        grad = self.positive_phase_gradients(samples_batch, bases_batch=bases_batch)
        vk = self.rbm_am.gibbs_steps(k, neg_batch)
        grad_model = self.rbm_am.effective_energy_gradient(vk)
        grad[0] -= grad_model / float(neg_batch.shape[0])
        return grad

    # training
    def _shuffle_data(self, pos_batch_size, neg_batch_size, num_batches, train_samples, input_bases, z_samples):
        pos_perm = torch.randperm(train_samples.shape[0])
        pos_samples = train_samples[pos_perm]
        if input_bases is None:
            neg_perm = pos_perm if neg_batch_size == pos_batch_size else torch.randint(
                train_samples.shape[0], size=(num_batches * neg_batch_size,), dtype=torch.long)
            neg_samples = train_samples[neg_perm]
        else:
            neg_perm = torch.randint(z_samples.shape[0], size=(num_batches * neg_batch_size,), dtype=torch.long)
            neg_samples = z_samples[neg_perm]
        pos_batches = [pos_samples[i:i + pos_batch_size] for i in range(0, len(pos_samples), pos_batch_size)]
        neg_batches = [neg_samples[i:i + neg_batch_size] for i in range(0, len(neg_samples), neg_batch_size)]
        if input_bases is not None:
            pos_bases = input_bases[pos_perm]
            pos_bases_batches = [pos_bases[i:i + pos_batch_size] for i in range(0, len(train_samples), pos_batch_size)]
            return zip(pos_batches, neg_batches, pos_bases_batches)
        else:
            return zip(pos_batches, neg_batches)

    def fit(self, data, epochs=100, pos_batch_size=100, neg_batch_size=None, k=1, lr=1e-3,
            input_bases=None, log_every=5, progbar=True, starting_epoch=1,
            optimizer=torch.optim.SGD, optimizer_args=None, scheduler=None,
            scheduler_args=None, target=None, bases=None, space=None, timeit=False,
            print_metrics=True, metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"):

        if input_bases is None:
            raise ValueError("input_bases must be provided to train ComplexWaveFunction!")
        if self.stop_training: return {"epoch": []}

        neg_batch_size = neg_batch_size or pos_batch_size
        optimizer_args = {} if optimizer_args is None else optimizer_args
        scheduler_args = {} if scheduler_args is None else scheduler_args

        train_samples = data.clone().detach().to(self.device, dtype=torch.double) if isinstance(data, torch.Tensor) \
            else torch.tensor(data, device=self.device, dtype=torch.double)
        z_samples = extract_refbasis_samples(train_samples, input_bases).to(self.device)

        all_params = list(chain.from_iterable(getattr(self, n).parameters() for n in self.networks))
        opt = optimizer(all_params, lr=lr, **optimizer_args)
        sch = scheduler(opt, **scheduler_args) if scheduler is not None else None

        history = {"epoch": []}
        want_metrics = target is not None
        if want_metrics:
            history["Fidelity"], history["KL"] = [], []
        if timeit: history["TimeSec"] = []
        if space is None: space = self.generate_hilbert_space()

        num_batches = ceil(train_samples.shape[0] / pos_batch_size)
        epoch_iter = tqdm(range(starting_epoch, epochs + 1), desc="Epochs", disable=not progbar)

        for ep in epoch_iter:
            t0 = time.time() if timeit else None
            data_iter = self._shuffle_data(pos_batch_size, neg_batch_size, num_batches,
                                           train_samples, input_bases, z_samples)
            for b, batch in enumerate(data_iter):
                grads = self.compute_batch_gradients(k, *batch)
                opt.zero_grad()
                for i, net in enumerate(self.networks):
                    rbm = getattr(self, net)
                    vector_to_grads(grads[i], rbm.parameters())
                opt.step()
                if self.stop_training: break
            if sch is not None: sch.step()

            if want_metrics and (ep % log_every == 0):
                with torch.no_grad():
                    fid_val = fidelity(self, target, space=space, bases=bases)
                    kl_val = KL(self, target, space=space, bases=bases)
                history["epoch"].append(ep)
                history["Fidelity"].append(fid_val)
                history["KL"].append(kl_val)
                if progbar:
                    epoch_iter.set_postfix(Fidelity=f"{fid_val:.4f}", KL=f"{kl_val:.4f}")
                if print_metrics:
                    print(metric_fmt.format(ep=ep, fid=fid_val, kl=kl_val))
            if timeit: history.setdefault("TimeSec", []).append(time.time() - t0)
            if self.stop_training: break
        return history

# -------------------------------
# Metrics
# -------------------------------
def fidelity(nn_state, target, space=None, **kwargs):
    space = nn_state.generate_hilbert_space() if space is None else space
    Z = nn_state.normalization(space)
    target = target.to(nn_state.device)
    psi = nn_state.psi(space) / Z.sqrt()
    overlap = inner_prod(target, psi)
    return absolute_value(overlap).pow_(2).item()

def _single_basis_KL(target_probs, nn_probs):
    return torch.sum(target_probs * probs_to_logits(target_probs)) - torch.sum(
        target_probs * probs_to_logits(nn_probs)
    )

def KL(nn_state, target, space=None, bases=None, **kwargs):
    space = nn_state.generate_hilbert_space() if space is None else space
    Z = nn_state.normalization(space)
    target = target.to(nn_state.device)
    KL_val = 0.0
    for basis in bases:
        tgt_psi_r = rotate_psi(nn_state, basis, space, psi=target)
        psi_r = rotate_psi(nn_state, basis, space)
        nn_probs_r = (absolute_value(psi_r) ** 2) / Z
        tgt_probs_r = absolute_value(tgt_psi_r) ** 2
        KL_val += _single_basis_KL(tgt_probs_r, nn_probs_r)
    return (KL_val / len(bases)).item()

# -------------------------------
# Training script (as before)
# -------------------------------
if __name__ == "__main__":
    train_path = "w_state_meas.txt"
    train_bases_path = "w_state_basis.txt"
    psi_path = "w_state_aug.txt"
    bases_path = "w_state_bases.txt"

    train_samples, true_psi, train_bases, bases = load_data(
        train_path, psi_path, train_bases_path, bases_path
    )

    one_hot_indices = [2**i for i in range(true_psi.shape[1].bit_length() - 1)]
    one_hot_true_psi = true_psi[:, one_hot_indices]

    true_phases_raw = torch.angle(one_hot_true_psi[0, :] + 1j * one_hot_true_psi[1, :])
    true_phases_wrapped = (true_phases_raw - true_phases_raw[0]) % (2 * np.pi)

    torch.manual_seed(1234)
    unitary_dict = create_dict()

    nv = train_samples.shape[-1]; nh = nv
    nn_state = ComplexWaveFunction(nv, nh, unitary_dict, gpu=False)

    epochs = 70; pbs = 100; nbs = 100; lr = 1e-1; k = 10; log_every = 5
    space = nn_state.generate_hilbert_space()

    history = nn_state.fit(
        train_samples, epochs=epochs, pos_batch_size=pbs,
        neg_batch_size=nbs, lr=lr, k=k, input_bases=train_bases,
        progbar=True, log_every=log_every, target=true_psi,
        bases=bases, space=space, timeit=True, print_metrics=True,
        metric_fmt="Epoch {ep}: Fidelity = {fid:.6f} | KL = {kl:.6f}"
    )

    fidelities = np.array(history.get("Fidelity", []))
    KLs = np.array(history.get("KL", []))
    epoch = np.array(history.get("epoch", []))

    full_hs = nn_state.generate_hilbert_space()
    one_hot_hs = full_hs[one_hot_indices, :]
    pred_phases_raw = nn_state.phase(one_hot_hs)
    pred_phases_wrapped = (pred_phases_raw - pred_phases_raw[0]) % (2 * np.pi)

    plt.rcParams.update({"font.family": "serif"})
    bitstrings = ["".join(str(int(b)) for b in row) for row in one_hot_hs.cpu().numpy()]
    indices = np.arange(len(pred_phases_wrapped))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
    ax.set_facecolor('white')
    ax.bar(indices - width/2, true_phases_wrapped, width, alpha=0.7, color='gray',
           label=r'$\phi_{\mathrm{true}}$', zorder=1)
    ax.bar(indices + width/2, pred_phases_wrapped, width, alpha=0.7, color='blue',
           label=r'$\phi_{\mathrm{predicted}}$', zorder=2)
    ax.set_xlabel("Basis State", fontsize=14)
    ax.set_ylabel("Phase (radians)", fontsize=14)
    ax.set_title("Phase Comparison: Phase-Augmented $W$ State", fontsize=16)
    ax.set_xticks(indices)
    ax.set_xticklabels([f"${b}$" for b in bitstrings], rotation=45)
    ax.set_ylim(0, 2 * np.pi + 0.2)
    ax.legend(frameon=True, framealpha=1, loc='best', fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
