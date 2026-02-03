import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class Hypernetwork(nn.Module):
    def __init__(self, num_v: int, num_h: int, cond_dim: int, hyper_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(cond_dim, hyper_dim)
        self.fc2 = nn.Linear(hyper_dim, 2 * (num_v + num_h))
        self.output_slicing = [num_v, num_v, num_h, num_h]

    def forward(self, cond: torch.Tensor):
        x = torch.tanh(self.fc1(cond))
        x = self.fc2(x)
        return torch.split(x, self.output_slicing, dim=-1)


class VanillaHyperRBM(nn.Module):
    """
    Vanilla RBM with FiLM-style hypernetwork conditioning of biases:
      b_mod(cond) = (1 + gamma_b) * b + beta_b
      c_mod(cond) = (1 + gamma_c) * c + beta_c

    Uses standard free energy and standard two-step Gibbs sampling.
    """
    def __init__(self, num_v: int, num_h: int, hyper_dim: int = 64, k: int = 10):
        super().__init__()
        self.num_v = num_v
        self.num_h = num_h
        self.k = k

        # physical reference temperature (also used for training unless you override)
        self.T = 1.0

        self.W = nn.Parameter(torch.empty(num_v, num_h))
        self.b = nn.Parameter(torch.zeros(num_v))
        self.c = nn.Parameter(torch.zeros(num_h))

        # keep cond_dim = 1 as in your original
        self.conditioner = Hypernetwork(num_v, num_h, 1, hyper_dim)

        self.initialize_weights()

    def initialize_weights(self, std: float = 0.01):
        nn.init.normal_(self.W, std=std)
        nn.init.constant_(self.b, 0.0)
        nn.init.constant_(self.c, 0.0)

    def _compute_mod_biases(self, cond: torch.Tensor):
        gamma_b, beta_b, gamma_c, beta_c = self.conditioner(cond)

        b_mod = (1.0 + gamma_b) * self.b.unsqueeze(0) + beta_b  # (B, V)
        c_mod = (1.0 + gamma_c) * self.c.unsqueeze(0) + beta_c  # (B, H)
        return b_mod, c_mod

    def _free_energy(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor) -> torch.Tensor:
        v = v.to(dtype=self.W.dtype, device=self.W.device)
        # F(v) = - v^T b - sum_j softplus((vW)_j + c_j)
        vb = (v * b_mod).sum(dim=-1)
        preact = v @ self.W + c_mod
        sp = F.softplus(preact).sum(dim=-1)
        return -vb - sp

    def _sample_h_given_v(
            self, v: torch.Tensor, c_mod: torch.Tensor, rng: Optional[torch.Generator], T: float
    ):
        logits = (v @ self.W + c_mod) / T
        p_h = torch.sigmoid(logits)
        h = torch.bernoulli(p_h, generator=rng)
        return h, p_h

    def _sample_v_given_h(
            self, h: torch.Tensor, b_mod: torch.Tensor, rng: Optional[torch.Generator], T: float
    ):
        logits = (h @ self.W.t() + b_mod) / T
        p_v = torch.sigmoid(logits)
        v = torch.bernoulli(p_v, generator=rng)
        return v, p_v

    def _gibbs_step(self, v: torch.Tensor, b_mod: torch.Tensor, c_mod: torch.Tensor,
                    rng: Optional[torch.Generator], T: float):
        h, _ = self._sample_h_given_v(v, c_mod, rng, T)
        v, _ = self._sample_v_given_h(h, b_mod, rng, T)
        return v, h

    def log_score(self, v: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Up to an additive constant (log Z). Kept consistent with your original scaling.
        """
        b_mod, c_mod = self._compute_mod_biases(cond)
        return -0.5 * self._free_energy(v, b_mod, c_mod) / self.T

    def forward(self, batch: Tuple[torch.Tensor, ...], aux_vars: Dict[str, Any]):
        """
        CD-k style training step.
        aux_vars may contain:
          - rng: torch.Generator
          - noise_frac: float (fraction of chains to random-restart)
          - T: float (optional override of temperature during training)
        """
        rng = aux_vars.get("rng")
        T = float(aux_vars.get("T", self.T))

        v_data, _, cond = batch
        v_data = v_data.to(device=self.W.device, dtype=self.W.dtype)
        cond = cond.to(device=self.W.device, dtype=self.W.dtype)

        b_mod, c_mod = self._compute_mod_biases(cond)

        v_model = v_data.clone()

        # optional random restarts for mixing
        noise_frac = float(aux_vars.get("noise_frac", 0.0))
        num_reset = int(v_data.size(0) * noise_frac)
        if num_reset > 0:
            v_model[:num_reset] = torch.bernoulli(
                torch.full_like(v_model[:num_reset], 0.5), generator=rng
            )

        h = torch.zeros((v_model.size(0), self.num_h), device=v_model.device, dtype=v_model.dtype)

        for _ in range(self.k):
            v_model, h = self._gibbs_step(v_model, b_mod, c_mod, rng, T=T)

        v_model = v_model.detach()

        loss = self._free_energy(v_data, b_mod, c_mod).mean() - self._free_energy(v_model, b_mod, c_mod).mean()
        return loss, {}

    @torch.no_grad()
    def generate(self, cond_batch: torch.Tensor, T_schedule: torch.Tensor,
                 rng: Optional[torch.Generator] = None) -> torch.Tensor:
        device = self.W.device
        dtype = self.W.dtype
        cond_batch = cond_batch.to(device=device, dtype=dtype)
        if cond_batch.dim() == 1:
            cond_batch = cond_batch.unsqueeze(0)

        B = cond_batch.shape[0]
        b_mod, c_mod = self._compute_mod_biases(cond_batch)

        v = torch.bernoulli(torch.full((B, self.num_v), 0.5, device=device, dtype=dtype), generator=rng)
        h = torch.zeros((B, self.num_h), device=device, dtype=dtype)

        for T_val in T_schedule:
            v, h = self._gibbs_step(v, b_mod, c_mod, rng, T=float(T_val))

        return v
