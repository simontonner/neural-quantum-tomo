import torch
import math
from typing import List, Tuple, Callable, Union


def compute_czz(samples: torch.Tensor, pairs: List[Tuple[int, int]]) -> Tuple[float, float]:
    spins_pm = 1.0 - 2.0 * samples.float()

    # vectorizing over pairs is faster
    idx_u = torch.tensor([p[0] for p in pairs], device=samples.device)
    idx_v = torch.tensor([p[1] for p in pairs], device=samples.device)
    spin_parities = spins_pm[:, idx_u] * spins_pm[:, idx_v]

    sample_czz = spin_parities.mean(dim=1)

    total_czz = sample_czz.mean().item()
    total_czz_err = sample_czz.std(unbiased=True).item() / math.sqrt(samples.shape[0])

    return total_czz, total_czz_err


def compute_cxx(samples: torch.Tensor, pairs: List[Tuple[int, int]],
                log_score_fn: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[float, float]:

    with torch.no_grad():
        log_scores_orig = log_score_fn(samples) # (B,)

        all_ratios = []
        for u, v in pairs:
            # Flip spins at u and v
            flipped = samples.clone()
            flipped[:, u] = 1 - flipped[:, u]
            flipped[:, v] = 1 - flipped[:, v]

            log_scores_flip = log_score_fn(flipped)

            # ratio = exp(log_flip - log_orig)
            ratios = torch.exp(log_scores_flip - log_scores_orig)
            all_ratios.append(ratios)

        # Stack -> (Batch, Num_Pairs)
        ratio_tensor = torch.stack(all_ratios, dim=1)

        # Spatial average per sample
        mean_per_sample = ratio_tensor.mean(dim=1)

        # Batch Statistics
        val = mean_per_sample.mean().item()
        err = mean_per_sample.std(unbiased=True).item() / math.sqrt(samples.shape[0])

        return val, err