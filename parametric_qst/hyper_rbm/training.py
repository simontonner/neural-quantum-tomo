import math
import torch
from torch.optim import Optimizer
from typing import Optional


def get_sigmoid_curve(high, low, steps, falloff, center=None):
    if center is None: center = steps / 2.0
    def fn(step):
        s = min(step, steps)
        return float(low + (high - low) / (1.0 + math.exp(falloff * (s - center))))
    return fn

def train_loop(
        model,
        optimizer: Optimizer,
        loader,
        num_epochs: int,
        lr_schedule_fn,
        noise_frac: float = 0.1,
        rng: Optional[torch.Generator] = None
):
    global_step = 0
    model.train()

    print(f"{'Epoch':<6} | {'Loss':<10} | {'LR':<10}")
    print("-" * 35)

    for epoch in range(num_epochs):
        tot_loss = 0.0
        for batch in loader:
            lr = lr_schedule_fn(global_step)
            optimizer.param_groups[0]["lr"] = lr        # we use a shared parameter group (0) over all parameters

            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch, {"rng": rng, "noise_frac": noise_frac})
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
            global_step += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = tot_loss / len(loader)
            print(f"{epoch+1:<6} | {avg_loss:+.4f}     | {lr:.6f}")

    return model