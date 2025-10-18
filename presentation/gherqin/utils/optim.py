import torch

def vector_to_grads(vec, parameters):
    """Write a flattened gradient vector into a module's .grad buffers."""
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {torch.typename(vec)}")

    offset = 0
    for p in parameters:
        n = p.numel()
        if offset + n > vec.numel():
            raise ValueError("Gradient vector is too short for parameter shapes.")
        g_slice = vec[offset: offset + n].view_as(p)
        if p.grad is None or p.grad.shape != p.shape or p.grad.dtype != p.dtype or p.grad.device != p.device:
            p.grad = torch.empty_like(p)
        p.grad.copy_(g_slice.to(dtype=p.dtype, device=p.device))
        offset += n

    if offset != vec.numel():
        raise ValueError(f"Gradient vector has extra elements: used {offset}, total {vec.numel()}.")
