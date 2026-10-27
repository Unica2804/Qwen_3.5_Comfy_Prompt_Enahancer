"""Sampling helpers (greedy / temperature / top-k / top-p)."""
import torch


@torch.no_grad()
def sample_logits(logits, temperature=0.0, top_k=0, top_p=1.0, generator=None):
    """logits: [vocab] tensor on GPU. Returns an int token id."""
    if temperature is None or temperature <= 1e-5:
        return int(torch.argmax(logits))
    x = logits.float() / float(temperature)
    if top_k and top_k > 0:
        k = min(int(top_k), x.numel())
        kth = torch.topk(x, k).values[-1]
        x = torch.where(x < kth, torch.full_like(x, float("-inf")), x)
    probs = torch.softmax(x, dim=-1)
    if top_p is not None and 0.0 < top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cs = torch.cumsum(sp, dim=-1)
        drop = (cs - sp) > top_p
        sp = torch.where(drop, torch.zeros_like(sp), sp)
        probs = torch.zeros_like(probs).scatter(0, si, sp)
    if probs.sum() <= 0:
        return int(torch.argmax(logits))
    return int(torch.multinomial(probs, 1, generator=generator))
