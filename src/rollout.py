from datasets import load_dataset
from typing import List
import torch
import torch.nn.functional as F

def sample_prompts(n: int) -> List[str]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    arr = []
    for r in ds:
        inst = (r.get("instruction", "").strip() or "").strip()
        inp = (r.get("input", "").strip() or "").strip()
        if inp:
            inst = inst + "\n\n" + inp
        if inst:
            arr.append(inst)
        if len(arr) >= n:
            break
    return arr


def shift_labels(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:].contiguous()

def gather_logprobs(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=-1)
    return logp.gather(-1, label.unsqueeze(-1)).squeeze(-1)

@torch.no_grad()
def model_logprobs(model, x: torch.Tensor) -> torch.Tensor:
    logits, _, _ = model.lm(x, None) if hasattr(model, "lm") else model(x, None)
    label = shift_labels(x)
    lp = gather_logprobs(logits, label)
    return lp

def approx_kl(policy_logprobs: torch.Tensor, ref_logprobs: torch.Tensor) -> torch.Tensor:
    return (policy_logprobs - ref_logprobs).mean()