import torch
import torch.nn as nn

class TopKGate(nn.Module):
    """
    top-k softmax gating with switch-style load-balancing aux loss
    """
    def  __init__(self, d_model: int, n_experts: int, k: int = 1):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.wg = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.wg(x)
        probs = torch.softmax(logits, dim=-1)
        topk_vals, topk_idx = torch.topk(probs, dim=-1, k=self.k)
        # load-balancing aux loss (switch):
        S, E = probs.size(), probs.size(1)
        importance = probs.mean(dim=0)
        hard1 = topk_idx[:, 0]
        load = torch.zeros(E, device=x.device)
        load.scatter_add(0, hard1, torch.ones_like(hard1, dtype=load.dtype))
        load = load / max(S, 1)
        aux_load = (E * (importance * load).sum())
        print("*" * 50)
        print(probs, importance, hard1, load, aux_load)
        print("*" * 50)
        return topk_idx, topk_vals, aux_load