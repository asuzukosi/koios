import torch
import torch.nn as nn
from torch.nn import functional as F
from src.gating import TopKGate
from src.ffn import ExpertMLP

class MoE(nn.Module):
    """
    mixture of experts layer (token-wise top-k routing)
    """
    def __init__(self, d_model: int, n_experts: int, k: int = 1, mult:int=4, swiglu: bool = True, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.k = k
        self.mult = mult
        self.swiglu = swiglu
        self.dropout = dropout
        self.gate = TopKGate(d_model, n_experts, k)
        self.experts = nn.ModuleList([ExpertMLP(d_model, mult, swiglu, dropout) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the input
        """
        B, T, d_model = x.shape
        S = B * T
        x_flat = x.view(S, d_model)
        idx, w, aux = self.gate(x_flat)

        y = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            for slot in range(self.k):
                sel = (idx[:, slot] == e) # returns true or false based on if the expert is selected for that slot
                if sel.any():
                    x_expert = x_flat[sel]
                    y_expert = self.experts[e](x_expert)
                    y[sel] += w[sel, slot: slot + 1] * y_expert # not all items in the batch will use the expert at the same intensity
        y = y.view(B, T, d_model)
        return y, aux # auxilary loss should be added to cross entropy loss before backpropagation
    
class HybridMoE(nn.Module):
    """
    hybrid mixture of experts layer (token-wise top-k routing)
    """
    def __init__(self, d_model: int, n_experts: int, alpha: float = 0.5, mult:int=4, k: int = 1, dropout: float = 0.0):
        super().__init__()
        self.alpha = alpha
        inner = mult * d_model
        self.dense = nn.Sequential(
            nn.Linear(d_model, inner, bias=False),
            nn.GELU(),
            nn.Linear(inner, d_model, bias=False),
            nn.Dropout(dropout)
        )
        self.moe = MoE(d_model, n_experts, k, mult, True, dropout)
        self.gate = TopKGate(d_model, n_experts, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_dense = self.dense(x)
        y_moe, aux_moe = self.moe(x)
        y = self.alpha * y_dense + (1 - self.alpha) * y_moe
        return y, aux_moe # auxilary loss should be added to cross entropy loss before backpropagation