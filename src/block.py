import torch
import torch.nn as nn
from src.ffn import PointwiseFeedForward, SwishdGELU
from src.mha import MultiHeadSelfAttention
from src.rmsnorm import RMSNorm
from typing import Optional

class Block(nn.Module):
    """
    single transformer block
    """
    def __init__(self, d_model: int, n_head: int, dropout: float=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PointwiseFeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the input
        """
        # implement residual connection
        x = x + self.attn(self.ln1(x)) # x is a single pathway being augmented by the attention mechanism
        x = x + self.ffn(self.ln2(x)) # x is a single pathway being augmented by the feed forward network
        return x
    


class BlockModern(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwishdGELU(d_model, mult=4, dropout=dropout)

    def forward(self, x: torch.Tensor, kv_cache: Optional[torch.Tensor] = None, start_pos: int = 0) -> torch.Tensor:
        x, new_kv_cache = x + self.attn(self.ln1(x), kv_cache=kv_cache, start_pos=start_pos) # x is a single pathway being augmented by the attention mechanism
        x = x + self.ffn(self.ln2(x)) # x is a single pathway being augmented by the feed forward network
        return x, new_kv_cache
