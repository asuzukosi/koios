import torch
import torch.nn as nn
from src.ffn import PointwiseFeedForward
from src.mha import MultiHeadSelfAttention

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