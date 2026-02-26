import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import GPTModern

class PolicyWithValue(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layer=4, 
                 n_head=4, d_model=128, dropout=0.1):
        super().__init__()
        self.lm = GPTModern(vocab_size, block_size, n_layer, n_head, d_model, dropout)
        self.val_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits, loss, _ = self.lm(x, y)
        value = self.val_head(logits).squeeze(-1)
        return logits, loss, value
    
    def generate(self, *args, **kwargs) -> torch.Tensor:
        return self.lm.generate(*args, **kwargs)
        