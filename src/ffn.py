import torch
import torch.nn as nn

class PointwiseFeedForward(nn.Module):
    """
    pointwise feed forward network of the transformer
    dimensionality flow:
    B =  batch size
    T =  sequence length
    d_model = model dimension
    x: (B, T, d_model) multiple batches, each batch has a sequence of T tokens, each token has a d_model dimension
    net: (B, T, d_model) <- linear(n_embed, mult * n_embed) -> relu -> dropout -> linear(mult * n_embed, n_embed)
    return: (B, T, d_model) <- project the net back to the model dimension
    """
    def __init__(self, d_model: int, mult: int=4, dropout: float=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult * d_model), # expand the model dimension
            nn.GELU(), # non-linearity
            nn.Dropout(dropout), # dropout
            nn.Linear(mult * d_model, d_model), # project back to the model dimension
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass of the input
        """
        return self.net(x)