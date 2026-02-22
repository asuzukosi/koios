import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attn_mask import causal_mask_sha
from typing import Optional
class SingleHeadSelfAttention(nn.Module):
    """
    single-head attention layer
    dimensionality flow:
    B =  batch size
    T =  sequence length
    d_model = model dimension
    x: (B, T, d_model) multiple batches, each batch has a sequence of T tokens, each token has a d_model dimension
    qkv: (B, T, 3 * d_model) multiple batches, each batch has a sequence of T tokens, each token has a 3 * d_model dimension
    q, k, v: (B, T, d_model) multiple batches, each batch has a sequence of T tokens, each token has a d_model dimension
    scores: (B, T, T) <- shows how each token relates to every other token in the sequence
    weights: (B, T, T) = softmax(scores)
    ctx: (B, T, d_model) <- weights @ v
    proj: (B, T, d_model) <- project the context back to the model dimension
    """
    def __init__(self, d_model:int, dropout: float=0.0, trace_shapes: bool=False):
        super().__init__()
        self.d_model: int = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.trace_shapes = trace_shapes

    def forward(self, x: torch.Tensor, cross_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        forward pass of the input
        """
        # validate the shape
        if self.trace_shapes:
            print(f"input x: {tuple(x.shape)} = B, T, d_model")
        B, T, d_model = x.shape
        assert d_model == self.d_model, "d_model must be equal to the model dimension"
        # compute the qkv
        qkv: torch.Tensor = self.qkv(x)
        # assert the shape of qkv
        assert qkv.shape == (B, T, 3 * d_model), "qkv must be of shape (B, T, 3 * d_model)"
        if self.trace_shapes:
            print(f"qkv: {tuple(qkv.shape)} = B, T, 3 * d_model")
        # split the qkv into q, k, v
        qkv = qkv.view(B, T, 3, d_model)
        if self.trace_shapes:
            print(f"qkv: {tuple(qkv.shape)} = B, T, 3, d_model")
        q, k, v = qkv.unbind(dim=2)
        print(f"q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)}")
        # assert the shape of q, k, v
        assert q.shape == (B, T, d_model), "q must be of shape (B, T, d_model)"
        assert k.shape == (B, T, d_model), "k must be of shape (B, T, d_model)"
        assert v.shape == (B, T, d_model), "v must be of shape (B, T, d_model)"
        # compute the scores
        scale = 1 / math.sqrt(d_model)
        scores = q @ k.transpose(-2, -1) * scale
        mask = causal_mask_sha(T, x.device)
        scores = scores.masked_fill(mask, float("-inf")) 
        if self.trace_shapes:
            print(f"scores: {tuple(scores.shape)} = B, T, T")
        # assert the shape of scores
        assert scores.shape == (B, T, T), "scores must be of shape (B, T, T)"
        # compute the weights
        weights = torch.softmax(scores, dim=-1)
        # assert the shape of weights
        assert weights.shape == (B, T, T), "weights must be of shape (B, T, T)"
        # compute the context
        if cross_v is None:
            assert v.shape == (B, T, d_model), "v must be of shape (B, T, d_model)"
            ctx: torch.Tensor = weights @ v
        else:
            assert cross_v.shape == (B, T, d_model), "cross_v must be of shape (B, T, d_model)"
            ctx: torch.Tensor = weights @ cross_v
        # assert the shape of ctx
        assert ctx.shape == (B, T, d_model), "ctx must be of shape (B, T, d_model)"
        # project the context back to the model dimension
        out: torch.Tensor = self.proj(ctx)
        # assert the shape of out
        assert out.shape == (B, T, d_model), "out must be of shape (B, T, d_model)"
        if self.trace_shapes:
            print(f"output out: {tuple(out.shape)} = B, T, d_model")
        # return the output
        return out

if __name__ == "__main__":
    # test the multi-head attention layer
    d_model = 12
    x = torch.randn(1, 5, d_model)
    attn = SingleHeadSelfAttention(d_model, trace_shapes=True)
    out: torch.Tensor = attn(x)
    print(f"output out: {tuple(out.shape)} = B, T, d_model")
