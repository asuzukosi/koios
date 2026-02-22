import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.attn_mask import causal_mask_mha
from typing import Optional
class MultiHeadSelfAttention(nn.Module):
    """
    multi-head attention layer
    dimensionality flow:
    B =  batch size
    T =  sequence length
    d_model = model dimension
    n_head = number of heads
    d_head = head dimension
    x: (B, T, d_model) multiple batches, each batch has a sequence of T tokens, each token has a d_model dimension
    qkv: (B, T, 3 * d_model) multiple batches, each batch has a sequence of T tokens, each token has a 3 * d_model dimension
    view:  (B, T, 3, n_heads, d_head) multiple batches, each batch has a sequence of T tokens, each token has a sequence of 3 values for query, key, and value, each query, key, and value has a sequence of n_heads, each head has a length of d_head
    split q, k, v: (B, T, n_heads, d_head) for each query, key, and value, we split the sequence of n_heads into n_heads, each head has a sequence of T tokens, each token has a length of d_head
    swap heads: (B, n_heads, T, d_head) we swap the sequence of n_heads into n_heads, each head has a sequence of T tokens, each token has a length of d_head
    scores: (B, n_heads, T, T) <- shows how each token relates to every other token in the sequence
    weights: (B, n_heads, T, T) = softmax(scores)
    ctx: (B, n_heads, T, d_head) <- weights @ v
    merge = (B, T, d_model) <- transpose(B, n_heads, T, d_head) to swap n_heads and T (B, T, n_heads, d_head) and then view(B, T, d_model)
    proj: (B, T, d_model) <- project the merged context back to the model dimension
    """
    def __init__(self, d_model:int, n_head:int, dropout: float=0.0, trace_shapes: bool=False):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model: int = d_model
        self.n_head: int = n_head
        self.d_head: int = d_model // n_head
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
        # view the qkv into (B, T, 3, n_head, d_head)
        qkv = qkv.view(B, T, 3, self.n_head, self.d_head)
        if self.trace_shapes:
            print(f"qkv: {tuple(qkv.shape)} = B, T, 3, self.n_head, self.d_head")
        # assert the shape of qkv
        assert qkv.shape == (B, T, 3, self.n_head, self.d_head), "qkv must be of shape (B, T, 3, self.n_head, self.d_head)"
        # split the qkv into q, k, v
        q, k, v = qkv.unbind(dim=2)
        # assert the shape of q, k, v
        assert q.shape == (B, T, self.n_head, self.d_head), "q must be of shape (B, T, self.n_head, self.d_head)"
        assert k.shape == (B, T, self.n_head, self.d_head), "k must be of shape (B, T, self.n_head, self.d_head)"
        assert v.shape == (B, T, self.n_head, self.d_head), "v must be of shape (B, T, self.n_head, self.d_head)"
        # transpose to swap the heads and tokens
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.trace_shapes:
            print(f"transpose heads q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)}") # this is the standard shapes for the attention calculation
        # assert the shape of q, k, v
        assert q.shape == (B, self.n_head, T, self.d_head), "q must be of shape (B, self.n_head, T, self.d_head)"
        assert k.shape == (B, self.n_head, T, self.d_head), "k must be of shape (B, self.n_head, T, self.d_head)"
        assert v.shape == (B, self.n_head, T, self.d_head), "v must be of shape (B, self.n_head, T, self.d_head)"
        # compute the scores
        scale = 1 / math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) * scale
        mask = causal_mask_mha(T, x.device)
        scores = scores.masked_fill(mask, float("-inf")) 
        if self.trace_shapes:
            print(f"scores: {tuple(scores.shape)} = B, self.n_head, T, T")
        # assert the shape of scores
        assert scores.shape == (B, self.n_head, T, T), "scores must be of shape (B, self.n_head, T, T)"
        # compute the weights
        weights = torch.softmax(scores, dim=-1)
        # assert the shape of weights
        assert weights.shape == (B, self.n_head, T, T), "weights must be of shape (B, self.n_head, T, T)"
        # compute the context
        if cross_v is None:
            assert v.shape == (B, self.n_head, T, self.d_head), "v must be of shape (B, self.n_head, T, self.d_head)"
            ctx: torch.Tensor = weights @ v
        else:
            assert cross_v.shape == (B, self.n_head, T, self.d_head), "cross_v must be of shape (B, self.n_head, T, self.d_head)"
            ctx: torch.Tensor = weights @ cross_v
        # assert the shape of ctx
        assert ctx.shape == (B, self.n_head, T, self.d_head), "ctx must be of shape (B, self.n_head, T, self.d_head)"
        # transpose to swap the heads and tokens
        ctx = ctx.transpose(1, 2)
        # assert the shape of ctx
        assert ctx.shape == (B, T, self.n_head, self.d_head), "ctx must be of shape (B, T, self.n_head, self.d_head)"
        # merge the heads and tokens
        ctx = ctx.contiguous().view(B, T, self.d_model)
        # assert the shape of ctx
        assert ctx.shape == (B, T, self.d_model), "ctx must be of shape (B, T, self.d_model)"
        # project the context back to the model dimension
        out: torch.Tensor = self.proj(ctx)
        # assert the shape of out
        assert out.shape == (B, T, self.d_model), "out must be of shape (B, T, self.d_model)"
        if self.trace_shapes:
            print(f"output out: {tuple(out.shape)} = B, T, d_model")
        # return the output
        return out

if __name__ == "__main__":
    # test the multi-head attention layer
    d_model = 12
    n_head = 3
    x = torch.randn(1, 5, d_model)
    attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=True)
    out: torch.Tensor = attn(x)
    print(f"output out: {tuple(out.shape)} = B, T, d_model")
