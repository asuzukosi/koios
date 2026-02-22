import os
import math
import torch
from src.mha import MultiHeadSelfAttention

OUT_TXT = os.path.join(os.path.dirname(__file__), "mha_shapes.txt")

def log(s: str):
    print(s)
    with open(OUT_TXT, "a") as f:
        f.write(s + "\n")



if __name__ == "__main__":
    # reset the output file
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)
    open(OUT_TXT, "w").close()

    B, T, d_model, n_head = 1, 5, 12, 3 # batch size, sequence length, model dimension, number of heads
    d_head = d_model // n_head # head dimension
    x = torch.randn(B, T, d_model) # input
    attn = MultiHeadSelfAttention(d_model, n_head, trace_shapes=True)
    log(f"input x: {tuple(x.shape)} = B, T, d_model")
    qkv: torch.Tensor = attn.qkv(x)
    log(f"qkv: {tuple(qkv.shape)} = B, T, 3 * d_model")
    qkv = qkv.view(B, T, 3, n_head, d_head) # (B, T, 3, n_head, d_head) 3 for q, k, v, n_head for number of heads, d_head for head dimension
    log(f"qkv: {tuple(qkv.shape)} = B, T, 3, n_head, d_head")
    q, k, v = qkv.unbind(dim=2)
    log(f"q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)}")
    q = q.transpose(1, 2) # (B, n_head, T, d_head)
    k = k.transpose(1, 2) # (B, n_head, T, d_head)
    v = v.transpose(1, 2) # (B, n_head, T, d_head)
    log(f"transpose heads q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)}")
    scale = 1 / math.sqrt(d_head)
    scores = q @ k.transpose(-2, -1) * scale
    log(f"scores q@k^T: {tuple(scores.shape)} = B, n_head, T, T")
    weights = torch.softmax(scores, dim=-1)
    log(f"softmax (weights): {tuple(weights.shape)} = B, n_head, T, d_head")
    ctx = weights @ v
    log(f"ctx: {tuple(ctx.shape)} = B, n_head, T, d_head")
    out = ctx.transpose(1, 2).contiguous().view(B, T, d_model)
    log(f"output out: {tuple(out.shape)} = B, T, d_model")
    out: torch.Tensor = attn.proj(out)
    log(f"output out: {tuple(out.shape)} = B, T, d_model")