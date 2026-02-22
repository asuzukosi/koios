import numpy as np
import torch
from src.sha import SingleHeadSelfAttention




def test_single_head_self_attention():
    torch.manual_seed(42)
    x = np.array([[[0.1, 0.2, 0.3, 0.4],
              [0.5, 0.4, 0.3, 0.2],
              [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)
    x: torch.Tensor = torch.tensor(x, dtype=torch.float32)
    attn = SingleHeadSelfAttention(d_model=4, trace_shapes=True)
    out: torch.Tensor = attn(x)
    assert out.shape == (1, 3, 4), "out shape must be (1, 3, 4)"
    assert torch.isfinite(out).all(), "out must be finite"

