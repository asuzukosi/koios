import numpy as np
import torch
from src.attn_mask import causal_mask_mha, causal_mask_sha


def test_causal_mask_mha():
    T = 3
    mask = causal_mask_mha(T)
    assert mask.shape == (1, 1, T, T)
    assert mask.dtype == torch.bool
    assert not mask.all()

def test_causal_mask_sha():
    T = 3
    mask = causal_mask_sha(T)
    assert mask.shape == (1, T, T)
    assert mask.dtype == torch.bool
    assert not mask.all()