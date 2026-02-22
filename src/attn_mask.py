import torch

def causal_mask_mha(T: int, device: torch.device = torch.device("cpu")) -> torch.Tensor: # (1, 1, T, T) torch.bool
    """
    create a causal mask for the attention calculation
    """
    m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return m.view(1, 1, T, T)


def causal_mask_sha(T: int, device: torch.device = torch.device("cpu")) -> torch.Tensor: 
    """
    create a causal mask for the attention calculation
    """
    m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
    return m.view(1, T, T)
