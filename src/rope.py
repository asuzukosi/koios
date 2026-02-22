import torch


class RopeCache:
    def __init__(self, head_dim:int, max_pos: int, base:float=10000.0, device: torch.device = torch.device("cpu")):
        assert head_dim % 2 == 0, "head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self.device = device
        self._build(max_pos)

    def _build(self, max_pos: int):
        self.max_pos = max_pos
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=self.device) / self.head_dim))
        t = torch.arange(max_pos, device=self.device).float()
        freqs = torch.outer(t, inv_freq)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

    def get(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 2:
            positions = positions[0]
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_pos:
            self._build(max(need, int(self.max_pos * 2)))
        cos = self.cos[positions]
        sin = self.sin[positions]
        return cos, sin

def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    rotate pairs along last dim for RoPE
    """
    assert x.size(-1) % 2 == 0, "head_dim must be even"
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    xr1 = x1 * cos  - x2 * sin
    xr2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    return out