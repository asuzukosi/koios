import torch
import torch.nn as nn
from src.logger import TBLogger


def _is_tb(logger) -> bool:
    return isinstance(logger, TBLogger)

def _log_hparams_tb(logger, args, total_steps) -> None:
    if not _is_tb(logger): return
    try:
        h = dict(
            vocab_size=args.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
            n_head=args.n_head, d_model=args.d_model, dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
            amp=args.amp, clip=args.clip, compile=args.compile, use_bpe=args.use_bpe, warmup_steps=args.warmup_steps,
            mixed_precision=args.mixed_precision, grad_accum_steps=args.grad_accum_steps, sample_every=args.sample_every
        )
        logger.hparams(h, {"meta/total_steps": total_steps})
    except Exception as e:
        pass


def _maybe_log_graph_tb(logger, model, xb, yb):
    if not hasattr(logger, "graph"): return
    try:
        class _TensorOnly(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m.eval()
            def forward(self, x, y=None):
                out = self.m(x, y) if y is not None else self.m(x)
                if isinstance(out, (list, tuple)):
                    for o in out:
                        if torch.is_tensor(o):
                            return o
                    return out[0]
                return out
        wrapped = _TensorOnly(model).to(xb.device)
        logger.graph(wrapped, (xb, yb))
    except Exception as e:
        pass
            