import time
from pathlib import Path
from typing import Any, Optional, Dict
import wandb

class NoopLogger:
    def log(self, **kwargs):
        pass
    def close(self):
        pass

class TBLogger(NoopLogger):
    def __init__(self, out_dir: str, flush_secs: int = 10, run_name: str | None = None):
        self.w = None
        self.hparams_logged = False
        run_name = run_name or time.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = Path(out_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.w = SummaryWriter(log_dir=str(run_dir), flush_secs=flush_secs)
        except Exception as e:
            print(f"[tblogger] tensorboard not availabe: {e}. logging disabled.")
        self._auto_hist_max_elems = 2048
        self.run_dir = str(run_dir)

    
    def log(self, step: Optional[int] = None, **kv: Any) -> None:
        if not self.w: return
        for k, v in kv.items():
            # text channel (opt-in via key prefix "text/")
            if isinstance(k, str) and k.startswith("text/"):
                try:
                    self.w.add_text(k[5:], str(v), global_step=step)
                except Exception:
                    pass
                continue

    def hparams(self, hparams: Dict[str, Any], metric_once: Optional[Dict[str, float]]) -> None:
        if not self.w or self.hparams_logged:
            return
        try:
            self.w.add_hparams(hparams, metric_once or {}, run_name="_hparams")
            self.hparams_logged = True
        except Exception as e:
            pass
    
    def flush(self):
        if self.w:
            try: self.w.flush()
            except Exception: pass

class WBLogger(NoopLogger):
    def __init__(self, project: str, run_name: str | None = None):
        try:
            wandb.init(project=project, name=run_name)
            self.wb = wandb
        except Exception as e:
            self.wb = None
    
    def log(self,  **kv: Any) -> None:
        if self.wb:
            self.wb.log(kv)



def init_logger(which: str, out_dir: str) -> NoopLogger | TBLogger | WBLogger:
    if which == "tensorboard":
        tb = TBLogger(out_dir)
        return tb if tb.w is not None else NoopLogger()
    elif which == "wandb":
        return WBLogger(project="koios")
    else:
        return NoopLogger()