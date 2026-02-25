from typing import List
from datasets import load_dataset
from src.bpe import BPETokenizer
from src.tokenizer import ByteTokenizer
import torch
from typing import Tuple

TEMPLATE = """
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>
"""

class PrefExample:
    prompt: str
    chosen: str
    rejected: str


def load_preferences(split: str = "train[:200]") -> list[PrefExample]:
    items: List[PrefExample] = []
    ds = load_dataset("Anthropic/hh-rlhf", split=split)
    for row in ds:
        ch = str(row.get("chosen", "")).strip()
        rj = str(row.get("rejected", "")).strip()
        if ch and rj:
            items.append(PrefExample(prompt="", chosen=ch, rejected=rj))


class PairCollator:
    """
    tokenize preference pairs into (pos, neg) input ids
    we format as the sft template with the 'chosen' or 'rejected' text as the response
    """
    def __init__(self, block_size: int = 256, bpe_dir: str | None = None, vocab_size: int | None = None):
        self.block_size = block_size
        self.tok = None
        self.tok = BPETokenizer(vocab_size=vocab_size or 32000)
        if bpe_dir:
            self.tok.load(bpe_dir)
        else:
            self.tok = ByteTokenizer()

    @property
    def vocab_size(self) -> int:
        return getattr(self.tok, 'vocab_size', 256)
    
    def _encode(self, text: str) -> list[int]:
        if hasattr(self.tok, 'encode'):
            ids = self.tok.encode(text)
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return ids
        else:
            return list(text.encode("utf-8"))
        
    def format_example(self, example: Tuple[str, str]) -> str:
        return TEMPLATE.format(prompt=example[0], response=example[1] or "")
        
    def collate(self, batch: List[Tuple[str, str, str]]):
        pos_ids, neg_ids = [], []
        for prompt, chosen, rejected in batch:
            pos_text = self.format_example((prompt, chosen))
            neg_text = self.format_example((prompt, rejected))
            pos_ids.append(self._encode(pos_text)[:self.block_size])
            neg_ids.append(self._encode(neg_text)[:self.block_size])
        # pad to block size
        def pad_to(ids, val):
            if len(ids) < self.block_size:
                ids = ids + [val] * (self.block_size - len(ids))
            return ids[:self.block_size]
        pos = [pad_to(it, 2) for it in pos_ids]
        neg = [pad_to(it, 2) for it in neg_ids]
        return torch.tensor(pos), torch.tensor(neg)
        