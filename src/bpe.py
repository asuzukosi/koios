import os, json
from pathlib import Path
from typing import List, Optional, Union

try:
    from tokenizers import ByteLevelBPETokenizer
except Exception:
    ByteLevelBPETokenizer = None


# NOTE: tokenization is one of the key considerations when training a model for coding as code usually uses unique sequences that may be valuable for effective tokenization

class BPETokenizer:
    """
    minimal bpe wrapper (huggingface tokenizers)
    trains on a text file or a folder of .txt files.saves merges/vocab to out_dir
    """
    def __init__(self, vocab_size: int = 32000, special_tokens: List[str] | None = None):
        if ByteLevelBPETokenizer is None:
            raise ImportError("please install huggingface tokenizers to use BPE tokenizer")
        self.vocab_size = vocab_size
        self.special_token=special_tokens or ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
        self._tok = None

    def train(self, data_path: Union[str, Path]):
        files = List[str] = []
        p = Path(data_path)
        if p.is_file():
            files.append(str(p))
        elif p.is_dir():
            files.extend(str(f) for f in p.glob("**/*.txt"))
        else:
            raise ValueError(f"Invalid data path: {data_path}")
        tok = ByteLevelBPETokenizer()
        tok.train(files, vocab_size=self.vocab_size, special_tokens=self.special_token)
        self._tok = tok

    def save(self, out_dir: Union[str, Path]):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        assert self._tok is not None, "train or load save()"
        self._tok.save_model(str(out / "bpe"))
        meta = {"vocab_size": self.vocab_size, "special_tokens": self.special_token}
        (out / "meta.json").write_text(json.dumps(meta))

    def load(self, dirpath: Union[str, Path]):
        dirp = Path(dirpath)
        vocab = dirp / "vocab.json"
        merges = dirp / "merges.txt"
        assert vocab.is_file(), f"Invalid vocab file: {vocab}"
        assert merges.is_file(), f"Invalid merges file: {merges}"
        self._tok = ByteLevelBPETokenizer.from_file(vocab, merges)
        return self
    
    def encode(self, text: str) -> List[int]:
        return self._tok.encode(text).ids
    
    def decode(self, ids: List[int]) -> str:
        return self._tok.decode(ids)
    

RLHFTokenizer = BPETokenizer