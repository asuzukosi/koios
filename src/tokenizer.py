import torch


class ByteTokenizer:
    """
    simplest possible tokenizer: byte tokenizer
    each character is encoded as a byte (0-255)
    """
    def __init__(self):
        self.stoi = {chr(i): i for i in range(256)}
        self.itos = {i: chr(i) for i in range(256)}
    
    def __len__(self) -> int:
        return len(self.stoi)
    
    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return bytes(tokens).decode("utf-8", errors="ignore")
    
    @property
    def vocab_size(self) -> int:
        return 256