import pathlib
import torch
from torch.utils.data import DataLoader, Dataset
from src.bpe import BPETokenizer
from pathlib import Path

class ByteDataset:
    """
    holds raw bytes of the text file and yields (x, y) blocks of LM
    - block size: sequence length ( context window )
    - split: fraction for trianing (rest for validation)
    """

    def __init__(self, path: str, block_size: int = 256, split: float = 0.9):
        data = pathlib.Path(path).read_bytes() # reading the file as bytes since we are using a byte tokenizer
        data = torch.Tensor(list(data), dtype=torch.long)
        n = int(split * len(data))
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(self, split: str = "train", batch_size: int = 32, device: torch.device = torch.device("cpu")) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        assert len(data) >= self.block_size, "data length must be greater than block size"
        ix = torch.randint(0, len(data) - self.block_size, (batch_size,)) # we are sampling random starting points for the blocks
        x = torch.stack([data[i:i+self.block_size] for i in ix]) # (batch_size, block_size)
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix]) # (batch_size, block_size) output is one token shifted to the right, during evaluation we would only be checking the last token of the block
        x, y = x.to(device), y.to(device) # move to device
        return x, y
    
    def get_vocab(self) -> dict[int, str]:
        return {i: chr(i) for i in range(256)}
    
class TextBPEDataset(Dataset):
    """
    holds BPE encoded tokens and yields (x, y) blocks of LM
    - block size: sequence length ( context window )
    - split: fraction for trianing (rest for validation)
    """
    def __init__(self, path: str, tokenizer: BPETokenizer, block_size: int = 256, split: float = 0.9):
        super().__init__()
        self.block_size = block_size
        text = Path(path).read_text(encoding="utf-8")
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self) -> int:
        return max(0, self.ids.numel() - self.block_size - 1)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.ids[idx:idx+self.block_size]
        y = self.ids[idx+1:idx+self.block_size+1]
        return x, y
    
def make_loader(path: str, tokenizer: BPETokenizer, block_size: int = 256, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    ds = TextBPEDataset(path, tokenizer, block_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
