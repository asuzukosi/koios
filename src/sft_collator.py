from src.bpe import BPETokenizer
from src.tokenizer import ByteTokenizer
from dataclasses import dataclass
import torch

TEMPLATE = """
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
{response}
<|im_end|>
"""

@dataclass
class Example:
    instruction: str
    response: str

class SFTCollator:
    """
    turn (instruction, response) into token ids and mask labels for causal lm
    labels for the prompt part are set to -100 so they do not contribute to the loss
    """
    def __init__(self, block_size: int = 256, bpe_dir: str | None = None):
        self.block_size = block_size
        self.tok = None
        self.tok = BPETokenizer(vocab_size=32000)
        if bpe_dir:
            self.tok.load(bpe_dir)
            print("loaded bpe tokenizer from", bpe_dir)
        else:
            print("no bpe tokenizer directory provided, training new tokenizer")

        if self.tok is None:
            self.tok = ByteTokenizer()
        if self.tok is None:
            raise RuntimeError("no tokenizer available, install tokenizers library")
        
    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size
    
    def collate(self, items: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        turn (instruction, response) into token ids and mask labels for causal lm
        labels for the prompt part are set to -100 so they do not contribute to the loss
        """
        input_ids = []
        labels = []
        for prompt, response in items:
            prefix_text = self.format_prompt_only(prompt)[:-10] # remove last 10 tokens in formater i.e <|im_end|>
            text = self.format_example((prompt, response))
            ids = self.tok.encode(text)[:self.block_size]
            prompt_ids = self.tok.encode(prefix_text)[:self.block_size]
            n_prompt = min(len(prompt_ids), len(ids))
            x = ids
            y = ids.copy()
            for t in range(len(y) - 1):
                y[t] = ids[t + 1]
            y[-1] = -100
            for i in range(n_prompt -1):
                y[i] = -100
            input_ids.append(x)
            labels.append(y)
        # pad to block size
        def pad_to(ids, val):
            if len(ids) < self.block_size:
                ids = ids + [val] * (self.block_size - len(ids))
            return ids[:self.block_size]
        x = [pad_to(it, 2) for it in input_ids]
        y = [pad_to(it, -100) for it in labels]
        return torch.tensor(x), torch.tensor(y)

    
    def format_prompt_only(self, prompt: str) -> str:
        """
        format the prompt only, no response
        """
        return TEMPLATE.format(prompt=prompt, response="")
    
    def format_example(self, example: tuple[str, str]) -> str:
        """
        format the example (instruction, response) into a string
        """
        return TEMPLATE.format(prompt=example[0], response=example[1])