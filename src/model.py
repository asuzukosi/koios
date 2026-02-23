import torch
import torch.nn as nn
import torch.nn.functional as F
from src.block import Block, BlockModern
from src.rmsnorm import RMSNorm
from src.pos_ecoding import SinusoidalPositionalEncoding
from typing import Optional
from src.utils.filtering import top_k_top_p_filtering
# implementation of tiny gpt model
class GPT(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layers: int=4, n_head: int=4, d_model: int=256, dropout: float=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositionalEncoding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(*[Block(d_model, n_head, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)


    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) # initialize the weight to a normal distribution with mean 0 and std 0.02
            if module.bias is not None:
                nn.init.zeros_(module.bias) # ensure the bias is initialized to 0
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) # initialize the weight to a normal distribution with mean 0 and std 0.02


    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        B, T = idx.shape # (batch_size, sequences) (B, T)
        assert T <= self.block_size, "sequence length must be less than or equal to block size"
        btd: torch.Tensor = self.tok_emb(idx) # (B, T, d_model) token embeddings
        pos: torch.Tensor = self.pos_emb(btd) # (B, T, d_model) positional embeddings
        x: torch.Tensor = self.drop(btd + pos) # (B, T, d_model)
        for block in self.blocks:
            # we can now do a bunch of stuff here
            x = block(x) # (B, T, d_model)
        x = self.ln_f(x) # (B, T, d_model)
        out: torch.Tensor = self.lm_head(x) # (B, T, vocab_size)
        loss: Optional[torch.Tensor] = None 
        if targets is not None:
            loss = F.cross_entropy(out.view(-1, out.size(-1)), targets.view(-1))
        return loss, out
    

    @torch.no_grad()
    def generate(self, idx: torch.Tensor,
                max_new_tokens:int=200, # used to limit the number of new tokens generated
                temperature: float=1.0, # used to control the randomness of selected tokens when sampling
                top_k: int=40, # used to limit the number of most likely tokens to consider
                top_p: float=0.9 # used to limit the cumulative probability of the most likely tokens to consider
                ) -> torch.Tensor:
        self.eval()
        # if prompt is empty start with new line byte token (i.e 10)
        B, T = idx.shape
        print(f"sampling for batch of size {B} with provided context of length {T}")
        if idx.size(1) == 0:
            # fill with new line byte token (i.e 10)
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)

        # TODO: seperate prefill and decode stages and make use of kv cache
        for _ in range(max_new_tokens):
            # we need to iteratively generate one token at a time and add it to the context
            idx_cond = idx[:, -self.block_size:] # (B, block_size) take the last block_size tokens from the context
            logits = self(idx_cond) # (B, block_size, vocab_size)
            logits = logits[:, -1, :] / max(temperature, 1e-6) # perform temperature scaling
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1) # (B, block_size, vocab_size)
            next_id = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, next_id], dim=1) # (B, block_size + 1) # older tokens are shifted to the left
        return idx

# TODO: use rmsnorm, rope, swiglu, sliding window, and kv cache


class GPTModern(nn.Module):
    def __init__(self, vocab_size: int, block_size: int, n_layers: int=4, n_head: int=4, d_model: int=256, dropout: float=0.0):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = SinusoidalPositionalEncoding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(*[BlockModern(d_model, n_head, dropout) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    # TODO: add implementation for lora adapter 
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, 
                kv_cache_list: Optional[list[torch.Tensor]] = None, start_pos: int = 0) -> tuple[torch.Tensor, Optional[torch.Tensor], list[torch.Tensor]]:
        B, T = idx.shape # (batch_size, sequences) (B, T)
        assert T <= self.block_size, "sequence length must be less than or equal to block size"
        btd: torch.Tensor = self.tok_emb(idx) # (B, T, d_model) token embeddings
        pos: torch.Tensor = self.pos_emb(btd) # (B, T, d_model) positional embeddings
        x: torch.Tensor = self.drop(btd + pos) # (B, T, d_model)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = None if kv_cache_list is None else kv_cache_list[i] # the kv cache for the i-th block
            x, cache = block(x, kv_cache=cache, start_pos=start_pos)
            new_caches.append(cache)
            # we can now do a bunch of stuff here
        x = self.ln_f(x) # (B, T, d_model)
        out: torch.Tensor = self.lm_head(x) # (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(out.view(-1, out.size(-1)), targets.view(-1))
        else:
            loss = None
        return out, loss, new_caches
    

    def generate(self, prompt: torch.Tensor, max_new_tokens:int=200, temperature: float=1.0, top_k: int=40,
                  top_p: float=0.9, sliding_window: Optional[int] = None, attention_sink: Optional[int] = None) -> torch.Tensor:
        self.eval()
        idx = prompt
        kvs = [None] * len(self.blocks)

        # TODO: clearly separate prefill and decode stages and make use of kv cache
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] if kvs[0] is None else idx[: -1:]
            # absolute start position from cache lenght (0 on first step)
            start_pos = 0 if kvs[0] is None else kvs[0].k.size(2)
            logits, _, kvs = self(idx_cond, kv_cache_list=kvs, start_pos=start_pos)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            next_logits = top_k_top_p_filtering(next_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True) if temperature == 0.0 else torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def generate_nocache(self, idx: torch.Tensor,
                max_new_tokens:int=200, # used to limit the number of new tokens generated
                temperature: float=1.0, # used to control the randomness of selected tokens when sampling
                top_k: int=40, # used to limit the number of most likely tokens to consider
                top_p: float=0.9 # used to limit the cumulative probability of the most likely tokens to consider
                ) -> torch.Tensor:
        self.eval()
        # if prompt is empty start with new line byte token (i.e 10)
        B, T = idx.shape
        print(f"sampling for batch of size {B} with provided context of length {T}")
        if idx.size(1) == 0:
            # fill with new line byte token (i.e 10)    
            idx = torch.full((idx.size(0), 1), 10, dtype=torch.long, device=idx.device)

        # TODO: seperate prefill and decode stages and make use of kv cache
        for _ in range(max_new_tokens):
            # we need to iteratively generate one token at a time and add it to the context
            idx_cond = idx[:, -self.block_size:] # (B, block_size) take the last block_size tokens from the context
            logits = self(idx_cond) # (B, block_size, vocab_size)
            logits = logits[:, -1, :] / max(temperature, 1e-6) # perform temperature scaling
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(logits, dim=-1) # (B, block_size, vocab_size)
            next_id = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat([idx, next_id], dim=1) # (B, block_size + 1) # older tokens are shifted to the left