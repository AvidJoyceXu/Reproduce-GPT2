import torch.nn as nn
import torch
import math
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024 # Size of context window.
    vocab_size: int = 50257 
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh") # Not required, only for precise reproduction of GPT-2
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"Each head has non-integer {config.n_embd/config.n_head} number of embeddings!"
        # K, Q, V projections for all heads (shared)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Configurations
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Attention mask where tokens don't cater to later tokens
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)) \
                              .view(1, 1, config.block_size, config.block_size))
        # torch.tril: get the lower triangle of a matrix (T, T)
    
    def forward(self, x):
        B, T, C = x.size() # C: n_embd
        # K, Q, V projections for all heads
        qkv:torch.Tensor = self.c_attn(x) # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2) # Each shape: (B, T, C), C = self.n_embd
        # Split the heads: hs for head size
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) 
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Attention
        att = torch.einsum('bhqd, bhkd -> bhqk', q, k) * (1/math.sqrt(k.size(-1))) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf')) # Queries should not access keys after them
        att = att.softmax(dim=-1) # softmax(-inf) = 0 
        y = att @ v # (B, nh, T, hs)
        # Rearrangement
        y = y.transpose(1, 2).contiguous().view(B, T, C) # contiguous for acceleration
        # Output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Reduce: Attention
        x = x + self.mlp(self.ln_2(x)) # Map: Feed forward
        # Take the residual stream out of the normalization
        return x