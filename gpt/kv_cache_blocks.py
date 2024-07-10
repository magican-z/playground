import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
import time


class SimpleSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True
        
    
    def forward(self, x, input_pos=None, kv_cache=None, mask=None):
        B, T, N = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N, dim=-1)
        q = q.view(B, T, self.config.n_headers, N // self.config.n_headers)
        k = k.view(B, T, self.config.n_headers, N // self.config.n_headers)
        v = v.view(B, T, self.config.n_headers, N // self.config.n_headers)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = k_cache.index_copy_(1, input_pos, k)
            v = v_cache.index_copy_(1, input_pos, v)
            kv_cache = k, v
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=input_pos is None)
        y = self.c_proj(y.transpose(1, 2).contiguous().view(B, T, N))
        return y, kv_cache
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SimpleSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, input_pos=None, kv_cache=None, mask=None):
        att_y, kv_cache = self.attn(self.ln_1(x), input_pos, kv_cache, mask)
        x = x + att_y
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache


@dataclass
class GPTConfig:
    n_layers: int = 12
    n_headers: int = 12
    n_embd: int = 768
    n_positions: int = 1024
    vocab_size: int = 50304
    block_size: int = 1024

    
def test_one_block():
    torch.set_grad_enabled(False)
    config = GPTConfig(n_layers=2, n_headers=3, n_embd=6, block_size=5, vocab_size=8)
    blk = Block(config)
    B = 2
    x = torch.rand(B, config.block_size, config.n_embd)
    y, *_ = blk(x)

    ones = torch.ones((config.block_size, config.block_size),  dtype=torch.bool, device=x.device)
    mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
    kv_cache_shape = (B, config.block_size, config.n_headers, config.n_embd // config.n_headers)
    kv_cache = (torch.zeros(kv_cache_shape, device=x.device), torch.zeros(kv_cache_shape, device=x.device))

    for i in range(config.block_size):
        input_pos = torch.tensor([i], device=x.device)
        x_i = x.index_select(1, input_pos)
        m_i = mask.index_select(2, input_pos) 
        y_i, kv_cache = blk(x_i, input_pos=input_pos, kv_cache=kv_cache, mask=m_i)
        mae = torch.mean(torch.abs(y_i[:,0,:] - y[:,i,:]))
        print(f'i={i}  MAE={mae}')


def test_2blocks():
    torch.set_grad_enabled(False)
    config = GPTConfig(n_layers=2, n_headers=3, n_embd=6, block_size=5, vocab_size=8)
    blks = [Block(config) for _ in range(config.n_layers)]
    B = 2
    x = torch.rand(B, config.block_size, config.n_embd)
    x0 = x # 这里要特别小心。如果不做这个修改，blk的循环会替换x，导致kvcache阶段输入的x已经不是最初的x
    ys = []
    for blk in blks:
        x0, *_ = blk(x0)
        ys.append(x0)

    ones = torch.ones((config.block_size, config.block_size),  dtype=torch.bool, device=x.device)
    mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
    kv_cache_shape = (B, config.block_size, config.n_headers, config.n_embd // config.n_headers)
    kv_cache = [(torch.zeros(kv_cache_shape, device=x.device), torch.zeros(kv_cache_shape, device=x.device)) for _ in range(config.n_layers)]
    for i in range(config.block_size):
        input_pos = torch.tensor([i], device=x.device)
        x_i = x.index_select(1, input_pos)
        m_i = mask.index_select(2, input_pos)
        for j, blk in enumerate(blks):
            x_i, kv_cache[j] = blk(x_i, input_pos=input_pos, kv_cache=kv_cache[j], mask=m_i)
            mae = torch.mean(torch.abs(x_i[:,0,:] - ys[j][:,i,:]))
            print(f'i={i} j={j} MAE={mae}')


test_2blocks()