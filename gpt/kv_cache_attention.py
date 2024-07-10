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
    


@dataclass
class GPTConfig:
    n_layers: int = 12
    n_headers: int = 12
    n_embd: int = 768
    n_positions: int = 1024
    vocab_size: int = 50304
    block_size: int = 1024

    
def test_y_equal():
    torch.set_grad_enabled(False)
    config = GPTConfig(n_layers=2, n_headers=3, n_embd=6, block_size=5)
    att = SimpleSelfAttention(config)
    B = 2
    x = torch.rand(B, config.block_size, config.n_embd)
    std_y, *_ = att(x)

    ones = torch.ones((config.block_size, config.block_size),  dtype=torch.bool, device=x.device)
    mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
    kv_cache_shape = (B, config.block_size, config.n_headers, config.n_embd // config.n_headers)
    kv_cache = (torch.zeros(kv_cache_shape, device=x.device), torch.zeros(kv_cache_shape, device=x.device))
    for i in range(config.block_size):
        input_pos = torch.tensor([i], device=x.device)
        x_i = x.index_select(1, input_pos)
        m_i = mask.index_select(2, input_pos) # 这里要特别小心。在attion模块中x的1-2维度会交换位置，而mask不换
        y_i, kv_cache = att(x_i, input_pos=input_pos, kv_cache=kv_cache, mask=m_i)
        mae = torch.mean(torch.abs(y_i[:,0,:] - std_y[:,i,:]))
        print(f'i={i} * MAE={mae}')


def test_speed():
    torch.set_grad_enabled(False)
    device = 'cuda:1'
    config = GPTConfig()
    attn = SimpleSelfAttention(config)
    attn = attn.to(device)
    B = 16
    x = torch.rand(B, config.block_size, config.n_embd, device=device)
    # 推理过程中，是逐个token推理出来的
    t0 = time.time()
    for i in range(config.block_size):
        std_y_i, *_ = attn(x[:, :i+1, :, :])
    torch.cuda.synchronize()
    print(f'standard attention: cost{time.time()-t0:.2f} seconds.')

    t1 = time.time()
    ones = torch.ones((config.block_size, config.block_size),  dtype=torch.bool, device=x.device)
    mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
    kv_cache_shape = (B, config.block_size, config.n_headers, config.n_embd // config.n_headers)
    kv_cache = (torch.zeros(kv_cache_shape, device=x.device), torch.zeros(kv_cache_shape, device=x.device))
    for i in range(config.block_size):
        input_pos = torch.tensor([i], device=x.device)
        x_i = x.index_select(1, input_pos)
        m_i = mask.index_select(2, input_pos) # 这里要特别小心。在attion模块中x的1-2维度会交换位置，而mask不换
        y_i, kv_cache = attn(x_i, input_pos=input_pos, kv_cache=kv_cache, mask=m_i)
    print(f'kv-cache attention: cost{time.time()-t1:.2f} seconds.')