import sys
import os
import torch
import math
import numpy as np
import time
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import inspect



class GPTConfig():
    is_set_TF32 = True
    is_set_Vocab = False
    is_set_Bf16 = False
    is_set_Flash = False
    is_set_Fuse = False
    is_set_Compile = False
    n_layers = 12
    n_headers = 12
    n_embd = 768
    n_positions  = 1024
    vocab_size = 50304 if is_set_Vocab else 50257
    block_size = 1024


class CauseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer("bias", 
                            torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, N = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N, dim=-1)
        q = q.view(B, T, self.config.n_headers, N // self.config.n_headers).transpose(1, 2)
        k = k.view(B, T, self.config.n_headers, N // self.config.n_headers).transpose(1, 2)
        v = v.view(B, T, self.config.n_headers, N // self.config.n_headers).transpose(1, 2)
        att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(T))
        att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = self.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))
        return y
    
class CauseAttention2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, N = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N, dim=-1)
        q = q.view(B, T, self.config.n_headers, N // self.config.n_headers).transpose(1, 2)
        k = k.view(B, T, self.config.n_headers, N // self.config.n_headers).transpose(1, 2)
        v = v.view(B, T, self.config.n_headers, N // self.config.n_headers).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.c_proj(y.transpose(1, 2).contiguous().view(B, T, -1))
        return y



class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CauseAttention2(config) if config.is_set_Flash else CauseAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.n_positions, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.weight)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens, targets=None):
        _, T = tokens.shape
        token_emb = self.transformer.wte(tokens)
        pos = torch.arange(T, device=tokens.device, dtype=torch.long)
        pos_emb = self.transformer.wpe(pos)
        x = token_emb + pos_emb
        
        for h in self.transformer.h:
            x = h(x)
        
        logits = self.lm_head(self.transformer.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class DataLoader(object):
    def __init__(self, B, T, fname):
        self.B, self.T = B, T
        npt = np.load(fname)
        npt = npt.astype(np.int32)
        self.buffer = torch.tensor(npt, dtype=torch.long)
        self.curr_idx = 0

    def get_batch(self):
        block_size = self.B * self.T
        x = self.buffer[self.curr_idx:self.curr_idx + block_size]
        y = self.buffer[self.curr_idx + 1:self.curr_idx + block_size + 1]
        self.curr_idx += block_size
        if self.curr_idx + block_size + 1 > len(self.buffer):
            self.curr_idx = 0
        return x.view(self.B, self.T), y.view(self.B, self.T)

'''
加速点：
1. bf16
2. compile
3. vocab_size
4. fuse
5. flashattention
'''
device = 'cuda:1'
if GPTConfig.is_set_TF32:
    torch.set_float32_matmul_precision('high')
model = GPT(GPTConfig)
model = model.to(device)
# compile
if GPTConfig.is_set_Compile:
    model = torch.compile(model)
# fuse
if GPTConfig.is_set_Fuse:
    fused_avaliable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_avaliable and 'cuda' in device
    print(f'use fuse: {use_fused}')

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)

total_batch_size = 2 ** 19
B = 16
T = 1024
grad_accum_steps = total_batch_size // (B * T )
max_steps = 20
dl = DataLoader(B, T, './shakespear.npy')
avg_speed = 0.0
for i_step in range(max_steps):
    t0 = time.time()
    x, y = dl.get_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # bf16
    if GPTConfig.is_set_Bf16:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
    else:
        _, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    proc_tokens = B * T
    dt = time.time() - t0
    speed = proc_tokens / dt 
    if i_step > 0:
        avg_speed += speed
    print(f'step:{i_step:04d} | loss:{loss.item():.4f} | cost:{dt * 1000:.1f}s | tokens:{speed}/s')
    
print(avg_speed / (max_steps-1))


