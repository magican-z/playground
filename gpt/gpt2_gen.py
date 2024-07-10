import sys
import os
import json
import math
import time
import random
from dataclasses import dataclass
import inspect
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    n_layers: int = 12
    n_headers: int = 12
    n_embd: int = 768
    n_positions: int = 1024
    #vocab_size: int = 50257
    vocab_size: int = 50304
    block_size: int = 1024


class SimpleSelfAttentionS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 2)
        self.c_attn_q = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True
        
    
    def forward(self, x, input_pos=None, kv_cache=None, mask=None):
        B, T, N = x.shape
        kv = self.c_attn(x)
        q = self.c_attn_q(x[:, T-1:, :])
        k, v = kv.split(N, dim=-1)
        q = q.view(B, 1, self.config.n_headers, N // self.config.n_headers)
        k = k.view(B, T, self.config.n_headers, N // self.config.n_headers)
        v = v.view(B, T, self.config.n_headers, N // self.config.n_headers)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=input_pos is None)
        y = self.c_proj(y.transpose(1, 2).contiguous().view(B, 1, N))
        return y, kv_cache


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
        self.attn = SimpleSelfAttentionS(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x, input_pos=None, kv_cache=None, mask=None):
        att_y, kv_cache = self.attn(self.ln_1(x), input_pos, kv_cache, mask)
        x = x + att_y
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        self.kv_cache = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idxs, targets=None, input_pos=None, max_len=None, mask=None):
        use_kvcache = False if input_pos is None else True
        b, t = idxs.shape
        token_emb = self.transformer.wte(idxs)
        if not use_kvcache:
            pos_emb = self.transformer.wpe(torch.arange(t, device=idxs.device, dtype=torch.long))
        else:
            pos_emb = self.transformer.wpe(input_pos)
            mask = mask.index_select(2, input_pos)
        x = token_emb + pos_emb

        if not use_kvcache:
            for b in self.transformer.h:
                x, *_ = b(x)
        else:
            if self.kv_cache is None:
                self.build_kvcache(b, max_len, idxs.device)
            for i, b in enumerate(self.transformer.h):
                x, self.kv_cache[i] = b(x, input_pos, self.kv_cache[i], mask)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def build_kvcache(self, B, max_len, device):
        cache_shape = (B, max_len, self.config.n_headers, self.config.n_embd // self.config.n_headers)
        self.kv_cache = [
            (torch.zeros(cache_shape, device=device), torch.zeros(cache_shape, device=device))
            for _ in range(self.config.n_layers)
        ]
        
    def gen(self, idxs, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self.forward(idxs)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            x = torch.multinomial(probs, num_samples=1)
            idxs = torch.cat((idxs, x), dim=1)
        return idxs
    
    def gen_with_kvcache(self, idxs, max_tokens):
        pos_idxs = torch.arange(max_tokens, dtype=torch.long, device=idxs.device)
        x = idxs
        pos = pos_idxs[:x.size(-1)]
        ones = torch.ones((max_tokens, max_tokens), dtype=torch.bool, device=idxs.device)
        mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
        start = x.size(-1)
        for i in range(start, max_tokens):
            logits, _ = self.forward(x, input_pos=pos, max_len=max_tokens, mask=mask)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            x = torch.multinomial(probs, num_samples=1)
            pos = pos_idxs[i:i+1]
            idxs = torch.cat((idxs, x), dim=1)
        return idxs

    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in no_decay_params)
        print(f'decay: {len(decay_params)} {num_decay_params}')
        print(f'nodecay: {len(no_decay_params)} {num_nodecay_params}')

        fused_avaliable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avaliable and 'cuda' in device
        print(f'use fuse: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=[0.9, 0.95], eps=1e-8, fused=use_fused)
        return optimizer

    def get_emb(self, idxs, targets=None, input_pos=None, max_len=None, mask=None):
        use_kvcache = False if input_pos is None else True
        b, t = idxs.shape
        token_emb = self.transformer.wte(idxs)
        if not use_kvcache:
            pos_emb = self.transformer.wpe(torch.arange(t, device=idxs.device, dtype=torch.long))
        else:
            pos_emb = self.transformer.wpe(input_pos)
            mask = mask.index_select(2, input_pos)
        x = token_emb + pos_emb
        return x


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


def test_blocks():
    torch.set_grad_enabled(False)
    config = GPTConfig(n_layers=2, n_headers=3, n_embd=6, block_size=5, vocab_size=8)
    blks = [Block(config) for _ in range(config.n_layers)]
    B = 2
    x = torch.rand(B, config.block_size, config.n_embd)
    x0 = x
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
            print(f'i={i} j={j}  MAE={mae}')


def test_gpt_emb():
    torch.set_grad_enabled(False)
    config = GPTConfig(n_layers=2, n_headers=3, n_embd=6, block_size=5, vocab_size=8)
    model = GPT(config)
    B = 2
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    x = x.reshape(B, -1)
    x_emb = model.get_emb(x)

    pos_idxs = torch.arange(x.size(-1), dtype=torch.long, device=x.device)
    ones = torch.ones((x.size(-1), x.size(-1)), dtype=torch.bool, device=x.device)
    mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
    x1_emb = model.get_emb(x[:, :2], input_pos=pos_idxs[:2], max_len=3, mask=mask)
    x2 = x.index_select(1, pos_idxs[2:2+1])
    x2_emb = model.get_emb(x2, input_pos=pos_idxs[2:2+1], max_len=3, mask=mask)

    print(x_emb)
    print(x1_emb)
    print(x2_emb)



def test_gpt():
    torch.set_grad_enabled(False)
    config = GPTConfig(n_layers=2, n_headers=3, n_embd=6, block_size=5, vocab_size=8)
    model = GPT(config)
    B = 2
    x = torch.tensor([[1, 2, 3, 4, 5, 6]])
    x = x.reshape(B, -1)
    y1, _ = model.forward(x[:, :2])
    y2, _ = model.forward(x)
    
    pos_idxs = torch.arange(x.size(-1), dtype=torch.long, device=x.device)
    ones = torch.ones((x.size(-1), x.size(-1)), dtype=torch.bool, device=x.device)
    mask = torch.tril(ones).unsqueeze(0).unsqueeze(0)
    y1_kvcache, _ = model.forward(x[:, :2], input_pos=pos_idxs[:2], max_len=3, mask=mask)
    x2 = x.index_select(1, pos_idxs[2:2+1])
    y2_kvcache, _ = model.forward(x2, input_pos=pos_idxs[2:2+1], max_len=3, mask=mask)

    mae1 = torch.mean(torch.abs(y1 - y1_kvcache))
    mae2 = torch.mean(torch.abs(y2[:, 2:,:] - y2_kvcache))
    print(f'mae1={mae1}, mae2={mae2}')


def test_speedup():
    import tiktoken
    cuda_id = 1
    device = f'cuda:{cuda_id}'
    checkpoint = torch.load(sys.argv[1], map_location=lambda storage, loc: storage.cuda(cuda_id))

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('high')
    model = GPT(GPTConfig)
    model.to(device)
    model = torch.compile(model)
    model.load_state_dict(checkpoint['model'])
    model.config = checkpoint['config']
    num_return_sequences = 5
    max_tokens = 1000

    prompt = 'Hello, earthling '
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    t0 = time.time()
    gen_tokens = model.gen(tokens, max_tokens)
    #gen_tokens = model.gen_with_kvcache(tokens, max_tokens)
    gen_sec = time.time() - t0
    for i in range(num_return_sequences):
        tokens = gen_tokens[i, :max_tokens].tolist()
        decoded = enc.decode(tokens)
        print(f"Sample {i}: {decoded}")
    print(f'generate cost:{gen_sec} sec')


if __name__ == "__main__":
    #test_y_equal()
    #test_blocks()
    #test_gpt_emb()
    #test_gpt()
    test_speedup()

    
    