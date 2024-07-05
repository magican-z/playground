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
import torch.optim as optim
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


@dataclass
class GPTConfig:
    n_layers: int = 12
    n_headers: int = 12
    n_embd: int = 768
    n_positions: int = 1024
    #vocab_size: int = 50257
    vocab_size: int = 50304
    block_size: int = 1024


class SimpleSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = True
        '''
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        '''
    
    def forward(self, x):
        B, T, N = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(N, dim=-1)
        q = q.view(B, T, self.config.n_headers, N // self.config.n_headers).transpose(1, 2)
        k = k.view(B, T, self.config.n_headers, N // self.config.n_headers).transpose(1, 2)
        v = v.view(B, T, self.config.n_headers, N // self.config.n_headers).transpose(1, 2)
        '''
        att = q @ k.transpose(-1, -2) * (1.0 / math.sqrt(T))
        att.masked_fill(self.bias[:, :, :T, :T]==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        '''
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = self.c_proj(y.transpose(1, 2).contiguous().view(B, T, N))
        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU()
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
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)

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

    def forward(self, idxs, targets=None):
        b, t = idxs.shape
        token_emb = self.transformer.wte(idxs)
        pos_emb = self.transformer.wpe(torch.arange(t, device=idxs.device, dtype=torch.long))
        x = token_emb + pos_emb

        for b in self.transformer.h:
            x = b(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
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
    


class DataLoaderSuper():
    def __init__(self, data_path, B, T, world_size, local_rank, data_type, shuffle=False):
        fnames = os.listdir(data_path)
        self.data_files = [os.path.join(data_path, fname) for fname in fnames if data_type in fname]
        self.B, self.T, self.world_size, self.local_rank = B, T, world_size, local_rank
        self.shuffle = shuffle
        if shuffle:
            random.shuffle(self.data_files)
        else:
            self.data_files.sort()
        self.curr_file_idx = -1
        self.start_points = []

    def restore(self, state):
        self.data_files = state['data_files']
        self.curr_file_idx = state['file_idx']
        self.shuffle = state['shuffle']
        self.start_points = state['start_points']
        self.tokens = self.load_tokens(self.data_files[self.curr_file_idx])

    def get_state(self):
        state = {
            'data_files': self.data_files,
            'file_idx': self.curr_file_idx,
            'shuffle': self.shuffle,
            'start_points': self.start_points,
        }
        return state

    def load_tokens(self, filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt
    
    def gen_start_idx(self):
        n_tokens = len(self.tokens)
        idx = 0
        batch_size = self.B * self.T * self.world_size
        my_offset = self.B * self.T * self.local_rank
        while True:
            my_point = idx + my_offset
            self.start_points.append(my_point)
            idx += batch_size
            if idx + batch_size + 1 >= n_tokens:
                break
        if self.shuffle:
            random.shuffle(self.start_points)

    def reset(self):
        if self.curr_file_idx >= len(self.data_files):
            if self.shuffle:
                random.shuffle(self.data_files)
            self.curr_file_idx = 0
        else:
            self.curr_file_idx += 1
        self.tokens = self.load_tokens(self.data_files[self.curr_file_idx])
        self.gen_start_idx()

    def get_batch(self):
        if not self.start_points:
            self.reset()
        start_idx = self.start_points.pop(0)
        end = start_idx + self.B * self.T
        x = self.tokens[start_idx:end]
        y = self.tokens[start_idx + 1: end + 1]
        return x.view(B, T), y.view(B, T)





#max_lr = 6e-4
max_lr = 1.8e-3
min_lr = max_lr * 0.1
warm_steps = 715
max_steps = 19073

def get_lr(it):
    if it < warm_steps:
        lr = max_lr * (it + 1) / warm_steps
        return lr
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warm_steps) / (max_steps - warm_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 1
    ddp_world_size = 1
    master_process = True
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)

device_type = 'cuda' if device.startswith('cuda') else 'cpu'

torch.manual_seed(9527)
if torch.cuda.is_available():
    torch.cuda.manual_seed(9527)

total_batch_size = 2 ** 19
B = 16
T = 1024
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f'total batch size: {total_batch_size}')
    print(f'grad_accum_steps: {grad_accum_steps}')

restore_checkpoint_path = sys.argv[1]
if restore_checkpoint_path and os.path.isfile(restore_checkpoint_path):
    checkpoint = torch.load(restore_checkpoint_path, map_location=lambda storage, loc: storage.cuda(ddp_local_rank))


#dl = DataLoader('./shakespear.txt', B, T, ddp_world_size, ddp_local_rank)
dl = DataLoaderSuper('./fineweb-edu-tokens', B, T, ddp_world_size, ddp_local_rank, 'val', shuffle=False)


torch.set_float32_matmul_precision('high')
model = GPT(GPTConfig)
model.to(device)
model = torch.compile(model)
if restore_checkpoint_path:
    model.load_state_dict(checkpoint['model'])
    model.config = checkpoint['config']
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

# torchrun --standalone --nproc_per_node=2 gpt2.py
#optimizer = raw_model.configure_optimizer(weight_decay=0.1,learning_rate=6e-4, device=device)
start_step = 0

if restore_checkpoint_path:
    start_step = checkpoint['step']
    #optimizer.load_state_dict(checkpoint['optimizer'])
    rng_state = checkpoint['rng_state'].to('cpu')
    torch.set_rng_state(rng_state)

model.eval()
with torch.no_grad():
    val_loss_accum = 0.0
    val_loss_steps = 20
    for _ in range(val_loss_steps):
        x, y = dl.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / val_loss_steps
        val_loss_accum += loss.detach()
if ddp:
    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
print(f'val_loss_accum:{val_loss_accum.item()}')

if ddp:
    destroy_process_group()
