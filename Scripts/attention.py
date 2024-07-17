import torch
from torch import nn
import torch.nn.functional as F 

from einops import rearrange

#Casual attention

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head
        self.scale = self.d_k ** -0.5

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.n_head, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = (attn_probs @ v).transpose(1, 2).reshape(B, T, C)
        attn_output = self.resid_dropout(self.out_proj(attn_output))

        return attn_output
    



#Flash Attention
class FlashAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout_p = config.dropout
        self.causal = True  # assuming causal for GPT-like model
        
    def forward(self, x):
        b, t, c = x.size()
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = [rearrange(x, 'b t h d -> (b h) t d') for x in (q, k, v)]
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = rearrange(attn_output, '(b h) t d -> b t (h d)', h=self.n_head)
        attn_output = self.proj(attn_output)
        # Save tensors for backward pass
        self.saved_tensors = (q, k, v, attn_weights)
        return attn_output

    def backward(self, dout):
        q, k, v, attn_weights = self.saved_tensors
        # Gradient of attention output
        datt = torch.matmul(dout, v.transpose(-2, -1))
        dv = torch.matmul(attn_weights.transpose(-2, -1), dout)
        # Gradient of attention weights
        datt_weights = dout @ v.transpose(-2, -1)
        # Masked fill in the gradient if causal
        if self.causal:
            mask = torch.triu(torch.ones(datt.size(-2), datt.size(-1), device=datt.device, dtype=torch.bool), diagonal=1)
            datt = datt.masked_fill(mask, 0.0)

        # Gradient of Q, K, V
        dq = datt @ k.transpose(-2, -1)
        dk = datt.transpose(-2, -1) @ q
        dq = dq.view_as(q)
        dk = dk.view_as(k)
        dv = dv.view_as(v)
        return dq, dk, dv
    

# Sparse attention
import numpy as np

#ref: https://github.com/kyegomez/SparseAttention/blob/main/sparse_attention.py

def get_attn_mask(n, attn_mode, local_attn_ctx=None):
    if attn_mode == 'all':
        b = torch.tril(torch.ones([n, n]))
    elif attn_mode == 'local':
        bandwidth = local_attn_ctx
        ctx = min(n - 1, bandwidth - 1)
        b = torch.tril(torch.ones([n, n]), ctx)
    elif attn_mode == 'strided':
        stride = local_attn_ctx
        x = torch.reshape(torch.arange(n, dtype=torch.int32), [n, 1])
        y = torch.transpose(x, 0, 1)
        z = torch.zeros([n, n], dtype=torch.int32)
        q = z + x
        k = z + y
        c1 = q >= k
        c2 = torch.eq(torch.fmod(q - k, stride), 0)
        c3 = torch.logical_and(c1, c2)
        b = c3.float()
    else:
        raise ValueError('Not yet implemented')
    b = torch.reshape(b, [1, 1, n, n])
    return b

def strided_transpose(x, n_ctx, local_attn_ctx, blocksize):
    bT_ctx = n_ctx // local_attn_ctx
    assert bT_ctx % blocksize == 0, f'{bT_ctx}, {blocksize}'
    n, t, embd = x.size()
    x = torch.reshape(x, [n, bT_ctx, local_attn_ctx, embd])
    x = torch.transpose(x, 1, 2)
    x = torch.reshape(x, [n, t, embd])
    return x

def split_heads(x, n):
    return torch.transpose(split_states(x, n), 1, 2)

def merge_heads(x):
    return merge_states(torch.transpose(x, 1, 2))

def split_states(x, n):
    x_shape = x.size()
    m = x_shape[-1]
    new_x_shape = x_shape[:-1] + [n, m // n]
    return torch.reshape(x, new_x_shape)

def merge_states(x):
    x_shape = x.size()
    new_x_shape = x_shape[:-2] + [np.prod(x_shape[-2:])]
    return torch.reshape(x, new_x_shape)

def blocksparse_attention_impl(q, k, v, heads, attn_mode, local_attn_ctx=None, blocksize=32):
    n_ctx = q.size()[1]
    if attn_mode == 'strided':
        q = strided_transpose(q, n_ctx, local_attn_ctx, blocksize)
        k = strided_transpose(k, n_ctx, local_attn_ctx, blocksize)
        v = strided_transpose(v, n_ctx, local_attn_ctx, blocksize)
    n_state = q.size()[-1] // heads
    scale_amount = 1.0 / np.sqrt(n_state)
    w = torch.matmul(q, k.transpose(-2, -1))
    w = F.softmax(w * scale_amount, dim=-1)
    a = torch.matmul(w, v)
    if attn_mode == 'strided':
        n, t, embd = a.size()
        bT_ctx = n_ctx // local_attn_ctx
        a = torch.reshape(a, [n, local_attn_ctx, bT_ctx, embd])
        a = torch.transpose(a, 1, 2)
        a = torch.reshape(a, [n, t, embd])
    return a

class SparseAttention(nn.Module):
    def __init__(self, config):
        super(SparseAttention, self).__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout_p = config.dropout
        self.blocksize = config.block_size
        self.local_attn_ctx = 32
        self.attn_mode = "all"
        

    def forward(self, x):
        b, t, c = x.size()
        qkv = self.qkv(x).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k, v = [rearrange(x, 'b t h d -> (b h) t d') for x in (q, k, v)]

        # Implement sparse attention logic
        attn_output = blocksparse_attention_impl(q, k, v, self.n_head, self.attn_mode, self.local_attn_ctx, self.blocksize)
        attn_output = rearrange(attn_output, '(b h) t d -> b t (h d)', h=self.n_head)
        attn_output = self.proj(attn_output)

        return attn_output