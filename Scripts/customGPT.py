import math

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import transformers

from einops import rearrange
from typing import Optional, Union

import tiktoken
import pandas as pd
import re

import os


class Block(nn.Module):
    def __init__(self, att, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = att(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# Assuming GPTConfig is defined somewhere in the code
class GPT(nn.Module):
    def __init__(self, att, config):
        super().__init__()
        assert config.vocab_size is not None, "Config must include vocab_size"
        assert config.block_size is not None, "Config must include block_size"
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            'h': nn.ModuleList([Block(att, config) for _ in range(config.n_layer)]),
            'ln_f': nn.LayerNorm(config.n_embd, eps=1e-5),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tie_weights()

        self.apply(self._init_weights)
        self.init_residuals()
        print(f"Number of parameters: {self.get_num_params()/1e6:.2f}M")

    def tie_weights(self):
        self.transformer.wte.weight = self.lm_head.weight

    def init_residuals(self):
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Sequence length {t} exceeds block size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = logits[:, [-1], :]  # Use only the last token's logits
            return logits, None

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, "Invalid model type"
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args), "Only 'dropout' can be overridden"

        print(f"Loading weights from pretrained model: {model_type}")

        config_args = cls.get_config_args(model_type)
        if 'dropout' in override_args:
            print(f"Overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        config = GPTConfig(**config_args)
        model = GPT(config)
        model.load_pretrained_weights(model_type)
        return model

    @staticmethod
    def get_config_args(model_type):
        config_map = {
            'gpt2': {'n_layer': 12, 'n_head': 12, 'n_embd': 768},
            'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'n_embd': 1024},
            'gpt2-large': {'n_layer': 36, 'n_head': 20, 'n_embd': 1280},
            'gpt2-xl': {'n_layer': 48, 'n_head': 25, 'n_embd': 1600},
        }
        config_args = config_map[model_type]
        config_args.update({'vocab_size': 50257, 'block_size': 1024, 'bias': True, 'dropout': 0.1})
        return config_args

    def load_pretrained_weights(self, model_type):
        model_hf = transformers.GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd = self.state_dict()
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k, v in sd_hf.items():
            if k in sd:
                if any(k.endswith(w) for w in transposed):
                    sd[k].copy_(v.t())
                else:
                    sd[k].copy_(v)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        N = self.get_num_params()
        L, H, Q, T = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx