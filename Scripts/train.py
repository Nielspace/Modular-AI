# Imports
import os
import math
import time
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext
import torch.distributed as dist
from tqdm import tqdm
import wandb
import argparse

from tokenise import TextDataset, pad_sequences
from attention import CausalSelfAttention, FlashAttention, SparseAttention
from customGPT import Block, GPT

def get_args():
    parser = argparse.ArgumentParser(description="Train GPT model with custom arguments")
    parser.add_argument('--data_path', type=str, default='wiki_medical_terms', help='Path to your text file')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--block_size', type=int, default=512, help='Block size')
    parser.add_argument('--n_layer', type=int, default=6, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=512, help='Embedding size')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--max_iters', type=int, default=5, help='Maximum iterations')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--eval_interval', type=int, default=2000, help='Evaluation interval')
    parser.add_argument('--eval_iters', type=int, default=200, help='Number of evaluation iterations')
    parser.add_argument('--log_interval', type=int, default=1, help='Logging interval')
    parser.add_argument('--init_from', type=str, default='scratch', help='Initialization mode: scratch, resume, gpt2*')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory for checkpoints')
    parser.add_argument('--backend', type=str, default='gloo', help='Backend for distributed training')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=40, help='Gradient accumulation steps')
    parser.add_argument('--local_attn_ctx', type=int, default=32, help='Local attention context')
    parser.add_argument('--attn_mode', type=str, default="local", help='Attention mode')
    parser.add_argument('--bias', type=bool, default=False, help='Use bias in model')
    parser.add_argument('--attention_type', type=str, default='sparse', choices=['causal', 'flash', 'sparse'], help='Type of attention mechanism to use')
    return parser.parse_args()

# System configuration
DTYPE = 'float32'
COMPILE = False
WEIGHT_DECAY = 1e-1
BETA1 = 0.9
BETA2 = 0.95
WARMUP_ITERS = 200
LR_DECAY_ITERS = 1000
MIN_LR = 6e-5
DECAY_LR = True
EVAL_ONLY = False
ALWAYS_SAVE_CHECKPOINT = True

# Functions and Classes
def load_data(data_path, block_size, batch_size):
    df = pd.read_parquet(data_path)
    articles = df.iloc[:, 1]
    dataset = TextDataset(articles, model="gpt2", seq_length=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequences)
    return dataloader, dataset

def init_ddp(args):
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        dist.init_process_group(backend=args.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'mps:{ddp_local_rank}' if torch.backends.mps.is_available() else f'cuda:{ddp_local_rank}'
        torch.mps.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert args.gradient_accumulation_steps % ddp_world_size == 0
        grad_accumulation_steps = args.gradient_accumulation_steps // ddp_world_size
    else:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        grad_accumulation_steps = args.gradient_accumulation_steps
    return ddp, master_process, device, seed_offset, ddp_world_size, grad_accumulation_steps

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    return model.configure_optimizers(weight_decay, learning_rate, betas, device_type)

def estimate_loss(model, dataloader, eval_iters, device, ctx):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        with tqdm(total=eval_iters, desc=f"Evaluating {split}") as pbar:
            for k in range(eval_iters):
                batch = next(iter(dataloader))
                X, Y = batch['input_ids'].to(device), batch['targets'].to(device)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
                pbar.update(1)
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout, local_attn_ctx, attn_mode, bias=True):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.local_attn_ctx = local_attn_ctx
        self.attn_mode = attn_mode

def main():
    args = get_args()
    
    # Load data
    dataloader, dataset = load_data(args.data_path, args.block_size, args.batch_size)
    print('Data Loaded')

    # Initialize DDP
    ddp, master_process, device, seed_offset, ddp_world_size, gradient_accumulation_steps = init_ddp(args)

    # Set seeds and device properties
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ctx = nullcontext()

    # Calculate vocab size based on the dataset
    vocab_size = len(dataset.tokens)

    # Select attention mechanism
    if args.attention_type == 'causal':
        attention_mechanism = CausalSelfAttention
    elif args.attention_type == 'flash':
        attention_mechanism = FlashAttention
    else:
        attention_mechanism = SparseAttention

    # Initialize model
    model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                      bias=args.bias, vocab_size=vocab_size, dropout=args.dropout,
                      local_attn_ctx=args.local_attn_ctx, attn_mode=args.attn_mode)
    if args.init_from == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_args)
        model = GPT(attention_mechanism, gptconf)
    elif args.init_from == 'resume':
        print(f"Resuming training from {args.out_dir}")
        ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(attention_mechanism, gptconf)
        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif args.init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {args.init_from}")
        model = GPT.from_pretrained(args.init_from, model_args)
        if args.block_size < model.config.block_size:
            model.crop_block_size(args.block_size)

    model.to(device)

    # Initialize optimizer and GradScaler
    optimizer = configure_optimizers(model, WEIGHT_DECAY, args.learning_rate, (BETA1, BETA2), device)
    scaler = GradScaler(enabled=(DTYPE == 'float16' and 'cuda' in device))

    # Wrap model in DDP if necessary
    if ddp:
        model = DDP(model, device_ids=[int(os.environ['LOCAL_RANK'])])

    # Training loop
    iter_num = 0
    best_val_loss = 1e9
    t0 = time.time()

    while iter_num <= args.max_iters:
        lr = get_lr(iter_num, args.learning_rate, WARMUP_ITERS, LR_DECAY_ITERS, MIN_LR) if DECAY_LR else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % args.eval_interval == 0 and master_process:
            losses = estimate_loss(model, dataloader, args.eval_iters, device, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss or ALWAYS_SAVE_CHECKPOINT:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss
                    }
                    print(f"saving checkpoint to {args.out_dir}")
                    torch.save(checkpoint, os.path.join(args.out_dir, 'ckpt.pt'))

        if iter_num == 0 and EVAL_ONLY:
            break

        with tqdm(total=gradient_accumulation_steps, desc=f"Training step {iter_num}") as pbar:
            total_train_loss = 0.0
            for micro_step in range(gradient_accumulation_steps):
                batch = next(iter(dataloader))
                X, Y = batch['input_ids'].to(device), batch['targets'].to(device)
                with ctx:
                    logits, loss = model(X, Y)
                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
                pbar.update(1)

        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.log_interval == 0 and master_process:
            print(f"iter {iter_num}: loss {loss.item() * gradient_accumulation_steps:.4f}, time {dt * 1000:.2f}ms")

        iter_num += 1

    if ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
