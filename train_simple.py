import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
from dataloader import DataLoader, TextDataLoader
import math
import time
import tiktoken

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 19073
total_batch_size = 1024
B = 4
T = 256
assert total_batch_size % (B*T) == 0
grad_accum_steps = total_batch_size // (B*T)

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 *(1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
# elif torch_directml.is_available():
#     device = torch_directml.device(0)
print(device)
# device = 'cpu'


train_loader = TextDataLoader(B, T, 'tinyshakespeare/input.txt')
torch.set_float32_matmul_precision('high')

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

model = GPT(GPTConfig())
model = model.to(device)
# if torch.cuda.is_available(): model.compile()
# else: model.compile(backend='eager')
# print("model compiled!")

enc = tiktoken.get_encoding('gpt2')

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(0.1, min_lr, device=device)

for step in range(max_steps):
    # --- train ---
    model.train()
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    # x, y = train_loader.next_batch()
    for i in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        # print('begin forward!')
        # with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x,y)
        # print('ended forward!')
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tok_per_sec = (train_loader.B * train_loader.T * grad_accum_steps) / (t1 - t0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'step{step} | loss: {loss_accum.item():.6f} | norm: {norm.item():.4f} | lr: {lr:.4e} | dt: {dt: .1f} | tok/sec: {tok_per_sec:.1f}')

    