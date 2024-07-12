import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
from dataloader import DataLoader
import math

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 *(1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.cuda.device(0)
# elif torch_directml.is_available():
#     device = torch_directml.device(0)
print(device)
# device = 'cpu'

loader = DataLoader(B = 4, T = 256)
torch.set_float32_matmul_precision('high')

torch.manual_seed(2024)

model = GPT(GPTConfig())
model.to(device)
model = torch.compile(model, backend='eager')
# print("model compiled!")


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)


for step in range(max_steps):
    x, y = loader.next_batch()
    x,y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x,y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'step{step} | loss: {loss.item():.6f} | norm: {norm.item():.4f} | lr: {lr:.4e}')