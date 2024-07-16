import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
from dataloader import DataLoader
import math
import time
import tiktoken

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
save_steps = 1000
total_batch_size = 524288
val_step = 100
B = 16
T = 1024
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


train_loader = DataLoader(B = B, T = T, split='train')
val_loader = DataLoader(B=B, T=T, split='val')
torch.set_float32_matmul_precision('high')

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

model = GPT(GPTConfig())
model = model.to(device)
if torch.cuda.is_available(): model.compile()
else: model.compile(backend='eager')
# print("model compiled!")

enc = tiktoken.get_encoding('gpt2')

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(0.1, 6e-4, device=device)

for step in range(max_steps):
    if step % val_step == 0: # validation
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        print(f"validation loss: {val_loss_accum.item():.4f}")

        num_return_sequences = 3
        max_length = 64
        prompt = "I know the moon, and this is an alien city."
        tokens = enc.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42)
        x = tokens.to(device)
        while x.size(1) < max_length:
            with torch.no_grad():
                logits = model(x)[0]
                logits = logits[:, -1, :]

                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol.to(device)), dim=1)

        x.to('cpu')
        for i in range(num_return_sequences):
            tokens = x[i,:max_length].tolist()
            decoded = enc.decode(tokens)
            print(f'Sample {i} >', decoded)


    if step % save_steps == 0: # save
        torch.save(model.state_dict(), f'./checkpoints/checkpoint_{step}.ckpt')


    # --- train ---
    model.train()
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for i in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x,y)
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

    