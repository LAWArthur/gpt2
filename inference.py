import torch
from torch.nn import functional as F
from model import GPT, GPTConfig

# ----- TEST ------
num_return_sequences = 5
max_length = 30
prompt = "I know the moon, and this is an alien city."

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
# if torch_directml.is_available():
#     device = torch_directml.device(0)
print(device)
#device = torch.device('cpu')

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.load_state_dict(state_dict=torch.load('checkpoints/checkpoint_500.ckpt'))

model.eval()
model.to(device)
print('model loaded and moved to gpu.')

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(prompt)
print(tokens)

tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)[0]
        logits = logits[:, -1, :]

        probs = F.softmax(logits, dim=-1).cpu()
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol.to(device)), dim=1)

x.to('cpu')
for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded)