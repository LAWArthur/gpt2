import torch
from torch.nn import functional as F
from model import GPT, GPTConfig

# ----- TEST ------
num_return_sequences = 3
max_length = 1024
prompt = ''

while True:
    try:
        l = input()
        prompt += '\n'
        prompt += l
    except:
        break


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
# if torch_directml.is_available():
#     device = torch_directml.device(0)
print(device)
#device = torch.device('cpu')

#model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
checkpoint = torch.load('out/ckpt.pt')
state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict=state_dict)

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

        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

x.to('cpu')
for i in range(num_return_sequences):
    tokens = x[i,:max_length].tolist()
    decoded = enc.decode(tokens)
    print('>', decoded, '\n')