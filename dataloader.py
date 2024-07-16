import tiktoken
import torch
import numpy as np
class DataLoader:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        # with open('tinyshakespeare/input.txt', 'r') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f'loaded {len(self.tokens)} tokens.')
        # print(f'1 epoch = {len(self.tokens) //  (B*T) } batches.')

        self.tokens = np.memmap(f'{split}.bin', dtype=np.uint16, mode='r')
        print(f'loaded {len(self.tokens)} tokens.')
        print(f'1 epoch = {len(self.tokens) //  (B*T) } batches.')

        self.current_position = 0

    def next_batch(self):
        B,T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        buf = torch.tensor(buf)
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)

        self.current_position += B*T

        if self.current_position + B*T+1 > len(self.tokens):
            self.current_position = 0
        return x,y
    
    def reset(self):
        self.current_position = 0