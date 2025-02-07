import torch.nn as nn
import torch
# import torch_directml
from torch.nn import functional as F
from dataclasses import dataclass
import inspect


@dataclass
class GPTConfig:
    block_size: int = 1024 # tmax
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd:int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_attn.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # used in attn mask
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                                .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B,T,C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B,T,self.n_head, C // self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head, C // self.n_head).transpose(1,2) # (B, nhead, T, head)

        # att = (q @ k.transpose(-2, -1)) * (k.size(-1) ** (-0.5)) # matmul & scale (B, nhead, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # mask out afterward tokens
        # att = F.softmax(att, dim=-1)
        # y = att @ v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1,2).contiguous().view(B,T,C) # concat heads
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
    
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.wpe(pos)
        tok_emb = self.wte(idx)
        x = pos_emb + tok_emb

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)

        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # curb the growth of residual stream
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, y=None):
        B,T = idx.size()
        assert T <= self.config.block_size, f"sequence too long({T})"
        x = self.transformer(idx)
        logits = self.lm_head(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss
    
    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in  {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print('loading weights from pretrained gpt: %s' % model_type)

        config_args = {
            'gpt2': dict(vocab_size = 50257, n_layer = 12, n_head = 12, n_embd = 768),
            'gpt2-medium': dict(vocab_size = 50257, n_layer = 24, n_head = 16, n_embd = 1024),
            'gpt2-large': dict(vocab_size = 50257, n_layer = 36, n_head = 20, n_embd = 1280),
            'gpt2-xl': dict(vocab_size = 50257, n_layer = 48, n_head = 25, n_embd = 1600)
        }[model_type]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # so its strange that bias is registered as a buffer...

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked.bias') and not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f'{k} shape {sd_hf[k].shape} doesnt match {sd[k].shape}'
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f'{k} shape {sd_hf[k].shape} doesnt match {sd[k].shape}'
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters')
        print(f'not decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters')
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f'using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8, fused=use_fused)
        return optimizer
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 82.58e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
