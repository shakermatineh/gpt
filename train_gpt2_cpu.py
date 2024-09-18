"""
Same as train_gpt, on shakespear dataset and without speed optimizations.
"""

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        hs = C // self.n_head
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2) # (B, nh, T, hs)
        
        # regular attention materializes the large (T,T) matrix
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)        
        
        # flash attention
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True) 
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y) # output projection
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # using approx version with tanh instead on exact version of gelu.
        # at the time erf was slow in tensorflow.
        # gpt2 used tanh approximate version, we stick to that.
        # addresses dead relu neuron problem. if flat 0, any activation gets 0 grad.
        # gelu always contribute a local gradient, always a change.
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        # two linear projections, gelu in between
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # difference with original transformer
        # prenorm: layernorm before attention
        # residual pathway doesn't go through normalization, direct addition.
        # clean residual pathways are desirable for optimization.
        x = x + self.attn(self.ln_1(x)) #communication. tokens communicate. aggreagation, pooling, weighted sum function, reduce operation.
        x = x + self.mlp(self.ln_2(x)) #computation. tokens think individually. map operation. transformers repeated applications of mapreduce.
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):
    # match the same schema of HF model

    def __init__(self, config):
        super().__init__()
        self.config = config

        # main module in HF model is transformer (in notebook)
        # nn. is a module that allows us to index into submodules. 
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), 
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # ModuleList to index using integers.
            ln_f = nn.LayerNorm(config.n_embd), # final layernorm before classifier
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # uses no bias for final projection.

        # weight sharing scheme. 
        # taking wte.weight and redirect it to point to lm_head 
        # copies data pointer, reference, wte.weight becomes orphan and python cleans it up.
        # 768 * 50257 = 40 million parameters, 30% of total params saving.
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # https://github.com/openai/gpt-2/blob/master/src/model.py#L147
        # in above code te initialized with std=0.02, and pe with std=0.01
        # we just use 0.02 for all embeddings
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # scale the weights of residual layers at initialization 
                # by a factor of 1/sqrt(N) based on GPT2 paper section 2.3
                std *= (2 * self.config.n_layer) ** -0.5 # in every block two times residuals added, in mlp and self attention 
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # std with Xavier init is 1/sqrt(num features in incoming layer)
            # either 768 or 1024 num features 1/sqrt(768) 0r 1/sqrt(1024) ~ 0.03
            # similar to 0.02 range. 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # by default bias init with uniform
            
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        # Careful to create pos on the right device. When calling forward:
        # model.to(device) and logits = model(x), no mismatch of one tensor on cpu other gpu.
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb # broadcasting hidden
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    

class DataLoaderLite:
    # create a simple dataloader to keep reading B*T batches from file to train on.
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # at init load tokens from disk and store in memory
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

#----------------------------------------------------------------------

def generate_sequences_with_huggingface(
    model, tokenizer, seed_text="Hello, I'm a language model,",
    num_sequences=5, max_length=30, device="cpu"):

    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    model.eval()

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_sequences,
            do_sample=True,           # Sampling instead of greedy
            top_k=50,                 # Top-k sampling
            top_p=0.95,               # Nucleus sampling
            temperature=1.0,          # Temperature controls randomness
            pad_token_id=tokenizer.eos_token_id  # Ensure padding is handled
        )

    for i in range(num_sequences):
        decoded = tokenizer.decode(output[i], skip_special_tokens=True)
        print(f"Sample {i + 1}: {decoded}")


def generate_sequences_from_model(
    model, seed_text="Hello, I'm a language model,",
    num_sequences=5, max_length=30, device="cpu"):

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(seed_text)
    tokens = torch.tensor(tokens, dtype=torch.long) # size [8] 
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1) # first size [1, 8] then [5, 8]
    
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(1337)
    model.eval()

    while xgen.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(xgen) # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (5, 50)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            sampled_idx = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
            next_token = torch.gather(topk_indices, -1, sampled_idx) # (B, 1)
            xgen = torch.cat((xgen, next_token), dim=1)
    
    for i in range(num_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(f"sample {i}: {decoded}")


def train_model_lm_objective(num_epochs=50, batch_size=4, block_size=32, lr=3e-4, device="cpu"):

    train_loader = DataLoaderLite(B=batch_size, T=block_size)
    model = GPT(GPTConfig())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for i in range(num_epochs):
        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad() # .backward adds to gradient (+=), so we must set to zero at the begining
        logits, loss = model(x, y) # init loss ~ -ln(1/50257) = 10.8
        loss.backward()
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = (train_loader.B * train_loader.T) / dt
        print(f"step {i:5d} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")
    torch.manual_seed(1337)
    if torch.cuda.is_available(): torch.cuda.manual_seed(1337)


    print("\nGenerating sequences from loaded pre-trained GPT-2 with manual sampling:")
    pretrained_model = GPT.from_pretrained('gpt2').to(device)
    generate_sequences_from_model(pretrained_model, device=device)


    print("\nGenerating sequences from pre-trained GPT-2 using Hugging Face pipelines:")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    pretrained_model_hf = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    tokenizer_hf = GPT2Tokenizer.from_pretrained('gpt2')
    generate_sequences_with_huggingface(pretrained_model_hf, tokenizer_hf, device=device)


    print("\nTraining custom GPT model on Shakespeare dataset:")
    custom_model = train_model_lm_objective(device=device)


    print("\nGenerating samples from custom-trained GPT model:")
    generate_sequences_from_model(custom_model, device=device)