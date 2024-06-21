import math
import inspect
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

#torch._dynamo.config.suppress_errors = True
# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.linear = nn.Linear(10, 10)

#     def forward(self, x):
#         return self.linear(x)

# model = SimpleModel()
# model = torch.compile(model)
# x = torch.randn(1, 10)
# print(model(x))
# import code; code.interact(local=locals())

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        
        # attention (materializes the large (T,T) matrix for all the queries and keys)
        # regular attention implementation:
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        # Flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
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
    # let's load the original gpt2 weights and generate tokens
    # match the same schema of HF model

    def __init__(self, config):
        super().__init__()
        self.config = config

        # main module in HF model is transformer (in notebook)
        # nn.ModuleDict is a module that allows us to index into submodules. 
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
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # std with Xavier init is 1/sqrt(num features in incoming layer)
            # either 768 or 1024 num features 1/sqrt(768) 0r 1/sqrt(1024) ~ 0.03
            # similar to 0.02 range. 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # by default bias init with uniform
            
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # decaying weights is a form of regularization, forcing optimizaton to not allow large weights
        # only decay embeddings and matmult participating weights.
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# ------------------------------------------------------------------------
import tiktoken

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
        # sample without replacement
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y

#----------------------------------------------------------------------
# attempt to autodetect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
# there's a bug for mps case
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps" # backend for apple silicon 
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

train_loader = DataLoaderLite(B=16, T=1024) # actual gpt2 context length is 1024 tokens
# if above doesn't fit into gpu and we get oom, keep decreasing batch size until fits.
# by default we want to max out batch size, use numbers that have many powers of two's.

torch.set_float32_matmul_precision('high')
# tells pytorch what kernels to run. By default it's highest (float32)
# high runs tf32. matmult with run on Tensor Cores.
# Available on A100, on older gpus might not be avaible.
# Based on specs it should increase throughput by 8x.
# in practice in this code we see 3x because we are memory-bound.

# get logits
model = GPT(GPTConfig(vocab_size=50304))
# increase ugly number to nearest number with many powers of 2's. very nice number, divides by 128
# functionally nothing breaks, it's like adding new tokens that will never be used. Nothing different from tokens never present in a dataset.
# we're adding calculations but it rans faster! cuda kernels work in powers of two numbers.
# The kernels chunk to powers to two then solve the nice parts and then come back to remaining parts. It's best to pad.

model.to(device)

# A NumPy version >=1.17.3 and <1.25.0 is required
# worked with this version: pip install numpy==1.22.4 & 1.24.1
# latest version less than 
model = torch.compile(model)

# learning rate scheuler in gpt3 is called cosine decay lr schedule with warmup.
# starts at zero, linearly ramps up over some amount of time and comes down with a cosine form.
max_lr = 6e-4 # page 8 of gpt3 paper
min_lr = max_lr * 0.1 # per description in paper
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad() # .backward adds to gradient (+=), so we must set to zero at the begining
    with torch.autocast(device_type=device, dtype=torch.bfloat16): # if we use float16 we need to do gradient scaling.
        logits, loss = model(x, y) # init loss ~ -ln(1/50257) = 10.8
        # with autocaset context manager logits are in bfloat16 but model.transformer.wte.weight.dtype is still float32
        # activations are converted but parameters aren't. That's the mixed precision part. It's not clear what is converted.
        # import code; code.interact(local=locals())
    loss.backward()
    # clip gradient to have maximum norm. square every gradient of parameters and add all up and square root.
    # sometimes we get unlucky during optimization, bad data batch, we get high loss and high gradient, it shocks the model.
    # it's best to visualize them. if norm of gradient is well behaved it's good, if climbing not stable training, sometimes there are spikes.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if torch.cuda.is_available(): 
        torch.cuda.synchronize() # wait for all scheduled gpu jobs to finish.
    t1 = time.time()
    dt = (t1 - t0) # time diff in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T) / dt
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

# at every batch we feed new data, so not overfitting on a single batch.
# each epoch is 2640 batches, we're only doing 50, so not expecting a lot of gain here.
# most of the gain is from canceling tokens that never occur
# pushing their bias to large negative number that makes the softmax probability to almost 0.
# with this code the loss comes down to 6.84

"""
watch -n 0.1 nvidia-smi

baseline with 1 A100 GPU with 40GB memory with FP32 tensors B=32 T=1024: 
time per iter: 1040ms, tokens_per_sec throughput: 15750

after changing from fp32 to tf32:
time per iter: 381ms, tokens_per_sec throughput: 42900

after mixed precision to bfloat16:
time per iter: 334ms, tokens_per_sec throughput: 49000
It has less precision. tradeoff. less accurate but can train longer and make up for it.

after torch.compile:
time per iter: 201ms, tokens_per_sec throughput: 81000

after flash attention:
time per iter: 155ms, tokens_per_sec throughput: 105000

after changing vocab size from 50257 to 50304:
time per iter: 122ms, tokens_per_sec throughput: 134000

after weight decay and fused AdamW optimizer
time per iter: 118ms, token_per_sec throughput: 138000
"""

import sys; sys.exit(0)


# generate tokens from the model
# identical to generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5) in notebook
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
#model = GPT(GPTConfig()) #random model
print("didn't crash yay!")
model.eval() # when just using the model, not training. models have different behaviors like Dropout.
model.to(device)
# what pytroch does for us internally when we do model.to(device)?

# prefix tokens
import tiktoken
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,) 8 tokens
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    # forward the model to get the logits
    #with torch.no_grad:
    logits = model(x) # (B, T, vocab_size)
    # take the logits at the last position
    logits = logits[:, -1, :] # (B, vocab_size)
    # get the probabilities
    probs = F.softmax(logits, dim=-1)
    # do top-k sampling of 50 (huggingface pipeline default) top50 of most likely tokens
    # https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/&amp;num;transformers.GenerationConfig.top_k

    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    # select a token from the top-k probabilities
    # note: multinomial does not demand the input to sum to 1
    ix = torch.multinomial(topk_probs, 1) # (B, 1)
    # gather the corresponding indices
    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    # append to the sequence
    x = torch.cat((x, xcol), dim=1)

# print the generated text
import tiktoken
enc = tiktoken.get_encoding('gpt2')
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)