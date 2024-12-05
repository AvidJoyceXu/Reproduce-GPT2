import torch.nn as nn
from sys import path
path.append('.')
path.append('./pretrain')

from pretrain.model_utils import Block
import torch
from pretrain.model_utils import GPTConfig
import torch.nn.functional as F
from dataloader import DataLoaderLite
import time
import inspect
import math

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd), # the num_embedding stands for the maximum unique embeddings it can generate
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd) # The final layer norm before the classifier head
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
        # Regress the embedding to vocabulary (require sigmoid to extract per-class scores)

    def forward(self, idx, targets=None):
        '''
        idx: (B, T)
        targets: (B, T)
        '''
        B, T = idx.size()
        assert T <= self.config.block_size, f"Text length {T} exceeds context window size {self.config.block_size}!"
        # Forward the position and token embeddingsx
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, config.n_embed)
        tok_emb = self.transformer.wte(idx) # (B, T, config.n_embed)
        x = pos_emb + tok_emb
        # Forward through the transformer
        for block in self.transformer.h:
            x = block(x)
        # Forward the last layer norm
        x = self.transformer.ln_f(x) # (B, T, config.n_embed)
        # Forward the regression head
        x = self.lm_head(x) # (B, T, config.vocab_size)
        loss = None
        
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, self.config.vocab_size), targets.view(-1)) # targets => (B*T)
        return x, loss


    @classmethod # Constructor
    def from_pretrained(cls, model_type):
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

        print("forcing vocab_size=50257, block_size=1024, bias=True")

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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
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
      
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

if __name__ == '__main__':
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig(vocab_size=50304))
    model.eval()

    if torch.backends.mps.is_available():
        device_type = 'mps'
    elif torch.cuda.is_available():
        device_type = 'cuda'
    else:
        device_type = 'cpu'
    device = torch.device(device_type)
    
    model = torch.compile(model)

    model.to(device)

    # num_return_sequences = 5
    # max_length = 50


    # enc = tiktoken.get_encoding('gpt2')
    # tokens = enc.encode("Hello, I'm a language model,")
    # tokens = torch.tensor(tokens, dtype=torch.long)
    # tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # repeat(1, 2, 1): repeat twice on the second dimension, don't repeat on the first or third dimension
    # x = tokens.to(device) # (5, 8)

    # torch.manual_seed(42)
    # # The next token problem.
    # while x.size(1) < max_length:
    #     with torch.no_grad():
    #         logits = model(x) # (B, T, vocab_size)
    #         logits = logits[:, -1, :] # Only accept the final token. 
    #         # (B, vocab_size)
    #         probs = torch.softmax(logits, dim=-1)
    #         topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
    #         # Select a token from the top-k probs
    #         ix = torch.multinomial(topk_probs, 1) # (B, 1)
    #         xcol = torch.gather(topk_indices, dim=-1, index=ix)
    #         x = torch.cat((x, xcol), dim=1) # Concat the newly sampled column to existing x.

    # for i in range(num_return_sequences):
    #     tokens = x[i, :max_length].tolist() # from tensor to list
    #     decoded = enc.decode(tokens)
    #     print(">", decoded)

    total_tokens_for_batch = 524288 # 2**19, ~0.5M in GPT-3 paper
    B = 16 # micro batch size
    T = 1024
    grad_accum_steps = total_tokens_for_batch // (B * T)
    print(f"Total token for one batch: {total_tokens_for_batch}, get {grad_accum_steps} grad accum steps")

    train_loader = DataLoaderLite(B=B, T=T)
    
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    warmup_steps = 25
    max_steps = 40

    
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
        
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

    for step in range(max_steps):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0 # for logging
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
            loss /= grad_accum_steps # scale the loss to account for accumulated gradient for greater batchsize
            loss_accum += loss.detach()
            loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()

        dt = (t1-t0)*1000 # time diff in miliseconds
        throughput = (train_loader.B * train_loader.T * grad_accum_steps) / (t1-t0)

        print(f"step {step}, lr: {lr:.4e}, loss: {loss_accum.item()}, dt:{dt:.2f}ms, norm: {norm:.4f}, throughput:{throughput} toks/sec")

