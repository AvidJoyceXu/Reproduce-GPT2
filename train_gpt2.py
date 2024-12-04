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

if __name__ == '__main__':
    # model = GPT.from_pretrained('gpt2')
    model = GPT(GPTConfig(vocab_size=50304))
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
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
    
    train_loader = DataLoaderLite(B=32, T=1024)
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

    # torch.set_float32_matmul_precision('high')
    for i in range(50):
        t0 = time.time()

        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        t1 = time.time()

        dt = (t1-t0)*1000 # time diff in miliseconds
        throughput = (train_loader.B * train_loader.T) / (t1-t0)

        print(f"step {i}, loss: {loss.item()}, dt:{dt:.2f}ms, throughput:{throughput} toks/sec")

