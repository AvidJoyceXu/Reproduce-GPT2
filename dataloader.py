import tiktoken
import torch

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split='train'):
        self.B = B
        self.T = T

        assert split in {'train', 'val'}
        print("Warning: split not implemented yet")

        with open('pretrain/input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_position = process_rank * B * T # pointer
        self.num_processes = num_processes

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        # Reset
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y