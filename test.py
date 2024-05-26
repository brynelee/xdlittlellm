from torch.nn import Embedding, Module, Parameter
import torch

""" emb = Embedding(10, 32)
s = 'a b c' # -> tokenizer -> ['a', 'b ', 'c']

r = emb(torch.tensor([
    [1,5,3]
]
    
))

print(r)
print(r.shape) """

class MyEmbedding(Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        # self.emb_matrix = Parameter(torch.randn(vocab_size, dim))
        self.emb_matrix = Parameter(torch.arange(0, vocab_size * dim, dtype=torch.float32).reshape(vocab_size, dim))

    def forward(self, ids): # 传进来的是一个批次，每一行都是一堆ID
        return self.emb_matrix[ids]
    
emb = MyEmbedding(10, 32)
r = emb(torch.tensor([
    [1,5,3]
]))
print(r)
print(r.shape)

