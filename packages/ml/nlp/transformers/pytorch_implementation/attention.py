import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads=1):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.W_k = nn.Linear(d_model, d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None): # Mask is needed to create masked attention
        batch_size = q.size(0)
        Q_ = self.W_q(q)
        K_ = self.W_q(k)
        V_ = self.W_v(v)
        breakpoint()
        Q, K, V = [x.transpose(1, 2) for x in (Q_, K_, V_)]
        scores = torch.matmul(Q, K.transpose(2,1))
        d_scores = scores/torch.sqrt(self.d_k)
        s_scores = F.softmax(d_scores)
        f_scores = torch.matmul(s_scores, V)
        return f_scores



if __name__ == '__main__':
    batch_size = 2
    max_int = 1000
    context_size = 10
    embedding_size = 100
    # Embedding usually has a shape of [1,embedding_size]
    samples = torch.randn(batch_size, context_size,embedding_size)
    breakpoint()
    attn = MultiHeadAttention(embedding_size)
    inp = samples
    print(attn(inp, inp,inp).shape)
