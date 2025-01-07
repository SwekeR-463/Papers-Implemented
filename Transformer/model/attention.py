import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return torch.matmul(attention, value), attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        assert embed_size % heads == 0
        self.d_k = embed_size // heads
        self.heads = heads
        self.embed_size = embed_size
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.query(query).view(batch_size, -1, self.heads, self.d_k).transpose(1,2)
        K = self.key(key).view(batch_size, -1, self.heads, self.d_k).transpose(1,2)
        V = self.value(value).view(batch_size, -1, self.heads, self.d_k).transpose(1,2)
        
        out, _ = ScaledDotProductAttention()(Q, K, V, mask)
        out = out.transpose(1,2).contiguous().view(batch_size, -1, self.embed_size)
        return self.fc_out(out)
        