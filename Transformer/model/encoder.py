import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.dropout(self.attention(x, x, x, mask))
        x = self.norm1(x + attn_out)
        ff_out = self.dropout(self.feed_forward(x))
        return self.norm2(x + ff_out)
