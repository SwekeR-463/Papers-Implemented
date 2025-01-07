import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.encoder_attention = MultiHeadAttention(embed_size, heads)
        self.feed_forward = FeedForward(embed_size, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        attn_out = self.dropout(self.attention(x, x, x, tgt_mask))
        x = self.norm1(x + attn_out)
        enc_attn_out = self.dropout(self.encoder_attention(x, enc_out, enc_out, src_mask))
        x = self.norm2(x + enc_attn_out)
        ff_out = self.dropout(self.feed_forward(x))
        return self.norm3(x + ff_out)