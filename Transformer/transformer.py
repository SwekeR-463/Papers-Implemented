import torch
import torch.nn as nn
from model.encoder import EncoderLayer
from model.decoder import DecoderLayer
from model.positional import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_layers, heads, hidden_dim, dropout, max_len):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_len)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_size, heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embed_size, heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        return (src != 0).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.size()
        mask = torch.tril(torch.ones((tgt_len, tgt_len))).bool()
        return mask.unsqueeze(0).expand(N, -1, -1)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        dec_out = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, enc_out, src_mask, tgt_mask)

        return self.fc_out(dec_out)