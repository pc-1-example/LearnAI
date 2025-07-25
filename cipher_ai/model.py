import torch
import torch.nn as nn

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) + self.pos_encoder[:, :src.size(1)]
        tgt_emb = self.embedding(tgt) + self.pos_encoder[:, :tgt.size(1)]
        memory = self.encoder(src_emb)
        output = self.decoder(tgt_emb, memory)
        return self.fc_out(output)
