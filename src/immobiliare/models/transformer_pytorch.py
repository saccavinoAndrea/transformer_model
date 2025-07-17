# src/immobiliare/models/transformer_pytorch.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # pos * sin
        pe[:, 1::2] = torch.cos(position * div_term)  # pos * cos

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


class TransformerForTokenClassification(nn.Module):
    def __init__(
        self,
        input_dim: int,       # es: 56 feature numeriche
        embedding_dim: int,   # es: 128
        num_classes: int,     # es: 10 etichette
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)

        # Verifica che la tua embedding layer sia correttamente dimensionata
        assert self.embedding.weight.shape == (embedding_dim, input_dim), (
            f"Embedding weight shape mismatch: "
            f"got {self.embedding.weight.shape}, "
            f"expected ({embedding_dim}, {input_dim})"
        )

        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, attention_mask=None):

        # x shape: [B, L, input_dim]
        assert x.dim() == 3, f"Expected 3‑D input, got {x.dim()}‑D"
        in_dim = x.size(-1)
        exp_in = self.embedding.in_features
        assert in_dim == exp_in, (
            f"Input feature dimension mismatch: "
            f"got {in_dim}, expected {exp_in}"
        )

        """
        x: [batch_size, seq_len, input_dim]
        attention_mask: [batch_size, seq_len] con 1 dove c'è token valido, 0 dove c'è PAD
        """
        x = self.embedding(x)                        # [B, L, D]
        x = self.positional_encoding(x)              # [B, L, D]
        x = self.dropout(x)

        x = x.permute(1, 0, 2)  # [L, B, D]

        if attention_mask is not None:
            # Trasforma in maschera booleana per PyTorch
            key_padding_mask = ~attention_mask.bool()  # [B, L] → True dove PAD
        else:
            key_padding_mask = None

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # [L, B, D]
        x = x.permute(1, 0, 2)  # [B, L, D]
        x = self.dropout(x)
        logits = self.classifier(x)  # [B, L, num_classes]
        return logits
