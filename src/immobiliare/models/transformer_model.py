# src/immobiliare/models/transformer_model.py

import torch
import torch.nn as nn
import math
from pathlib import Path
from typing import Any

from immobiliare.core_interfaces.model.imodel import IModel
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.utils.logging.decorators import log_exec


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(IModel, nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        num_classes: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        device: str = "cpu"
    ):
        nn.Module.__init__(self)
        self.logger = LoggerFactory.get_logger("transformer_model")
        self.device = torch.device(device)
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.pos_enc = PositionalEncoding(embedding_dim, max_len=max_seq_len)
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
        self.to(self.device)

    @log_exec(logger_name="transformer_model", method_name="train")
    def train(self, train_data: Any, val_data: Any = None) -> None:
        """
        train_data: DataLoader di (features, mask, labels)
        """
        optimizer = torch.optim.Adam(self.parameters())
        loss_fn = nn.CrossEntropyLoss()
        self.train()
        for epoch, loader in enumerate(train_data, 1):
            total_loss = 0.0
            for x, mask, y in loader:
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(x, mask)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.logger.log_info(f"Epoch {epoch} training loss: {total_loss/len(train_data):.4f}")
        # validazione opzionale...

    @log_exec(logger_name="transformer_model", method_name="predict")
    def predict(self, inputs: Any) -> Any:
        """
        inputs: tuple(features_tensor, mask_tensor)
        ritorna: logits o labels
        """
        x, mask = inputs
        self.eval()
        with torch.no_grad():
            x, mask = x.to(self.device), mask.to(self.device)
            logits = self.forward(x, mask)
        return logits

    @log_exec(logger_name="transformer_model", method_name="save")
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        self.logger.log_info(f"Model salvato in {path}")

    @log_exec(logger_name="transformer_model", method_name="load")
    def load(self, path: Path) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.logger.log_info(f"Model caricato da {path}")

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = x.permute(1, 0, 2)
        x = self.dropout(x)
        return self.classifier(x)
