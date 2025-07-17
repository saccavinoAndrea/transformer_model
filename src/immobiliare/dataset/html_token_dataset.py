import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Any


class HTMLTokenDataset(Dataset):
    def __init__(
        self,
        #jsonl_path: str,
        label2id: Optional[Dict[str, int]] = None,
        feature_keys_to_use: Optional[List[str]] = None,
    ):
        self.samples = []
        self.feature_keys_to_use = feature_keys_to_use
        self.label2id = label2id

        """
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if "label" not in record:
                    continue  # scarta token non etichettati
                self.samples.append(record)

        # Costruzione dizionario label2id (se non fornito)
        if not label2id:
            all_labels = sorted(set(rec["label"] for rec in self.samples))
            self.label2id = {label: idx for idx, label in enumerate(all_labels)}

        # Determina feature numeriche valide
        if not self.feature_keys_to_use:
            numeric_keys = [
                k for k in self.samples[0].keys()
                if isinstance(self.samples[0][k], (int, float)) and k != "position"
            ]
            self.feature_keys_to_use = numeric_keys
        """

    def retrieve_labeled_samples(self, jsonl_path: str):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if "label" not in record:
                    continue  # scarta token non etichettati
                self.samples.append(record)

    def calculate_feature_keys(self, jsonl_path: str):

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                self.samples.append(record)

        # Determina feature numeriche valide
        if not self.feature_keys_to_use:
            numeric_keys = [
                k for k in self.samples[0].keys()
                if isinstance(self.samples[0][k], (int, float)) and k != "position"
            ]
            self.feature_keys_to_use = numeric_keys

    def calculate_label2id(self, row: List[Dict[str, Any]]):
        # Costruzione dizionario label2id (se non fornito)
        if not self.label2id:
            all_labels = sorted(set(rec["label"] for rec in row))
            self.label2id = {label: idx for idx, label in enumerate(all_labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        x = torch.tensor([rec[k] for k in self.feature_keys_to_use], dtype=torch.float)
        y = torch.tensor(self.label2id[rec["label"]], dtype=torch.long)
        return x, y

