import torch

def html_collate_fn(batch):
    """
    Collate function per HTMLTokenDataset che:
    - batcha vettori di feature 1‑D di lunghezza fissa
    - batcha label scalari
    Restituisce:
      features: Tensor [batch_size, D]
      labels:   Tensor [batch_size]
    """
    features, labels = zip(*batch)
    # stack dei feature: da tuple(batch_size) di [D] -> [batch_size, D]
    features = torch.stack(features, dim=0)
    # stack delle label scalari: da tuple(batch_size) di [] -> [batch_size]
    labels = torch.stack(labels, dim=0)
    return features, labels

