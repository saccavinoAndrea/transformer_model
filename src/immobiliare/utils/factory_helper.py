# src/immobiliare/utils

import json
from pathlib import Path

def load_label2id() -> dict:
    # es: data/label2id.json salvata dopo il preprocess
    p = Path("data/artifacts/label2id.json")
    return json.loads(p.read_text(encoding="utf-8"))

def load_feature_keys() -> list:
    # es: data/artifacts/feature_keys.json
    p = Path("data/artifacts/feature_keys.json")
    return json.loads(p.read_text(encoding="utf-8"))
