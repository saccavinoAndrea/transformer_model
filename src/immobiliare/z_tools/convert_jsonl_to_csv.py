# src/immobiliare/z_tools/convert_jsonl_to_csv.py

import json
import csv
import os
from pathlib import Path

def convert_jsonl_to_csv(directory: str, filename: str):
    directory = Path(directory)
    jsonl_path = directory / filename

    if not jsonl_path.exists():
        raise FileNotFoundError(f"File non trovato: {jsonl_path}")

    if not jsonl_path.suffix == ".jsonl":
        raise ValueError("Il file deve avere estensione .jsonl")

    csv_path = jsonl_path.with_suffix(".csv")

    with open(jsonl_path, 'r', encoding='utf-8') as jsonl_file:
        lines = [json.loads(line) for line in jsonl_file if line.strip()]

    if not lines:
        raise ValueError("Il file JSONL è vuoto")

    # Usa le chiavi del primo dizionario come intestazioni
    fieldnames = list(lines[0].keys())

    with open(csv_path, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lines)

    print(f"File CSV salvato in: {csv_path}")



if __name__ == "__main__":
    root_path = "data/transformer_model/labeling/artifacts/old/"
    a_pages = "20_pages_training/100_ft/tokens/analysis"
    b_pages = "40_pages_training/tokens/analysis"
    c_pages = "50_pages_training/100_ft/tokens/analysis"
    d_pages = "100_pages_training/tokens/"


    #convert_jsonl_to_csv(root_path + a_pages, "tokens_filtered_20250716_090226.jsonl")
    #convert_jsonl_to_csv(root_path + b_pages, "tokens_filtered_20250716_005043.jsonl")
    #convert_jsonl_to_csv(root_path + c_pages, "tokens_filtered_20250716_094636.jsonl")
    #convert_jsonl_to_csv(root_path + d_pages, "tokens_filtered_20250717_092439.jsonl")
    convert_jsonl_to_csv(root_path + d_pages, "token_embeddings_dense_labeled_20250720_232648.jsonl")