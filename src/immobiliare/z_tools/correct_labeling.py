import csv
import json
from pathlib import Path

from utils import resolve_versioned_jsonl, timestamped_path


def substitute_label_by_word(input_path: Path, word: str) -> (int, int):
    """
    - Legge il CSV con DictReader
    - Conta quante righe hanno la prima colonna == word
    - Per ognuna, se l'ultima colonna != "O", la imposta a "O" e incrementa subs
    - Riscrive il CSV in-place con DictWriter
    - Ritorna (matches, substitutions)
    """

    # 1) Leggi tutto
    with open(input_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("CSV senza header o vuoto")
        rows = list(reader)

    key0 = fieldnames[0]
    last_key = fieldnames[-1]

    matches = sum(1 for r in rows if r[key0] == word)
    substitutions = 0

    for r in rows:
        if word != "Solo Cucina Arredata":
            if r[key0] == word and r[last_key] != "O":
                r[last_key] = "O"
                substitutions += 1
        elif word == "Solo Cucina Arredata":
            if r[key0] == word and r[last_key] != "FEATURE_ARREDATO":
                r[last_key] = "FEATURE_ARREDATO"
                substitutions += 1


    # 2) Riscrivi il CSV in-place
    with open(input_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"→ '{word}': {matches} righe trovate, {substitutions} etichette modificate")
    return matches, substitutions

def cast_value(val: str):
    val = val.strip()
    if val == "":
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val

def convert_csv_to_json_with_numbers(output_csv_path: Path) -> int:
    """
    Converte in JSONL con cast automatico di numeri.
    """
    jsonl_path = output_csv_path.with_suffix(".jsonl")
    count = 0
    with open(output_csv_path, mode='r', encoding='utf-8') as cf, \
         open(jsonl_path, mode='w', encoding='utf-8') as jf:
        reader = csv.DictReader(cf)
        for row in reader:
            record = {k: cast_value(v) for k, v in row.items()}
            json.dump(record, jf, ensure_ascii=False)
            jf.write("\n")
            count += 1
    print(f"✅ {count} record scritti in JSONL: {jsonl_path}")
    return count

def process(csv_file: str, words_to_substitute: list[str]):
    csv_path_resolved = resolve_versioned_jsonl(csv_file)
    path = Path(csv_path_resolved)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    total_matches = 0
    total_subs = 0

    for w in words_to_substitute:
        m, s = substitute_label_by_word(path, w)
        total_matches += m
        total_subs += s

    print(f"🎯 Totale righe matching: {total_matches}, totali etichette modificate: {total_subs}\n")

    convert_csv_to_json_with_numbers(path)


if __name__ == "__main__":
    # ===== ESEMPIO DI USO =====
    #csv_path = "data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled_20250720_232648.csv"
    csv_path = "data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled.csv"
    parole = ["ascensore", "terrazzo", "cantina", "Solo Cucina Arredata"]  # metti qui la tua lista di parole
    process(csv_path, parole)
