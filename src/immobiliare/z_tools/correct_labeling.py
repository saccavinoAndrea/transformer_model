import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from immobiliare.dataset.html_token_dataset import HTMLTokenDataset
from immobiliare.utils import resolve_versioned_jsonl, timestamped_path

# Pattern per agenzia “immobiliare”
AGENCY_KEYWORD = "Immobiliare"
# Sequenza righe che vogliamo intercettare
FIRST_MARKER  = "Annunci in zona"
FIRST_MARKER_ADD  = "Agenzie di zona"
SECOND_MARKERS = {"Gold", "Silver"}

# Pattern che identifica la riga di terminazione: "<Numero> risultati per:"
RESULTS_PATTERN = re.compile(r"^\s*\d+(\.\d+)?\s+risultati per:", re.IGNORECASE)

PATTERN_CASE_IN_VENDITA = re.compile(r"^case in vendita\s+\S+", flags=re.IGNORECASE)
PATTERN_IMMOBILIARE = re.compile(r'immobiliare\.it', re.IGNORECASE)
PATTERN_ANNUNCI = re.compile(r'annunci immobiliari a \w+', re.IGNORECASE)
PATTERN_CASE_VENDITA = re.compile(r'case in vendita (a\s)?\w+.*', re.IGNORECASE)
PATTERN_PROVINCIA = re.compile(r'provincia di \w+', re.IGNORECASE)


def substitute_label_by_word(rows: List[Dict[str, Any]], word: str) -> Tuple[int, int]:
    key0 = list(rows[0].keys())[0]
    last_key = list(rows[0].keys())[-1]

    matches = sum(1 for r in rows if r[key0] == word)
    substitutions = 0

    for r in rows:
        if word != "Solo Cucina Arredata" and word != "Descrizione_dettagliata":
            if r[key0] == word and r[last_key] != "O":
                r[last_key] = "O"
                substitutions += 1
        elif word == "Solo Cucina Arredata":
            if r[key0] == word and r[last_key] != "FEATURE_ARREDATO":
                r[last_key] = "FEATURE_ARREDATO"
                substitutions += 1
        elif word == "Descrizione_dettagliata":
            text = r[key0].lower()
            if (
                    len(r[key0]) >= 100
                    and "function" not in text
                    and "cookie" not in text
                    and r[last_key] != "FEATURE_DESCRIZIONE_DETTAGLIATA"
            ):
                r[last_key] = "FEATURE_DESCRIZIONE_DETTAGLIATA"
                substitutions += 1

    print(f"→ '{word}': {matches} righe trovate, {substitutions} etichette modificate")
    return matches, substitutions

def substitute_label_agency_sequence(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Scorre i record 4-a-4 e, quando trova la sequenza:
      1) O / "Annunci in zona" o "Agenzie di zona"
      2) O / "Gold" or "Silver"
      3) O / qualsiasi text
      4) O / text contiene "Immobiliare"
    rietichetta il 3° record come FEATURE_AGENZIA_IMMOBILIARE.

    Ritorna (match trovati, sostituzioni effettuate).
    """
    matches = 0
    subs = 0

    for i in range(len(rows) - 3):
        try:
            a, b, c, d = rows[i], rows[i+1], rows[i+2], rows[i+3]
            if (
                a.get("label") == "O" and (a.get("text") == FIRST_MARKER or FIRST_MARKER_ADD in a.get("text"))
                and b.get("label") == "O" and b.get("text") in SECOND_MARKERS
                and c.get("label") == "O"
                and d.get("label") == "O" and AGENCY_KEYWORD in d.get("text", "")
            ):
                matches += 1
                if c["label"] != "FEATURE_AGENZIA_IMMOBILIARE":
                    c["label"] = "FEATURE_AGENZIA_IMMOBILIARE"
                    subs += 1
        except Exception:
            continue

    print(f"→ Sequence agency: {matches} matches, {subs} substitutions")
    return matches, subs

def substitute_feature_zona_sequence(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Cerca "Scegli la zona" e rietichetta i token successivi con label "O" come FEATURE_ZONA,
    fino a trovare un token che matcha RESULTS_PATTERN.

    Modifica in-place la lista rows.

    Ritorna:
        matches: numero di occorrenze trovate di "Scegli la zona"
        substitutions: numero totale di token rietichettati
    """
    matches = 0
    substitutions = 0
    n = len(rows)
    i = 0

    while i < n:
        try:
            token = rows[i]
            if token.get("label") == "O" and token.get("text") == "Scegli la zona":
                matches += 1
                j = i + 1
                while j < n:
                    t = rows[j]
                    txt = t.get("text", "")
                    lbl = t.get("label", "")
                    if RESULTS_PATTERN.match(txt):
                        break
                    if lbl == "O":
                        rows[j]["label"] = "FEATURE_ZONA"
                        substitutions += 1
                    j += 1
                i = j
            else:
                i += 1
        except Exception:
            i += 1
            continue

    print(f"→ Sequence Zona: {matches} matches, {substitutions} substitutions")
    return matches, substitutions

def substitute_feature_banner_sequence(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Modifica in-place i token con label 'O' e text corrispondente a pattern/banner noti,
    rietichettandoli come FEATURE_BANNER_*.

    Ritorna una tupla (match trovati, sostituzioni effettuate).
    """
    exact_matches = {
        "RSS": "FEATURE_BANNER_RSS",
        "Vendi casa velocemente": "FEATURE_BANNER_PUBBLICITA",
        "Ti aiutiamo a trovare le agenzie di zona per vendere casa": "FEATURE_BANNER_PUBBLICITA",
        "Vendi con agenzia": "FEATURE_BANNER_PUBBLICITA",
        "Agenzie e costruttori": "FEATURE_BANNER_PUBBLICITA",
        "Privati": "FEATURE_BANNER_PUBBLICITA",
        "Ricerche correlate": "FEATURE_BANNER_PUBBLICITA",
        "Scegli la zona": "FEATURE_BANNER_PUBBLICITA",
        # "Prezzi immobili": "FEATURE_BANNER_PUBBLICITA",
        # "prezzi case": "FEATURE_BANNER_PUBBLICITA"
    }

    matches = 0
    substitutions = 0
    n = len(rows)

    for row in rows:
        try:
            text = row.get("text", "")
            label = row.get("label", "")
            if label != "O":
                continue

            new_label = None
            if text in exact_matches:
                new_label = exact_matches[text]
            elif "Comune" in text:
                new_label = "FEATURE_BANNER_COMUNE"
            elif (
                    PATTERN_CASE_IN_VENDITA.match(text) or
                    "risultati per:" in text or
                    text == "Più rilevanti" or
                    "Immobiliare.it" in text or
                    # "Prezzi case in vendita" in text or
                    PATTERN_ANNUNCI.match(text) or
                    PATTERN_CASE_VENDITA.match(text) or
                    PATTERN_PROVINCIA.match(text)
            ):
                new_label = "FEATURE_BANNER_PUBBLICITA"

            if new_label:
                row["label"] = new_label
                matches += 1
                substitutions += 1

        except Exception:
            continue

    print(f"→ Sequence Banner: {matches} matches, {substitutions} substitutions")
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

def save_label2id(input_path: Path, output_path: Path):

    rows: List[Dict[str, Any]] = []

    jsonl_path = resolve_versioned_jsonl(input_path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)

    # 5) salva la mappa tg label2id
    ds = HTMLTokenDataset()
    ds.calculate_label2id(rows)
    label2id_final_path = timestamped_path(output_path)
    with open(label2id_final_path, "w", encoding="utf-8") as f:
        json.dump(ds.label2id, f, ensure_ascii=False, indent=2)

    print("✅ Artifacts label2Id salvato in " + str(label2id_final_path))

def process(csv_file: str, output_csv: str, output_jsonl: str, output_label2id: str, words_to_substitute: list[str]):
    csv_path_resolved = resolve_versioned_jsonl(csv_file)
    path = Path(csv_path_resolved)
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    # Carica CSV in memoria
    with open(path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    total_matches = 0
    total_subs = 0

    for w in words_to_substitute:
        m, s = substitute_label_by_word(rows, w)
        total_matches += m
        total_subs += s

    m_seq, s_seq = substitute_label_agency_sequence(rows)
    total_matches += m_seq
    total_subs += s_seq

    zona_match, zona_count = substitute_feature_zona_sequence(rows)
    total_matches += zona_match
    total_subs += zona_count

    banner_match, banner_count = substitute_feature_banner_sequence(rows)
    total_matches += banner_match
    total_subs += banner_count

    print(f"🎯 Totale righe matching: {total_matches}, totali etichette modificate: {total_subs}\n")

    # Salva su output_csv_resolved
    output_csv_time = timestamped_path(output_csv)
    with open(output_csv_time, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Salva su output_jsonl_resolved
    output_jsonl_time= timestamped_path(output_jsonl)
    with open(output_jsonl_time, mode='w', encoding='utf-8') as jf:
        for row in rows:
            record = {k: cast_value(v) for k, v in row.items()}
            json.dump(record, jf, ensure_ascii=False)
            jf.write("\n")

    print(f"✅ Scrittura file completata: {output_csv}, {output_jsonl}")

    # Salva label2id
    save_label2id(Path(output_jsonl), Path(output_label2id))

def main():
    # ===== ESEMPIO DI USO =====
    csv_path = "data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled.csv"
    csv_output = "data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled_refine.csv"
    jsonl_output = "data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled_refine.jsonl"
    label2id_path = "data/transformer_model/labeling/artifacts/label2id_refine.json"
    parole = ["ascensore", "terrazzo", "cantina", "Solo Cucina Arredata", "Descrizione_dettagliata"]  # metti qui la tua lista di parole
    process(csv_path, csv_output, jsonl_output, label2id_path, parole)

if __name__ == "__main__":
    main()
