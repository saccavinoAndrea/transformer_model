import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from dataset.html_token_dataset import HTMLTokenDataset
from utils import resolve_versioned_jsonl, timestamped_path

# Pattern per agenzia “immobiliare”
AGENCY_KEYWORD = "Immobiliare"
# Sequenza righe che vogliamo intercettare
FIRST_MARKER  = "Annunci in zona"
FIRST_MARKER_ADD  = "Agenzie di zona"
SECOND_MARKERS = {"Gold", "Silver"}

# Pattern che identifica la riga di terminazione: "<Numero> risultati per:"
RESULTS_PATTERN = re.compile(r"^\s*\d+(\.\d+)?\s+risultati per:", re.IGNORECASE)

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


    # 2) Riscrivi il CSV in-place
    with open(input_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"→ '{word}': {matches} righe trovate, {substitutions} etichette modificate")
    return matches, substitutions

def substitute_label_agency_sequence(input_path: Path) -> (int, int):
    """
    Scorre i record 4-a-4 e, quando trova la sequenza:
      1) O / "Annunci in zona" o "Agenzie di zona"
      2) O / "Gold" or "Silver"
      3) O / qualsiasi text
      4) O / text contiene "Immobiliare"
    rietichetta il 3° record come FEATURE_AGENZIA_IMMOBILIARE.
    """
    matches = 0
    subs    = 0

    # 1) Leggi tutto
    with open(input_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # 2) Scorri in finestra
    for i in range(len(rows) - 3):
        try:
            a, b, c, d = rows[i], rows[i+1], rows[i+2], rows[i+3]
            if (a.get("label") == "O" and (a.get("text") == FIRST_MARKER or FIRST_MARKER_ADD in a.get("text"))
                and b.get("label") == "O" and b.get("text") in SECOND_MARKERS
                and c.get("label") == "O"
                and d.get("label") == "O" and AGENCY_KEYWORD in d.get("text", "")
            ):
                matches += 1
                if c["label"] != "FEATURE_AGENZIA_IMMOBILIARE":
                    c["label"] = "FEATURE_AGENZIA_IMMOBILIARE"
                    subs += 1
        except Exception as e:
            # salta finestre malformate senza interrompere
            continue

    # 3) Riscrivi in-place
    with open(input_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"→ Sequence agency: {matches} matches, {subs} substitutions")
    return matches, subs

def substitute_feature_zona_sequence(input_path: Path):
    """
    Scorre la lista di token (ogni elemento è dict con 'label' e 'text'),
    quando trova un elemento con text == "Scegli la zona",
    etichetta come FEATURE_ZONA tutti i token successivi con label "O"
    fino a incontrare un text che matcha RESULTS_PATTERN.

    Modifica in-place rows, e ritorna il numero di token rietichettati.
    """
    # 1) Leggi tutto
    with open(input_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    count = 0
    n = len(rows)
    i = 0

    while i < n:
        try:
            token = rows[i]
            if token.get("label") == "O" and token.get("text") == "Scegli la zona":
                # da qui in poi, etichetta fino al break
                j = i + 1
                while j < n:
                    t = rows[j]
                    txt = t.get("text", "")
                    lbl = t.get("label", "")
                    if RESULTS_PATTERN.match(txt):
                        break
                    if lbl == "O":
                        rows[j]["label"] = "FEATURE_ZONA"
                        count += 1
                    j += 1
                # posiziona i su j per continuare la scansione oltre il blocco
                i = j
            else:
                i += 1
        except Exception:
            # in caso di riga malformata o chiavi mancanti, saltala
            i += 1
            continue

    # 3) Riscrivi in-place
    with open(input_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"→ FEATURE_ZONA applied to {count} tokens")
    return count

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

def process(csv_file: str, output_path: str, words_to_substitute: list[str]):
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

    # 2) rietichettatura agency a finestra di 4 record
    m_seq, s_seq = substitute_label_agency_sequence(path)
    total_matches += m_seq
    total_subs += s_seq

    # 3) rietichettatura zona
    zona_count = substitute_feature_zona_sequence(path)
    total_matches += zona_count
    total_subs += zona_count

    print(f"🎯 Totale righe matching: {total_matches}, totali etichette modificate: {total_subs}\n")

    convert_csv_to_json_with_numbers(path)
    input_path = str(path).replace("csv", "jsonl")
    save_label2id(Path(input_path), Path(output_path))


if __name__ == "__main__":
    # ===== ESEMPIO DI USO =====
    csv_path = "data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled.csv"
    label2id_path = "data/transformer_model/labeling/artifacts/label2id.json"
    parole = ["ascensore", "terrazzo", "cantina", "Solo Cucina Arredata"]  # metti qui la tua lista di parole
    process(csv_path, label2id_path, parole)
