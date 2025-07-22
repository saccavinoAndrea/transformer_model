import re
import csv
from collections import Counter
from pathlib import Path

from immobiliare.utils import resolve_versioned_jsonl

# Marker e pattern per le sostituzioni di sequenza
FIRST_MARKER = "Annunci in zona"
FIRST_MARKER_ADD = "Agenzie di zona"
SECOND_MARKERS = {"Gold", "Silver"}
AGENCY_KEYWORD = "Immobiliare"
RESULTS_PATTERN = re.compile(r'\d+[\.,]?\d*\s*risultati per:')

def substitute_label_agency_sequence(input_path: Path) -> (int, int):
    """
    Rietichetta il 3° elemento di ogni sequenza 4-a-4 come FEATURE_AGENZIA_IMMOBILIARE.
    """
    matches = subs = 0
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    for i in range(len(rows) - 3):
        a, b, c, d = rows[i], rows[i+1], rows[i+2], rows[i+3]
        if (a["true_label"] == "O" and a["text"] in {FIRST_MARKER, FIRST_MARKER_ADD}
            and b["true_label"] == "O" and b["text"] in SECOND_MARKERS
            and c["true_label"] == "O"
            and d["true_label"] == "O" and AGENCY_KEYWORD in d["text"]):
            matches += 1
            if c["true_label"] != "FEATURE_AGENZIA_IMMOBILIARE":
                c["true_label"] = "FEATURE_AGENZIA_IMMOBILIARE"
                subs += 1

    with open(input_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"→ Sequence agency: {matches} matches, {subs} substitutions")
    return matches, subs

def substitute_feature_zona_sequence(input_path: Path) -> int:
    """
    Etichetta FEATURE_ZONA per tutti i token dopo "Scegli la zona" fino a un risultato.
    """
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    count = 0
    i = 0
    while i < len(rows):
        if rows[i]["true_label"] == "O" and rows[i]["text"] == "Scegli la zona":
            j = i + 1
            while j < len(rows) and not RESULTS_PATTERN.match(rows[j]["text"]):
                if rows[j]["true_label"] == "O":
                    rows[j]["true_label"] = "FEATURE_ZONA"
                    count += 1
                j += 1
            i = j
        else:
            i += 1

    with open(input_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"→ FEATURE_ZONA applied to {count} tokens")
    return count

def process_and_label_csv(input_path: Path):
    """
    1) Aggiunge header 'true_label' (inizializzato a 'O').
    2) Applica substitute_label_agency_sequence.
    3) Applica substitute_feature_zona_sequence.
    4) Blocchi immobiliari: €…, titolo, locali, superficie, bagni, lusso, ascensore, balcone, cantina.
    """
    # leggi e aggiungi true_label
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "true_label" not in fieldnames:
            fieldnames.append("true_label")
        rows = []
        for row in reader:
            row["true_label"] = "O"
            rows.append(row)

    # riscrivi con header aggiornato
    with open(input_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # applica sequenze
    substitute_label_agency_sequence(input_path)
    substitute_feature_zona_sequence(input_path)

    # leggi di nuovo per blocchi immobiliari
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # pattern e relative etichette in ordine
    patterns = [
        (r'€\s*\d', "FEATURE_PREZZO"),
        (r'"', "FEATURE_TITOLO"),
        (r'\d+\+?\s+locali', "FEATURE_LOCALI"),
        (r'\d+\s*m²', "FEATURE_SUPERFICIE"),
        (r'\d+\s+bagni?', "FEATURE_BAGNI"),
        (r'^Lusso$', "FEATURE_LUSSO"),
        (r'^(Ascensore|No Ascensore)$', "FEATURE_ASCENSORE"),
        (r'^Balcone$', "FEATURE_BALCONE"),
        (r'^Cantina$', "FEATURE_CANTINA"),
    ]

    i = 0
    while i < len(rows):
        # se inizia blocco di prezzo
        if re.search(patterns[0][0], rows[i]["text"]):
            for pat, label in patterns:
                if i < len(rows) and re.search(pat, rows[i]["text"]):
                    rows[i]["true_label"] = label
                i += 1
            continue
        i += 1

    # riscrivi il file finale
    with open(input_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("→ Blocchi FEATURE_* etichettati.")

def refine_announcement_labels(input_path: Path) -> int:
    """
    Rileggere il CSV etichettato, trova i blocchi di annuncio che iniziano con FEATURE_PREZZO
    e sostituisce:
      - il primo O dopo FEATURE_PREZZO con FEATURE_TITOLO
      - eventuali O su "Piano…" con FEATURE_PIANO
      - eventuali O su "Locali" o "<numero> locali" con FEATURE_LOCALI
      - eventuali O su "<numero> m²" con FEATURE_SUPERFICIE
      - eventuali O su "Terrazzo" con FEATURE_TERRAZZO
      - eventuali O su "X bagni" o "X+ bagni" con FEATURE_BAGNI
      - eventuali O su "Lusso" con FEATURE_LUSSO
      - eventuali O su "Ascensore"/"No Ascensore" con FEATURE_ASCENSORE
      - eventuali O su "Balcone" con FEATURE_BALCONE
      - eventuali O su "Cantina" con FEATURE_CANTINA
      - eventuali O contenenti "Arredato" con FEATURE_ARREDATO

    Generalizza su tutte le sequenze di annuncio fino al prossimo FEATURE_PREZZO.
    Restituisce il numero totale di sostituzioni effettuate.
    """
    PRICE_LABEL = "FEATURE_PREZZO"

    # feature patterns per rietichettare O
    feature_patterns = [
        (r'^[Pp]iano\s+terra$',          "FEATURE_PIANO"),
        (r'^[Uu]ltimo\s+[Pp]iano$',      "FEATURE_PIANO"),
        (r'^[Pp]iano\s+\S+',             "FEATURE_PIANO"),
        (r'^(?:[Ll]ocali|\d+\+?\s*locali)$', "FEATURE_LOCALI"),
        (r'^\d+\s*m²$', "FEATURE_SUPERFICIE"),
        (r'^Terrazzo$',                  "FEATURE_TERRAZZO"),
        (r'\d+\+?\s*bagni?',              "FEATURE_BAGNI"),
        (r'^Lusso$',                      "FEATURE_LUSSO"),
        (r'^(Ascensore|No Ascensore)$',   "FEATURE_ASCENSORE"),
        (r'^Balcone$',                    "FEATURE_BALCONE"),
        (r'^Cantina$',                    "FEATURE_CANTINA"),
        (r'[Aa]rredato',                  "FEATURE_ARREDATO"),
        (r'Solo Cucina Arredata', "FEATURE_ARREDATO"),
    ]

    # 1) Leggi tutto
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    subs = 0
    i = 0
    n = len(rows)

    while i < n:
        if rows[i]["true_label"] == PRICE_LABEL:
            # 1) primo O diventa titolo
            j = i + 1
            while j < n and rows[j]["true_label"] != "O":
                j += 1
            if j < n:
                rows[j]["true_label"] = "FEATURE_TITOLO"
                subs += 1

            # 2) scansiona blocco fino a prossimo PRICE_LABEL
            k = j + 1
            while k < n and rows[k]["true_label"] != PRICE_LABEL:
                if rows[k]["true_label"] == "O":
                    txt = rows[k]["text"]
                    # prova ciascun pattern in ordine
                    for pat, label in feature_patterns:
                        if re.search(pat, txt):
                            rows[k]["true_label"] = label
                            subs += 1
                            break
                k += 1
            # salta oltre il blocco
            i = k
        else:
            i += 1

    # 3) Riscrivi in-place
    with open(input_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"→ Refine announcements: {subs} labels updated")
    return subs

def print_label_stats(input_path: Path):
    """
    Stampa le etichette distinte assegnate e il conteggio di esempi per label.
    """
    with open(input_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        labels = [row['true_label'] for row in reader]

    counter = Counter(labels)
    distinct = list(counter.keys())

    # Print distinct labels
    print("Etichette distinte trovate:")
    for lbl in distinct:
        print(f"  - {lbl}")
    print(f"\nTotale etichette distinte: {len(distinct)}\n")

    # Print counts per label
    print("Conteggio esempi per label:")
    for lbl, cnt in counter.most_common():
        print(f"  {lbl}: {cnt}")

if __name__ == "__main__":
    # ===== ESEMPIO DI USO =====
    csv_path_resolved = resolve_versioned_jsonl("data/transformer_model/inference/ground_truth/ground_truth_generate.csv")
    process_and_label_csv(csv_path_resolved)
    refine_announcement_labels(csv_path_resolved)
    print_label_stats(csv_path_resolved)