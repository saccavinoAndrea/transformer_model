import csv
from pathlib import Path

def substitute_label_by_word(csv_path: str):
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {csv_path}")

    with open(path, newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))

    if not reader:
        raise ValueError("Il file CSV è vuoto")

    has_header = not reader[0][0].replace('.', '', 1).lstrip('-').isdigit()
    header = reader[0] if has_header else None
    rows = reader[1:] if has_header else reader

    num_colonne = len(rows[0])
    num_righe_originali = len(rows)
    sostituzioni = 0

    for row in rows:
        if len(row) != num_colonne:
            raise ValueError("Riga con numero di colonne incoerente")
        if row[0].strip() == "terrazzo" and row[-1] != "O":
            row[-1] = "O"
            sostituzioni += 1

    assert len(rows) == num_righe_originali, "Numero di righe non corrispondente!"

    out_path = path.with_name(path.stem + "_modificato.csv")

    with open(out_path, mode="w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)

    print(f"✅ CSV modificato salvato in: {out_path}")
    print(f"🔁 Sostituzioni effettuate: {sostituzioni}")
