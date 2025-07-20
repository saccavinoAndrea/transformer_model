import csv
import json
from pathlib import Path

def cast_value(val: str):
    val = val.strip()
    if val == "":
        return None
    # Prova a castare a int
    try:
        iv = int(val)
        return iv
    except ValueError:
        pass
    # Prova a castare a float
    try:
        fv = float(val)
        return fv
    except ValueError:
        pass
    # Resta stringa
    return val

def convert_csv_to_json_with_numbers(csv_file_path: str):
    csv_path = Path(csv_file_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"File non trovato: {csv_path}")

    json_path = csv_path.with_suffix(".jsonl")
    count = 0

    with open(csv_path, mode="r", encoding="utf-8") as csv_file, \
         open(json_path, mode="w", encoding="utf-8") as json_file:

        reader = csv.DictReader(csv_file)
        for row in reader:
            # Applica cast su ogni valore
            record = {k: cast_value(v) for k, v in row.items()}
            json.dump(record, json_file, ensure_ascii=False)
            json_file.write("\n")
            count += 1

    print(f"✅ File convertito in JSON salvato in: {json_path} ({count} record).")