# src/immobiliare/labeling/auto_labeler.py

import csv, json, re
from pathlib import Path
from typing import List, Dict, Any

from bs4 import BeautifulSoup

from dataset.html_token_dataset import HTMLTokenDataset
from immobiliare.core_interfaces.labeler.ilabeler import ILabeler
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.utils.logging.decorators import log_exec
from utils import resolve_versioned_jsonl, timestamped_path


class AutoLabeler(ILabeler):
    def __init__(
        self,
        html_dir: Path,
        html_file_name: str,
        input_jsonl: Path,
        output_jsonl: Path,
        output_csv: Path,
        label_2id_dir: Path
    ):
        self.logger = LoggerFactory.get_logger("auto_labeler")
        self.html_dir = html_dir
        self.html_file_name = html_file_name
        self.input_jsonl = input_jsonl
        self.output_jsonl = output_jsonl
        self.output_csv = output_csv
        self.label_2id_dir = label_2id_dir


    @staticmethod
    def extract_labels_from_html(html_text: str) -> List[Dict[str, str]]:
        # (stessa logica tua, senza modifiche)
        soup = BeautifulSoup(html_text, "html.parser")
        annunci = soup.find_all("div", class_="nd-mediaObject__content")
        labeled_tokens = []
        for annuncio in annunci:
            # --- Prezzo ---
            prezzo = annuncio.select_one(".styles_in-listingCardPrice__earBq span")
            if prezzo:
                text = prezzo.get_text(strip=True)
                labeled_tokens.append({"token": text, "label": "FEATURE_PREZZO"})

            # --- Titolo ---
            titolo = annuncio.select_one("a.styles_in-listingCardTitle__Wy437")
            if titolo:
                text = titolo.get_text(strip=True)
                labeled_tokens.append({"token": text, "label": "FEATURE_TITOLO"})

            # --- Feature list ---
            feature_items = annuncio.select(".styles_in-listingCardFeatureList__item__CKRyT span")
            for item in feature_items:
                text = re.sub(r"(?<![€$])\s+(?![€$])", " ", item.get_text(strip=True))
                label = "O"
                tl = text.lower()
                if re.search(r"\d+\s*(m²|mq)", tl):
                    label = "FEATURE_SUPERFICIE"
                elif re.search(r"\d+\s*\+?\s*local.?", tl):
                    label = "FEATURE_LOCALI"
                elif re.search(r"(\d+|\b(un|uno|una|due|tre|quattro|cinque|sei|sette|otto|nove|dieci)\b)\s*\+?\s*bagn.?\b.*", tl, re.IGNORECASE):
                    label = "FEATURE_BAGNI"
                elif re.search(r"piano\s*\w+", tl):
                    label = "FEATURE_PIANO"
                elif "ascensore" in tl:
                    label = "FEATURE_ASCENSORE"
                elif "balcone" in tl:
                    label = "FEATURE_BALCONE"
                elif "terrazzo" in tl:
                    label = "FEATURE_TERRAZZO"
                elif "cantina" in tl:
                    label = "FEATURE_CANTINA"
                elif "arredato" in tl:
                    label = "FEATURE_ARREDATO"
                elif "lusso" in tl:
                    label = "FEATURE_LUSSO"
                elif re.search(r"(€|eur|euro)", tl):
                    label = "FEATURE_PREZZO"
                elif re.search(r"\d{2}/\d{2}/\d{4}", text):
                    label = "FEATURE_DATA"
                elif "asta" in tl:
                    label = "FEATURE_TIPOLOGIA"

                labeled_tokens.append({"token": text, "label": label})

        return labeled_tokens

    @log_exec(logger_name="auto_labeler", method_name="run")
    def run(self) -> List[Dict[str, Any]]:
        # 1) costruisci label_map
        label_map: Dict[str, str] = {}
        html_files = sorted(self.html_dir.glob(self.html_file_name))
        for p in html_files:
            txt = p.read_text(encoding="utf-8")
            for item in self.extract_labels_from_html(txt):
                key = item["token"].strip().lower()
                label_map[key] = item["label"]

        # ------------------------------------------------------------
        # 2) Leggi token_embeddings_dense_<timestamp>.jsonl e fai join
        # ------------------------------------------------------------
        rows: List[Dict[str, Any]] = []

        jsonl_path = resolve_versioned_jsonl(self.input_jsonl)

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                # 🆕 fallback tra 'token' e 'text'
                token_key = row.get("token") or row.get("text", "")
                key = token_key.strip().lower()

                row["label"] = label_map.get(key, "O")
                rows.append(row)

        # `rows` ora contiene l’output con label assegnate

        # 3) salva JSONL
        final_path = timestamped_path(self.output_jsonl)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.logger.log_info(f"JSONL etichettato salvato in {final_path}")

        # 4) salva CSV
        final_path_csv = timestamped_path(self.output_csv)
        final_path_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        self.logger.log_info(f"CSV etichette salvato in {final_path_csv}")

        # 5) salva la mappa tg label2id
        ds = HTMLTokenDataset()
        ds.calculate_label2id(rows)
        label2id_final_path = timestamped_path(self.label_2id_dir)
        with open(label2id_final_path, "w", encoding="utf-8") as f:
            json.dump(ds.label2id, f, ensure_ascii=False, indent=2)

        self.logger.log_info("✅ Artifacts label2Id salvato in " +  str(label2id_final_path))

        return rows
