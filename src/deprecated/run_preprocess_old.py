import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import joblib
from bs4 import BeautifulSoup


from immobiliare.config_loader.loader import ConfigLoader
from immobiliare.preprocess.html.feature.feature_extractor import FeatureExtractor
from immobiliare.preprocess.html.feature.feature_normalize import FeatureNormalizer
from immobiliare.preprocess.html.tokenizer.html_tokenizer import HTMLAnnuncioTokenizer
from immobiliare.scraping.file_fetcher import FileFetcher
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils.logging.logger_factory import LoggerFactory

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
HTML_DIR = BASE_DIR / "data" / "html_pages_dir"  #da dove recuperiamo le pagine per il training
JSONL_PATH = BASE_DIR / "data" / "output" / "output_1.jsonl"
CSV_PATH = BASE_DIR / "data" / "output" / "output_1.csv"
NORMALIZER_PATH = BASE_DIR / "data" / "artifacts" / "normalizer_1.pkl"
ANNUNCIO_NAME_TRAIN = "annuncio_for_training_*.html"


def compute_max_dom_depth(tag):
    ## Utility ricorsiva per calcolare la profondità massima del DOM.##
    max_d = 0
    for child in tag.find_all(recursive=False):
        d = 1 + compute_max_dom_depth(child)
        if d > max_d:
            max_d = d
    return max_d

class RunPreProcess:
    def __init__(self):
        config = ConfigLoader.load()
        self.pages_number_for_training = 60
        self.input_dir = Path(HTML_DIR)
        self.output_jsonl_path = Path(JSONL_PATH)
        self.output_csv_path = Path(CSV_PATH)
        self.normalizer_path = Path(NORMALIZER_PATH)
        self.filename_pattern = ANNUNCIO_NAME_TRAIN
        self.logger = LoggerFactory.get_logger("preprocess_pipeline_old")

        # Componenti della pipeline
        self.fetcher = FileFetcher(self.input_dir)
        self.tokenizer = HTMLAnnuncioTokenizer()
        self.extractor = FeatureExtractor()
        self.normalizer = FeatureNormalizer()

    def _save_jsonl(self, data: List[Dict[str, Any]], path: Path):
        if not data:
            self.logger.log_info("⚠️ Nessun dato da salvare in JSONL.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _save_csv(self, data: List[Dict[str, Any]], path: Path):
        if not data:
            self.logger.log_info("⚠️ Nessun dato da salvare in CSV.")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    @log_exec(logger_name="preprocess_pipeline_old", method_name="run")
    def run(self):
        all_tokens_with_features = []

        # 🆕 Modifica: passa il numero di pagine da elaborare
        pages = self.fetcher.fetch(limit=self.pages_number_for_training)

        if not pages:
            self.logger.log_info("❌ Nessun file HTML trovato. Interruzione.")
            return

        for page in pages:
            filename = page["filename"]
            content = page["content"]
            self.logger.log_info(f"📄 Processing: {filename}")
            try:
                soup = BeautifulSoup(content, "html.parser")
                token_data = self.tokenizer.tokenize(soup)
                if not token_data:
                    continue

                total_tokens = len(token_data)
                max_depth = compute_max_dom_depth(soup.body or soup)
                all_texts = [text for text, _ in token_data]

                for i, (token_text, tag) in enumerate(token_data):
                    feats = self.extractor.extract_features(
                        token_text,
                        tag,
                        position=i,
                        total_tokens=total_tokens,
                        max_depth_in_doc=max_depth,
                        all_tokens=all_texts
                    )
                    all_tokens_with_features.append(feats)

            except Exception as e:
                self.logger.log_exception(f"❌ Errore durante il processing di {filename}", e)

        # Normalizzazione
        tokens_normalized = self.normalizer.fit_transform(all_tokens_with_features)

        # Salvataggio normalizer
        self.normalizer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.normalizer, self.normalizer_path)
        self.logger.log_info(f"✅ Normalizer salvato in: {self.normalizer_path}")

        # Output finale
        self._save_jsonl(tokens_normalized, self.output_jsonl_path)
        self._save_csv(tokens_normalized, self.output_csv_path)

        self.logger.log_info(f"✅ Preprocessing completato: {len(tokens_normalized)} token da {len(pages)} pagine")
        self.logger.log_info(f"   • JSONL ➜ {self.output_jsonl_path}")
        self.logger.log_info(f"   • CSV   ➜ {self.output_csv_path}")

if __name__ == "__main__":
    preprocess = RunPreProcess()
    preprocess.run()