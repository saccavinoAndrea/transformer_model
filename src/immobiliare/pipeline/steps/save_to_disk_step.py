# src/immobiliare/pipeline/steps/save_to_disk_step.py

import json
import csv
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, List, Dict

from dataset.html_token_dataset import HTMLTokenDataset
from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.utils.logging.decorators import log_exec
from utils import timestamped_path, resolve_versioned_jsonl


class SaveToDiskStep(IPipelineStep):
    """
    Pipeline Step: salva i dati normalizzati su JSONL e CSV su disco.
    """

    def __init__(self,
                 jsonl_path: str,
                 csv_path: str,
                 features_keys_dir: str = None):

        self.logger = LoggerFactory.get_logger("save_to_disk_step")
        self.jsonl_path = jsonl_path
        self.csv_path = csv_path
        self.features_keys_dir = features_keys_dir



    def _serialize_record(self, record: Any) -> Dict[str, Any]:
        """
        Converte un oggetto dataclass (es. Token) in dizionario.
        Rimuove campi non serializzabili come 'tag'.
        """
        if is_dataclass(record):
            d = asdict(record)
            d.pop("tag", None)  # rimuovi BeautifulSoup Tag
            return d
        elif isinstance(record, dict):
            return {k: v for k, v in record.items() if k != "tag"}
        else:
            raise ValueError("Record non serializzabile")


    def _save_jsonl(self, data: List[Any], path: Path):
        final_path = timestamped_path(path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, "w", encoding="utf-8") as f:
            for record in data:
                serializable = self._serialize_record(record)
                f.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        self.logger.log_info(f"✔️ JSONL salvato in: {final_path}")


    def _save_csv(self, data: List[Any], path: Path):
        final_path = timestamped_path(path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, "w", encoding="utf-8", newline="") as f:
            serializables = [self._serialize_record(r) for r in data]
            writer = csv.DictWriter(f, fieldnames=serializables[0].keys())
            writer.writeheader()
            writer.writerows(serializables)
        self.logger.log_info(f"✔️ CSV salvato in: {final_path}")

    def _save_feature_keys_json(self):
        if self.features_keys_dir:
            jsonl_resolved_path = resolve_versioned_jsonl(self.jsonl_path)
            ds = HTMLTokenDataset()
            ds.calculate_feature_keys(str(jsonl_resolved_path))
            feature_keys_final_path = timestamped_path(Path(self.features_keys_dir) / "feature_keys.json")
            with open(feature_keys_final_path, "w", encoding="utf-8") as f:
                json.dump(ds.feature_keys_to_use, f, ensure_ascii=False, indent=2)

            self.logger.log_info(f"✔️ feature keys salvate in: {feature_keys_final_path}")


    @log_exec(logger_name="save_to_disk", method_name="run")
    def run(self, data: List[Any]) -> List[Any]:

        self.logger.log_info("Start save to disk step ...")
        start_time = time.time()  # ← inizio cronometro

        if not data:
            self.logger.log_info("Nessun dato da salvare.")
            return []

        self._save_jsonl(data, Path(self.jsonl_path))
        self._save_csv(data, Path(self.csv_path))
        self._save_feature_keys_json()

        total_time = time.time() - start_time
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        self.logger.log_info(f"\n⏱️  Tempo totale di save to disk step: {int(h)}h {int(m)}m {int(s)}s")

        return data  # Pass-through
