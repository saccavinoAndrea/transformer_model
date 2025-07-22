# src/immobiliare/analysis/analyzer.py

import json
import pandas as pd

from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Any

from immobiliare.core_interfaces.analisys.ianalyzer import IAnalyzer
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils import resolve_versioned_jsonl, timestamped_path


class Analyzer(IAnalyzer):
    def __init__(
        self,
        input_jsonl: Path,
        filtered_output: Path,
        distribution_output: Path
    ):
        self.input_jsonl = input_jsonl
        self.filtered_output = filtered_output
        self.distribution_output = distribution_output
        self.logger = LoggerFactory.get_logger("analyzer")

    @log_exec(logger_name="analyzer", method_name="run")
    def run(self) -> Tuple[Counter, List[Dict[str, Any]]]:
        counter = Counter()
        filtered: List[Dict[str, Any]] = []

        jsonl_path = resolve_versioned_jsonl(self.input_jsonl)

        # 1) Leggi e conta
        if not jsonl_path.exists():
            msg = f"File non trovato: {jsonl_path}"
            self.logger.log_exception(msg, FileNotFoundError(msg))
            raise FileNotFoundError(msg)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)

                    text = obj.get("text", "")
                    label = obj.get("label", "O")

                    # se troviamo un nuovo "html", resettiamo il contatore a 1
                    if text == "html":
                        pos_counter = 1

                    # aggiorniamo il counter delle label
                    counter[label] += 1

                    label = obj.get("label", "O")
                    filtered.append({
                        "position": pos_counter,
                        "relative_position": obj.get("relative_position", ""),
                        "label": label,
                        "text": obj.get("text", ""),
                    })

                    # incrementiamo pos_counter per la prossima riga
                    pos_counter += 1

                except Exception as e:
                    self.logger.log_exception("Errore parsing JSONL", e)

        # 2) Salva distribuzione
        final_path = timestamped_path(self.distribution_output)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path, "w", encoding="utf-8") as f:
            for lbl, cnt in counter.most_common():
                f.write(f"{lbl}: {cnt}\n")
        self.logger.log_info(f"Distribuzione salvata in {final_path}")

        # 3) Salva JSONL filtrato
        final_path_filtered = timestamped_path(self.filtered_output)
        final_path_filtered.parent.mkdir(parents=True, exist_ok=True)
        with open(final_path_filtered, "w", encoding="utf-8") as f:
            for rec in filtered:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.logger.log_info(f"Token filtrati salvati in {final_path_filtered}")

        #df = pd.read_json(final_path_filtered, lines=True)
        #df.to_csv("data/analysis/tokens_filtered.csv", index=False)

        return counter, filtered
