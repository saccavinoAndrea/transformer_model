# src/immobiliare/pipeline/steps/normalization_step.py
import json

import joblib
from pathlib import Path
import time
from typing import Any, List, Union, Dict

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.preprocess.html.feature.feature_normalize import FeatureNormalizer
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils import timestamped_path


class NormalizationStep(IPipelineStep):
    """
    Pipeline Step: normalizza le feature di Token o dict usando FeatureNormalizer
    e salva l’istanza fit su disco (normalizer.pkl).
    """

    def __init__(self, normalizer_path: str, normalizer_json_path: str):
        self.normalizer = FeatureNormalizer()
        self.normalizer_path = normalizer_path
        self.normalizer_json_path = normalizer_json_path
        self.logger = LoggerFactory.get_logger("normalization_step")

    @log_exec(logger_name="normalization_step", method_name="run")
    def run(self, data: List[Union[Dict, Any]]) -> List[Union[Dict, Any]]:

        self.logger.log_info("Start normalizer data ...")
        start_time = time.time()  # ← inizio cronometro

        """
        :param data: lista di dict o di dataclass Token con campi numerici
        :return: lista di dict o Token con feature normalizzate
        """
        try:
            if not data:
                self.logger.log_info("Nessun dato da normalizzare.")
                return []

            # Fit + transform
            normalized = self.normalizer.fit_transform(data)

            # Estrai i parametri della normalizzazione
            params = {
                "mean": self.normalizer.means,
                "scale": self.normalizer.stds,
            }

            # Scrivi in JSON (leggibile e indipendente da joblib)
            final_normalizer_json_path = timestamped_path(Path(self.normalizer_json_path))
            final_normalizer_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(final_normalizer_json_path, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)

            self.logger.log_info(f"✅ Normalizer params saved to: {final_normalizer_json_path}")

            final_path = timestamped_path(Path(self.normalizer_path))
            # Salvataggio normalizer per usi futuri
            final_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.normalizer, final_path)
            self.logger.log_info(f"✅ Normalizer salvato in: {final_path}")

            total_time = time.time() - start_time
            m, s = divmod(total_time, 60)
            h, m = divmod(m, 60)
            self.logger.log_info(f"\n⏱️  Tempo totale di normalizer data: {int(h)}h {int(m)}m {int(s)}s")

            return normalized

        except Exception as e:
            self.logger.log_exception("Errore in NormalizationStep.run", e)
            raise
