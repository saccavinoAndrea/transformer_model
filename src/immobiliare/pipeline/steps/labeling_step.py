# src/immobiliare/pipeline/steps/labeling_step.py

from pathlib import Path
from typing import Any, List, Dict
import time

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.labeling.auto_labeler import AutoLabeler
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils.logging.logger_factory import LoggerFactory


class LabelingStep(IPipelineStep):
    """
    Pipeline Step: etichetta i token basandosi sugli HTML e su un JSONL di input.
    """

    def __init__(
            self,
            html_dir: str,
            html_file_name: str,
            input_jsonl: str,
            output_jsonl: str,
            output_csv: str,
            label_2id_dir: str):

        # converte da str a Path
        self.html_dir = Path(html_dir)
        self.html_file_name = html_file_name
        self.input_jsonl = Path(input_jsonl)
        self.output_jsonl = Path(output_jsonl)
        self.output_csv = Path(output_csv)
        self.label_2id_dir = Path(label_2id_dir)
        self.logger = LoggerFactory.get_logger("labeling_step")

        self.labeler = AutoLabeler(
            html_dir=self.html_dir,
            html_file_name=self.html_file_name,
            input_jsonl=self.input_jsonl,
            output_jsonl=self.output_jsonl,
            output_csv=self.output_csv,
            label_2id_dir=self.label_2id_dir
        )

    @log_exec(logger_name="labeling_step", method_name="run")
    def run(self, data: Any = None) -> List[Dict[str, Any]]:

        self.logger.log_info("Start labeling data ...")
        start_time = time.time()  # ← inizio cronometro

        """
        Ignora l’input, esegue AutoLabeler.run() e restituisce le righe etichettate.
        """
        try:
            labeled_rows = self.labeler.run()
            self.logger.log_info(f"Labeling completato: {len(labeled_rows)} record")

            total_time = time.time() - start_time
            m, s = divmod(total_time, 60)
            h, m = divmod(m, 60)
            self.logger.log_info(f"\n⏱️  Tempo totale di labeling data: {int(h)}h {int(m)}m {int(s)}s")

            return labeled_rows
        except Exception as e:
            self.logger.log_exception("Errore in LabelingStep", e)
            raise
