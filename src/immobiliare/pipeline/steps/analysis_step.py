# src/immobiliare/pipeline/steps/analysis_step.py

from typing import Any, Tuple, List, Dict
from pathlib import Path

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.analisys.analyzer import Analyzer
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils.logging.logger_factory import LoggerFactory

class AnalysisStep(IPipelineStep):
    """
    Step di pipeline che esegue l’analisi e restituisce Counter + filtered list.
    """
    def __init__(
        self,
        input_jsonl: str,
        filtered_output: str,
        distribution_output: str
    ):
        self.input_jsonl = Path(input_jsonl)
        self.filtered_output = Path(filtered_output)
        self.distribution_output = Path(distribution_output)
        self.logger = LoggerFactory.get_logger("analysis_step")
        self.analyzer = Analyzer(
            input_jsonl=self.input_jsonl,
            filtered_output=self.filtered_output,
            distribution_output=self.distribution_output
        )

    @log_exec(logger_name="analysis_step", method_name="run")
    def run(self, data: Any = None) -> Tuple[Any, Any]:
        try:
            counter, filtered = self.analyzer.run()
            self.logger.log_info(f"Analisi completata: {len(filtered)} token, {len(counter)} categorie")
            return counter, filtered
        except Exception as e:
            self.logger.log_exception("Errore in AnalysisStep", e)
            raise
