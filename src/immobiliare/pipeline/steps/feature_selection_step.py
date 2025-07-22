# preprocess/feature_selector.py
import time
from pathlib import Path
from typing import Any

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.labeling.feature_selectors import RandomForestSelector
from immobiliare.utils import log_exec, resolve_versioned_jsonl
from immobiliare.utils.logging.logger_factory import LoggerFactory


class FeatureSelectionStep(IPipelineStep):
    def __init__(self,
                 feature_labelled_dir: str,
                 feature_selected_dir: str,
                 label_column_name: str,
                 ):

        self.logger = LoggerFactory.get_logger("feature_selection_step")
        self.feature_labelled_dir = feature_labelled_dir
        self.feature_selected_dir = feature_selected_dir
        self.label_column_name = label_column_name


    @log_exec(logger_name="feature_selection_step", method_name="run")
    def run(self, data: Any) -> Any:

        self.logger.log_info("Start feature selection step ...")
        start_time = time.time()  # ← inizio cronometro

        try:
            random_forest_selector = RandomForestSelector(
                Path(self.feature_labelled_dir),
                Path(self.feature_selected_dir),
                self.label_column_name,
            )

            random_forest_selector.execute_selection()

            total_time = time.time() - start_time
            m, s = divmod(total_time, 60)
            h, m = divmod(m, 60)
            self.logger.log_info(f"\n⏱️  Tempo totale di feature selection: {int(h)}h {int(m)}m {int(s)}s")

        except Exception as e:
            self.logger.log_exception("Errore in FeatureSelectionStep", e)
            raise


