# src/immobiliare/pipeline/steps/file_loading_step.py

from typing import Any, List, Dict
from pathlib import Path
import time

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.scraping.file_fetcher import FileFetcher
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.utils.logging.decorators import log_exec


class FileLoadingStep(IPipelineStep):
    """
    Pipeline Step: carica file HTML da disco usando FileFetcher.
    """

    def __init__(self, input_dir: str, limit: int = None):
        # Costruisce internamente il FileFetcher
        self.fetcher = FileFetcher(Path(input_dir))
        self.limit = limit
        self.logger = LoggerFactory.get_logger("file_loading_step")

    @log_exec(logger_name="file_loading_step", method_name="run")
    def run(self, data: Any = None) -> List[Dict[str, str]]:
        self.logger.log_info("Start loading data ...")
        start_time = time.time()  # ← inizio cronometro

        """
        Ignora l’input e restituisce la lista di pagine HTML:
        [{'filename': ..., 'content': ...}, ...]
        """
        try:
            pages = self.fetcher.fetch(limit=self.limit)
            if not pages:
                self.logger.log_info("Nessuna pagina trovata dal FileFetcher.")
            else:
                self.logger.log_info(f"{len(pages)} pagine caricate dal FileFetcher.")


            total_time = time.time() - start_time
            m, s = divmod(total_time, 60)
            h, m = divmod(m, 60)
            self.logger.log_info(f"\n⏱️  Tempo totale di loading data: {int(h)}h {int(m)}m {int(s)}s")

            return pages
        except Exception as e:
            self.logger.log_exception("Errore in FileLoadingStep.run", e)
            # Propaga l'errore per far terminare la pipeline o gestirlo a monte
            raise
