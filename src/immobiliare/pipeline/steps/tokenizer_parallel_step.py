# src/immobiliare/pipeline/steps/tokenizer_parallel_step.py
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.domain.token import Token
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.pipeline.steps.tokenizer_runner import _process_single_page
from immobiliare.utils.token_utils import build_token_from_dict


class TokenizerParallelStep(IPipelineStep):
    """
    Step combinato: tokenizza HTML + estrae feature numeriche + crea Token.
    """
    def __init__(self):
        self.logger = LoggerFactory.get_logger("tokenizer_parallel_step")

    @log_exec(logger_name="tokenizer_parallel_step", method_name="run")
    def run(self, data: List[Dict[str, str]]) -> List[Token]:
        self.logger.log_info("Start parallel tokenizer data ...")
        start_time = time.time()

        all_tokens: List[Token] = []

        with ProcessPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(_process_single_page, page) for page in data]

            for future in futures:
                try:
                    token_dicts = future.result()
                    for d in token_dicts:
                        token = build_token_from_dict(d)
                        all_tokens.append(token)
                except Exception as e:
                    self.logger.log_exception("Errore durante l'elaborazione di una pagina", e)

        self.logger.log_info(f"Totale Token estratti e processati in parallelo: {len(all_tokens)}")
        total_time = time.time() - start_time
        h, m = divmod(total_time, 3600)
        m, s = divmod(m, 60)
        self.logger.log_info(f"\n⏱️  Tempo totale di parallel tokenizer data: {int(h)}h {int(m)}m {int(s)}s")

        return all_tokens



