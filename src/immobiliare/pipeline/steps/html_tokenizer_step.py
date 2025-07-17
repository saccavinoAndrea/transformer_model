# src/immobiliare/pipeline/steps/html_tokenizer_step.py

from typing import List, Tuple

from bs4 import Tag, BeautifulSoup

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.preprocess.html.tokenizer.html_tokenizer import HTMLAnnuncioTokenizer
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils.logging.logger_factory import LoggerFactory


class HTMLTokenizerStep(IPipelineStep):
    """
    Pipeline Step: trasforma il contenuto HTML grezzo in una lista di token grezzi (text, tag).
    """

    def __init__(self):
        self.tokenizer = HTMLAnnuncioTokenizer()
        self.logger = LoggerFactory.get_logger("html_tokenizer_step")

    @log_exec(logger_name="html_tokenizer_step", method_name="run")
    def run(self, data: List[dict]) -> List[Tuple[str, Tag]]:
        """
        data: lista di dict [{'filename': ..., 'content': ...}, ...]
        restituisce: lista concatenata di token (text, tag) con metadati filename in meta
        """
        all_tokens: List[Tuple[str, Tag]] = []
        for page in data:
            filename = page.get("filename", "<unknown>")
            content = page.get("content", "")
            try:
                soup = BeautifulSoup(content, "html.parser")
                tokens = self.tokenizer.tokenize(soup)
                # opzionale: potresti aggiungere filename nei tuple o
                # gestire filename in un successivo step
                all_tokens.extend(tokens)
                # self.logger.log_info(f"{len(tokens)} token da {filename}")
            except Exception as e:
                self.logger.log_exception(f"Errore in HTMLTokenizerStep su {filename}", e)
        self.logger.log_info(f"Totale token grezzi generati: {len(all_tokens)}")
        return all_tokens
