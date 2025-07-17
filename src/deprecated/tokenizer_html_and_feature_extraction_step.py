# src/immobiliare/pipeline/steps/tokenizer_html_and_feature_extraction_step.py
import time
from typing import List, Dict, Any
from bs4 import BeautifulSoup

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.domain.token import Token
from immobiliare.preprocess.html.feature.feature_extractor import FeatureExtractor
from immobiliare.preprocess.html.tokenizer.html_tokenizer import HTMLAnnuncioTokenizer
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.utils.logging.decorators import log_exec


class TokenizerHtmlAndFeatureExtractionStep(IPipelineStep):
    """
    Step combinato: tokenizza HTML + estrae feature numeriche + crea Token.
    """
    def __init__(self):
        self.tokenizer = HTMLAnnuncioTokenizer()
        self.extractor = FeatureExtractor()
        self.logger = LoggerFactory.get_logger("tokenizer_feature_step")

    @log_exec(logger_name="tokenizer_feature_step", method_name="run")
    def run(self, data: List[Dict[str, str]]) -> List[Token]:

        self.logger.log_info("Start tokenizer data ...")
        start_time = time.time()  # ← inizio cronometro

        """
        :param data: lista di dict con chiavi 'filename' e 'content'
        :return: lista di Token (con tutte le feature calcolate)
        """
        all_tokens: List[Token] = []

        for page in data:
            filename = page.get("filename", "<unknown>")
            content = page.get("content", "")
            try:
                soup = BeautifulSoup(content, "html.parser")
                token_tuples = self.tokenizer.tokenize(soup)

                if not token_tuples:
                    continue

                # --- qui richiami process_all ---
                # process_all si occupa di:
                # 1) estrarre per ciascun token tutte le feature (incluse window-stats)
                # 2) ritornare una lista di Token già istanziati
                page_tokens: List[Token] = self.extractor.process_all(
                    token_tuples=token_tuples
                )
                # aggiungi filename nella meta di ciascun token
                for tok in page_tokens:
                    tok.meta["filename"] = filename
                all_tokens.extend(page_tokens)

            except Exception as e:
                self.logger.log_exception(f"Errore nella tokenizzazione/estrazione di {filename}", e)

        self.logger.log_info(f"Totale Token estratti e processati: {len(all_tokens)}")

        total_time = time.time() - start_time
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        self.logger.log_info(f"\n⏱️  Tempo totale di tokenizer data: {int(h)}h {int(m)}m {int(s)}s")

        return all_tokens
