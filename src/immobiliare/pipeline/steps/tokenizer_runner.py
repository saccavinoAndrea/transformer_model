from typing import Dict, List, Any

from bs4 import BeautifulSoup

from immobiliare.preprocess.html.feature.feature_extractor import FeatureExtractor
from immobiliare.preprocess.html.tokenizer.html_tokenizer import HTMLAnnuncioTokenizer
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.utils.token_utils import serialize_tokens


def _process_single_page(page: Dict[str, str]) -> List[Dict[str, Any]]:
    logger = LoggerFactory.get_logger("tokenizer_parallel_step")
    filename = page.get("filename", "<unknown>")
    content = page.get("content", "")

    try:
        # Inizializza tokenizer ed extractor all’interno del processo
        tokenizer = HTMLAnnuncioTokenizer()
        extractor = FeatureExtractor()

        soup = BeautifulSoup(content, "html.parser")
        token_tuples = tokenizer.tokenize(soup)
        if not token_tuples:
            return []

        page_tokens = extractor.process_all(token_tuples=token_tuples)

        for tok in page_tokens:
            tok.meta["filename"] = filename

        return serialize_tokens(page_tokens, filename)

    except Exception as e:
        # Puoi loggare su file, o restituire errore
        print(f"[ERROR _process_single_page] {e}")
        logger.log_exception(f"[ERROR _process_single_page]", e)
        return []
