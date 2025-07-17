# src/immobiliare/preprocess/tokenizer/html_annuncio_tokenizer.py

from typing import List, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

from immobiliare.core_interfaces.tokenizer.itokenizer import ITokenizer
from immobiliare.utils.logging.logger_factory import LoggerFactory


class HTMLAnnuncioTokenizer(ITokenizer):
    """
    Tokenizer che estrae coppie (testo, tag) dai nodi testuali di un BeautifulSoup.
    Offre anche un metodo avanzato che restituisce Token dataclass.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger("tokenizer_html_annuncio")

    def tokenize(self, soup: BeautifulSoup) -> List[Tuple[str, Tag]]:
        """
        Estrae coppie (testo, tag) per tutti i nodi testuali dell'HTML.
        Restituisce solo token testuali non vuoti e lunghi <= 500 caratteri,
        insieme al tag genitore.
        """
        # self.logger.log_info("Inizio tokenizzazione HTMLAnnuncioTokenizer")
        tokens: List[Tuple[str, Tag]] = []

        try:
            for descendant in soup.descendants:
                if isinstance(descendant, NavigableString):
                    text = descendant.strip()
                    if text and len(text) <= 500:
                        parent = descendant.parent
                        if isinstance(parent, Tag):
                            tokens.append((text, parent))
            # self.logger.log_info(f"{len(tokens)} token estratti da HTMLAnnuncioTokenizer")
            return tokens

        except Exception as e:
            self.logger.log_exception("Errore in HTMLAnnuncioTokenizer.tokenize", e)
            raise
