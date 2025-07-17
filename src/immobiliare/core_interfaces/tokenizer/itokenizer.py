# src/immobiliare/core_interfaces/tokenizer/itokenizer.py

from abc import ABC, abstractmethod
from typing import List, Tuple

from bs4 import BeautifulSoup, Tag


class ITokenizer(ABC):
    @abstractmethod
    def tokenize(self, soup: BeautifulSoup) -> List[Tuple[str, Tag]]:
        """
        Tokenizza il contenuto testuale restituendo coppie (testo, tag).
        """
        pass


