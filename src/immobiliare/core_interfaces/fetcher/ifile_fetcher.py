# src/immobiliare/core_interfaces/fetcher/ifile_fetcher.py

from abc import ABC, abstractmethod
from typing import List, Dict

class IFileFetcher(ABC):
    """
    Contratto per fetcher che leggono file HTML da disco.
    """

    @abstractmethod
    def fetch(self) -> List[Dict[str, str]]:
        """
        Restituisce una lista di dizionari con chiavi:
        - 'filename'
        - 'content'
        """
        pass
