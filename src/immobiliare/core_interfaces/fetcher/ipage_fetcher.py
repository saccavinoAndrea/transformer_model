# src/immobiliare/core_interfaces/fetcher/ipage_fetcher.py

from abc import ABC, abstractmethod
from pathlib import Path

class IPageFetcher(ABC):
    """
    Contratto per fetcher che scaricano pagine HTML da remoto.
    """

    @abstractmethod
    async def fetch(self, n_pages: int, out_dir: Path) -> None:
        """
        Scarica fino a `n_pages` pagine HTML in `out_dir`.
        """
        pass
