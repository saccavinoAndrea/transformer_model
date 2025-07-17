# src/immobiliare/scraping/file_fetcher.py

from pathlib import Path
from typing import List, Dict

from immobiliare.core_interfaces.fetcher.abstract_ifetcher import IFetcher
from immobiliare.core_interfaces.fetcher.ifile_fetcher import IFileFetcher
from immobiliare.utils.logging.logger_factory import LoggerFactory


class FileFetcher(IFetcher, IFileFetcher):
    """
    Fetcher che carica file HTML da disco locale (es. per preprocessing).
    """

    def __init__(self, input_dir: Path = None):
        self.input_dir = input_dir
        self.logger = LoggerFactory.get_logger("file_fetcher")

    def fetch(self, limit: int = None) -> List[Dict[str, str]]:
        """
        Legge tutti i file .html dalla directory e restituisce
        una lista di dict: {'filename': ..., 'content': ...}
        """
        self.logger.log_info(f"Lettura file HTML da: {self.input_dir}")

        if not self.input_dir.exists() or not self.input_dir.is_dir():
            msg = f"La directory {self.input_dir} non esiste o non è valida."
            self.logger.log_exception(msg, FileNotFoundError(msg))
            raise FileNotFoundError(msg)

        html_files = list(self.input_dir.glob("*.html"))

        if limit:
            html_files = html_files[:limit]

        if not html_files:
            self.logger.log_info("Nessun file HTML trovato nella directory.")

        pages = []
        for file_path in html_files:
            try:
                content = file_path.read_text(encoding="utf-8")
                pages.append({"filename": file_path.name, "content": content})
                # self.logger.log_info(f"✔ File letto: {file_path.name}")
            except Exception as e:
                self.logger.log_exception(f"❌ Errore lettura file {file_path.name}", e)

        self.logger.log_info(f"Totale file HTML letti: {len(pages)}")
        return pages
