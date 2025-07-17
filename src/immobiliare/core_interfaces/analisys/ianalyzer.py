# src/immobiliare/core_interfaces/analysis/ianalyzer.py

from abc import ABC, abstractmethod
from typing import Tuple, List, Any
from collections import Counter

class IAnalyzer(ABC):
    """
    Contratto per un “analyzer” che legge un JSONL di token etichettati,
    restituisce:
      - Counter delle label
      - lista di dict filtrati (token, position, label)
    """
    @abstractmethod
    def run(self) -> Tuple[Counter, List[dict]]:
        pass
