# src/immobiliare/core_interfaces/labeler/ilabeler.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class ILabeler(ABC):
    """
    Definisce il contratto per un “labeler”:
    - carica input (HTML + JSONL)
    - estrae le label
    - restituisce lista di dict { …, 'label': … }
    """
    @abstractmethod
    def run(self) -> List[Dict[str, Any]]:
        """
        Esegue tutto il flusso di labeling.
        """
        pass
