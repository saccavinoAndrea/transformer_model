# src/immobiliare/core_interfaces/feature/ifeature_extractor.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from bs4 import Tag

class IFeatureExtractor(ABC):
    """
    Contratto per estrarre feature da un singolo token HTML.
    """

    @abstractmethod
    def extract_features(
        self,
        token: str,
        tag: Tag,
        position: int,
        *,
        total_tokens: int,
        max_depth_in_doc: int,
        all_tokens: List[str],
        total_text_len: int = None,
        page_token_counts: dict = None,
        parent_tokens_positions: List[int] = None
    ) -> Dict[str, float]:
        """
        Calcola un dizionario di feature numeriche/boolean per un token.
        """
        pass
