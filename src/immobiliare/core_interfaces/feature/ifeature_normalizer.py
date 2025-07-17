# src/immobiliare/core_interfaces/metrics/ifeature_normalizer.py

from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any

class IFeatureNormalizer(ABC):
    """
    Interfaccia per normalizzatori di feature numeriche (es. Z-score).
    Può operare su dizionari o dataclass compatibili.
    """

    @abstractmethod
    def fit(self, data: List[Union[Dict[str, Any], Any]]) -> None:
        """
        Calcola media e deviazione standard delle feature numeriche.
        """
        pass

    @abstractmethod
    def transform(self, data: List[Union[Dict[str, Any], Any]]) -> List[Union[Dict[str, Any], Any]]:
        """
        Applica la normalizzazione alle feature selezionate.
        """
        pass

    @abstractmethod
    def fit_transform(self, data: List[Union[Dict[str, Any], Any]]) -> List[Union[Dict[str, Any], Any]]:
        """
        Esegue `fit` e `transform` in sequenza.
        """
        pass
