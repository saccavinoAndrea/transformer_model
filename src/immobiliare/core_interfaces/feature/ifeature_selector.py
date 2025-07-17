# src/immobiliare/core_interfaces/metrics/ifeature_normalizer.py

from abc import ABC, abstractmethod
from typing import Any


class IFeatureSelector(ABC):
    """
    Interfaccia per selettori di feature.
    Può operare su dizionari o dataclass compatibili.
    """

    @abstractmethod
    def execute_selection(self, data: Any) -> None:
        """
        :param data:
        :param Any:
        """
        pass

