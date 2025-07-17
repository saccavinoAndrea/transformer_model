# src/immobiliare/core_interfaces/pipeline/ipipeline_step.py

from abc import ABC, abstractmethod
from typing import Any

class IPipelineStep(ABC):
    """
    Un singolo step della pipeline: riceve un input, lo elabora e restituisce un output.
    """

    @abstractmethod
    def run(self, data: Any) -> Any:
        """
        :param data: l’input di questo step (può essere None per il primo step)
        :return: l’output da passare allo step successivo
        """
        pass
