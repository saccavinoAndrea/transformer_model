# src/immobiliare/core_interfaces/pipeline/ipipeline.py:

from abc import ABC, abstractmethod


class IPipeline(ABC):
    @abstractmethod
    def run(self):
        """
        Metodo principale per eseguire la pipeline.
        """
        pass
