# src/immobiliare/core_interfaces/trainer/itrainer.py

from abc import ABC, abstractmethod


class ITrainer(ABC):
    @abstractmethod
    def run(self):
        """
        Tokenizza il contenuto testuale restituendo coppie (testo, tag).
        """
        pass


