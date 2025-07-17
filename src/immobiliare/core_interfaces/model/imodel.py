# src/immobiliare/core_interfaces/model/imodel.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Tuple

class IModel(ABC):
    """
    Contratto per i modelli del progetto:
      - train
      - predict
      - save / load
    """

    @abstractmethod
    def train(self, train_data: Any, val_data: Any = None) -> None:
        pass

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass
