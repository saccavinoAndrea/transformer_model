# src/immobiliare/core_interfaces/imetricparser.py

from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path

class IMetricParser(ABC):
    @abstractmethod
    def load_metrics(self, path: Path) -> List[Dict]: pass

    @abstractmethod
    def filter_by_method(self, metrics: List[Dict], method_name: str) -> List[Dict]: pass
