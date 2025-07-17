# src/immobiliare/core_interfaces/ilogger.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class ILogger(ABC):
    @abstractmethod
    def log_info(self, message: str):
        pass

    @abstractmethod
    def log_exception(self, message: str, exc: Exception):
        pass

    @abstractmethod
    def log_metrics_json(self, method_name: str, status_code: int,
                        duration_ms: float, duration_readable: str = "", message: str = "",
                        out_dir: Path = None):
        pass