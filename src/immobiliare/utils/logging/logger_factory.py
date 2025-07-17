# src/immobiliare/utils/logger_factory.py
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict
from core_interfaces.logger.ilogger import ILogger

class LoggerFactory:
    _loggers: Dict[str, ILogger] = {}
    _log_levels: Dict[str, str] = {}

    @classmethod
    def configure(cls, levels: Dict[str, str]):
        cls._log_levels = levels or {}

    @classmethod
    def get_logger(cls, name: str = "default") -> ILogger:
        if name in cls._loggers:
            return cls._loggers[name]

        # Crea un nuovo logger se non esiste
        logger = logging.getLogger(name)
        level_str = cls._log_levels.get(name, "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
        logger.setLevel(level)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(f"[%(asctime)s] [{name.upper()}] %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Wrap nel nostro ILogger
        wrapped_logger = _WrappedLogger(logger)
        cls._loggers[name] = wrapped_logger
        return wrapped_logger


class _WrappedLogger(ILogger):
    """Adatta un logging.Logger standard all'interfaccia ILogger."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def log_info(self, message: str):
        self._logger.info(message)

    def log_exception(self, message: str, exc: Exception):
        self._logger.error(message)
        self._logger.error("Exception type: %s", type(exc).__name__)
        self._logger.error("Exception message: %s", str(exc))

    def log_metrics_json(self, method_name: str, status_code: int, duration_ms: float, duration_readable: str = "",
                         message: str = "", out_dir: Path = None):
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            file_path = out_dir / f"{self._logger.name}_metrics.jsonl"
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "logger": self._logger.name,
                    "method": method_name,
                    "status_code": status_code,
                    "duration_ms": int(round(duration_ms)),
                    "duration_readable": f"{duration_ms / 1000:.2f} sec",
                    "message": message
                }) + "\n")
        except Exception as e:
            self._logger.warning(f"Errore scrittura log_metrics_json: {e}")
