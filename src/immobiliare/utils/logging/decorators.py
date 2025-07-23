# src/immobiliare/utils/decorators.py

import time
import functools
from pathlib import Path
from typing import Callable

from immobiliare.config_loader import ConfigLoader
from immobiliare.utils.logging.logger_factory import LoggerFactory

def log_exec(logger_name: str, method_name: str = None):
    """
    Decoratore per loggare automaticamente info e metriche in JSON.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = LoggerFactory.get_logger(logger_name)

            config = ConfigLoader.load()
            log_dir = Path(config.logger_out_dir)

            name = method_name or func.__name__
            start = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = int(round((time.time() - start) * 1000))
                #logger.log_info(f"{name} completato.")
                logger.log_metrics_json(method_name=name, status_code=200, duration_ms=duration_ms,
                                        duration_readable="", message="OK", out_dir=log_dir)
                return result
            except Exception as e:
                duration_ms = int(round((time.time() - start) * 1000))
                logger.log_exception(f"{name} fallito", e)
                logger.log_metrics_json(method_name=name, status_code=500, duration_ms=duration_ms,
                                        duration_readable="", message=str(e), out_dir=log_dir)
                raise
        return wrapper
    return decorator


def alog_exec(logger_name: str, method_name: str = None):
    """
    Decoratore asincrono per loggare durata, stato e messaggi in JSON.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()

            logger = LoggerFactory.get_logger(logger_name)
            config = ConfigLoader.load()
            log_dir = Path(config.logger_out_dir)

            name = method_name or func.__name__

            try:
                result = await func(*args, **kwargs)
                duration_ms = int(round((time.perf_counter() - start) * 1000))
                logger.log_info(f"{name} completato.")
                logger.log_metrics_json(method_name=name, status_code=200, duration_ms=duration_ms,
                                        duration_readable="", message="OK", out_dir=log_dir)
                return result
            except Exception as e:
                duration_ms = int(round((time.perf_counter() - start) * 1000))
                logger.log_exception(f"{name} fallito", e)
                logger.log_metrics_json(method_name=name, status_code=500, duration_ms=duration_ms,
                                        duration_readable="", message=str(e), out_dir=log_dir)
                raise
        return wrapper
    return decorator