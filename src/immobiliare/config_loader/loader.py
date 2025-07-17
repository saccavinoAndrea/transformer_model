# src/immobiliare/config_loader/loader.py
from dataclasses import dataclass
from typing import Dict

import yaml


@dataclass
class Config:
    loggers: Dict[str, str]
    logger_out_dir: str
    logger_out_file_name: str
    n_pages: int
    raw_html_dir: str
    from_page: int
    pipeline_preprocess_config: str
    pipeline_preprocess_section: str
    pipeline_labeling_config: str
    pipeline_labeling_section: str
    pipeline_model_config: str
    pipeline_model_section: str
    pipeline_training_config: str
    pipeline_training_section: str
    pipeline_inference_config: str
    pipeline_inference_section: str


class ConfigLoader:
    @staticmethod
    def load(config_path: str = "config/config.yaml") -> Config:
        """Carica il file YAML in modo semplice, relativo alla root del progetto"""
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)
        return Config(**raw)
