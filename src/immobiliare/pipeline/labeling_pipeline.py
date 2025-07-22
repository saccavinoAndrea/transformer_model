# src/immobiliare/pipeline/labeling_pipeline.py

from typing import Any

from immobiliare.config_loader import ConfigLoader
from immobiliare.core_interfaces.pipeline.ipipeline import IPipeline
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.config_loader.pipeline_loader import load_pipeline_from_yaml


class LabelingPipeline(IPipeline):
    def __init__(self, config_path: str = "config/labeling_pipeline_config.yaml"):
        config = ConfigLoader.load()
        self.logger = LoggerFactory.get_logger("labeling_pipeline")
        # carica la lista di step dalla sezione `labeling_pipeline` del YAML
        self.steps = load_pipeline_from_yaml(config.pipeline_labeling_config, config.pipeline_labeling_section)

    @log_exec(logger_name="labeling_pipeline", method_name="run")
    def run(self, _input: Any = None) -> Any:
        data = None
        for step in self.steps:
            data = step.run(data)
        return data
