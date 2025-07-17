# src/immobiliare/pipeline/model_pipeline.py

from config_loader import ConfigLoader
from immobiliare.config_loader.pipeline_loader import load_pipeline_from_yaml
from immobiliare.core_interfaces.pipeline.ipipeline import IPipeline
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils.logging.logger_factory import LoggerFactory


class ModelPipeline(IPipeline):
    def __init__(self, config_path: str = "config/model_pipeline_config.yaml"):
        config = ConfigLoader.load()
        self.logger = LoggerFactory.get_logger("model_pipeline")
        self.steps = load_pipeline_from_yaml(config.pipeline_preprocess_config, config.pipeline_preprocess_section)

    @log_exec(logger_name="model_pipeline", method_name="run")
    def run(self, _):
        data = None
        for step in self.steps:
            data = step.run(data)
        return data
