# src/immobiliare/pipeline/training_pipeline.py

from immobiliare.config_loader import ConfigLoader
from immobiliare.config_loader.pipeline_loader import load_pipeline_from_yaml
from immobiliare.core_interfaces.pipeline.ipipeline import IPipeline
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.utils.logging.logger_factory import LoggerFactory


class TrainingPipeline(IPipeline):
    def __init__(self, config_path: str = "config/training_pipeline_config.yaml"):
        config = ConfigLoader.load()
        self.logger = LoggerFactory.get_logger("training_pipeline")
        self.steps = load_pipeline_from_yaml(config.pipeline_training_config, config.pipeline_training_section)

    @log_exec(logger_name="training_pipeline", method_name="run")
    def run(self, _):
        data = None
        for step in self.steps:
            data = step.run(data)
        return data
