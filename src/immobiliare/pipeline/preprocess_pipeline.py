# src/immobiliare/pipeline/preprocess_pipeline.py

from immobiliare.config_loader import load_pipeline_from_yaml
from immobiliare.core_interfaces.pipeline.ipipeline import IPipeline
from immobiliare.config_loader.loader import ConfigLoader
from immobiliare.utils.logging.logger_factory import LoggerFactory

from utils import log_exec


class PreprocessPipeline(IPipeline):
    def __init__(self, config_path: str = "config/pipeline_preprocess.yaml"):
        config = ConfigLoader.load()
        self.logger = LoggerFactory.get_logger("preprocess_pipeline")
        self.steps = load_pipeline_from_yaml(config.pipeline_preprocess_config, config.pipeline_preprocess_section)

    @log_exec(logger_name="preprocess_pipeline", method_name="run")
    def run(self, _):
        data = None
        for step in self.steps:
            data = step.run(data)