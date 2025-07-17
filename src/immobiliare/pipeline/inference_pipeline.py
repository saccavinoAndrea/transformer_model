from config_loader import ConfigLoader, load_pipeline_from_yaml
from core_interfaces.pipeline.ipipeline import IPipeline
from utils import log_exec
from utils.logging.logger_factory import LoggerFactory


class InferencePipeline(IPipeline):
    def __init__(self, config_path: str = "config/pipeline_inference.yaml"):
        config = ConfigLoader.load()
        self.logger = LoggerFactory.get_logger("inference_pipeline")
        self.steps = load_pipeline_from_yaml(config.pipeline_inference_config, config.pipeline_inference_section)

    @log_exec(logger_name="inference_pipeline", method_name="run")
    def run(self, _):
        data = None
        for step in self.steps:
            data = step.run(data)
        return data