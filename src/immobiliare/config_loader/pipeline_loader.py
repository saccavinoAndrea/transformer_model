# src/immobiliare/config_loader/pipeline_loader.py

import importlib
import yaml

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep

def load_pipeline_from_yaml(
        yaml_path: str,
        section: str = "pipeline_preprocess") -> list[IPipelineStep]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    steps_config = config.get(section, [])
    steps = []

    for step in steps_config:
        class_path = step["class"]
        params = step.get("params", {})

        # Dynamically import the class
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        # Instantiate step with params
        steps.append(cls(**params))

    return steps
