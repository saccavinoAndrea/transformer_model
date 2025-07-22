# src/immobiliare/pipeline/steps/training_step.py
import json
from pathlib import Path
from typing import Any

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.utils.logging.decorators import log_exec
from immobiliare.training.trainer import Trainer
from immobiliare.utils import resolve_versioned_jsonl
from immobiliare.utils.logging.logger_factory import LoggerFactory


class TrainingStep(IPipelineStep):
    def __init__(
            self,
            jsonl_labeled_path: str,
            label_2id_dir: str,
            features_keys_dir: str,
            selected_features_path: str,
            model_dir: str,
            report_dir: str,
            artifact_dir: str,
            training_with_selected_features: bool,
            batch_size: int,
            lr: float,
            patience: int,
            val_split: float,
            max_epochs: int,
    ):
        self.logger = LoggerFactory.get_logger("training_step")
        self.jsonl_labeled_path = jsonl_labeled_path
        self.label_2id_dir = label_2id_dir
        self.features_keys_dir = features_keys_dir
        self.selected_features_path = selected_features_path
        self.model_dir = model_dir
        self.report_dir = report_dir
        self.artifact_dir = artifact_dir
        self.training_with_selected_features = training_with_selected_features
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.val_split = val_split
        self.max_epochs = max_epochs


    @log_exec(logger_name="model_training_step", method_name="run")
    def run(self, data: Any = None) -> None:


        # 1) Carica la lista delle feature keys calcolate nel preprocess
        features_keys_path_resolved = resolve_versioned_jsonl(self.features_keys_dir)
        with open(features_keys_path_resolved, "r", encoding="utf-8") as f:
            features_keys = json.load(f)

        # 1) Carica la lista di feature selezionate dopo Random Forest classification, o altri modelli
        selected_features_path_resolved = resolve_versioned_jsonl(self.selected_features_path)
        with open(selected_features_path_resolved, "r", encoding="utf-8") as f:
            selected = json.load(f)

        # 2) Carica la lista di label2Id
        label_2id_dir_resolved = str(resolve_versioned_jsonl(self.label_2id_dir))
        with open(label_2id_dir_resolved, "r", encoding="utf-8") as f:
            label2id = json.load(f)

        jsonl_labeled_path_resolved = str(resolve_versioned_jsonl(self.jsonl_labeled_path))


        trainer = Trainer(
            jsonl_path=jsonl_labeled_path_resolved,
            label2id=label2id,
            feature_keys_to_use= selected if self.training_with_selected_features else features_keys,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            val_split=self.val_split,
            model_save_path=self.model_dir,
            report_dir=Path(self.report_dir),
            artifact_dir=Path(self.artifact_dir),
            max_epochs=self.max_epochs,
        )
        trainer.run()