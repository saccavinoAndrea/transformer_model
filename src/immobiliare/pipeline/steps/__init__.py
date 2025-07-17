# ---- Fase di preprocess
from .file_loading_step import FileLoadingStep
from .tokenizer_parallel_step import TokenizerParallelStep
from .normalization_step import NormalizationStep
from .save_to_disk_step import SaveToDiskStep

# ---- Fase di labeling
from .labeling_step import LabelingStep
from .feature_selection_step import FeatureSelectionStep
from .analysis_step import AnalysisStep

# ---- Fase di training
from .training_step import TrainingStep

# ---- Fase di inferenza
from .inference_step import InferenceStep
from .evaluation_step import EvaluationStep


# ---- racchiusi nell'unico step TokenizerParallelStep ---- #
from .html_tokenizer_step import HTMLTokenizerStep
from .feature_extraction_step import FeatureExtractionStep
