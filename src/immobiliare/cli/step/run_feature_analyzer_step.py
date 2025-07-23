# src/immobiliare/cli/run_inference.py
from pathlib import Path

from immobiliare.analisys import AnalyzerFeatureDistribution
from immobiliare.pipeline.steps import AnalysisStep
from immobiliare.utils import resolve_versioned_jsonl


def main():

    """
    feature_analyzer = AnalyzerFeatureDistribution(
        input_dir=Path("data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled_20250717_092412.jsonl"),
        output_dir=Path("data/transformer_model/labeling/artifacts/tokens/analysis"),
        compare=Path(""),
    )
    feature_analyzer.run()
    """
    resolved_path = resolve_versioned_jsonl("data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled.jsonl")
    feature_analyzer = AnalysisStep(
        input_jsonl=str(resolved_path),
        filtered_output="data/transformer_model/labeling/artifacts/tokens/analysis/tokens_filtered.jsonl",
        distribution_output="data/transformer_model/labeling/artifacts/tokens/analysis/token_type_distribution.txt",
    )
    feature_analyzer.run()

if __name__ == "__main__":
    main()
