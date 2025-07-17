# src/immobiliare/cli/run_inference.py
from pathlib import Path

from analisys import AnalyzerFeatureDistribution


def main():

    feature_analyzer = AnalyzerFeatureDistribution(
        input_dir=Path("data/labels/labeled.jsonl"),
        output_dir=Path("data/analysis/feature_distribution"),
        compare=Path(""),
    )
    feature_analyzer.run()

if __name__ == "__main__":
    main()
