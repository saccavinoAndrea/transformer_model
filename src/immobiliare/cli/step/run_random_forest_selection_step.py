# src/immobiliare/cli/run_inference.py
from pathlib import Path

from labeling import RandomForestSelector


def main():

    random_forest = (
        RandomForestSelector(
            Path("data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled.csv"),
            Path("data/transformer_model/labeling/artifacts/feature/selected_features.json"),
            "label",
        )
    )
    random_forest.execute_selection()

if __name__ == "__main__":
    main()
