import json
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from core_interfaces.feature.ifeature_selector import IFeatureSelector
from utils import resolve_versioned_jsonl, timestamped_path
from utils.logging.logger_factory import LoggerFactory

from typing import Tuple, Union

ArrayLike = Union[np.ndarray, pd.Series, list]

def _filter_rare_classes(
    X: ArrayLike,
    y: ArrayLike,
    min_samples: int = 2
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Rimuove le istanze delle classi con meno di `min_samples` occorrenze,
    restituendo X_filtered e y_filtered allineati.
    """
    # Conta le occorrenze di y (gestisce list e pd.Series)
    counts = Counter(y) if not isinstance(y, pd.Series) else y.value_counts().to_dict()

    # Se X è lista Python, filtriamo con list comprehension insieme a y
    if isinstance(X, list):
        filtered = [(x_i, y_i) for x_i, y_i in zip(X, y) if counts[y_i] >= min_samples]
        if not filtered:
            return [], []
        X_filt, y_filt = zip(*filtered)
        return list(X_filt), list(y_filt)

    # Se X è pandas DataFrame o Series, usiamo mask booleana su y (pd.Series)
    if isinstance(y, pd.Series):
        mask = y.map(lambda lbl: counts.get(lbl, 0) >= min_samples)
        X_filt = X.loc[mask] if hasattr(X, "loc") else X[mask]
        y_filt = y.loc[mask]
        return X_filt, y_filt

    # Se qui, presumiamo X e y siano NumPy array
    mask = np.array([counts.get(lbl, 0) >= min_samples for lbl in y])
    X_filt = X[mask]
    y_filt = y[mask]
    return X_filt, y_filt


class RandomForestSelector(IFeatureSelector):

    def __init__(
            self,
            feature_labelled_dir: Path,
            feature_selected_dir: Path,
            label_column_name: str):

        self.logger = LoggerFactory.get_logger("random_forest_selector")
        self.feature_labelled_dir = resolve_versioned_jsonl(feature_labelled_dir)
        self.feature_selected_dir = timestamped_path(feature_selected_dir)
        self.label_column_name = label_column_name

        # 1) Carica tutto il DataFrame
        df = pd.read_csv(self.feature_labelled_dir)

        # 2) Estrai la serie delle etichette
        if self.label_column_name not in df.columns:
            raise ValueError(f"Colonna label '{self.label_column_name}' non trovata in {self.feature_labelled_dir}")
        self.y = df[self.label_column_name]

        # 3) Seleziona solo le colonne numeriche (float, int)
        numeric_df = df.select_dtypes(include=[np.number])

        # 4) Rimuovi la colonna delle label dagli input
        if self.label_column_name in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[self.label_column_name])

        # 5) Memorizza nomi e dati
        self.feature_names = numeric_df.columns.tolist()
        self.X = numeric_df.values


    def execute_selection(self, test_size: float = 0.2, random_state: int = 42, n_estimators: int = 100):

        x_filtered, y_filtered = _filter_rare_classes(self.X, self.y, min_samples=2)

        # Split stratificato
        x_train, x_test, y_train, y_test = train_test_split(
            x_filtered, y_filtered,
            test_size=test_size,
            stratify=y_filtered,
            random_state=random_state
        )

        # Allena la foresta
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(x_train, y_train)

        # Estrai importanze globali
        importances = pd.Series(rf.feature_importances_, index=self.feature_names)
        importances = importances.sort_values(ascending=False)

        cum_importance = importances.cumsum()
        selected = cum_importance[cum_importance <= 0.99].index.tolist()
        selected = sorted(selected, key=lambda col: importances[col], reverse=True)

        with open(self.feature_selected_dir, "w", encoding="utf-8") as f:
            json.dump(selected, f, ensure_ascii=False, indent=2)
        self.logger.log_info("✅ Artifacts selected_features.json salvato in " + str(self.feature_selected_dir))

        # Stampa prime n selected
        self.logger.log_info(("=== Top " + str(len(selected)) + " feature importances globali ==="))
        #print(importances.head(len(selected)))

        # Grafico
        plt.figure(figsize=(8, 6))
        importances.head(20).plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title("=== Top " + str(len(selected)) + " feature importances globali ===")
        plt.tight_layout()
        #plt.show()

        return rf, importances