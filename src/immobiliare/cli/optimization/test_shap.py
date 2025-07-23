# src/immobiliare/analysis/shap_explain_and_params.py

import json
import pandas as pd
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import warnings
from torch import nn
from typing import Union

from immobiliare.models.transformer_pytorch import TransformerForTokenClassification

# ✅ Ignora warning da SHAP su np.random.seed
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded by calling `np.random.seed`")

def count_model_parameters(model: nn.Module) -> int:
    total = sum(p.numel() for p in model.parameters())
    print(f"🔢 Numero totale di parametri nel modello: {total}")
    return total


def explain_with_shap_fast(model: nn.Module,
                           test_data: Union[np.ndarray, pd.DataFrame],
                           sample_size: int = 50,
                           background_size: int = 20,
                           save_path: Union[str, None] = None,
                           topk_csv: str = "top_features.csv",
                           barplot_path: str = "shap_barplot.png",
                           force_plot_path: str = "shap_force_plot.html",
                           device: str = 'cpu'):
    model.eval().to(device)

    if isinstance(test_data, pd.DataFrame):
        test_data = test_data.values
    X = test_data.astype(np.float32)
    X_test = X[:sample_size]
    X_bg   = X[:background_size]

    Xb = X_bg.reshape(-1, 1, X.shape[1]).astype(np.float32)
    Xt = X_test.reshape(-1, 1, X.shape[1]).astype(np.float32)

    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(x_np, dtype=torch.float32)
            if x_t.dim() == 2:
                x_t = x_t.unsqueeze(1)
            logits = model(x_t)
            preds = logits[:, 0, :]
            return preds.numpy()

    Xb = Xb[:, 0, :]
    Xt = Xt[:, 0, :]

    print("Xb shape:", Xb.shape, type(Xb))
    print("Xt shape:", Xt.shape, type(Xt))

    masker = shap.maskers.Independent(Xb)
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")
    shap_result = explainer(Xt)

    # 1️⃣ Grafico SHAP summary
    shap.summary_plot(shap_result.values, Xt,
                      feature_names=FEATURE_NAMES,
                      show=save_path is None)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"📈 Grafico SHAP salvato in: {save_path}")
    plt.close()

    # 2️⃣ Salva Top-K feature importance
    print("shap_result.values shape:", shap_result.values.shape)  # es. (20, 14, 100)
    mean_abs_vals = np.abs(shap_result.values).mean(axis=(0, 2)) # media su samples e classi
    print("mean_abs_vals shape:", mean_abs_vals.shape)  # es. (100,)
    print("FEATURE_NAMES len:", len(FEATURE_NAMES))  # 100

    shap_df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': mean_abs_vals
    }).sort_values(by='importance', ascending=False)
    shap_df.to_csv(topk_csv, index=False)
    print(f"📄 Top feature SHAP salvate in: {topk_csv}")

    # 3️⃣ Grafico Barplot
    topk = shap_df.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(topk['feature'][::-1], topk['importance'][::-1])
    plt.xlabel("Importanza media |SHAP value|")
    plt.title("Top 20 Feature Importanti (SHAP)")
    plt.tight_layout()
    plt.savefig(barplot_path)
    print(f"📊 Barplot SHAP salvato in: {barplot_path}")
    plt.close()


if __name__ == "__main__":
    model = TransformerForTokenClassification(
        input_dim=100,
        embedding_dim=64,
        num_classes=14,
        num_layers=2,
        nhead=8,
    )
    model.load_state_dict(torch.load(
        "data/transformer_model/training/artifacts/model/pytorch_transformer/best_model_20250717_092449.pt",
        map_location='cpu'
    ))

    with open("data/transformer_model/preprocess/artifacts/features/feature_keys_20250717_092306.json", "r") as f:
        FEATURE_NAMES = json.load(f)

    df = pd.read_csv(
        "data/transformer_model/labeling/artifacts/tokens/token_embeddings_dense_labeled_20250717_092412.csv"
    )
    df_sel = df[FEATURE_NAMES].dropna(axis=0, how="any")

    expected_dim = model.embedding.in_features
    actual_dim   = df_sel.shape[1]
    assert actual_dim == expected_dim, (
        f"Mi aspettavo {expected_dim} feature numeriche pulite, "
        f"ma ne ho trovate {actual_dim}. "
        f"Colonne escluse: {set(df.columns) - set(df_sel.columns)}"
    )

    X_test = df_sel.values

    count_model_parameters(model)
    explain_with_shap_fast(
        model,
        X_test,
        sample_size=20,
        background_size=10,
        save_path="shap_fast.png",
        topk_csv="top_features.csv",
        barplot_path="shap_barplot.png",
        force_plot_path="shap_force_plot.html",
        device='cpu'
    )
