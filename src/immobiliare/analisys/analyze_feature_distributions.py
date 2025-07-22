"""
src/immobiliare/analysis/analyze_feature_distributions.py
--------------------------------------
Analizza statistiche di tutte le feature numeriche:
• mean, std, min, max, count per label
• confronta (opzionale) due modelli/jsonl (old vs new)
• esporta un CSV riepilogativo
• genera box‑plot PNG per ogni feature

Esempio d’uso:
python tools/analyze_feature_distributions.py \
    --input data/inference_test/data_to_predict.jsonl \
    --outdir report_feat \
    --compare data/prev_run/data_to_predict.jsonl
"""

import json
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from immobiliare.core_interfaces.analisys.ianalyzer import IAnalyzer
from immobiliare.utils.logging.logger_factory import LoggerFactory


# ---------- utils -----------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def dataframe_from_jsonl(path: Path, tag: str) -> pd.DataFrame:
    df = pd.DataFrame(load_jsonl(path))
    df["__src__"] = tag  # distinzione old/new
    return df


def numeric_cols(df: pd.DataFrame, exclude=("text", "label", "tag", "meta", "__src__")) -> List[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

# ---------- main analysis ---------------------------------------------------


def analyze(df: pd.DataFrame, outdir: Path, prefix: str = "") -> pd.DataFrame:
    """
    Ritorna un DataFrame (feature × label) con mean, std, min, max, n_
    e genera box‑plot PNG.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    feat_cols = numeric_cols(df)
    labels = df["label"].unique()

    records = []
    for feat in feat_cols:
        for lbl in labels:
            subset = df[df["label"] == lbl][feat]
            if subset.empty:
                continue
            records.append(
                dict(
                    feature=feat,
                    label=lbl,
                    mean=subset.mean(),
                    std=subset.std(),
                    min=subset.min(),
                    max=subset.max(),
                    count=len(subset),
                )
            )

        # ---------- box‑plot ----------
        plt.figure(figsize=(6, 3))
        data = [df[df["label"] == lbl][feat] for lbl in labels]
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.title(f"{prefix}{feat}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / f"{prefix}{feat}_boxplot.png")
        plt.close()

    summary_df = pd.DataFrame(records)
    summary_df.to_csv(outdir / f"{prefix}feature_stats.csv", index=False)
    print(f"✔  Salvato CSV riepilogo in: {outdir / f'{prefix}feature_stats.csv'}")
    return summary_df

class AnalyzerFeatureDistribution(IAnalyzer):
    def __init__(
            self,
            input_dir: Path,
            output_dir: Path,
            compare: Path):

        self.logger = LoggerFactory.get_logger("analyze_feature_distributions")
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.compare = None

    def run(self) -> None: #Tuple[Counter, List[dict]]:

        parser = argparse.ArgumentParser(description="Analizza distribuzione feature numeriche per label")
        parser.add_argument("--input", required=True, type=Path, help="jsonl con features+label")
        parser.add_argument("--compare", type=Path, default=None, help="secondo jsonl per confronto (old vs new)")
        parser.add_argument("--output_dir", type=Path, default=Path("feature_report"), help="cartella output")
        #args = parser.parse_args()

        #output_dir: Path = args.output_dir

        # -------- dataset principale --------
        df_main = dataframe_from_jsonl(self.input_dir, tag="NEW")
        summary_new = analyze(df_main, self.output_dir, prefix="new_")

        # -------- confronto opzionale -------
        if self.compare:
            df_old = dataframe_from_jsonl(self.compare, tag="OLD")
            summary_old = analyze(df_old, self.output_dir, prefix="old_")

            # merge (inner su feature+label) e Δ mean
            merged = summary_new.merge(
                summary_old,
                on=["feature", "label"],
                suffixes=("_new", "_old"),
                how="inner",
            )
            merged["delta_mean"] = merged["mean_new"] - merged["mean_old"]
            merged.to_csv(self.output_dir / "diff_old_vs_new.csv", index=False)
            print(f"✔  Salvato confronto old/new in: {self.output_dir / 'diff_old_vs_new.csv'}")

        print("\n✅ Analisi completata.")
