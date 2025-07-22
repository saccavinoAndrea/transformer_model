import json
import time
from pathlib import Path

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.utils import resolve_versioned_jsonl, timestamped_path
from immobiliare.utils.logging.logger_factory import LoggerFactory


class EvaluationStep(IPipelineStep):
    def __init__(self,
                 predictions_complete_path,
                 ground_truth_path,
                 report_json,
                 report_csv,
                 cm_png):

        self.logger = LoggerFactory.get_logger("evaluation_step")
        self.predictions_complete_path = Path(predictions_complete_path)
        self.ground_truth_path = Path(ground_truth_path)
        self.report_json = Path(report_json)
        self.report_csv = Path(report_csv)
        self.cm_png = Path(cm_png)

    def run(self, data=None):

        self.logger.log_info("Start inference evaluation ...")
        start_time = time.time()  # ← inizio cronometro

        import csv
        # 1) Carica dati
        # predizioni: ci aspettiamo colonne “text” e “predicted_label”
        predictions_complete_path_resolved = resolve_versioned_jsonl(self.predictions_complete_path)
        preds = list(csv.DictReader(open(predictions_complete_path_resolved, encoding="utf-8")))
        self.logger.log_info("predictions_complete_len: " + str(len(preds)))

        # ground‑truth: ci aspettiamo colonne “text” e “true_label”
        ground_truth_path_resolved = resolve_versioned_jsonl(self.ground_truth_path)
        gt = list(csv.DictReader(open(ground_truth_path_resolved, encoding="utf-8")))
        self.logger.log_info("ground_truth_len: " + str(len(gt)))

        # 1. Costruisci un dizionario di lookup
        pred_map = {p["text"].strip(): p["predicted_label"] for p in preds}

        # 2) Allinea e costruisci liste
        y_true, y_pred = [], []
        for r in gt:
            key = r["text"].strip()
            y_true.append(r["true_label"])
            if key in pred_map:
                y_pred.append(pred_map[key])
            else:
                y_pred.append("O")  # <‑‑ gestisci come preferisci]

        # ... dopo aver costruito y_true e y_pred …
        # --- 1) Pulizia / default sui None in y_pred e y_true ---
        y_true_clean, y_pred_clean = [], []
        for t, p in zip(y_true, y_pred):
            # Filtra o mappa i None in y_true
            if t is None:
                # Se preferisci saltare:
                continue
            # Filtra o mappa i None in y_pred
            if p is None:
                p = "O"

            y_true_clean.append(t)
            y_pred_clean.append(p)

        # --- 2) Sanity check ---
        bad_true = [x for x in y_true_clean if not isinstance(x, str)]
        bad_pred = [x for x in y_pred_clean if not isinstance(x, str)]
        assert not bad_true, f"Valori non stringa in y_true_clean: {bad_true}"
        assert not bad_pred, f"Valori non stringa in y_pred_clean: {bad_pred}"

        # --- 3) Costruisci LABELS filtrando ogni None residuo ---
        all_labels = set(y_true_clean + y_pred_clean)
        LABELS = sorted(lbl for lbl in all_labels if lbl is not None)

        # --- 4) Calcolo report esplicito con labels pulite ---
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(
            y_true_clean,
            y_pred_clean,
            labels=LABELS,
            zero_division=0,
            output_dict=True
        )

        report_json_final_path = timestamped_path(self.report_json)
        with open(report_json_final_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 4) Error analysis CSV
        rows = []
        for t, p, r in zip(gt, y_pred, y_true):
            rows.append({
                "text": t["text"],
                "true_label": t["true_label"],
                "predicted_label": p,
                "correct": p == t["true_label"]
            })

        import csv
        report_csv_final_path = timestamped_path(self.report_csv)
        with open(report_csv_final_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # 5) Confusion matrix plot
        cm_png_final_path = timestamped_path(self.cm_png)

        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=list(report.keys())[:-3])
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(cm, aspect='auto')
        ax.set_xticks(range(len(cm)))
        ax.set_yticks(range(len(cm)))
        ax.set_xticklabels(list(report.keys())[:-3], rotation=90)
        ax.set_yticklabels(list(report.keys())[:-3])
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(cm_png_final_path)

        self.logger.log_info(f"✅ Evaluation report saved to {self.report_json}, {self.report_csv}, {self.cm_png}")

        total_time = time.time() - start_time
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        self.logger.log_info(f"\n⏱️  Tempo totale di inference evaluation: {int(h)}h {int(m)}m {int(s)}s")
