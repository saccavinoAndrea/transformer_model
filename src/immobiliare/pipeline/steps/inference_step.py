# src/immobiliare/pipeline/steps/inference_step.py
import csv
import json
import time
from pathlib import Path
from typing import Any, List

import joblib
import torch

from core_interfaces.pipeline.ipipeline_step import IPipelineStep
from models.transformer_pytorch import TransformerForTokenClassification

from pipeline.steps import FileLoadingStep, TokenizerParallelStep, SaveToDiskStep
from utils import timestamped_path, resolve_versioned_jsonl
from utils.logging.logger_factory import LoggerFactory


def save_ground_truth_csv(data: List[Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["text"])  # intestazione
        for d in data:
            # se il dizionario non ha "text" salta la riga
            token_dict = d.to_dict()
            if "text" in token_dict:
                writer.writerow([token_dict["text"]])


class InferenceStep(IPipelineStep):

    def __init__(
            self,
            html_pages_dir_to_predict: str,
            file_name_to_predict: str,
            normalizer_dir: str,
            jsonl_report_dir: str,
            csv_report_dir: str,
            feature_keys_dir: str,
            label2id_dir: str,
            model_dir: str,
            predicted_rumor_path: str,
            predicted_feature_path: str,
            predicted_complete_path: str,
            page_number_to_predict: int,
            file_loader: None,
            tokenizer_extractor: None,
            normalizer: None,
            writer_report: None
    ):
        self.logger = LoggerFactory.get_logger("inference_step")
        self.html_pages_dir_to_predict = Path(html_pages_dir_to_predict)
        self.file_name_to_predict = file_name_to_predict
        self.normalizer_dir = normalizer_dir
        self.jsonl_report_dir = jsonl_report_dir
        self.csv_report_dir = csv_report_dir
        self.feature_keys_dir = feature_keys_dir
        self.label2id_dir = label2id_dir
        self.model_dir = model_dir
        self.predicted_rumor_path = predicted_rumor_path
        self.predicted_feature_path = predicted_feature_path
        self.predicted_complete_path = predicted_complete_path
        self.page_number_to_predict = page_number_to_predict
        self.file_loader = file_loader
        self.tokenizer_extractor = tokenizer_extractor
        self.normalizer = normalizer
        self.writer_report = writer_report
        self.all_tokens_with_features = None
        self.tokens_normalized = []


    def run(self, data: Any) -> Any:
        self.logger.log_info("Start inference ...")
        start_time = time.time()  # ← inizio cronometro

        normalizer_resolved_dir = resolve_versioned_jsonl(self.normalizer_dir)

        jsonl_report_final_dir = str(timestamped_path(self.jsonl_report_dir))
        csv_report_final_dir = str(timestamped_path(self.csv_report_dir))

        self.file_loader = FileLoadingStep(input_dir=str(self.html_pages_dir_to_predict), limit=self.page_number_to_predict)
        self.tokenizer_extractor = TokenizerParallelStep()
        self.normalizer = joblib.load(normalizer_resolved_dir)
        self.writer_report = SaveToDiskStep(jsonl_path=jsonl_report_final_dir, csv_path=csv_report_final_dir)


        html_page_to_predict = self.file_loader.run()
        self.all_tokens_with_features = self.tokenizer_extractor.run(html_page_to_predict)

        self.predict()

        total_time = time.time() - start_time
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        self.logger.log_info(f"\n⏱️  Tempo totale di inference: {int(h)}h {int(m)}m {int(s)}s")

    def predict(self) -> Any:

        feature_keys = None
        label2id = None

        # 2) carica feature_keys
        feature_keys_resolved_dir = resolve_versioned_jsonl(self.feature_keys_dir)
        with open(feature_keys_resolved_dir, "r", encoding="utf-8") as f:
            feature_keys = json.load(f)
        self.logger.log_info("lunghezza feature_keys: " + str(len(feature_keys))) # 2) lunghezza feature_keys

        # 2) Normalizzazione su TUTTI i token (eccetto 'position')
        self.tokens_normalized = self.normalizer.transform(self.all_tokens_with_features)


        # 2.1) Salvataggio aggregato
        jsonl_report_final_dir = str(timestamped_path(self.jsonl_report_dir))
        csv_report_final_dir = str(timestamped_path(self.csv_report_dir))

        self.writer_report.run(self.tokens_normalized)
        self.logger.log_info(f"✅ Estratti e normalizzati {len(self.tokens_normalized)} token da {self.page_number_to_predict} pagine")
        self.logger.log_info(f"   • JSONL → {jsonl_report_final_dir}")
        self.logger.log_info(f"   • CSV   → {csv_report_final_dir}")

        # 2.2) Salvataggio aggiuntivo per generazione ground_truth.csv
        #save_ground_truth_csv(all_tokens_with_features, Path("data/inference_test/ground_truth/ground_truth.csv"))

        # 3) carica label2id e costruisci id2label
        label2id_resolved_dir = str(resolve_versioned_jsonl(self.label2id_dir))
        with open(label2id_resolved_dir, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        label2id = {v: k for k, v in label2id.items()}

        # 4) build modello (stessi hyperparam del training!)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = len(feature_keys)
        num_classes = len(label2id)

        self.logger.log_info("NUMERO DI FEATURE SELEZIONATE " + str(input_dim))
        self.logger.log_info("NUMERO DI CLASSI SELEZIONATE "+ str(num_classes))


        model = TransformerForTokenClassification(
            input_dim=input_dim,
            embedding_dim=64,  # o il valore che hai usato
            num_classes=num_classes,
            nhead=8                   # idem
        ).to(device)

        model_resolved_dir = str(resolve_versioned_jsonl(self.model_dir))
        #model.load_state_dict(torch.load(model_final_dir, map_location=device))

        ckpt = torch.load(model_resolved_dir, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        in_keys = [k for k in state_dict.keys() if "embedding.weight" in k or "linear" in k and "weight" in k]
        for k in in_keys:
            if "embedding.weight" == k:
                self.logger.log_info(k + str(state_dict[k].shape))

        model.load_state_dict(state_dict)
        model.eval()

        normed = [[feat.to_dict()[k] for k in feature_keys] for feat in self.tokens_normalized]
        self.logger.log_info("normed execute")

        # Costruisci tensore
        inputs = (
            torch.tensor(
            normed,
            dtype=torch.float
        ).unsqueeze(1).to(device))

        # Inferenza
        with torch.no_grad():
            logits = model(inputs).squeeze(1)
            predictions = logits.argmax(-1).cpu().tolist()

        # carica il JSONL di input
        jsonl_report_resolved_dir = resolve_versioned_jsonl(self.jsonl_report_dir)
        samples = []
        with open(jsonl_report_resolved_dir, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                samples.append(rec)

        token_plus_label =  [{"text": rec["text"], "predicted_label": label2id[p]}
                             for rec, p in zip(samples, predictions)]

        # Preparo due liste
        o_rows = []
        feat_rows = []
        for item in token_plus_label:
            row = (item["text"], item["predicted_label"])
            if item["predicted_label"] == "O":
                o_rows.append(row)
            else:
                feat_rows.append(row)

        header = ["text", "predicted_label"]
        # Scrivo CSV per gli “O”
        csv_o_final_dir = str(timestamped_path(self.predicted_rumor_path))
        #csv_o = "data/inference_test/predictions/predicted_O.csv"
        with open(csv_o_final_dir, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(o_rows)

        self.logger.log_info(f"? Salvati {len(o_rows)} token “O” in: {csv_o_final_dir}")

        # Scrivo CSV per le feature
        csv_feat_final_dir = str(timestamped_path(self.predicted_feature_path))
        #csv_feat = "data/inference_test/predictions/predicted_features.csv"
        with open(csv_feat_final_dir, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(feat_rows)

        self.logger.log_info(f"? Salvate {len(feat_rows)} token di feature in: {csv_feat_final_dir}")

        # Salvataggio CSV completo O + feature
        csv_complete_final_dir = str(timestamped_path(self.predicted_complete_path))
        with open(csv_complete_final_dir, "w", encoding="utf-8", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "predicted_label"])
            writer.writeheader()
            writer.writerows(token_plus_label)

        self.logger.log_info(f"✅ Predizioni salvate anche in {csv_complete_final_dir}")


    def debugging_log(self, feature_keys: str, logits: torch.Tensor, label2id: dict):

        # Prima di normalizzare, dump dei valori “grezzi” per il primo token titolo:

        for i, feat_dict in enumerate(self.all_tokens_with_features):
            # identifica un titolo nel ground-truth o dal testo
            text = feat_dict.to_dict()["text"]
            if i == 99:  # una semplice heuristic per trovare i tuoi titoli
                print(f"\n>>> Raw features for potential TITLE token #{i}:")
                for k in ['char_entropy', 'length_zscore', 'n_words_zscore', 'tf_log', 'distinct_ratio', 'uniq_ratio', 'vowel_ratio', 'cv_balance', 'max_digit_run', 'uniq_bigram_ratio', 'neigh_len_diff', 'cap_run_norm']:
                    print(f"   {k} = {feat_dict.to_dict().get(k)}")
                break

        print("")

        # 2) Normalizzazione su TUTTI i token (eccetto 'position')
        print("Normed type:", type(self.tokens_normalized), "len:", len(self.tokens_normalized))
        print("First element:", self.tokens_normalized[99], type(self.tokens_normalized[99]))
        print("")

        for i, vec in enumerate(self.tokens_normalized):
            text = vec.to_dict()["text"]
            if i == 99:
                print(f"\n>>> Normalized features for TITLE token #{i}:")
                for k in ['char_entropy', 'length_zscore', 'n_words_zscore', 'tf_log', 'distinct_ratio', 'uniq_ratio', 'vowel_ratio', 'cv_balance', 'max_digit_run', 'uniq_bigram_ratio', 'neigh_len_diff', 'cap_run_norm']:
                    idx = feature_keys.index(k)
                    print(f"   {k} (col {idx}) = {vec.to_dict()[k]}")
                break

        print("")

        # --- INIZIO DEBUG per TOKEN TITLE ---
        # trova l’indice del tuo token “titolo”
        title_idx = None
        for i, tok in enumerate(self.all_tokens_with_features):
            if i == 99:
                title_idx = i
                break

        if title_idx is not None:
            import torch.nn.functional as F

            # Stampa raw logits
            logit_vec = logits[title_idx].cpu().tolist()
            print("")
            print(f"\n>>> Raw logits for TITLE token #{title_idx}:")
            for cls_id, logit in enumerate(logit_vec):
                label = label2id[cls_id]
                print(f"  {cls_id:2d} {label:20s}: {logit:+.4f}")

            # Stampa probabilità softmax
            probs = F.softmax(logits, dim=-1)[title_idx].cpu().tolist()
            print("")
            print(f"\n>>> Softmax probs for TITLE token #{title_idx}:")
            for cls_id, prob in enumerate(probs):
                label = label2id[cls_id]
                print(f"  {cls_id:2d} {label:20s}: {prob * 100:5.2f}%")
        else:
            print(">>> WARNING: nessun token TITLE trovato per il debug logits.")
        # --- FINE DEBUG ---
