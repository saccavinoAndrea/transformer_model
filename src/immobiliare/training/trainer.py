import csv
import json
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split

from core_interfaces.trainer.itrainer import ITrainer
from dataset.collate import html_collate_fn
from dataset.html_token_dataset import HTMLTokenDataset
from models.transformer_pytorch import TransformerForTokenClassification
from utils import timestamped_path
from utils.logging.logger_factory import LoggerFactory


class Trainer(ITrainer):

    def __init__(
        self,
        jsonl_path: str,
        label2id: dict,
        feature_keys_to_use: List[str],
        batch_size: int = 32,
        lr: float = 1e-4,
        patience: int = 20,
        val_split: float = None,
        model_save_path: str = None,
        report_dir: Path = None,
        artifact_dir: Path = None,
        max_epochs: int = None,
    ):
        self.logger = LoggerFactory.get_logger("training_step")
        self.jsonl_path = jsonl_path
        self.label2id = label2id
        self.feature_keys_to_use = feature_keys_to_use
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.val_split = val_split
        self.max_epochs = max_epochs
        self.report_dir = report_dir
        self.artifact_dir = artifact_dir
        self.model_save_path = Path(model_save_path) if model_save_path else None

        # per raccogliere metriche di training e validation
        self.train_losses = []
        self.val_losses   = []
        self.val_f1s      = []

    def run(self):
        self.train()

    def train(self):
        self.logger.log_info("Start training ...")
        start_time = time.time()  # ← inizio cronometro

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #ds = HTMLTokenDataset()
        #label2id = ds.label2id

        # 1) carico l'intero dataset
        self.logger.log_info(f" feature_keys_to_use length: {len(self.feature_keys_to_use)}")
        full_ds = HTMLTokenDataset(label2id=self.label2id, feature_keys_to_use=self.feature_keys_to_use)
        full_ds.retrieve_labeled_samples(self.jsonl_path)

        # 🚀 Salvo anche le mappe label<->id
        #self.label2id = label2id

        total = len(full_ds)
        n_val = int(total * self.val_split)
        n_train = total - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        """
        # --- OVERSAMPLING SETUP --------------------------------------
        # costruiamo un elenco di label_id per ogni sample in train_ds
        train_labels = [train_ds.dataset.samples[i]["label"] for i in train_ds.indices]
        train_label_ids = [label2id[l] for l in train_labels]

        # conteggi per classe
        counts = torch.bincount(torch.tensor(train_label_ids, dtype=torch.long))
        # pesi per sample = inverso alla frequenza della sua classe
        sample_weights = [1.0 / counts[label_id].item() for label_id in train_label_ids]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        # ------------------------------------------------------------

        # 2) costruisco i DataLoader
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=html_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=html_collate_fn,
        )

        # 3) modello: input_dim = numero di feature selezionate!
        input_dim = len(full_ds.feature_keys_to_use)
        print("NUMERO DI FEATURE SELEZIONATE ", input_dim)
        num_classes = len(label2id)
        print("NUMERO DI CLASSI SELEZIONATE ", num_classes)
        # tieni embedding_dim multiplo di nhead
        model = TransformerForTokenClassification(
            input_dim=input_dim,
            embedding_dim=64,
            num_classes=num_classes,
            nhead=8
        ).to(device)

        # 4) class weights per il loss (opzionale ma consigliato anche con oversampling)

        # estraiamo gli id (interi) delle label dai sample di full_ds
        full_label_ids = [label2id[s["label"]] for s in full_ds.samples]
        full_counts = torch.bincount(torch.tensor(full_label_ids, dtype=torch.long))

            # peso inverso alla frequenza, normalizzato per num_classi
        class_weights = full_counts.sum() / (full_counts * full_counts.numel())
        criterion = CrossEntropyLoss(weight=class_weights.to(device))
        """

        seed = 12345
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # --- OVERSAMPLING SETUP --------------------------------------
        train_labels = [train_ds.dataset.samples[i]["label"] for i in train_ds.indices]
        #train_label_ids = [label2id[l] for l in train_labels]
        train_label_ids = [self.label2id[l] for l in train_labels]

        counts = torch.bincount(torch.tensor(train_label_ids, dtype=torch.long))

        # PESI PER LOSS
        class_weights = counts.sum() / (counts + 1e-6)
        class_weights = class_weights / class_weights.mean()
        class_weights = torch.clamp(class_weights, max=25.0)

        # PESI PER SAMPLER
        sample_weights = [1.0 / counts[label_id].item() for label_id in train_label_ids]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        # ------------------------------------------------------------

        # DataLoader (con il sampler)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            sampler=sampler,  # <- cambia shuffle=True con sampler!
            collate_fn=html_collate_fn,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=html_collate_fn,
        )

        # modello
        input_dim = len(self.feature_keys_to_use)
        #num_classes = len(label2id)
        num_classes = len(self.label2id)
        model = TransformerForTokenClassification(
            input_dim=input_dim,
            embedding_dim=64,
            num_classes=num_classes,
            nhead=8
        ).to(device)

        # criterio
        criterion = CrossEntropyLoss(weight=class_weights.to(device))

        # 5) optimizer + scheduler
        optimizer = Adam(model.parameters(), lr=self.lr, weight_decay=1e-5)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )

        self.artifact_dir.mkdir(exist_ok=True)

        model_save_final_path = timestamped_path(self.model_save_path)

        best_val_f1 = 0.0
        best_epoch_f1 = 0
        patience_counter = 0
        epoch = 0

        while True:
            epoch += 1
            # — TRAIN —
            model.train()
            total_train_loss = 0
            for feats, labels in train_loader:
                feats, labels = feats.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(feats.unsqueeze(1)).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            avg_train = total_train_loss / len(train_loader)
            self.train_losses.append(avg_train)

            # — VALIDATION —
            model.eval()
            total_val_loss = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats, labels = feats.to(device), labels.to(device)
                    logits = model(feats.unsqueeze(1)).squeeze(1)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()
                    preds = logits.argmax(dim=-1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())
            avg_val = total_val_loss / len(val_loader)
            self.val_losses.append(avg_val)

            f1_macro = f1_score(all_labels, all_preds, average='macro')
            self.val_f1s.append(f1_macro)

            # Scheduler su validation loss
            scheduler.step(avg_val)

            # — stampa e report —
            print(f"Epoch {epoch} — epoch"
                  f"train_loss: {avg_train:.4f}  "
                  f"val_loss: {avg_val:.4f}  "
                  f"val_F1: {f1_macro:.4f}")

            # 🚀 Salvo anche le mappe label<->id
            id2label = {v: k for k, v in self.label2id.items()}

            # salva report di dettaglio per questo epoch
            # 1) prendi la lista completa di id di classe
            labels = list(sorted(id2label.keys()))
            # 2) costruisci i nomi nella stessa esatta posizione
            target_names = [id2label[i] for i in labels]

            # 3) ora chiama classification_report passando entrambi
            rep = classification_report(
                all_labels,
                all_preds,
                labels=labels,
                target_names=target_names,
                output_dict=True,
                zero_division=0  # evita warning su classi non presenti in predizioni
            )

            # --- 1) Salvo lo stesso report in CSV ---
            # Trasformo il dict in DataFrame (righe=classi e metrica)
            df_rep = pd.DataFrame(rep).T
            # (opzionale: riordino le colonne)
            df_rep = df_rep[["precision", "recall", "f1-score", "support"]]

            csv_final_path = timestamped_path(self.report_dir / f"report_epoch_{epoch}.csv")
            #csv_path = self.report_dir / f"report_epoch_{epoch}.csv"
            df_rep.to_csv(csv_final_path, index=True, float_format="%.4f")

            # calcolo f1_macro sui dati di validazione
            f1_macro = f1_score(all_labels, all_preds, average='macro')

            # --- Early‑stopping & checkpointing basato su F1‑macro ---
            if f1_macro > best_val_f1:
                best_val_f1 = f1_macro
                best_epoch_f1 = epoch
                patience_counter = 0

                if model_save_final_path:
                    torch.save(model.state_dict(), str(model_save_final_path))
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.log_info(f"🔹 Early stopping triggered at epoch {epoch}.")
                    break

            # Safety cap sul numero massimo di epoche
            if epoch >= self.max_epochs:
                self.logger.log_info(f"⏹ Reached max_epochs={self.max_epochs}, stopping.")
                break

        self.logger.log_info(f"   Best model (max F1) era epoch {best_epoch_f1} con val_F1={best_val_f1:.4f}")

        # Report per classe
        print("🔍 Classification Report per classe:\n")
        print(classification_report(
            all_labels,
            all_preds,
            labels=labels,
            target_names=target_names,
            digits=3,
            zero_division=0
        ))

        # alla fine del training, salvo di nuovo il best model
        if model_save_final_path:
            torch.save(model.state_dict(), str(model_save_final_path))
            self.logger.log_info(f"Model saved to {model_save_final_path}")

        # 2) dopo il loop, genero i grafici
        self._plot_metrics()

        # 3) ed esporto tutte le metriche in un CSV
        self._export_metrics_csv()


        # 2) salva la mappa tg label2id
        #label2id_final_path = timestamped_path(self.artifact_dir / "label2id.json")
        #with open(label2id_final_path, "w", encoding="utf-8") as f:
        #    json.dump(self.label2id, f, ensure_ascii=False, indent=2)

        #self.logger.log_info("✅ Artifacts salvati in " + """str(feature_keys_final_path)""" + " and " +  str(label2id_final_path))

        total_time = time.time() - start_time
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        self.logger.log_info(f"\n⏱️  Tempo totale di training: {int(h)}h {int(m)}m {int(s)}s")

    def _plot_metrics(self):
        """Disegna curves di loss e F1‐macro e salva le figure."""
        epochs = list(range(1, len(self.train_losses) + 1))

        # Loss
        loss_curve_final_path = timestamped_path(self.report_dir / "loss_curve.png")

        plt.figure()
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses,   label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.savefig(loss_curve_final_path)

        # F1‐macro
        f1_macro_final_path = timestamped_path(self.report_dir / "f1_macro_curve.png")

        plt.figure()
        plt.plot(epochs, self.val_f1s, label="Val F1‐macro")
        plt.xlabel("Epoch")
        plt.ylabel("F1‐macro")
        plt.title("Validation F1‐macro")
        plt.savefig(f1_macro_final_path)
        plt.close("all")

    def _export_metrics_csv(self):
        """Esporta train/val loss e val F1‐macro in un CSV."""
        csv_final_path = timestamped_path(self.report_dir / "training_metrics.csv")
        #csv_path = self.report_dir / "training_metrics.csv"
        with open(csv_final_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_f1_macro"])
            for e, (tl, vl, f1) in enumerate(zip(
                    self.train_losses,
                    self.val_losses,
                    self.val_f1s
                ), start=1):
                writer.writerow([e, f"{tl:.6f}", f"{vl:.6f}", f"{f1:.6f}"])

        self.logger.log_info("✅ Metrics CSV salvato: " + str(csv_final_path))
