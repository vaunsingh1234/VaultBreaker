import copy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from vaultbreaker.utils.logger import get_logger
from vaultbreaker.utils.io import ensure_dir
from vaultbreaker.models.unified_net import VaultBreakerUnifiedNet
from vaultbreaker.training.dataset import MultiModalStegoDataset, BalancedFormatBatchSampler

logger = get_logger("Trainer")

class UnifiedTrainer:
    """
    Manages joint multi-modal training of VaultBreakerUnifiedNet with:
    - Format-balanced batching
    - AdamW optimizer and CosineAnnealingLR schedule
    - Multi-task binary BCE + auxiliary method CrossEntropyLoss
    - Early stopping tracking validation ROC-AUC
    """
    def __init__(
        self,
        model: VaultBreakerUnifiedNet,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        aux_loss_weight: float = 0.3,
        device: Optional[torch.device] = None
    ):
        self.device = device or (
            torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = model.to(self.device)
        self.aux_loss_weight = aux_loss_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def train_epoch(self, dataset: MultiModalStegoDataset, batch_size: int = 32) -> float:
        """Run one training epoch across formats."""
        self.model.train()
        sampler = BalancedFormatBatchSampler(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0.0
        n_batches = 0

        for batch_indices in sampler:
            # Group batch items by media_type
            batch_items = [dataset[i] for i in batch_indices]
            by_format: Dict[str, List[Dict]] = {}
            for item in batch_items:
                m_type = item["media_type"]
                by_format.setdefault(m_type, []).append(item)

            batch_loss = 0.0
            self.optimizer.zero_grad()

            for m_type, items in by_format.items():
                x = torch.stack([it["x"] for it in items]).to(self.device)
                y = torch.stack([it["y"] for it in items]).to(self.device).unsqueeze(1)
                m_idx = torch.stack([it["method_idx"] for it in items]).to(self.device)

                is_single = (x.size(0) == 1)
                x_in = torch.cat([x, x], dim=0) if is_single else x

                logits, aux_logits, _ = self.model(x_in, m_type)
                if is_single:
                    logits = logits[:1]
                    if aux_logits is not None:
                        aux_logits = aux_logits[:1]
                loss_binary = self.bce_loss(logits, y)

                if self.model.aux_head is not None and aux_logits is not None:
                    loss_aux = self.ce_loss(aux_logits, m_idx)
                    sub_loss = loss_binary + self.aux_loss_weight * loss_aux
                else:
                    sub_loss = loss_binary

                batch_loss += sub_loss * (len(items) / len(batch_items))

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += batch_loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def evaluate(
        self,
        dataset: MultiModalStegoDataset
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model on a dataset.
        Returns overall metrics, logits array, and ground truth labels.
        """
        self.model.eval()
        all_logits = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for item in dataset.samples:
                x = item["x"].to(self.device)
                m_type = item["media_type"]
                label = item["y"].item()

                logit, _, _ = self.model(x, m_type)
                logit_val = logit.item()
                prob = torch.sigmoid(torch.tensor(logit_val)).item()

                all_logits.append(logit_val)
                all_probs.append(prob)
                all_labels.append(label)

        y_true = np.array(all_labels, dtype=int)
        y_prob = np.array(all_probs, dtype=float)
        y_logits = np.array(all_logits, dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.5

        metrics = {"accuracy": acc, "f1": f1, "roc_auc": auc}
        return metrics, y_logits, y_true

    def fit(
        self,
        train_ds: MultiModalStegoDataset,
        val_ds: MultiModalStegoDataset,
        epochs: int = 30,
        batch_size: int = 32,
        early_stopping_patience: int = 6,
        save_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Complete training loop with CosineAnnealingLR and early stopping."""
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        best_auc = -1.0
        best_state = None
        patience_counter = 0
        history = []

        logger.info(f"Starting joint multi-modal training on device: {self.device} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_ds, batch_size=batch_size)
            scheduler.step()

            val_metrics, val_logits, val_labels = self.evaluate(val_ds)
            val_auc = val_metrics["roc_auc"]
            val_acc = val_metrics["accuracy"]

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_auc": val_auc
            })

            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (best val AUC: {best_auc:.4f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if save_path:
            ensure_dir(save_path.parent)
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"Saved best model weights to {save_path}")

        # Final evaluation on val with best model
        final_metrics, final_logits, final_labels = self.evaluate(val_ds)
        return {
            "best_metrics": final_metrics,
            "val_logits": final_logits,
            "val_labels": final_labels,
            "history": history
        }
