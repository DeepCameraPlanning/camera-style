from typing import Any, Dict, List, Tuple

from omegaconf.dictconfig import DictConfig
import torch
from torch.nn import TripletMarginLoss
from pytorch_lightning import LightningModule
import torchmetrics

from src.models.modules.i3d import make_flow_i3d


class TripletI3DModel(LightningModule):
    def __init__(
        self,
        pretrained_path: str,
        model_config: DictConfig,
        num_classes: int,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
    ):
        super().__init__()

        self._optimizer = optimizer
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._momentum = momentum
        self.criterion = TripletMarginLoss()

        self._model_config = model_config
        self._num_classes = num_classes
        self.model = make_flow_i3d(pretrained_path)

        metric_params = {"num_classes": 1, "multiclass": False}
        self.train_precision = torchmetrics.Precision(**metric_params)
        self.val_precision = torchmetrics.Precision(**metric_params)
        self.test_precision = torchmetrics.Precision(**metric_params)
        self.train_recall = torchmetrics.Recall(**metric_params)
        self.val_recall = torchmetrics.Recall(**metric_params)
        self.test_recall = torchmetrics.Recall(**metric_params)
        self.train_f1 = torchmetrics.F1(**metric_params)
        self.val_f1 = torchmetrics.F1(**metric_params)
        self.test_f1 = torchmetrics.F1(**metric_params)
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.test_confusion = torchmetrics.ConfusionMatrix(num_classes=2)

    def _shared_log_step(
        self,
        mode: str,
        loss: torch.Tensor,
        accuracy: torch.Tensor,
        precision: torch.Tensor,
        recall: torch.Tensor,
        f1: torch.Tensor,
    ):
        """Log metrics at each epoch and each step for the training."""
        on_step = True if mode == "train" else False
        self.log(
            f"{mode}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{mode}/accuracy",
            accuracy,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{mode}/precision",
            precision,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            f"{mode}/recall",
            recall,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{mode}/f1",
            f1,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

    def _shared_eval_step(self, batch: torch.Tensor, batch_idx: int):
        """ """
        anchor_flows = batch["anchor_flows"].float()
        positive_flows = batch["positive_flows"].float()
        negative_flows = batch["negative_flows"].float()

        anchor_out = self.model.extract_features(anchor_flows)
        positive_out = self.model.extract_features(positive_flows)
        negative_out = self.model.extract_features(negative_flows)

        loss = self.criterion(anchor_out, positive_out, negative_out)

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training loop."""
        loss, preds, targets = self._shared_eval_step(batch, batch_idx)
        accuracy = self.train_accuracy(preds, targets)
        precision = self.train_precision(preds, targets)
        recall = self.train_recall(preds, targets)
        f1 = self.train_f1(preds, targets)
        self._shared_log_step("train", loss, accuracy, precision, recall, f1)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Validation loop."""
        loss, preds, targets = self._shared_eval_step(batch, batch_idx)
        accuracy = self.val_accuracy(preds, targets)
        precision = self.val_precision(preds, targets)
        recall = self.val_recall(preds, targets)
        f1 = self.val_f1(preds, targets)
        self._shared_log_step("val", loss, accuracy, precision, recall, f1)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Test loop."""
        video_name = batch["video_name"]
        is_cut = batch["is_cut"]
        loss, preds, targets = self._shared_eval_step(batch, batch_idx)

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "video_name": video_name,
            "is_cut": is_cut,
        }

    def test_epoch_end(self, outputs: torch.Tensor):
        """Compute metrics over all testing samples"""
        preds, targets, is_cuts = [], [], []
        for out in outputs:
            preds.append(out["preds"])
            targets.append(out["targets"])
            is_cuts.append(out["is_cut"])
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        is_cuts = torch.cat(is_cuts)

        accuracy = self.test_accuracy(preds, targets)
        precision = self.test_precision(preds, targets)
        recall = self.test_recall(preds, targets)
        f1 = self.test_f1(preds, targets)
        confusion_matrix = self.test_confusion(preds, targets)

        self.test_outputs = {
            "preds": preds.cpu().tolist(),
            "targets": targets.cpu().tolist(),
            "is_cuts": is_cuts.cpu().tolist(),
            "accuracy": accuracy.cpu().tolist(),
            "precision": precision.cpu().tolist(),
            "recall": recall.cpu().tolist(),
            "f1": f1.cpu().tolist(),
            "confusion_matrix": confusion_matrix.cpu().tolist(),
            "n_samples": targets.shape[0],
            "n_positives": targets.sum().cpu().tolist(),
        }

        return outputs

    def configure_optimizers(self) -> Tuple[List[Any], List[Any]]:
        """Define optimizers and LR schedulers."""
        if self._optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                weight_decay=self._weight_decay,
                momentum=self._momentum,
                lr=self._lr,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", 0.1, verbose=False
            )

        if self._optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), self._lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda x: x  # Identity, only to monitor
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }
