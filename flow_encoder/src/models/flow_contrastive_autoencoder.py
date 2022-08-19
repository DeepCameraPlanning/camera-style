from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig
import torch
from torch.nn import MSELoss, TripletMarginLoss
from pytorch_lightning import LightningModule

from flow_encoder.src.models.metrics.ranking import RankingLoss
from flow_encoder.src.models.modules.i3d import make_flow_autoencoder


class I3DContrastiveAutoencoderModel(LightningModule):
    def __init__(
        self,
        pretrained_path: str,
        model_size: str,
        contrastive_mode: str,
        optimizer: str,
        margin: float,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        batch_size: int,
        config: DictConfig,
    ):
        super().__init__()

        self._config = config
        self._optimizer = optimizer
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._momentum = momentum
        self._batch_size = batch_size

        self.contrastive_mode = contrastive_mode
        if contrastive_mode == "triplet":
            self.contrastive_loss = TripletMarginLoss(margin)
        elif contrastive_mode == "ranking":
            self.contrastive_loss = RankingLoss(margin)
        self.reconstruction_loss = MSELoss()
        self.sqrt_loss_weights = torch.nn.Parameter(torch.ones(2))
        self.loss_weights = self.sqrt_loss_weights.square()

        self.model = make_flow_autoencoder(pretrained_path, model_size)

    @staticmethod
    def get_pair_labels(
        pos_feats: torch.Tensor, neg_feats: torch.Tensor
    ) -> torch.Tensor:
        """Build a binary label vector: 1 for positive sample pairs, 0 otws."""
        y = torch.cat(
            [
                torch.ones(pos_feats.shape[0], device=pos_feats.device),
                torch.zeros(neg_feats.shape[0], device=pos_feats.device),
            ]
        )
        return y

    def backward(self, loss, optimizer, idx):
        loss.backward(retain_graph=True)

    def _shared_eval_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Extract features from anchor, positive and negative flows, and compute
        the triplet loss between these outputs.
        """
        anchor_flows = batch["anchor_flows"].float()
        positive_flows = batch["positive_flows"].float()
        negative_flows = batch["negative_flows"].float()

        anchor_features, anchor_out = self.model(anchor_flows)
        positive_features, _ = self.model(positive_flows)
        negative_features, _ = self.model(negative_flows)

        if self.contrastive_mode == "triplet":
            contrastive_value = self.contrastive_loss(
                anchor_features, positive_features, negative_features
            )
        elif self.contrastive_mode == "ranking":
            anchor_features = torch.cat([anchor_features, anchor_features])
            all_features = torch.cat([positive_features, negative_features])
            labels = self.get_pair_labels(positive_features, negative_features)
            contrastive_value = self.contrastive_loss(
                anchor_features, all_features, labels
            )

        reconstruction_value = self.reconstruction_loss(
            anchor_out, anchor_flows
        )
        self.task_loss = torch.stack([contrastive_value, reconstruction_value])
        raw_loss = self.task_loss.sum()
        self.loss_weights = self.sqrt_loss_weights.square()
        self.total_loss = torch.mul(self.loss_weights, self.task_loss).sum()
        outputs = {
            "positive_clipname": batch["positive_clipname"],
            "negative_clipname": batch["negative_clipname"],
            "anchor_features": anchor_features,
            "positive_features": positive_features,
            "negative_features": negative_features,
            "gt_flows": anchor_flows,
            "pred_flows": anchor_out,
        }

        return self.total_loss, raw_loss, self.task_loss, outputs

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training loop."""
        out = self._shared_eval_step(batch, batch_idx)
        total_loss, raw_loss, task_loss, outputs = out
        contrastive_value, reconstruction_value = task_loss

        metric_dict = {
            "loss": total_loss.detach(),
            "raw_loss": raw_loss.detach(),
            "contrastive_loss": contrastive_value.detach(),
            "reconstruction_loss": reconstruction_value.detach(),
        }
        weight_dict = {
            "contrastive_weight": self.loss_weights[0].detach(),
            "reconstruction_weight": self.loss_weights[1].detach(),
        }
        flow_dict = {
            "gt_flows": outputs["gt_flows"].detach(),
            "pred_flows": outputs["pred_flows"].detach(),
        }

        return {
            "loss": total_loss,
            "metric_dict": metric_dict,
            "weight_dict": weight_dict,
            "flow_dict": flow_dict,
        }

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Validation loop."""
        out = self._shared_eval_step(batch, batch_idx)
        total_loss, raw_loss, task_loss, outputs = out
        contrastive_value, reconstruction_value = task_loss

        metric_dict = {
            "loss": total_loss,
            "raw_loss": raw_loss,
            "contrastive_loss": contrastive_value,
            "reconstruction_loss": reconstruction_value,
        }
        flow_dict = {
            "gt_flows": outputs["gt_flows"].detach(),
            "pred_flows": outputs["pred_flows"].detach(),
        }

        return {
            "loss": total_loss,
            "metric_dict": metric_dict,
            "flow_dict": flow_dict,
        }

    def test_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Test loop."""
        out = self._shared_eval_step(batch, batch_idx)
        total_loss, raw_loss, task_loss, outputs = out
        contrastive_value, reconstruction_value = task_loss

        return {
            "loss": total_loss,
            "raw_loss": raw_loss,
            "contrastive_value": contrastive_value,
            "reconstruction_value": reconstruction_value,
            "out": outputs,
        }

    def test_epoch_end(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Gather all test outputs."""
        positive_clipnames, negative_clipnames = [], []
        anchor_out, positive_out, negative_out = [], [], []
        for out in outputs:
            positive_clipnames.extend(out["out"]["positive_clipname"])
            negative_clipnames.extend(out["out"]["negative_clipname"])
            anchor_out.append(out["out"]["anchor_features"])
            positive_out.append(out["out"]["positive_features"])
            negative_out.append(out["out"]["negative_features"])
        anchor = torch.cat(anchor_out)
        positive = torch.cat(positive_out)
        negative = torch.cat(negative_out)

        self.test_outputs = {
            "positive_clipnames": positive_clipnames,
            "negative_clipnames": negative_clipnames,
            "anchor": anchor.cpu(),
            "positive": positive.cpu(),
            "negative": negative.cpu(),
        }

        return outputs

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["config"] = self._config

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
                optimizer, lr_lambda=lambda x: 1  # Identity, only to monitor
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val/loss",
        }
