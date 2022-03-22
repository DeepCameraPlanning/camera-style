from typing import Any, Dict, List, Tuple

from omegaconf.dictconfig import DictConfig
import torch
from torch.nn import TripletMarginLoss
from pytorch_lightning import LightningModule

from flow_encoder.src.models.modules.i3d import make_flow_i3d


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
        batch_size: int,
    ):
        super().__init__()

        self._optimizer = optimizer
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._momentum = momentum
        self._batch_size = batch_size
        self.criterion = TripletMarginLoss()

        self._model_config = model_config
        self._num_classes = num_classes
        self.model = make_flow_i3d(pretrained_path)

    def _shared_log_step(self, mode: str, loss: torch.Tensor):
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
            batch_size=self._batch_size,
        )

    def _shared_eval_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract features from anchor, positive and negative flows, and compute
        the triplet loss between these outputs.
        """
        anchor_flows = batch["anchor_flows"].float()
        positive_flows = batch["positive_flows"].float()
        negative_flows = batch["negative_flows"].float()

        anchor_out = self.model.extract_features(anchor_flows)
        positive_out = self.model.extract_features(positive_flows)
        negative_out = self.model.extract_features(negative_flows)

        loss = self.criterion(anchor_out, positive_out, negative_out)

        outputs = {
            "positive_clipname": batch["positive_clipname"],
            "negative_clipname": batch["negative_clipname"],
            "anchor": anchor_out,
            "positive": positive_out,
            "negative": negative_out,
        }

        return loss, outputs

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training loop."""
        loss, _ = self._shared_eval_step(batch, batch_idx)
        self._shared_log_step("train", loss)

        return {"loss": loss}

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Validation loop."""
        loss, _ = self._shared_eval_step(batch, batch_idx)
        self._shared_log_step("val", loss)

        return {"loss": loss}

    def test_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Test loop."""
        loss, outputs = self._shared_eval_step(batch, batch_idx)

        return {"loss": loss, "out": outputs}

    def test_epoch_end(self, outputs: torch.Tensor):
        """Gather all test outputs."""
        positive_clipnames, negative_clipnames = [], []
        anchor_out, positive_out, negative_out = [], [], []
        for out in outputs:
            positive_clipnames.extend(out["out"]["positive_clipname"])
            negative_clipnames.extend(out["out"]["negative_clipname"])
            anchor_out.append(out["out"]["anchor"])
            positive_out.append(out["out"]["positive"])
            negative_out.append(out["out"]["negative"])
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
