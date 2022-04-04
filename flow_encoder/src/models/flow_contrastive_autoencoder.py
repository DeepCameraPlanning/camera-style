import os.path as osp
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss, TripletMarginLoss, L1Loss
from pytorch_lightning import LightningModule

from flow_encoder.src.models.modules.i3d import make_flow_autoencoder
from utils.flow_utils import FlowUtils
from utils.file_utils import create_dir


class I3DContrastiveAutoencoderModel(LightningModule):
    def __init__(
        self,
        pretrained_path: str,
        check_dir: str,
        histogram: bool,
        triplet_coef: float,
        reconstruction_coef: float,
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

        self._triplet_coef = triplet_coef
        self.triplet_loss = TripletMarginLoss()
        self._reconstruction_coef = reconstruction_coef
        self.reconstruction_loss = MSELoss()
        # self.reconstruction_loss = L1Loss()

        self.model = make_flow_autoencoder(pretrained_path)

        self._check_dir = check_dir
        self.flow_utils = FlowUtils()

    def _shared_log_step(
        self,
        mode: str,
        total_loss: torch.Tensor,
        triplet_loss: torch.Tensor,
        reconstruction_loss: torch.Tensor,
    ):
        """Log metrics at each epoch and each step for the training."""
        on_step = True if mode == "train" else False
        self.log(
            f"{mode}/loss",
            total_loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self._batch_size,
        )
        self.log(
            f"{mode}/triplet_loss",
            triplet_loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self._batch_size,
        )
        self.log(
            f"{mode}/reconstruction_loss",
            reconstruction_loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self._batch_size,
        )

    def _log_check(self, gt_flows: torch.Tensor, pred_flows: torch.Tensor):
        """Log pred and gt flow."""
        n_frames = min(self._batch_size, 10)
        current_check_dir = osp.join(
            self._check_dir, "epoch-" + str(self.current_epoch).zfill(4)
        )
        create_dir(current_check_dir)
        for frame_index in range(n_frames):
            gt_check_path = osp.join(
                current_check_dir, str(frame_index).zfill(3) + "_gt.png"
            )
            gt_frame = self.flow_utils.flow_to_frame(
                gt_flows[frame_index, :, 0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            plt.imsave(gt_check_path, gt_frame)
            pred_check_path = osp.join(
                current_check_dir, str(frame_index).zfill(3) + "_pred.png"
            )
            pred_frame = self.flow_utils.flow_to_frame(
                pred_flows[frame_index, :, 0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            plt.imsave(pred_check_path, pred_frame)

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

        anchor_features, anchor_out = self.model(anchor_flows)
        positive_features, _ = self.model(positive_flows)
        negative_features, _ = self.model(negative_flows)

        triplet_value = self._triplet_coef * self.triplet_loss(
            anchor_features, positive_features, negative_features
        )
        reconstruction_value = (
            self._reconstruction_coef
            * self.reconstruction_loss(anchor_out, anchor_flows)
        )
        total_loss = triplet_value + reconstruction_value
        outputs = {
            "positive_clipname": batch["positive_clipname"],
            "negative_clipname": batch["negative_clipname"],
            "anchor_features": anchor_features,
            "positive_features": positive_features,
            "negative_features": negative_features,
            "gt_flow": anchor_flows,
            "pred_flow": anchor_out,
        }

        if (
            batch_idx == 0
            and self._check_dir is not None
            and self.current_epoch % 5 == 0
        ):
            self._log_check(anchor_flows, anchor_out)

        return total_loss, triplet_value, reconstruction_value, outputs

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training loop."""
        out = self._shared_eval_step(batch, batch_idx)
        total_loss, triplet_value, reconstruction_value, _ = out
        self._shared_log_step(
            "train", total_loss, triplet_value, reconstruction_value
        )

        return {"loss": total_loss}

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Validation loop."""
        out = self._shared_eval_step(batch, batch_idx)
        total_loss, triplet_value, reconstruction_value, _ = out
        self._shared_log_step(
            "val", total_loss, triplet_value, reconstruction_value
        )

        return {"loss": total_loss}

    def test_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Test loop."""
        out = self._shared_eval_step(batch, batch_idx)
        total_loss, triplet_value, reconstruction_value, outputs = out

        return {
            "loss": total_loss,
            "triplet_value": triplet_value,
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
