import os.path as osp
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss  # , L1Loss()
from pytorch_lightning import LightningModule

from flow_encoder.src.models.modules.i3d import make_flow_autoencoder
from utils.flow_utils import FlowUtils
from utils.file_utils import create_dir


class I3DAutoencoderModel(LightningModule):
    def __init__(
        self,
        pretrained_path: str,
        model_size: str,
        flow_type: str,
        check_dir: str,
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
        self._flow_type = flow_type

        # self.loss = L1Loss()
        self.loss = MSELoss()

        self.model = make_flow_autoencoder(pretrained_path, model_size)

        self._check_dir = check_dir
        self.flow_utils = FlowUtils()

    def _shared_log_step(
        self,
        mode: str,
        loss: torch.Tensor,
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
        """ """
        input_flows = batch[f"{self._flow_type}_flows"].float()
        _, out_flows = self.model(input_flows)

        loss = self.loss(out_flows, input_flows)
        outputs = {
            "clipname": batch[f"{self._flow_type}_flows"],
            "gt_flow": input_flows,
            "pred_flow": out_flows,
        }

        if (
            batch_idx == 0
            and self._check_dir is not None
            and self.current_epoch % 5 == 0
        ):
            self._log_check(input_flows, out_flows)

        return loss, outputs

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training loop."""
        out = self._shared_eval_step(batch, batch_idx)
        loss, _ = out
        self._shared_log_step("train", loss)

        return {"loss": loss}

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Validation loop."""
        out = self._shared_eval_step(batch, batch_idx)
        loss, _ = out
        self._shared_log_step("val", loss)

        return {"loss": loss}

    def test_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Test loop."""
        loss, outputs = self._shared_eval_step(batch, batch_idx)

        return {
            "loss": loss,
            "out": outputs,
        }

    def test_epoch_end(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Gather all test outputs."""
        clipnames = []
        input_flows, out_flows = [], []
        for out in outputs:
            clipnames.extend(out["out"]["clipname"])
            input_flows.append(out["out"]["input_flows"])
            out_flows.append(out["out"]["out_flows"])

        input_flows = torch.cat(input_flows)
        out_flows = torch.cat(out_flows)

        self.test_outputs = {
            "clipnames": clipnames,
            "input_flows": input_flows.cpu(),
            "out_flows": out_flows.cpu(),
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
