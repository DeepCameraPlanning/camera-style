from typing import Any, Dict

import numpy as np
from pytorch_lightning import Callback, LightningModule, Trainer
import torch
import wandb

from utils.flow_utils import FlowUtils


class TrainingLogger(Callback):
    @staticmethod
    def _log_metrics(
        metric_dict: Dict[str, torch.Tensor],
        pl_module: LightningModule,
        mode: str,
    ):
        """ """
        on_step = True if mode == "train" else False
        for metric_name, metric_value in metric_dict.items():
            pl_module.log(
                f"{mode}/{metric_name}",
                metric_value,
                on_step=on_step,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=pl_module._batch_size,
            )

    @staticmethod
    def _log_weights(
        weight_dict: Dict[str, torch.Tensor],
        pl_module: LightningModule,
    ):
        """ """
        for weight_name, weight_value in weight_dict.items():
            pl_module.log(
                weight_name,
                weight_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
                batch_size=pl_module._batch_size,
            )

    @staticmethod
    def _log_reconstruction(
        flow_dict: Dict[str, torch.Tensor],
        trainer: Trainer,
        pl_module: LightningModule,
        mode: str,
    ):
        """Log pred and gt flow."""
        flow_utils = FlowUtils()
        epoch_index = str(pl_module.current_epoch).zfill(4)
        n_frames = min(pl_module._batch_size, 10)
        image_arrays = []
        for frame_index in range(n_frames):
            gt_frame = flow_utils.flow_to_frame(
                flow_dict["gt_flows"][frame_index, :, 0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            pred_frame = flow_utils.flow_to_frame(
                flow_dict["pred_flows"][frame_index, :, 0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            image_arrays.append(
                wandb.Image(
                    np.vstack([gt_frame, pred_frame]),
                    caption=f"[gt/pred] epoch: {epoch_index}",
                )
            )

        trainer.logger.experiment.log(
            {f"reconstruction/{mode}": image_arrays}, commit=False
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
    ):
        # Log metrics
        self._log_metrics(outputs["metric_dict"], pl_module, "train")
        if "weight_dict" in outputs:
            self._log_weights(outputs["weight_dict"], pl_module)

        # Log reconstructed flows every 5 epochs
        if batch_idx == 0 and pl_module.current_epoch % 5 == 0:
            self._log_reconstruction(
                outputs["flow_dict"], trainer, pl_module, "train"
            )

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ):
        # Skip sanity check steps
        if trainer.sanity_checking:
            return

        # Log metrics
        self._log_metrics(outputs["metric_dict"], pl_module, "val")

        # Log reconstructed flows every 5 epochs
        if batch_idx == 0 and pl_module.current_epoch % 5 == 0:
            self._log_reconstruction(
                outputs["flow_dict"], trainer, pl_module, "val"
            )
