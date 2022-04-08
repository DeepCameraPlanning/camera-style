from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from omegaconf.dictconfig import DictConfig
import torch
from torch.nn import SmoothL1Loss, MSELoss
from torchvision.utils import draw_bounding_boxes
from pytorch_lightning import LightningModule

from character_encoder.src.models.metrics.global_iou import GIOULoss
from character_encoder.src.models.metrics.iou import IOULoss
from character_encoder.src.models.modules.perceiver import make_latent_ca
from flow_encoder.src.models.flow_contrastive_autoencoder import (
    I3DContrastiveAutoencoderModel,
)
from utils.file_utils import create_dir
from utils.flow_utils import FlowUtils


class LatentCharacterModel(LightningModule):
    def __init__(
        self,
        autoencoder_ckpt_path: str,
        model_config: DictConfig,
        bbox_coef: float,
        reconstruction_coef: float,
        optimizer: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        batch_size: int,
        check_dir: str,
    ):
        super().__init__()

        self._optimizer = optimizer
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._momentum = momentum
        self._batch_size = batch_size

        self._bbox_coef = bbox_coef
        self.bbox_loss = SmoothL1Loss(reduction="mean")
        # self.bbox_loss = GIOULoss()
        self._reconstruction_coef = reconstruction_coef
        self.reconstruction_loss = MSELoss(reduction="mean")

        self.giou = GIOULoss()
        self.iou = IOULoss()
        self.l1 = SmoothL1Loss(reduction="mean")

        self._model_config = model_config
        self.model = make_latent_ca(**model_config)
        self.autoencoder = I3DContrastiveAutoencoderModel.load_from_checkpoint(
            autoencoder_ckpt_path,
            pretrained_path=None,
            check_dir="",
            histogram=False,
            triplet_coef=0,
            reconstruction_coef=0,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
            batch_size=batch_size,
        ).model

        self._check_dir = check_dir
        self._flow_utils = FlowUtils()

    @staticmethod
    def _annotate_bbox(
        batch: Dict[str, Any],
        frame_index: int,
        target_bbox: torch.Tensor,
        pred_bbox: torch.Tensor,
        save_path: str,
    ) -> np.array:
        """Annotate and save target and predicted bbox on the given frame."""
        frame = batch["target_frame"][frame_index].permute(2, 0, 1).cpu()

        # Scale bboxes
        _, height, width = frame.shape
        cpu_target_bbox = target_bbox[frame_index].cpu()
        # cpu_target_bbox[[0, 2]] *= width
        # cpu_target_bbox[[1, 3]] *= height
        cpu_target_bbox = cpu_target_bbox[None]
        cpu_pred_bbox = pred_bbox[frame_index].cpu()
        # cpu_pred_bbox[[0, 2]] *= width
        # cpu_pred_bbox[[1, 3]] *= height
        cpu_pred_bbox = cpu_pred_bbox[None]
        # Draw bboxes
        frame = draw_bounding_boxes(
            frame, cpu_target_bbox, colors="green", width=4
        )
        frame = draw_bounding_boxes(
            frame, cpu_pred_bbox, colors="red", width=4
        )
        # Save frame
        plt.imsave(save_path, frame.permute(1, 2, 0).numpy())

    def _log_bbox_check(
        self,
        batch: torch.Tensor,
        target_bbox: torch.Tensor,
        pred_bbox: torch.Tensor,
    ):
        """Log pred and gt bounding-boxes."""
        n_frames = min(self._batch_size, 10)
        current_check_dir = osp.join(
            self._check_dir, "epoch-" + str(self.current_epoch).zfill(4)
        )
        create_dir(current_check_dir)
        for frame_index in range(n_frames):
            check_path = osp.join(
                current_check_dir, str(frame_index).zfill(3) + "_bbox.png"
            )
            self._annotate_bbox(
                batch,
                frame_index,
                target_bbox * 224,
                pred_bbox * 224,
                check_path,
            )

    def _log_flow_check(
        self, target_flows: torch.Tensor, pred_flows: torch.Tensor
    ):
        """Log pred and gt flow."""
        n_frames = min(self._batch_size, 10)
        current_check_dir = osp.join(
            self._check_dir, "epoch-" + str(self.current_epoch).zfill(4)
        )
        create_dir(current_check_dir)
        for frame_index in range(n_frames):
            gt_check_path = osp.join(
                current_check_dir,
                str(frame_index).zfill(3) + "_flow_target.png",
            )
            gt_frame = self._flow_utils.flow_to_frame(
                target_flows[frame_index, :, 0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            plt.imsave(gt_check_path, gt_frame)
            pred_check_path = osp.join(
                current_check_dir, str(frame_index).zfill(3) + "_flow_pred.png"
            )
            pred_frame = self._flow_utils.flow_to_frame(
                pred_flows[frame_index, :, 0]
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            plt.imsave(pred_check_path, pred_frame)

    def _shared_log_step(
        self,
        mode: str,
        loss: torch.Tensor,
        iou: torch.Tensor,
        giou: torch.Tensor,
        l1: torch.Tensor,
        reconstruction: torch.Tensor,
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
        self.log(
            f"{mode}/iou",
            iou,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self._batch_size,
        )
        self.log(
            f"{mode}/giou",
            giou,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self._batch_size,
        )
        self.log(
            f"{mode}/l1",
            l1,
            on_step=on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=self._batch_size,
        )
        self.log(
            f"{mode}/reconstruction",
            reconstruction,
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
        """Input flow feature and character mask, output bbox coordinates."""
        input_flows = batch["input_flows"]
        input_character_mask = batch["input_character_mask"]
        target_bbox = batch["target_bbox"]

        input_feature = self.autoencoder.encoder.extract_features(
            input_flows, avg_out=False
        )
        import ipdb

        ipdb.set_trace()
        out_feature, pred_bbox = self.model(
            input_feature.view(input_feature.shape[0], -1),
            input_character_mask,
        )
        pred_flows = self.autoencoder.decoder(
            out_feature.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        target_bbox = target_bbox.type(pred_bbox.dtype)
        bbox_loss = self.bbox_loss(pred_bbox, target_bbox)
        reconstruction_loss = self.reconstruction_loss(input_flows, pred_flows)
        loss = (
            self._bbox_coef * bbox_loss
            + self._reconstruction_coef * reconstruction_loss
        )

        iou = self.iou(pred_bbox, target_bbox)
        giou = self.giou(pred_bbox, target_bbox)
        l1 = self.l1(pred_bbox, target_bbox)

        outputs = {
            "clip_name": batch["clip_name"],
            "keyframe_indices": batch["keyframe_indices"],
            "input_feature": input_feature,
            "input_character_mask": input_character_mask,
            "target_bbox": target_bbox,
            "pred_bbox": pred_bbox,
            "iou": iou,
            "giou": giou,
            "l1": l1,
            "reconstruction": reconstruction_loss,
        }

        if (
            batch_idx == 0
            and self._check_dir is not None
            and self.current_epoch % 5 == 0
        ):
            self._log_bbox_check(batch, target_bbox, pred_bbox)
            self._log_flow_check(input_flows, pred_flows)

        return loss, outputs

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training loop."""
        loss, outputs = self._shared_eval_step(batch, batch_idx)
        self._shared_log_step(
            "train",
            loss,
            outputs["iou"],
            outputs["giou"],
            outputs["l1"],
            outputs["reconstruction"],
        )

        return {"loss": loss}

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Validation loop."""
        loss, outputs = self._shared_eval_step(batch, batch_idx)
        self._shared_log_step(
            "val",
            loss,
            outputs["iou"],
            outputs["giou"],
            outputs["l1"],
            outputs["reconstruction"],
        )

        return {"loss": loss}

    def test_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Test loop."""
        loss, outputs = self._shared_eval_step(batch, batch_idx)

        return {"loss": loss, "out": outputs}

    def test_epoch_end(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Gather all test outputs."""
        clip_names, keyframe_indices = [], []
        input_features, input_character_masks = [], []
        target_bboxes, pred_bboxes = [], []
        for out in outputs:
            clip_names.extend(out["out"]["clip_name"])
            keyframe_indices.extend(out["out"]["keyframe_indices"])
            input_features.append(out["out"]["input_feature"])
            input_character_masks.append(out["out"]["input_character_mask"])
            target_bboxes.append(out["out"]["target_bbox"])
            pred_bboxes.append(out["out"]["pred_bbox"])
        input_features = torch.cat(input_features)
        input_character_masks = torch.cat(input_character_masks)
        target_bboxes = torch.cat(target_bboxes)
        pred_bboxes = torch.cat(pred_bboxes)

        self.test_outputs = {
            "clip_names": clip_names,
            "keyframe_indices": keyframe_indices,
            "input_features": input_features.cpu(),
            "input_character_masks": input_character_masks.cpu(),
            "target_bboxes": target_bboxes.cpu(),
            "pred_bboxes": pred_bboxes.cpu(),
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
