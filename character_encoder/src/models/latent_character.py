from typing import Any, Dict, List, Tuple

from omegaconf.dictconfig import DictConfig
import torch
from torch.nn import SmoothL1Loss
from pytorch_lightning import LightningModule

from character_encoder.src.models.modules.global_iou import GIOULoss
from character_encoder.src.models.modules.perceiver import make_latent_ca


class LatentCharacterModel(LightningModule):
    def __init__(
        self,
        model_config: DictConfig,
        loss: str,
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
        if loss == "smoothl1":
            self.criterion = SmoothL1Loss()
        elif loss == "giou":
            self.criterion = GIOULoss()

        self._model_config = model_config
        self.model = make_latent_ca()

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
        """Input flow feature and character mask, output bbox coordinates."""
        input_feature = batch["input_feature"]
        input_character_mask = batch["input_character_mask"]
        target_bbox = batch["target_bbox"]

        predicted_bbox = self.model(input_feature, input_character_mask)
        target_bbox = target_bbox.type(predicted_bbox.dtype)
        loss = self.criterion(predicted_bbox, target_bbox)

        outputs = {
            "clip_name": batch["clip_name"],
            "keyframe_indices": batch["keyframe_indices"],
            "input_feature": input_feature,
            "input_character_mask": input_character_mask,
            "target_bbox": target_bbox,
            "predicted_bbox": predicted_bbox,
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

    def test_epoch_end(self, outputs: torch.Tensor) -> Dict[str, Any]:
        """Gather all test outputs."""
        clip_names, keyframe_indices = [], []
        input_features, input_character_masks = [], []
        target_bboxes, predicted_bboxes = [], []
        for out in outputs:
            clip_names.extend(out["out"]["clip_name"])
            keyframe_indices.extend(out["out"]["keyframe_indices"])
            input_features.append(out["out"]["input_feature"])
            input_character_masks.append(out["out"]["input_character_mask"])
            target_bboxes.append(out["out"]["target_bbox"])
            predicted_bboxes.append(out["out"]["predicted_bbox"])
        input_features = torch.cat(input_features)
        input_character_masks = torch.cat(input_character_masks)
        target_bboxes = torch.cat(target_bboxes)
        predicted_bboxes = torch.cat(predicted_bboxes)

        self.test_outputs = {
            "clip_names": clip_names,
            "keyframe_indices": keyframe_indices,
            "input_features": input_features.cpu(),
            "input_character_masks": input_character_masks.cpu(),
            "target_bboxes": target_bboxes.cpu(),
            "predicted_bboxes": predicted_bboxes.cpu(),
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
