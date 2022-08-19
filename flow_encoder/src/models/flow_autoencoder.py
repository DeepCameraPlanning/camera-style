from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig
import torch
from torch.nn import MSELoss
from pytorch_lightning import LightningModule

from flow_encoder.src.models.modules.i3d import make_flow_autoencoder


class I3DAutoencoderModel(LightningModule):
    def __init__(
        self,
        pretrained_path: str,
        model_size: str,
        flow_type: str,
        optimizer: str,
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
        self._flow_type = flow_type

        self.loss = MSELoss()

        self.model = make_flow_autoencoder(pretrained_path, model_size)

    def _shared_eval_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """ """
        input_flows = batch[f"{self._flow_type}_flows"].float()
        _, out_flows = self.model(input_flows)

        loss = self.loss(out_flows, input_flows)
        outputs = {
            "clipname": batch[f"{self._flow_type}_flows"],
            "gt_flows": input_flows,
            "pred_flows": out_flows,
        }

        return loss, outputs

    def training_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Training loop."""
        out = self._shared_eval_step(batch, batch_idx)
        loss, outputs = out

        metric_dict = {"loss": loss}
        flow_dict = {
            "gt_flows": outputs["gt_flows"].detach(),
            "pred_flows": outputs["pred_flows"].detach(),
        }

        return {
            "loss": loss,
            "metric_dict": metric_dict,
            "flow_dict": flow_dict,
        }

    def validation_step(
        self, batch: torch.Tensor, batch_idx: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Validation loop."""
        out = self._shared_eval_step(batch, batch_idx)
        loss, outputs = out

        metric_dict = {"loss": loss}
        flow_dict = {
            "gt_flows": outputs["gt_flows"].detach(),
            "pred_flows": outputs["pred_flows"].detach(),
        }

        return {
            "loss": loss,
            "metric_dict": metric_dict,
            "flow_dict": flow_dict,
        }

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
