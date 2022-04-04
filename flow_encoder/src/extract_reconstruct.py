import os.path as osp
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader

from flow_encoder.src.datamodules.datasets.single_dataset import (
    SingleFlowDataset,
)
from flow_encoder.src.datamodules.datasets.mask_dataset import (
    MaskFlowDataset,
)
from flow_encoder.src.models.flow_autoencoder import I3DAutoencoderModel
from flow_encoder.src.models.flow_contrastive_autoencoder import (
    I3DContrastiveAutoencoderModel,
)
from utils.file_utils import create_dir, write_clip
from utils.flow_utils import FlowUtils


def extract_reconstruct(config: DictConfig):
    flow_utils = FlowUtils()

    # Initialize dataset and dataloader
    if config.datamodule.masked:
        data_set = MaskFlowDataset(
            flow_dir=config.datamodule.raft_dir,
            n_frames=config.model.n_frames,
            stride=config.model.stride,
            n_temporal_mask=config.datamodule.n_temporal_mask,
            spatial_mask_size=config.datamodule.spatial_mask_size,
        )
    else:
        data_set = SingleFlowDataset(
            flow_dir=config.datamodule.raft_dir,
            n_frames=config.model.n_frames,
            stride=config.model.stride,
        )
    data_loader = DataLoader(
        data_set,
        batch_size=config.compnode.batch_size,
        num_workers=config.compnode.num_workers,
    )
    # Initialize model
    model_params = {
        "histogram": config.model.histogram,
        "pretrained_path": None,
        "optimizer": config.model.optimizer,
        "learning_rate": config.model.learning_rate,
        "weight_decay": config.model.weight_decay,
        "momentum": config.model.momentum,
        "batch_size": config.compnode.batch_size,
    }
    device = "cuda" if config.compnode.num_gpus > 0 else "cpu"

    if config.model.module_name == "autoencoder_i3d":
        model_params["triplet_coef"] = config.model.triplet_coef
        model_params["reconstruction_coef"] = config.model.reconstruction_coef
        model_params["check_dir"] = config.datamodule.check_dir
        model_params["checkpoint_path"] = config.model.checkpoint_path
        extractor = I3DContrastiveAutoencoderModel.load_from_checkpoint(
            **model_params
        )
        extractor = extractor.to(device)
    elif config.model.module_name == "autoencoder_i3d":
        model_params["check_dir"] = config.datamodule.check_dir
        model_params["flow_type"] = config.datamodule.flow_type
        extractor = I3DAutoencoderModel(**model_params)

    reconstructions, gt_flows = [], []
    for batch in data_loader:
        break
    flows = batch["flows"].to(device)
    with torch.no_grad():
        _, batch_output = extractor.model(flows)
    for clipname, reconstruction, gt_flow in zip(
        batch["clipname"], batch_output, flows
    ):
        if clipname.split("/")[0] != "0_-D3PMCmZot0":
            break
        reconstructions.append(reconstruction.permute([1, 2, 3, 0]))
        gt_flows.append(gt_flow.permute([1, 2, 3, 0]))
    reconstructions = torch.vstack(reconstructions).to("cpu").numpy()
    reconstructions_rgb = [
        flow_utils.flow_to_frame(flow) for flow in reconstructions
    ]
    gt_flows = torch.vstack(gt_flows).to("cpu").numpy()
    gt_flows_rgb = [flow_utils.flow_to_frame(flow) for flow in gt_flows]

    # Save the test ouptuts
    create_dir(config.result_dir)
    save_path = osp.join(
        config.result_dir, f"{config.xp_name}_0_-D3PMCmZot0.mp4"
    )
    write_clip(reconstructions_rgb, save_path, 5)
    save_path = osp.join(config.result_dir, "gt_0_-D3PMCmZot0.mp4")
    write_clip(gt_flows_rgb, save_path, 5)
