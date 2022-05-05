import os.path as osp

from omegaconf import DictConfig
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from flow_encoder.src.datamodules.datasets.single_dataset import (
    SingleFlowDataset,
)
from flow_encoder.src.datamodules.datasets.mask_dataset import (
    MaskFlowDataset,
)
from flow_encoder.src.models.flow_contrastive_encoder import (
    I3DContrastiveEncoderModel,
)
from flow_encoder.src.models.flow_autoencoder import I3DAutoencoderModel
from flow_encoder.src.models.flow_contrastive_autoencoder import (
    I3DContrastiveAutoencoderModel,
)
from utils.file_utils import create_dir, save_pickle


def extract_features(config: DictConfig):
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
        "pretrained_path": None,
        "model_size": config.model.model_size,
        "optimizer": config.model.optimizer,
        "learning_rate": config.model.learning_rate,
        "weight_decay": config.model.weight_decay,
        "momentum": config.model.momentum,
        "batch_size": config.compnode.batch_size,
    }
    device = "cuda" if config.compnode.num_gpus > 0 else "cpu"

    if config.model.histogram:
        model_params["histogram"] = config.model.histogram
        extractor = I3DContrastiveEncoderModel(**model_params)
    elif config.model.module_name == "contrastive_encoder_i3d":
        model_params["histogram"] = config.model.histogram
        model_params["margin"] = config.model.margin
        model_params["checkpoint_path"] = config.model.checkpoint_path
        extractor = I3DContrastiveEncoderModel.load_from_checkpoint(
            **model_params, strict=False
        )
        extractor = extractor.to(device)
    elif config.model.module_name == "contrastive_autoencoder_i3d":
        model_params["margin"] = config.model.margin
        model_params["checkpoint_path"] = config.model.checkpoint_path
        model_params["check_dir"] = config.datamodule.check_dir
        extractor = I3DContrastiveAutoencoderModel.load_from_checkpoint(
            **model_params, strict=False
        )
        extractor = extractor.to(device)
    elif config.model.module_name == "autoencoder_i3d":
        model_params["checkpoint_path"] = config.model.checkpoint_path
        model_params["check_dir"] = config.datamodule.check_dir
        model_params["flow_type"] = config.datamodule.flow_type
        extractor = I3DAutoencoderModel(**model_params)
        extractor = extractor.to(device)

    extracted_features = {}
    for batch in tqdm(data_loader):
        flows = batch["flows"].to(device)
        with torch.no_grad():
            batch_output = extractor.model.extract_features(flows)
        for clipname, feature in zip(batch["clipname"], batch_output):
            extracted_features[clipname] = feature.to("cpu")

    # Save the test ouptuts
    create_dir(config.result_dir)
    save_path = osp.join(config.result_dir, f"{config.xp_name}_external.pk")
    print("saving into", save_path)
    save_pickle(extracted_features, save_path)
