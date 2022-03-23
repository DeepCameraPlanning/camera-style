import os.path as osp
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from flow_encoder.src.datamodules.datasets.single_dataset import (
    SingleFlowDataset,
)
from flow_encoder.src.models.triplet_i3d import TripletI3DModel
from utils.file_utils import create_dir, save_pickle


def extract_features(config: DictConfig):
    # Initialize dataset and dataloader
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

    if config.model.histogram:
        extractor = TripletI3DModel(**model_params)
    else:
        model_params["checkpoint_path"] = config.model.checkpoint_path
        extractor = TripletI3DModel.load_from_checkpoint(**model_params)
        extractor = extractor.to(device)

    extracted_features = {}
    for batch in tqdm(data_loader):
        flows = batch["flows"].to(device)
        with torch.no_grad():
            batch_output = extractor.model.extract_features(flows).squeeze()
        for clipname, feature in zip(batch["clipname"], batch_output):
            extracted_features[clipname] = feature.to("cpu")

    # Save the test ouptuts
    create_dir(config.result_dir)
    save_path = osp.join(config.result_dir, f"{config.xp_name}_external.pk")
    save_pickle(extracted_features, save_path)
