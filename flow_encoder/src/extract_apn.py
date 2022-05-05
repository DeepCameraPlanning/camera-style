from omegaconf import DictConfig
import os.path as osp

from pytorch_lightning import Trainer

from flow_encoder.src.datamodules.flow_datamodule import TripletFlowDataModule
from flow_encoder.src.models.contrastive_flow_encoder import I3DEncoderModel
from utils.file_utils import create_dir, save_pickle


def extract_features(config: DictConfig):
    # Initialize dataset
    data_module = TripletFlowDataModule(
        split_dir=config.datamodule.split_dir,
        unity_dir=config.datamodule.unity_dir,
        prcpt_dir=config.datamodule.raft_dir,
        n_frames=config.model.n_frames,
        stride=config.model.stride,
        frame_size=config.model.frame_size,
        batch_size=config.compnode.batch_size,
        num_workers=config.compnode.num_workers,
    )

    # Initialize model
    model_params = {
        "checkpoint_path": config.model.checkpoint_path,
        "pretrained_path": None,
        "model_size": config.model.model_size,
        "optimizer": config.model.optimizer,
        "learning_rate": config.model.learning_rate,
        "weight_decay": config.model.weight_decay,
        "momentum": config.model.momentum,
        "batch_size": config.compnode.batch_size,
    }
    model = I3DEncoderModel.load_from_checkpoint(**model_params)

    trainer = Trainer(
        gpus=config.compnode.num_gpus,
        num_nodes=config.compnode.num_nodes,
        accelerator=config.compnode.accelerator,
        precision=16,
    )

    # Launch model training
    trainer.test(model, data_module)

    # Save the test ouptuts
    create_dir(config.result_dir)
    save_path = osp.join(config.result_dir, f"{config.xp_name}_extracted.pk")
    save_pickle(model.test_outputs, save_path)
