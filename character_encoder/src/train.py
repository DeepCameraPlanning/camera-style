from omegaconf import DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from character_encoder.src.datamodules.latent_bbox_datamodule import (
    LatentBboxDataModule,
)
from character_encoder.src.models.latent_character import LatentCharacterModel


def train(config: DictConfig):
    # Initialize dataset
    data_module = LatentBboxDataModule(
        split_dir=config.datamodule.split_dir,
        bbox_dir=config.datamodule.bbox_dir,
        flow_dir=config.datamodule.flow_dir,
        frame_dir=config.datamodule.frame_dir,
        stride=config.datamodule.stride,
        n_frames=config.datamodule.n_frames,
        batch_size=config.compnode.batch_size,
        num_workers=config.compnode.num_workers,
    )

    # Initialize callbacks
    wandb_logger = WandbLogger(
        name=config.xp_name,
        project=config.project_name,
        offline=config.log_offline,
    )
    checkpoint = ModelCheckpoint(
        monitor=config.checkpoint_metric,
        mode="min",
        save_last=True,
        dirpath=config.character_checkpoint_dirpath,
        filename=config.xp_name + "-{epoch}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize model
    model_params = {
        "model_config": config.model,
        "bbox_coef": config.model.bbox_coef,
        "reconstruction_coef": config.model.reconstruction_coef,
        "autoencoder_ckpt_path": config.model.autoencoder_ckpt_path,
        "optimizer": config.model.optimizer,
        "learning_rate": config.model.learning_rate,
        "weight_decay": config.model.weight_decay,
        "momentum": config.model.momentum,
        "batch_size": config.compnode.batch_size,
        "check_dir": config.datamodule.check_dir,
    }
    model = LatentCharacterModel(**model_params)

    trainer = Trainer(
        gpus=config.compnode.num_gpus,
        num_nodes=config.compnode.num_nodes,
        accelerator=config.compnode.accelerator,
        max_epochs=config.num_epochs,
        callbacks=[lr_monitor, checkpoint],
        logger=wandb_logger,
        log_every_n_steps=5,
        # precision=16,
    )

    # Launch model training
    trainer.fit(model, data_module)
