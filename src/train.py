from omegaconf import DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from src.datamodules.flow_datamodule import TripletFlowDataModule
from src.models.triplet_i3d import TripletI3DModel


def train(config: DictConfig):
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
        dirpath=config.checkpoint_dirpath,
        filename=config.xp_name + "-{epoch}-{val_loss:.2f}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize model
    model_params = {
        "pretrained_path": config.model.pretrained_path,
        "model_config": config.model,
        "num_classes": config.datamodule.n_classes,
        "optimizer": config.model.optimizer,
        "learning_rate": config.model.learning_rate,
        "weight_decay": config.model.weight_decay,
        "momentum": config.model.momentum,
        "batch_size": config.compnode.batch_size,
    }
    model = TripletI3DModel(**model_params)

    trainer = Trainer(
        gpus=config.compnode.num_gpus,
        num_nodes=config.compnode.num_nodes,
        accelerator=config.compnode.accelerator,
        max_epochs=config.num_epochs,
        callbacks=[lr_monitor, checkpoint],
        logger=wandb_logger,
        log_every_n_steps=5,
        precision=16,
    )

    # Launch model training
    trainer.fit(model, data_module)
