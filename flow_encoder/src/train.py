from omegaconf import DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from flow_encoder.src.datamodules.flow_datamodule import TripletFlowDataModule
from flow_encoder.src.models.callbacks.gradnorm import GradNorm
from flow_encoder.src.models.callbacks.logger import TrainingLogger
from flow_encoder.src.models.flow_contrastive_encoder import (
    I3DContrastiveEncoderModel,
)
from flow_encoder.src.models.flow_autoencoder import I3DAutoencoderModel
from flow_encoder.src.models.flow_contrastive_autoencoder import (
    I3DContrastiveAutoencoderModel,
)
from flow_encoder.src.models.flow_contrastive_vqvae import (
    I3DContrastiveVQVAEModel,
)


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
    log_momentum = True if config.model.optimizer == "adam" else False
    lr_monitor = LearningRateMonitor(
        logging_interval="epoch", log_momentum=log_momentum
    )
    grad_norm = GradNorm(config.model.grad_norm_alpha)
    callbacks = [lr_monitor, checkpoint, grad_norm, TrainingLogger()]

    # Initialize model
    model_params = {
        "pretrained_path": config.model.pretrained_path,
        "model_size": config.model.model_size,
        "optimizer": config.model.optimizer,
        "learning_rate": config.model.learning_rate,
        "weight_decay": config.model.weight_decay,
        "momentum": config.model.momentum,
        "batch_size": config.compnode.batch_size,
        "config": config,
    }

    if config.model.module_name == "contrastive_encoder_i3d":
        model_params["margin"] = config.model.margin
        model_params["histogram"] = config.model.histogram
        model = I3DContrastiveEncoderModel(**model_params)
    elif config.model.module_name == "contrastive_autoencoder_i3d":
        model_params["contrastive_mode"] = config.model.contrastive_mode
        model_params["margin"] = config.model.margin
        model = I3DContrastiveAutoencoderModel(**model_params)
    elif config.model.module_name == "contrastive_vqvae_i3d":
        model_params["margin"] = config.model.margin
        model_params["commitment_cost"] = config.model.commitment_cost
        model_params["n_embeddings"] = config.model.n_embeddings
        model_params["contrastive_mode"] = config.model.contrastive_mode
        model = I3DContrastiveVQVAEModel(**model_params)
    elif config.model.module_name == "autoencoder_i3d":
        model_params["flow_type"] = config.datamodule.flow_type
        model = I3DAutoencoderModel(**model_params)

    trainer = Trainer(
        gpus=config.compnode.num_gpus,
        num_nodes=config.compnode.num_nodes,
        accelerator=config.compnode.accelerator,
        max_epochs=config.num_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=5,
        # precision=16,
    )

    # Launch model training
    trainer.fit(model, data_module)
