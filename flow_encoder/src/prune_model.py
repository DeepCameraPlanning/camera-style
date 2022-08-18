import os
import os.path as osp

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from flow_encoder.src.models.flow_autoencoder import I3DAutoencoderModel
from utils.file_utils import load_pth, create_dir
from utils.flow_utils import FlowUtils


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    flow_utils = FlowUtils()

    # Load and save test chunk
    clip_name = os.listdir(config.datamodule.unity_dir)[0]
    clip_dir = osp.join(config.datamodule.unity_dir, clip_name)
    flow_filenames = sorted(os.listdir(clip_dir))[:16]
    flow_chunk = torch.stack(
        [
            load_pth(osp.join(clip_dir, filename))
            for filename in sorted(flow_filenames)
        ]
    ).permute([3, 0, 1, 2])
    pruning_res_dir = osp.join(config.viz_dir, "pruning_results")
    ref_res_dir = osp.join(pruning_res_dir, "ref")
    create_dir(ref_res_dir)
    for k, flow in enumerate(flow_chunk.permute([1, 2, 3, 0])):
        frame = flow_utils.flow_to_frame(flow.numpy())
        plt.imsave(osp.join(ref_res_dir, str(k).zfill(2) + ".png"), frame)

    model_params = {
        "histogram": config.model.histogram,
        "pretrained_path": None,
        "optimizer": config.model.optimizer,
        "learning_rate": config.model.learning_rate,
        "weight_decay": config.model.weight_decay,
        "momentum": config.model.momentum,
        "batch_size": config.compnode.batch_size,
        "flow_type": config.datamodule.flow_type,
        "checkpoint_path": config.model.checkpoint_path,
    }

    # for pruning_rate in [0.0, 0.2, 0.4, 0.6, 0.8, 0.99]:
    for pruning_rate in [0.8]:
        # Initialize model
        model = I3DAutoencoderModel.load_from_checkpoint(**model_params)
        encoder = model.model.encoder
        parameters_to_prune = []
        # Prune model
        for mod in encoder.modules():
            if isinstance(mod, nn.Conv3d):
                parameters_to_prune.append((mod, "weight"))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_rate,
        )
        n_zeros, n_elements = 0, 0

        for mod_index, (name, mod) in enumerate(
            model.model.encoder.named_modules()
        ):
            if isinstance(mod, nn.Conv3d):
                print(
                    name,
                    float(torch.sum(mod.weight == 0) / mod.weight.nelement()),
                    mod,
                )
                n_zeros += torch.sum(mod.weight == 0)
                n_elements += mod.weight.nelement()

        sparsity = n_zeros / n_elements
        print(f"Sparsity: {sparsity:.2f}")
        # Infer model and save results
        _, reconstructed_flow = model.model(flow_chunk[None])
        current_res_dir = osp.join(pruning_res_dir, str(pruning_rate).zfill(3))
        create_dir(current_res_dir)
        for k, flow in enumerate(reconstructed_flow[0].permute([1, 2, 3, 0])):
            frame = flow_utils.flow_to_frame(flow.detach().numpy())
            plt.imsave(
                osp.join(current_res_dir, str(k).zfill(2) + ".png"), frame
            )


if __name__ == "__main__":
    main()
