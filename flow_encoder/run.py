import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    if config.get("print_config"):
        from utils.diverse_utils import print_config

        print_config(config, resolve=True)

    if config.run_type == "train":
        from flow_encoder.src.train import train

        train(config)

    elif config.run_type == "extract_apn":
        from flow_encoder.src.extract_apn import extract_features

        extract_features(config)

    elif config.run_type == "extract_external":
        from flow_encoder.src.extract_external import extract_features

        extract_features(config)

    elif config.run_type == "debug":
        from flow_encoder.src.debug import debug

        debug(config)


if __name__ == "__main__":
    main()
