import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    if config.get("print_config"):
        from src.utils import utils

        utils.print_config(config, resolve=True)

    if config.run_type == "train":
        from src.train import train

        train(config)

    if config.run_type == "extract":
        from src.extract_features import extract_features

        extract_features(config)

    # if config.run_type == "debug":
    #     debug(config)


if __name__ == "__main__":
    main()
