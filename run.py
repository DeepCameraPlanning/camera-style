import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # from src.evaluate import evaluate
    from src.train import train

    # from src.debug import debug
    from src.utils import utils

    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if config.run_type == "train":
        train(config)

    # if config.run_type == "evaluate":
    #     evaluate(config)

    # if config.run_type == "debug":
    #     debug(config)


if __name__ == "__main__":
    main()
