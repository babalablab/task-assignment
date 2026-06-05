import hydra
import numpy as np
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from config import Config
from train.train import train_common_confusions


@hydra.main(version_base=None, config_path="../config/", config_name="config")
def main(config: Config):
    np.random.seed(config.trainer.seed)
    tags = [config.dataset.name]
    if config.debug:
        tags.append("debug")
    if "tags" in config.keys():
        tags.append(config.tags)
    compiled_config = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )

    if config.wandb_enabled:
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.name,
            tags=tags,
            config=compiled_config,
        )
    try:
        if config.method == "icrowd":
            print("icrowd")
            if config.debug:
                config.name = "debug"
            logger = instantiate(config.logger) if config.wandb_enabled else None
            train_dataset = instantiate(config.dataset.train_dataset)
            _ = instantiate(config.dataset.valid_dataset)
            test_dataset = instantiate(config.dataset.test_dataset)
            trainer = instantiate(config.trainer, logger=logger)
            trainer(train_dataset, test_dataset)
        elif config.mode == "train":
            train_common_confusions(config)
        else:
            raise NotImplementedError(f"mode {config.mode} is not implemented")
    finally:
        if config.wandb_enabled:
            wandb.finish()


if __name__ == "__main__":
    main()
