from pathlib import Path


class DatasetConfig:
    dataset_name: str = "???"
    num_workers: int = 4
    path: Path = Path("???")
    around_text: bool = False
    artificial: bool = False
    all_system: bool = True
    system_num: int = 5


class MatchingConfig:
    system_ratio: float = 0.4
    crowd_ratio: float = 0.4
    expert_ratio: float = 0.2


class TrainConfig:
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.2
    epoch: int = 800
    batch_size: int = 8
    out_dim: int = 2
    learning_rate: float = 0.079
    hidden_dim: int = 20
    dropout: bool = False
    batch_norm: bool = False
    dropout_rate: float = 0.3
    weight_decay: float = 0.078


class Config:
    seed: int = 10
    method: str = "???"
    name: str = "???"
    debug: bool = True
    model: str = "???"
    loss: str = "???"
    output_dir: Path = Path("???")
    abci: bool = False
    mode: str = "???"
    commit_hash: str = "${commit_hash: ${debug}}"
    wandb_enabled: bool = True
    wandb_entity: str = "???"
    wandb_project: str = "???"
    dataset: DatasetConfig = DatasetConfig()
    matching: MatchingConfig = MatchingConfig()
    train: TrainConfig = TrainConfig()
