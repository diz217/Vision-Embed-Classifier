from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class DataConfig:
    root: str | Path
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    val_ratio: float = 0.1
    seed: int = 42
    download: bool = False
    pin_memory: bool = True


@dataclass
class ModelConfig:
    backbone_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    freeze_backbone: bool = True
    head_type: str = "cosine"   # linear / cosine 
    cosine_scale: float = 20


@dataclass
class TrainerConfig:
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"
    checkpoint_dir: str = "artifacts/checkpoints"
    best_checkpoint_name: str = "best.pt"
    last_checkpoint_name: str = "last.pt"
    log_every_n_steps: int = 20

@dataclass
class InferConfig:
    image_path: str = "data/sample.jpg"
    top_k: int = 5
    
@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
    infer: InferConfig | None = None
    
def _resolve_path(path_str: str, config_path: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    data_cfg = DataConfig(**raw["data"])
    model_cfg = ModelConfig(**raw["model"])
    trainer_cfg = TrainerConfig(**raw["trainer"])
    infer_cfg = InferConfig(**raw["infer"]) if "infer" in raw else None

    data_cfg.root = str(_resolve_path(data_cfg.root, config_path))
    trainer_cfg.checkpoint_dir = str(_resolve_path(trainer_cfg.checkpoint_dir, config_path))

    if infer_cfg is not None:
        infer_cfg.image_path = str(_resolve_path(infer_cfg.image_path, config_path))

    return ExperimentConfig(experiment_name=raw["experiment_name"],
                            seed=raw["seed"],
                            data=data_cfg,
                            model=model_cfg,
                            trainer=trainer_cfg,
                            infer=infer_cfg)