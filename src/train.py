import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from data.datamodule import OxfordPetDataModule
from models.model_builder import build_model
from engine.trainer import Trainer
from utils.config import load_experiment_config
from utils.seed import set_seed
from utils.logger import get_logger
from utils.visualization import plot_training_history

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    
    config_path = repo_root / "configs" / "experiment" / "train_baseline.yaml"
    cfg = load_experiment_config(config_path)

    log_file = repo_root / "artifacts" / "logs" / f"{cfg.experiment_name}.log"
    logger = get_logger("train", log_file=str(log_file))
    
    logger.info("Starting training pipeline.")
    logger.info("Experiment name: %s", cfg.experiment_name)
    logger.info("Config path: %s", config_path)
    
    set_seed(cfg.seed)

    # build data
    datamodule = OxfordPetDataModule(cfg.data)
    datamodule.setup()

    logger.info("Data module setup complete.")
    logger.info("Number of classes: %d", datamodule.num_classes)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = build_model(
        num_classes=datamodule.num_classes,
        backbone_name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        head_type=cfg.model.head_type,
        cosine_scale=cfg.model.cosine_scale)

    logger.info("Model built successfully.")

    trainer = Trainer(model, cfg.trainer)

    logger.info("Trainer initialized. Starting fit().")

    history = trainer.fit(train_loader=train_loader, val_loader=val_loader)

    logger.info("Training finished.")
    logger.info("Best validation accuracy: %.4f", trainer.best_val_acc)

    figure_path = repo_root / "artifacts" / "figures" / f"{cfg.experiment_name}_history.png"
    plot_training_history(history, figure_path)

    logger.info("Training history figure saved to %s", figure_path)


if __name__ == "__main__":
    main()