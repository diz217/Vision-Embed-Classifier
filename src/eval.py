from pathlib import Path
from data.datamodule import OxfordPetDataModule
from models.model_builder import build_model
from engine.checkpoint import load_checkpoint
from engine.trainer import Trainer
from utils.config import load_experiment_config
from utils.seed import set_seed
from utils.logger import get_logger

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    
    config_path = repo_root / "configs" / "experiment" / "eval_baseline.yaml"
    cfg = load_experiment_config(config_path)

    log_file = repo_root / "artifacts" / "logs" / f"{cfg.experiment_name}_eval.log"
    logger = get_logger("evaluate", log_file=str(log_file))

    logger.info("Starting evaluation.")
    set_seed(cfg.seed)

    datamodule = OxfordPetDataModule(cfg.data)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()

    logger.info("Data module setup complete.")
    logger.info("Number of classes: %d", datamodule.num_classes)
    logger.info("Config path: %s", config_path)

    model = build_model(
        num_classes=datamodule.num_classes,
        backbone_name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        head_type=cfg.model.head_type,
        cosine_scale=cfg.model.cosine_scale)

    logger.info("Model built successfully.")

    trainer = Trainer(model, cfg.trainer)

    checkpoint_path = Path(cfg.trainer.checkpoint_dir) / cfg.trainer.best_checkpoint_name
    logger.info("Loading checkpoint from %s", checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path=checkpoint_path,model=trainer.model,optimizer=None,map_location=trainer.device)
    logger.info("Checkpoint loaded. Epoch=%s, best_val_acc=%.4f",checkpoint.get("epoch", -1),checkpoint.get("best_val_acc", -1.0))

    test_loss, test_acc = trainer.validate_one_epoch(val_loader=test_loader,epoch=0,stage='Test')

    logger.info("Evaluation finished.")
    logger.info("Test loss: %.4f", test_loss)
    logger.info("Test accuracy: %.4f", test_acc)

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()