from pathlib import Path
import torch
from PIL import Image
from data.datamodule import OxfordPetDataModule
from data.transforms import build_eval_transform
from models.model_builder import build_model
from engine.checkpoint import load_checkpoint
from utils.config import load_experiment_config
from utils.seed import set_seed
from utils.logger import get_logger

@torch.no_grad()
def predict_single_image(model, image_tensor, class_names, device, top_k: int = 5):
    model.eval()

    image_tensor = image_tensor.unsqueeze(0).to(device)
    logits = model(image_tensor)

    probs = torch.softmax(logits, dim=1)
    top_k = min(top_k, probs.shape[1])
    top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    top_probs = top_probs.squeeze(0).cpu().tolist()
    top_indices = top_indices.squeeze(0).cpu().tolist()

    predictions = []
    for prob, idx in zip(top_probs, top_indices):
        predictions.append({"class_index": idx,"class_name": class_names[idx],"probability": prob,})
    return predictions

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    
    config_path = repo_root / "configs" / "experiment" / "infer_baseline.yaml"
    cfg = load_experiment_config(config_path)

    log_file = repo_root / "artifacts" / "logs" / f"{cfg.experiment_name}_infer.log"
    logger = get_logger("infer", log_file=str(log_file))

    logger.info("Starting inference.")
    set_seed(cfg.seed)
    # user-editable input image path
    image_path = repo_root / "data" / "sample.jpg"

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    datamodule = OxfordPetDataModule(cfg.data)
    datamodule.setup()

    class_names = datamodule.class_names

    model = build_model(
        num_classes=datamodule.num_classes,
        backbone_name=cfg.model.backbone_name,
        pretrained=cfg.model.pretrained,
        freeze_backbone=cfg.model.freeze_backbone,
        head_type=cfg.model.head_type,
        cosine_scale=cfg.model.cosine_scale)

    checkpoint_path = Path(cfg.trainer.checkpoint_dir) / cfg.trainer.best_checkpoint_name
    logger.info("Loading checkpoint from %s", checkpoint_path)

    device = torch.device(cfg.trainer.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    load_checkpoint(checkpoint_path=checkpoint_path,model=model,optimizer=None,map_location=device)

    transform = build_eval_transform(image_size=cfg.data.image_size)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)

    predictions = predict_single_image(model=model,image_tensor=image_tensor,class_names=class_names,device=device,top_k=5)
    logger.info("Inference finished for image: %s", image_path)
    
    print(f"Image: {image_path}")
    print("Top-5 predictions:")
    for i, pred in enumerate(predictions, start=1):
        print(f"{i}. {pred['class_name']} " f"(index={pred['class_index']}, prob={pred['probability']:.4f})")

if __name__ == "__main__":
    main()