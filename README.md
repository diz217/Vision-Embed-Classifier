# Vision-Embed-Classifier
Efficient pipeline to extract image embeddings from pretrained VLMs (CLIP) and train custom classification heads.


## Project Structure
```bash
Vision-Embed-Classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── experiment/
│        ├── eval_baseline.yaml
│        ├── infer_baseline.yaml
│        └── train_baseline.yaml
├── data/
│   └── raw/
├── scripts/
├── src/
│    ├── train.py
│    ├── evaluate.py
│    ├── infer.py
│    ├── data/
│    │   ├── datasets.py
│    │   ├── transforms.py
│    │   └── datamodule.py
│    ├── models/
│    │   ├── backbone.py
│    │   ├── classifier.py
│    │   └── model_builder.py
│    ├── engine/
│    │   ├── trainer.py
│    │   ├── checkpoint.py
│    │   └── metrics.py
│    └── utils/
│        ├── config.py
│        ├── logger.py
│        ├── seed.py
│        ├── paths.py
│        └── visualization.py
├── artifacts/
│    ├── checkpoints/
│    │    ├── best.pt
│    │    └── last.pt
│    ├── logs/
|    │    ├── clip_oxford_pet_L14_eval.log
|    │    └── clip_oxford_pet_L14.log
|    └── figures/
|         ├── clip_oxford_pet_L14_history_acc.png
|         └── clip_oxford_pet_L14_history_loss.png
└── tests/
```
