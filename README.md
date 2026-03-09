# Vision-Embed-Classifier
Efficient pipeline to extract image embeddings from pretrained VLMs (CLIP) and train custom classification heads.


## Project Structure
```bash
Vision-Embed-Classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/                       # Config-driven experiment definitions
│    └── experiment/
│         ├── train_baseline.yaml  # Training configuration (data/model/training hyperparameters)
│         ├── eval_baseline.yaml   # Evaluation configuration (checkpoint + evaluation settings)
│         └── infer_baseline.yaml  # Inference configuration (image input + prediction settings)
├── data/
│     └── raw/                     # Dataset root (Oxford-IIIT Pet dataset)
├── scripts/                       # Optional helper scripts for running experiments
├── src/                           # Core source code
│    ├── train.py                  # Training entrypoint: builds config → data → model → trainer
│    ├── evaluate.py               # Evaluation entrypoint: loads best checkpoint and computes metrics
│    ├── infer.py                  # Inference entrypoint: runs prediction on a single image
│    ├── data/                     # Data pipeline
│    │    ├── datasets.py          # Dataset definition and label handling
│    │    ├── transforms.py        # Image preprocessing and augmentation
│    │    └── datamodule.py        # DataModule abstraction for train/val/test loaders
│    ├── models/                   # Model architecture
│    │    ├── backbone.py          # CLIP/OpenCLIP visual encoder wrapper
│    │    ├── classifier.py        # Trainable classifier head (cosine / linear)
│    │    └── model_builder.py     # Builds full model from backbone + classifier head
│    ├── engine/                   # Training engine
│    │    ├── trainer.py           # Training loop (forward/backward/optimization)
│    │    ├── checkpoint.py        # Model checkpoint saving and loading
│    │    └── metrics.py           # Accuracy and evaluation metrics
│    └── utils/                    # Shared infrastructure utilities
│         ├── config.py            # YAML → structured experiment configuration loader
│         ├── logger.py            # Experiment logging (file + console)
│         ├── seed.py              # Global seed control for reproducibility
│         ├── paths.py             # Path resolution utilities (repo-relative paths)
│         └── visualization.py     # Training curve visualization utilities
├── artifacts/                     # Generated outputs from experiments
│    ├── checkpoints/
│    │    ├── best.pt              # Best validation checkpoint
│    │    └── last.pt              # Final checkpoint after training
│    ├── logs/
│    │    ├── clip_oxford_pet_L14.log        # Training log
│    │    └── clip_oxford_pet_L14_eval.log   # Evaluation log
│    └── figures/
│         ├── clip_oxford_pet_L14_history_acc.png   # Training / validation accuracy curves
│         └── clip_oxford_pet_L14_history_loss.png  # Training / validation loss curves
└── tests/                         # Unit tests for key components
```

## System Design

## Configuration system

## Usage 

## Results 

## Outlooks
