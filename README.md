# Vision-Embed-Classifier
Vision-Embed-Classifier is a modular deep learning training system for image classification built around pretrained visual encoders.

Instead of implementing a single model script, the project demonstrates how to structure a clean, reproducible ML pipeline with configuration-driven experiments, modular data and model components, and a clear training → evaluation → inference workflow.

The classifier leverages a large pretrained vision encoder to generate embeddings and trains a lightweight classification head for downstream tasks.


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

## Core Design Principles 

This repository is designed to demonstrate **clean and reproducible ML engineering practices** rather than a single experimental model script.  
The system emphasizes modularity, configurability, and clear separation of responsibilities across the training pipeline.

### 1. Modular ML Architecture

The project is organized into independent components responsible for different stages of the ML workflow:
```bash
DataModule → Model Builder → Trainer → Evaluator → Inference
```
Each module encapsulates a specific responsibility:

- **DataModule** handles dataset loading, preprocessing, and dataloader construction.
- **Model Builder** assembles the full model from backbone and classifier components.
- **Trainer** manages the training loop, optimization, and checkpointing.
- **Evaluator** loads trained checkpoints and computes evaluation metrics.
- **Inference** provides prediction utilities for single-image inputs.

This modular structure makes it easy to modify individual parts of the pipeline without affecting the rest of the system.

### 2. Configuration-Driven Experiments

All experiments are defined using YAML configuration files:
```bash
configs/experiment/
train_baseline.yaml
eval_baseline.yaml
infer_baseline.yaml
```
Configurations are parsed into structured dataclasses:
```bash
ExperimentConfig
├── DataConfig
├── ModelConfig
├── TrainerConfig
└── InferConfig
```
This design enables:
- reproducible experiments
- clean separation between configuration and code
- easy hyperparameter tuning without modifying source files

### 3. Pretrained Encoder + Lightweight Classifier
The model architecture separates **representation learning** from **task-specific classification**.

A large pretrained visual encoder generates image embeddings, while a lightweight classifier head is trained for the downstream task.

Benefits of this design include:
- strong feature representations from pretrained models
- faster convergence
- reduced training cost

### 4. Reproducible ML Workflow
The repository provides a clear experiment lifecycle:
```bash
train.py → evaluate.py → infer.py
```
Each stage loads the same experiment configuration, ensuring consistent model construction and evaluation.

Outputs are organized into structured artifact directories:
```bash
artifacts/
├── checkpoints/
├── logs/
└── figures/
```
This makes experiments easy to reproduce and inspect.

### 5. Structured Experiment Outputs
Training automatically produces several artifacts:
- **checkpoints** – best and last model states
- **logs** – detailed training and evaluation logs
- **figures** – training curves for loss and accuracy

These artifacts provide transparent experiment tracking and make debugging or comparison across runs straightforward.

## Configuration system

## Usage 

## Results 

## Outlooks
