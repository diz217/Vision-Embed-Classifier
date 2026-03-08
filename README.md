# Vision-Embed-Classifier
Efficient pipeline to extract image embeddings from pretrained VLMs (CLIP) and train custom classification heads.


## Project Structure
```bash
Vision-Embed-Classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── model/
│   ├── data/
│   ├── train/
│   └── experiment/
├── scripts/
│   ├── train.sh
│   ├── eval.sh
│   └── infer.sh
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
│        └── visualization.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_config.py
└── artifacts/
    ├── checkpoints/
    ├── logs/
    ├── metrics/
    └── figures/
```
