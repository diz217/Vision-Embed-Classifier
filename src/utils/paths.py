from pathlib import Path

# repo root (project/)
REPO_ROOT = Path(__file__).resolve().parents[3]

# canonical data directories
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# datasets
OXFORD_PET_DIR = RAW_DIR / "oxford-iiit-pet"