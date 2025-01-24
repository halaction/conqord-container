from pathlib import Path


MODULE_DIR = Path(__file__).absolute().parent
BASE_DIR = MODULE_DIR.parent

CONQORD_DIR = MODULE_DIR / "conqord"
DATASET_DIR = CONQORD_DIR / "datasets"
MODEL_DIR = CONQORD_DIR / "model_pth"
