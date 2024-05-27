import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
DATA_ROOT_PATH = PROJECT_ROOT  # modify this to your data root path
sys.path.append(str(PROJECT_ROOT))

# Data related
DATASET_PATH = DATA_ROOT_PATH / "data" / "dataset"

# Experiments and checkpoints related
EXPERIMENT_DIR = DATA_ROOT_PATH / "experiments"
