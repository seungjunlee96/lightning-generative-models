import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

# Data related
DATASET_PATH = PROJECT_ROOT / "data" / "dataset"

# Experiments and checkpoints related
EXPERIMENT_DIR = PROJECT_ROOT / "experiments"
