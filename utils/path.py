import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

# Dataset
DATASET_PATH = PROJECT_ROOT / "data" / "dataset"
