from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "data" / "chest_xray"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
RANDOM_SEED = 42

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"


def ensure_output_dirs() -> None:
    """Create output directories if they do not already exist."""
    for directory in (FIGURES_DIR, RESULTS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
