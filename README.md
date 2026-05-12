# Pneumonia Detection from Chest X-ray Images (Checkpoint 2)

## Project Goal
This project builds a reproducible baseline pipeline for binary image classification:
- **Input:** chest X-ray images
- **Classes:** `NORMAL` and `PNEUMONIA`
- **Focus for Checkpoint 2:** data understanding, preprocessing, and baseline model performance

## Dataset Source
Dataset: **Kaggle Chest X-Ray Pneumonia** by Paul Mooney  
Link: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Do not commit the dataset to this repository. Download it manually and place it under `data/chest_xray/`.

## Expected Folder Structure
```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -r requirements.txt
```

## Run Data Exploration
```bash
python src/data_exploration.py
```

## Train the Baseline Model
```bash
python src/train_baseline.py
```

## Generated Outputs
Scripts create outputs automatically:
- `outputs/figures/class_distribution.png`
- `outputs/figures/example_images_grid.png` (if samples are available)
- `outputs/figures/training_accuracy.png`
- `outputs/figures/training_loss.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/results/dataset_summary.csv`
- `outputs/results/training_history.csv`
- `outputs/results/baseline_metrics.txt`
- `outputs/models/baseline_cnn.keras`
