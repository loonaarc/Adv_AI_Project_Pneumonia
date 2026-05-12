# Checkpoint 2 Report: Pneumonia Detection from Chest X-ray Images

## Data Understanding
- Task: Binary image classification (`NORMAL` vs `PNEUMONIA`) using chest X-ray images.
- Data source: Kaggle Chest X-Ray Pneumonia dataset (Paul Mooney).
- Objective for this checkpoint: establish a clean, reproducible baseline pipeline.

## Dataset Structure
- Root path used in this project: `data/chest_xray/`
- Expected structure:
  - `train/NORMAL`, `train/PNEUMONIA`
  - `val/NORMAL`, `val/PNEUMONIA`
  - `test/NORMAL`, `test/PNEUMONIA`
- Dataset summary CSV path: `outputs/results/dataset_summary.csv`

## Class Distribution
- Class distribution plot path: `outputs/figures/class_distribution.png`
- Key counts (fill after running data exploration):
  - Train NORMAL: **[INSERT COUNT]**
  - Train PNEUMONIA: **[INSERT COUNT]**
  - Val NORMAL: **[INSERT COUNT]**
  - Val PNEUMONIA: **[INSERT COUNT]**
  - Test NORMAL: **[INSERT COUNT]**
  - Test PNEUMONIA: **[INSERT COUNT]**

## Data Preprocessing
- Images are resized to **224 x 224**.
- Pixel values are normalized to **[0, 1]**.
- Grayscale images are converted to **3 channels** for CNN input compatibility.
- TensorFlow `tf.data` pipelines are used for efficient loading and prefetching.

## Augmentation Strategy
- Applied on training data only:
  - Random rotation
  - Random zoom
  - Random horizontal flip
  - Random contrast

## Train/Validation/Test Split
- Uses existing dataset folders provided by source dataset:
  - `train/` for model fitting
  - `val/` for model selection and early stopping
  - `test/` for final baseline evaluation

## Baseline Model
- Custom CNN architecture:
  - Conv2D + MaxPooling blocks
  - Dropout regularization
  - Dense hidden layer
  - Sigmoid output neuron for binary classification
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics: Accuracy, Precision, Recall

## Initial Baseline Performance
- Training accuracy: **[INSERT TRAINING ACCURACY]**
- Validation accuracy: **[INSERT VALIDATION ACCURACY]**
- Test accuracy: **[INSERT TEST ACCURACY]**
- Test precision: **[INSERT TEST PRECISION]**
- Test recall: **[INSERT TEST RECALL]**
- Confusion matrix image: **[INSERT CONFUSION MATRIX IMAGE]**

## Observations and Challenges
- **[INSERT OBSERVATIONS ABOUT CLASS IMBALANCE, OVERFITTING, OR IMAGE QUALITY]**
- **[INSERT CHALLENGES ENCOUNTERED DURING TRAINING OR PREPROCESSING]**

## Next Steps
- Tune hyperparameters (learning rate, batch size, dropout, epochs).
- Try transfer learning (e.g., MobileNetV2, EfficientNet) for stronger performance.
- Add more evaluation metrics (F1-score, ROC-AUC).
- Explore class balancing techniques if needed.
