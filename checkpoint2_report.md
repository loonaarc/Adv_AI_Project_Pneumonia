# Checkpoint 2 Report: Pneumonia Detection from Chest X-Rays

## Project Goal

The goal is to build a binary image classifier that distinguishes **NORMAL** and **PNEUMONIA** chest X-ray images.

Dataset: Kaggle - Chest X-Ray Images (Pneumonia) by Paul Mooney.

## Repository and Notebook

GitHub repository:
https://github.com/loonaarc/Adv_AI_Project_Pneumonia/tree/main

Notebook:
https://github.com/loonaarc/Adv_AI_Project_Pneumonia/blob/main/pneumonia_detection_checkpoint2.ipynb

Open notebook in Google Colab:
https://colab.research.google.com/github/loonaarc/Adv_AI_Project_Pneumonia/blob/main/pneumonia_detection_checkpoint2.ipynb

The notebook is designed to run both locally in VS Code and in Google Colab. It downloads the dataset through the Kaggle API, so a Kaggle API token is required. In Colab, the recommended setup is to add `KAGGLE_API_TOKEN` in the Colab Secrets panel.

## Data Understanding

The original Kaggle dataset contains separate `train`, `val`, and `test` folders. The provided validation split is very small:

| Original split | NORMAL | PNEUMONIA | Total |
| --- | ---: | ---: | ---: |
| train | 1341 | 3875 | 5216 |
| val | 8 | 8 | 16 |
| test | 234 | 390 | 624 |

Because 16 validation images are not enough for reliable model selection, the notebook creates a new reproducible stratified 80/10/10 split across all downloaded images.

| Working split | NORMAL | PNEUMONIA | Total |
| --- | ---: | ---: | ---: |
| train | 1266 | 3418 | 4684 |
| val | 159 | 427 | 586 |
| test | 158 | 428 | 586 |

The split is saved to `outputs/results/dataset_split_80_10_10.csv`.

Class distribution for the working split:

![Class distribution by split](outputs/figures/class_distribution.png)

Example training images:

![Example training images](outputs/figures/example_images_grid.png)

## Data Preprocessing

Images are loaded from the split manifest and processed as follows:

- decode image files as grayscale
- resize to `224 x 224`
- convert grayscale images to 3 channels for CNN compatibility
- normalize pixel values to `[0, 1]`
- apply data augmentation only to the training split

Training augmentation includes small rotations, zoom, horizontal flip, and contrast variation.

## Baseline Model

The baseline is a small convolutional neural network:

- 3 convolution + max pooling blocks
- dropout for regularization
- dense hidden layer
- sigmoid output for binary classification

The model uses Adam, binary cross-entropy loss, and reports accuracy, precision, and recall.

## Baseline Results

The baseline CNN was trained for 5 epochs with early stopping monitoring validation loss. The validation curves are not fully smooth: training loss decreases steadily, while validation loss improves at first and then becomes unstable, including one epoch with a clear validation accuracy drop. This suggests that the baseline model is useful as a first reference, but not yet robust enough for final project conclusions.

This instability is a key checkpoint finding. It may be caused by the class imbalance, the single random split, the small CNN architecture, or difficult individual X-ray images. Therefore, the test metrics below are treated as baseline evidence, not as final proof that the model is reliable.

| Metric | Test result |
| --- | ---: |
| Loss | 0.1909 |
| Accuracy | 0.9283 |
| Precision | 0.9488 |
| Recall | 0.9533 |

Training curves:

![Training curves](outputs/figures/training_curves.png)

Detailed test classification report:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| NORMAL | 0.8718 | 0.8608 | 0.8662 | 158 |
| PNEUMONIA | 0.9488 | 0.9533 | 0.9510 | 428 |
| Macro avg | 0.9103 | 0.9070 | 0.9086 | 586 |
| Weighted avg | 0.9281 | 0.9283 | 0.9282 | 586 |

Confusion matrix on the test split:

| True class | Predicted NORMAL | Predicted PNEUMONIA |
| --- | ---: | ---: |
| NORMAL | 136 | 22 |
| PNEUMONIA | 20 | 408 |

Confusion matrix figure:

![Confusion matrix](outputs/figures/confusion_matrix.png)

Generated result files:

- `outputs/results/baseline_metrics.txt`
- `outputs/results/classification_report.txt`
- `outputs/figures/training_curves.png`
- `outputs/figures/confusion_matrix.png`

## Initial Observations

- The dataset is strongly imbalanced toward pneumonia cases.
- The original validation split is too small, so an 80/10/10 stratified split is used for a more stable validation signal.
- The baseline CNN reaches promising test performance, especially for pneumonia recall, but the validation curves are unstable.
- The confusion matrix shows 20 false negatives for pneumonia and 22 false positives for pneumonia. For a medical screening task, false negatives are especially important and should be reduced further.
- The baseline CNN provides a first performance reference, but the final project should compare stronger approaches such as transfer learning.

## Next Steps

- Inspect the confusion matrix, especially false negatives for pneumonia.
- Consider class weighting or threshold tuning because the classes are imbalanced.
- Repeat training or use cross-validation/stronger validation checks to confirm that the baseline performance is stable.
- Try transfer learning with a pretrained CNN for the final submission.
