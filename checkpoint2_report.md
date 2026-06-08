# Checkpoint 2 Report: Pneumonia Detection from Chest X-Rays

Lilie Lin, Junu Rahman, Arooj Shahzadi

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

The dataset is imbalanced toward pneumonia images. In the working split, the
training set contains 1,266 NORMAL images and 3,418 PNEUMONIA images. This
imbalance is important because a model can achieve high accuracy while still
performing worse on the minority class.

Patient-level leakage is an important limitation. This Kaggle release does not
provide a separate patient metadata table, and the filenames are not documented
as reliable patient identifiers. Therefore, the split used here is an
image-level stratified split. If patient IDs were available, all images from the
same patient should be assigned to only one split using a grouped split.

Class distribution for the working split:

![Class distribution by split](outputs/figures/class_distribution.png)

Example training images:

![Example training images](outputs/figures/example_images_grid.png)

Manual visual inspection of the sample grid shows that the images are generally
recognizable chest X-rays, but they vary in brightness, contrast, size, and
positioning. Some cases are visually harder than others. From a non-expert
perspective, the difference between NORMAL and PNEUMONIA is not always obvious:
pneumonia cases may show cloudier or more opaque lung regions, but the pattern
is often subtle. The checkpoint feedback notebook also includes a small image-quality sample
check for contrast and a sharpness proxy.

## Data Preprocessing

Images are loaded from the split manifest and processed as follows:

- decode image files as grayscale
- resize to `224 x 224`
- convert grayscale images to 3 channels for CNN compatibility
- normalize pixel values to `[0, 1]`
- apply online data augmentation only to the training split

Training augmentation includes small rotations, zoom, horizontal flip, and
contrast variation. Augmentation is online, so no extra image files are written
to disk. Each epoch still iterates over 4,684 training images, but the model may
see randomly transformed versions across epochs. Because augmentation is applied
uniformly to training batches, the original class distribution is preserved; it
does not oversample NORMAL images.

Horizontal flipping is used cautiously. For this binary task, the label only
indicates whether pneumonia is present, not whether it is left-sided or
right-sided. A flip should therefore not change the target label. However,
flipping would be less appropriate for medical tasks where laterality or
anatomical orientation is part of the prediction target.

## Baseline Model

The baseline is a small convolutional neural network:

- 3 convolution + max pooling blocks
- dropout for regularization
- dense hidden layer
- sigmoid output for binary classification

The model uses Adam, binary cross-entropy loss, and reports accuracy, precision, and recall.

The checkpoint feedback notebook now exports a baseline architecture table to
`outputs/results/baseline_architecture.csv`, including each layer, output shape,
and parameter count.

## Baseline Results

The baseline CNN was trained for 5 epochs with early stopping monitoring validation loss. The validation curves are not fully smooth: training loss decreases steadily, while validation loss improves at first and then becomes unstable, including one epoch with a clear validation accuracy drop. This suggests that the baseline model is useful as a first reference, but not yet robust enough for final conclusions.

This instability is a key checkpoint finding. It may be caused by the class imbalance, the single random split, the small CNN architecture, or difficult individual X-ray images. Therefore, the test metrics below are treated as baseline evidence, not as final proof that the model is reliable.

| Metric | Test result |
| --- | ---: |
| Loss | 0.1646 |
| Accuracy | 0.9386 |
| Precision | 0.9712 |
| Recall | 0.9439 |

The checkpoint feedback notebook also computes ROC-AUC, Precision-Recall AUC, and a
threshold analysis table. The default threshold of 0.5 is used for the baseline
classification report, but this is only a starting point. In a medical screening
setting, false negatives are especially costly, so threshold selection should be
performed on the validation set using sensitivity, specificity, and clinical
cost trade-offs.

Baseline training curves:

![Training curves](outputs/figures/training_curves.png)

The unstable validation curve in this baseline run shows a limitation of relying on one validation split. Simpler ways to reduce this issue would be to repeat training with different random seeds, apply class weighting, or tune the classification threshold. Stratified k-fold cross-validation could also help if more compute is available, but it is not required for this baseline checkpoint.

Detailed test classification report:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| NORMAL | 0.8588 | 0.9241 | 0.8902 | 158 |
| PNEUMONIA | 0.9712 | 0.9439 | 0.9573 | 428 |
| Macro avg | 0.9150 | 0.9340 | 0.9238 | 586 |
| Weighted avg | 0.9409 | 0.9386 | 0.9393 | 586 |

Confusion matrix on the test split:

| True class | Predicted NORMAL | Predicted PNEUMONIA |
| --- | ---: | ---: |
| NORMAL | 146 | 12 |
| PNEUMONIA | 24 | 404 |

Confusion matrix figure:

![Confusion matrix](outputs/figures/confusion_matrix.png)

Generated result files:

- `outputs/results/baseline_metrics.txt`
- `outputs/results/classification_report.txt`
- `outputs/figures/training_curves.png`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/roc_curve.png`
- `outputs/figures/precision_recall_curve.png`
- `outputs/results/threshold_analysis.csv`
Transfer-learning model comparison and validation-based final model selection
are continued in the separate draft notebook
`pneumonia_detection_project_submission.ipynb`. This keeps checkpoint 2 focused
on the implemented feedback for data understanding, preprocessing, baseline
architecture, and baseline evaluation.

## Initial Observations

- The dataset is strongly imbalanced toward pneumonia cases.
- The original validation split is too small, so an 80/10/10 stratified split is used for a more stable validation signal.
- The baseline CNN reaches promising test performance, especially for pneumonia recall, but the validation curves are unstable.
- The unstable validation curve shows that the single validation split should be interpreted cautiously; class weighting, threshold tuning, or repeated runs could make the estimate more reliable.
- The baseline confusion matrix shows 24 false negatives for pneumonia and 12 false positives for pneumonia at threshold 0.5. For a medical screening task, false negatives are especially important and should be reduced further.
- The separate project-submission notebook continues with validation-based model comparison and validation-based threshold selection before final selected-model test evaluation.
- The baseline CNN provides a first performance reference, but it should not be treated as a final reliable medical model.
- Reproducibility files are included as `requirements.txt` and `python_version.txt`.

