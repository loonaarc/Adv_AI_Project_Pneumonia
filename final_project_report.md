# Final Project Report: Pneumonia Detection from Chest X-Rays

Lilie Lin, Junu Rahman, Arooj Shahzadi

## 1. Motivation and Problem Understanding

This project investigates binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA**. Pneumonia screening is clinically relevant because missed pneumonia cases can delay treatment. The model is a controlled course project and is not intended to replace medical diagnosis.

## 2. Related Work and Method Choice

Convolutional neural networks are commonly used for image classification because they learn local visual patterns from images. For medical imaging tasks with limited data, transfer learning is often useful because pretrained CNNs can reuse general low-level visual features.

Related work and benchmark context:

- Wang et al. introduced ChestX-ray14, a large public chest X-ray dataset with 112,120 frontal X-ray images and 14 thoracic disease labels: https://arxiv.org/abs/1705.02315
- Rajpurkar et al. proposed CheXNet, a DenseNet-121 model for pneumonia detection on chest X-rays: https://arxiv.org/abs/1711.05225
- Kermany et al. published a pediatric chest X-ray pneumonia dataset and showed that deep learning can identify medical diagnoses from retinal OCT and chest X-ray images. The Kaggle dataset used here is based on this data source: https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5

This project compares a custom CNN baseline with MobileNetV2 and DenseNet121 transfer-learning models.

## 3. Dataset Understanding

Dataset: Kaggle Chest X-Ray Images (Pneumonia) by Paul Mooney.

The original Kaggle validation split contains only 16 images, so the project creates a reproducible stratified 80/10/10 split across all downloaded images. The dataset is imbalanced toward pneumonia images, which makes recall, PR-AUC, and false negatives especially important.

Patient-level leakage cannot be fully audited because reliable patient IDs are not provided. If patient IDs were available, a grouped split should keep all images from one patient in only one split.

## 4. Preprocessing and Augmentation

Images are decoded as grayscale, resized to 224x224, converted to three channels, and scaled to `[0, 1]`. Augmentation is online and training-only. Each epoch sees 4,684 original training images; no augmented image files are written to disk.

Augmentations are intentionally mild: small rotation, zoom, horizontal flip, and contrast variation. Horizontal flipping is justified only because the task predicts pneumonia presence/absence rather than laterality.

## 5. Models and Training

Models compared:

| Model | Description |
| --- | --- |
| Baseline CNN | Custom CNN trained from scratch with class weights, dropout, and early stopping |
| MobileNetV2 | Frozen ImageNet-pretrained base with task-specific classification head |
| DenseNet121 | Frozen ImageNet-pretrained base with task-specific classification head |

Model selection is based on validation metrics. The test split is reserved for the selected final model and threshold.

## 6. Evaluation Metrics

Metrics used:

- accuracy
- precision
- recall / sensitivity
- F1-score
- ROC-AUC
- PR-AUC
- false negatives and false positives
- confusion matrix

For pneumonia screening, false negatives are especially important because they represent pneumonia images predicted as normal.

## 7. Results and Model Comparison

To be filled after running `final_project.ipynb`.

Expected generated files:

- `outputs/results/validation_model_comparison.csv`
- `outputs/results/model_selection_summary.csv`
- `outputs/results/selected_model_test_metrics.csv`
- `outputs/figures/selected_model_test_confusion_matrix.png`

## 8. Explainability

Grad-CAM is used to visualize image regions that influenced model predictions. The heatmaps should be interpreted cautiously: they can show whether the model appears to focus on plausible lung regions, but they do not prove clinical correctness.

Expected generated files:

- `outputs/results/gradcam_summary.csv`
- `outputs/figures/gradcam_*.png`

## 9. Limitations and Future Work

Limitations:

- Image-level split only; patient-level leakage cannot be fully ruled out.
- Dataset imbalance remains clinically relevant.
- The dataset may not represent all patient populations, hospitals, scanners, or acquisition protocols.
- External validation on another dataset was not performed.
- Grad-CAM interpretation was not reviewed by medical experts.
- Frozen transfer-learning models could be improved through fine-tuning.

Future work:

- fine-tune pretrained models
- test additional architectures and hyperparameters
- use patient-level metadata if available
- validate on external datasets
- perform subgroup/fairness analysis and calibration
- include medical expert review

## 10. Reproducibility

The repository includes:

- executable notebooks
- fixed random seed
- saved split manifest
- saved metrics, figures, and model artifacts
- `requirements.txt`
- `python_version.txt`
