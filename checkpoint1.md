# Checkpoint 1: Pneumonia Detection from Chest X-Rays

## Introduction

This project develops deep learning models to detect pneumonia from chest X-ray images. The task is a binary image classification problem with two output classes: **NORMAL** and **PNEUMONIA**.

The project compares a custom convolutional neural network (CNN) with transfer learning approaches based on pre-trained computer vision architectures. It also uses Grad-CAM to visualize which image regions influence the model predictions.

## Project Goal

The goal is to develop and compare deep learning models for automatic pneumonia detection from chest X-ray images, with attention to both predictive performance and interpretability.

The comparison between a custom CNN and pre-trained architectures allows us to analyze trade-offs between:

- model complexity
- training efficiency
- generalization performance
- interpretability of predictions

## Motivation

Pneumonia is a serious respiratory disease, and early detection is important for effective treatment. Manual diagnosis from chest X-rays can be time-consuming and may vary between readers.

AI-based systems can support clinicians by providing faster and more consistent prediction assistance. However, in medical AI applications, accuracy alone is not enough. Interpretability is also important because users need to understand whether the model focuses on clinically meaningful image regions.

## State of the Art

Recent advances in deep learning, especially convolutional neural networks, have achieved strong results in medical image classification tasks such as pneumonia detection from chest X-rays.

Rajpurkar et al. introduced **CheXNet**, a DenseNet121-based model trained on the NIH ChestX-ray14 dataset, and reported performance comparable to radiologists for pneumonia detection.

Transfer learning approaches using architectures such as **ResNet50**, **VGG16**, and **DenseNet121** are widely used because they benefit from pre-training on large image datasets such as ImageNet. This is especially useful when the available medical dataset is limited.

Many deep learning models are difficult to interpret, which is a limitation in clinical contexts. Therefore, explainable AI methods such as **Grad-CAM** are commonly used to highlight image regions that contribute most strongly to a prediction.

## Example References

- Rajpurkar et al., *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning*, 2017.
- He et al., *Deep Residual Learning for Image Recognition*, 2015.
- Simonyan and Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, 2014.

## ML Task Definition

| Item | Description |
| --- | --- |
| Task type | Supervised learning: binary classification |
| Input | Grayscale chest X-ray images |
| Output | `NORMAL` or `PNEUMONIA` |
| Main dataset | Kaggle Chest X-ray Pneumonia dataset |
| Dataset link | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |
| Optional dataset | NIH Chest X-ray dataset |
| Optional dataset link | https://www.kaggle.com/datasets/nih-chest-xrays/data |

## Data Preparation and Training Pipeline

Before training, all images are preprocessed to ensure consistent input for the models.

Planned preprocessing steps:

- resize images to a fixed resolution, for example `224 x 224`
- normalize pixel values
- convert grayscale images to 3 channels if required by pre-trained models
- create a reproducible train/validation/test split

Planned data augmentation techniques:

- random rotation
- horizontal flipping
- zooming
- small shifts or translations
- brightness or contrast adjustments

These augmentation methods help reduce overfitting and improve generalization.

## Dataset Split

The original Kaggle dataset contains train, validation, and test folders, but the provided validation split is very small. Therefore, a reproducible stratified split will be used for model training and evaluation.

Possible split:

- training set: 80%
- validation set: 10%
- test set: 10%

The same split must be used for all model comparisons.

## Handling Class Imbalance

Medical datasets are often imbalanced. In this dataset, pneumonia images are more frequent than normal images. To reduce bias toward the majority class, the project may use:

- class weighting
- oversampling
- balanced batch sampling

Class weighting is the preferred first approach because it is simple and does not duplicate training images.

## Models

Two model families will be implemented and compared.

### 1. Custom CNN

The custom CNN is designed from scratch. It provides a simple baseline and helps us understand CNN architecture choices and performance trade-offs.

Expected components:

- convolution layers
- max pooling
- dropout regularization
- dense classification head
- sigmoid output for binary classification

### 2. Transfer Learning Models

Pre-trained architectures will be used as stronger comparison models.

Possible architectures:

- ResNet50
- VGG16
- DenseNet121

Transfer learning is expected to improve performance and training efficiency because the base models already learned general visual features from large image datasets.

## Fair Model Comparison

To ensure a fair comparison between models:

- use the same dataset split for all models
- apply identical preprocessing and augmentation steps where appropriate
- use the same evaluation metrics
- use similar training settings where possible, such as optimizer, batch size, and number of epochs
- report both aggregate metrics and confusion matrices

This helps ensure that performance differences mainly reflect model architecture or training choices rather than inconsistent experimental setup.

## Evaluation

Model performance will be evaluated using:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- training and validation loss curves
- training and validation accuracy curves

These metrics are important in medical diagnosis tasks because false negatives and false positives have different practical consequences. In pneumonia screening, false negatives are especially important because a missed pneumonia case could delay treatment.

## Explainability Approach

Grad-CAM will be used to generate heatmaps showing which regions of the X-ray influence the model predictions.

This is important because:

- it supports transparency
- it helps check whether the model focuses on lung regions
- it can reveal whether the model is using irrelevant image artifacts

Grad-CAM results will be interpreted cautiously. A heatmap can show where the model focuses, but it does not prove true clinical reasoning.

## Ethical Considerations and Dataset Bias

Medical imaging datasets may contain biases related to:

- patient demographics
- imaging devices
- hospital-specific acquisition procedures
- dataset collection process

As a result, a model trained on one public dataset may not generalize reliably to real clinical environments. This project therefore focuses on model comparison, evaluation, and interpretability rather than claiming clinical applicability.

## Conclusion

This project evaluates performance and interpretability in deep learning-based pneumonia detection from chest X-rays. By comparing a custom CNN with transfer learning models and applying Grad-CAM, the project aims to show how deep learning can support medical image classification while remaining transparent about limitations.
