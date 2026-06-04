Checkpoint 1
Introduction
This project aims to develop a deep learning model to detect pneumonia from chest X-ray images using Convolutional Neural Networks (CNNs). The project compares a custom CNN with transfer learning approaches based on pre-trained architectures and uses Grad-CAM to visualize how the models make their predictions.

Project Goal
Develop and compare deep learning models for automatic pneumonia detection from chest X-ray images, focusing on both performance and interpretability. The comparison between custom and pre-trained architectures allows analysis of trade-offs between model complexity, training efficiency, and accuracy.

Motivation
Pneumonia is a serious respiratory disease, and early detection is crucial for effective treatment. Manual diagnosis from chest X-rays can be time-consuming and subjective. AI-based systems can assist clinicians by providing faster and more consistent predictions. In addition to prediction accuracy, interpretability is especially important in medical AI applications to increase trust and transparency.

State of the Art
Recent advances in deep learning, particularly Convolutional Neural Networks (CNNs), have achieved strong performance in medical image classification tasks such as pneumonia detection from chest X-rays. Rajpurkar et al. (2017) introduced CheXNet, a DenseNet121-based model trained on the NIH ChestX-ray14 dataset, achieving performance comparable to radiologists for pneumonia detection.
Transfer learning approaches using architectures such as ResNet50, VGG16, and DenseNet121 are widely used because they benefit from pre-training on large datasets such as ImageNet. Previous research has shown that transfer learning improves accuracy and reduces overfitting, especially when working with limited medical datasets.
However, many deep learning models lack interpretability, which is critical in clinical applications. Therefore, explainable AI techniques such as Grad-CAM are increasingly used to visualize which image regions influence model predictions.
Example References
Rajpurkar et al., CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning (2017)
He et al., Deep Residual Learning for Image Recognition (ResNet)
Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG16)

ML Task Definition
Task Type
Supervised Learning → Binary Classification

Input
Chest X-ray images (grayscale medical images)
Public dataset:
Kaggle Chest X-ray Pneumonia dataset
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Optional:
NIH Chest X-ray dataset
https://www.kaggle.com/datasets/nih-chest-xrays/data

Output
Binary classification:
Pneumonia
Normal

Data Preparation and Training Pipeline
Before training, all images will be preprocessed to ensure consistent input for the models.
Planned preprocessing steps:
Resize images to a fixed resolution (e.g., 224×224)
Normalize pixel values
Convert grayscale images into 3-channel format if required by pre-trained models
Planned data augmentation techniques:
Random rotation
Horizontal flipping
Zooming
Small shifts/translations
Brightness adjustments
These augmentation methods help reduce overfitting and improve model generalization.
Dataset Split
The dataset will be divided into:
Training set
Validation set
Test set
A split such as 70/15/15 or 80/10/10 will be used.
Handling Class Imbalance
Since medical datasets are often imbalanced, techniques such as:
class weighting
oversampling
balanced batch sampling
may be used to reduce bias toward the majority class.

Models
Two approaches will be implemented and compared.
1. Custom CNN
Designed from scratch
Allows full control over architecture and hyperparameters
Useful for understanding CNN design principles and performance trade-offs
2. Transfer Learning Models
Pre-trained architectures such as:
ResNet50
VGG16
optionally DenseNet121
will be fine-tuned for pneumonia classification.
Transfer learning is expected to improve performance and training efficiency due to pre-trained feature extraction capabilities.

Fair Model Comparison
To ensure a fair comparison between models:
the same dataset split will be used for all models
identical preprocessing and augmentation steps will be applied
the same evaluation metrics will be used
similar training settings such as optimizer, batch size, and number of epochs will be used where possible
This ensures that performance differences mainly reflect differences in model architecture rather than experimental setup.

Evaluation
Model performance will be evaluated using:
Accuracy
Precision
Recall
F1-score
Additionally:
Confusion Matrix for detailed analysis
Training vs validation accuracy/loss curves for overfitting analysis
These metrics are particularly important in medical diagnosis tasks, where false negatives and false positives have significant consequences.

Explainability Approach
Grad-CAM (Gradient-weighted Class Activation Mapping) will be used to generate heatmaps highlighting which regions of the X-ray images influence the model’s predictions.
This is particularly important in medical applications, where transparency and trust are required. Grad-CAM visualizations help verify whether the model focuses on clinically relevant lung regions instead of irrelevant image features.

Ethical Considerations and Dataset Bias
Medical imaging datasets may contain biases related to:
patient demographics
imaging devices
hospital-specific acquisition procedures
As a result, models trained on one dataset may not generalize perfectly to other clinical environments.
Additionally, explainability methods such as Grad-CAM provide visual indications of important regions but do not guarantee true clinical reasoning.
Therefore, this project focuses on careful evaluation and interpretability rather than claiming clinical applicability.

Conclusion
This project evaluates both performance and interpretability in deep learning-based pneumonia detection from chest X-rays. By comparing custom CNN architectures with transfer learning approaches and applying explainable AI techniques such as Grad-CAM, the project aims to contribute to understanding how deep learning can support medical diagnosis while maintaining transparency and trustworthiness.


