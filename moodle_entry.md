Moodle Entry
ML Task
Binary image classification for medical diagnosis (pneumonia detection from chest X-ray images)

Input
Grayscale chest X-ray images (2D medical images)

Output
Binary class label:
Pneumonia
Normal

Model Architecture(s)
Custom Convolutional Neural Network (CNN)
Transfer Learning models:
ResNet50
VGG16
optionally DenseNet121
Fine-tuning of pre-trained models

Planned Pipeline
Image resizing and normalization
Data augmentation
Train/validation/test split
Handling class imbalance through class weighting or oversampling

Evaluation Metrics
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Training vs Validation loss/accuracy curves

Data Source(s)
Kaggle Chest X-ray Pneumonia dataset
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Optional:
NIH Chest X-ray dataset
https://www.kaggle.com/datasets/nih-chest-xrays/data

Explainability Approach
Grad-CAM (Gradient-weighted Class Activation Mapping)
Visualization of relevant image regions influencing predictions
Analysis of model interpretability in medical imaging tasks
