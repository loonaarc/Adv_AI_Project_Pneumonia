# Moodle Entry: Pneumonia Detection from Chest X-Rays

## ML Task

Binary image classification for medical diagnosis support.

The model receives a chest X-ray image and predicts whether the image belongs to one of two classes:

- `NORMAL`
- `PNEUMONIA`

## Input and Output

| Item | Description |
| --- | --- |
| Input | Grayscale chest X-ray image |
| Output | Binary class label |
| Classes | `NORMAL`, `PNEUMONIA` |
| Main dataset | Kaggle Chest X-ray Pneumonia dataset |
| Dataset link | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |

## Planned Model Architectures

The project compares at least two model approaches:

1. **Custom CNN baseline**
   - trained from scratch
   - used as a simple reference model

2. **Transfer learning model**
   - pre-trained architecture such as ResNet50, VGG16, or DenseNet121
   - adapted to binary pneumonia classification

## Planned Pipeline

- load and inspect the chest X-ray dataset
- create a reproducible train/validation/test split
- resize images to `224 x 224`
- normalize pixel values
- convert grayscale images to 3 channels when needed
- apply data augmentation to the training split
- handle class imbalance with class weighting or a similar method
- train and compare multiple models
- evaluate results on a held-out test set

## Evaluation Metrics

The models will be evaluated using:

- accuracy
- precision
- recall
- F1-score
- confusion matrix
- training and validation loss curves
- training and validation accuracy curves

Recall is especially important because false negatives in pneumonia detection can be medically risky.

## Explainability

Grad-CAM will be used to visualize which image regions influence the model predictions.

The Grad-CAM heatmaps will help check whether the model focuses on clinically relevant lung regions instead of irrelevant artifacts or borders.

## Expected Outcome

The final submission will compare the baseline CNN with a transfer learning model, discuss model performance and limitations, and include Grad-CAM examples for interpretability.
