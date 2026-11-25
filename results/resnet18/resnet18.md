
# ResNet18 Baseline Results

## Model
- Architecture: ResNet18
- Training: From scratch
- Input: SAR images (VV and VH)
- Image resolution: 256 × 256
- Classes: Oil, No_oil, Lookalike

## Validation Results
- Accuracy: 83.01%
- Macro F1-score: 0.8180
- Multiclass Log Loss: 0.5016
- Training epochs: 5

## Confusion Matrix
See `resnet18_confusion_matrix.png`.

## Observations
The model performed best at identifying No_oil images.
It had more difficulty distinguishing Lookalike images
from Oil and No_oil images.
