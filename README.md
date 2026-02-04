# Ensemble Learning for Medical Image Classification

This repository provides algorithmic pseudocode and reference implementations
for an ensemble learning framework using deep convolutional neural networks
(InceptionV3, ResNet50, and VGG16) combined with multiple ensemble strategies.

The proposed framework integrates:
- Support Vector Machine (SVM) stacking
- Multi-Layer Perceptron (MLP) stacking
- Confidence-weighted inverted bell curve ensemble fusion

## Key Features
- Transfer learning–based CNN models
- Probability-based ensemble learning
- True 5-fold cross-validation at the ensemble (meta-model) level
- Out-of-fold probability usage for leak-free ensemble training
- Macro-averaged evaluation metrics
- Fold-wise confusion matrix analysis

## Repository Structure
- `pseudocode/` : Journal-ready, algorithm-level pseudocode
- `code/`       : Python reference implementations

## Reproducibility
The repository includes detailed pseudocode and implementation scripts to
ensure transparency and reproducibility. Due to data size and privacy
constraints, datasets and trained model weights are not included.

## Note
This repository is intended solely for academic and research purposes.
