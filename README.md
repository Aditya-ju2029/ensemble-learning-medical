# ensemble-learning-medical
Ensemble learning with CNNs using stacking and confidence-weighted fusion
# Ensemble Learning for Medical Image Classification

This repository contains pseudocode and implementation for an ensemble learning
framework using deep CNN models (InceptionV3, ResNet50) combined with:

- SVM stacking
- MLP stacking
- Confidence-weighted inverted bell curve ensemble

## Key Features
- True 5-fold cross-validation
- Out-of-fold probability generation
- Macro-averaged evaluation metrics
- Fold-wise confusion matrix analysis

## Repository Structure
- `pseudocode/` : Algorithm-level pseudocode (journal-ready)
- `code/`       : Python implementations
- `results/`    : Fold-wise evaluation outputs

## Note
This repository is intended for academic and research purposes.
