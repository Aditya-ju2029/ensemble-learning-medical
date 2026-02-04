# Ensemble Learning for Medical Image Classification

This repository provides algorithmic pseudocode and reference implementations
for an ensemble learning framework using deep convolutional neural networks
(InceptionV3, ResNet50, and VGG16).

The framework combines multiple CNN models with probability-based ensemble
strategies to improve classification robustness.

## Methods
The proposed framework integrates:
- Support Vector Machine (SVM) stacking
- Multi-Layer Perceptron (MLP) stacking
- Confidence-weighted inverted bell curve ensemble fusion

## Experimental Protocol
- CNN base models are trained using transfer learning and a fixed train–test split.
- Ensemble models are evaluated using true 5-fold cross-validation at the
  meta-model level.
- Out-of-fold probability usage is employed to prevent data leakage.
- Performance is reported using macro-averaged Accuracy, Precision, Recall,
  and F1-score, along with fold-wise confusion matrices.

## Repository Structure
- `pseudocode/` : Algorithm-level pseudocode (journal-ready)
- `code/`       : Python reference implementations

## Reproducibility
All algorithmic steps are documented through pseudocode and reference
implementations. Software dependencies are listed in `requirements.txt`.
Due to data size and privacy considerations, datasets and trained model
weights are not included.

## Citation
If you use this repository in your research, please cite it. A `CITATION.cff`
file is provided for automated citation generation.

## Note
This repository is intended solely for academic and research purposes.
