# Ensemble Learning for Medical Image Classification

This repository provides algorithmic pseudocode and reference implementations
for an ensemble learning framework using deep convolutional neural networks
(InceptionV3, ResNet50, and VGG16).

The framework combines multiple CNN models with probability-based ensemble
strategies to improve classification robustness in medical image analysis.

---

## Methods

The proposed framework integrates the following ensemble strategies:

- Support Vector Machine (SVM) stacking  
- Multi-Layer Perceptron (MLP) stacking  
- Confidence-weighted inverted bell curve ensemble fusion  

---

## Experimental Protocol

- CNN base models are trained using transfer learning and a fixed train–test split.
- Ensemble (meta-)models are evaluated using **true 5-fold cross-validation**.
- Out-of-fold probability usage is employed to prevent data leakage during
  ensemble training.
- Performance is reported using **macro-averaged Accuracy, Precision, Recall,
  and F1-score**, along with **fold-wise confusion matrices**.

---

## Dataset and Class Naming

The dataset consists of four semantic classes: **Cyst**, **Normal**, **Stone**,
and **Tumor**.  
Class ordering in the code follows the directory structure used during data
loading and **does not affect evaluation metrics or experimental conclusions**.

---

## Repository Structure

- `pseudocode/` : Algorithm-level pseudocode (journal-ready)
- `code/`       : Python reference implementations

---

## Reproducibility

All algorithmic steps are documented through pseudocode and reference
implementations. Software dependencies are listed in `requirements.txt`.
Due to data size and privacy considerations, datasets and trained model
weights are not included in this repository.

---

## Citation

If you use this repository in your research, please cite it.
A `CITATION.cff` file is provided to enable automated citation generation
via GitHub.

---

## Relation to Associated Manuscript

This repository provides a reproducible implementation of the methods
described in the associated manuscript:

“Weighted Deep Classifier Fusion and Soft Classifier Output Fusion for
Kidney Disease Detection from CT Scan Images.”

The implementation extends the manuscript by incorporating true
5-fold cross-validation for all ensemble (fusion) models, providing
more robust and unbiased performance estimation.


---

## Note

This repository is intended solely for academic and research purposes.
