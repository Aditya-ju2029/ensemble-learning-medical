Algorithm 4: SVM-Based Stacking Ensemble with True 5-Fold Cross-Validation

Input:
    - Out-of-fold probability features from base CNN models
    - True class labels Y
    - Number of folds K

Output:
    - Fold-wise ensemble predictions and performance metrics

Steps:
1. Split the probability feature matrix and label vector into K folds.

2. For each fold k = 1 to K:
       a. Use fold k as the validation set.
       b. Use the remaining K−1 folds as the training set.
       c. Train a linear SVM classifier using only the training folds.
          Model training is performed exclusively on training folds
          within each cross-validation iteration.
       d. Predict class labels for the validation fold.
       e. Compute Accuracy, Precision, Recall, and F1-score.

3. End For

4. Aggregate fold-wise results and report the mean ± standard deviation
   of the evaluation metrics.
