Algorithm 6: Inverted Bell Curve Ensemble with True 5-Fold CV

Input:
    - Out-of-fold probability predictions from multiple CNN models
    - True class labels Y
    - Number of folds K

Output:
    - Ensemble predictions and performance metrics

Steps:
1. Define inverted bell curve weighting function:
       w(x) = (1 / a) * exp(−(x − b)^2 / (2c^2))
2. Split dataset into K folds.
3. For each fold k = 1 to K:
       a. Use fold k as validation set.
       b. Use remaining folds as training set.
       c. Optimize parameters (a, b, c) using training folds.
       d. Apply optimized parameters to validation fold probabilities.
       e. Compute weighted ensemble predictions.
       f. Calculate Accuracy, Precision, Recall, and F1-score.
4. End For
5. Aggregate fold-wise results and report mean ± standard deviation.
