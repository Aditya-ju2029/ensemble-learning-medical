Algorithm 4: MLP-Based Stacking Ensemble (5-Fold CV)

Input:
    - Out-of-fold probability matrix P
    - True class labels Y
    - Number of folds K

Output:
    - Fold-wise performance metrics

Steps:
1. Split (P, Y) into K folds.
2. For each fold k = 1 to K:
       a. Use fold k as validation set.
       b. Use remaining folds as training set.
       c. Apply feature standardization on training probabilities.
       d. Train MLP classifier on standardized probabilities.
       e. Predict class labels for validation fold.
       f. Compute Accuracy, Precision, Recall, and F1-score.
3. End For
4. Aggregate fold-wise metrics and report mean ± standard deviation.
