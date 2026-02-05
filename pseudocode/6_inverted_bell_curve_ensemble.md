## Algorithm 6: Inverted Bell Curve Ensemble with True 5-Fold Cross-Validation

### Input
- Out-of-fold probability predictions from multiple CNN models  
- True class labels \( Y \)  
- Number of folds \( K \)  

### Output
- Ensemble predictions  
- Performance metrics  

### Steps

1. Define the inverted bell curve weighting function:
   w(x) = (1 / a) * exp(−(x − b)^2 / (2c^2))

2. Split the dataset into \( K \) folds.  

3. For each fold \( k = 1 \) to \( K \):  
   a. Use fold \( k \) as the validation set.  
   b. Use the remaining \( K - 1 \) folds as the training set.  
   c. Optimize parameters \( (a, b, c) \) using **only** the training folds.  
      Parameter optimization is performed exclusively within the training
      folds of each cross-validation iteration.  
   d. Apply the optimized parameters to the validation fold probability
      predictions.  
   e. Compute weighted ensemble predictions for the validation fold.  
   f. Calculate Accuracy, Precision, Recall, and F1-score.  

4. End For  

5. Aggregate fold-wise results and report the mean ± standard deviation
   of the evaluation metrics.  

**End Algorithm**
