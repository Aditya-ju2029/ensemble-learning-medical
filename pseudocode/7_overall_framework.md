Algorithm 7: Overall Ensemble Learning Framework

Input:
    - Medical image dataset
    - Pre-trained CNN architectures

Output:
    - Final ensemble evaluation results

Steps:
1. Train base CNN models using 5-fold cross-validation.
2. Generate out-of-fold probability predictions.
3. Apply ensemble strategies:
       a. SVM stacking
       b. MLP stacking
       c. Inverted bell curve ensemble
4. Perform true 5-fold evaluation.
5. Report mean ± standard deviation of results.

The framework is modular and allows independent replacement
of base models or ensemble strategies.
