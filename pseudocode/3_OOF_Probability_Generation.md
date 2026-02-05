## Algorithm 3: Out-of-Fold (OOF) Probability Usage for Ensemble Learning

### Input
- Probability prediction CSV files generated from trained base CNN models  
- True class labels for all samples  
- Number of folds \( K \) for ensemble-level cross-validation  

### Output
- Fold-wise training and validation splits of probability features  
- Out-of-fold (OOF) predictions for ensemble meta-models  

### Steps

1. Load probability prediction files produced by the base CNN models:  
   a. Each file contains sample identifiers, true labels, and class
      probability scores.  
   b. No retraining of CNN models is performed at this stage.  

2. Construct a probability-based feature matrix \( P \) by concatenating
   probability vectors from all base models for each sample.  

3. Initialize \( K \)-fold cross-validation at the ensemble (meta-model) level.  

4. For each fold \( k = 1 \) to \( K \):  
   a. Split the probability feature matrix \( P \) and label vector \( Y \) into:  
      - Training subset consisting of \( K - 1 \) folds  
      - Validation subset consisting of 1 fold  

   b. Train the ensemble meta-model using **only** the training subset.  

   c. Generate predictions for the validation subset.  

   d. Store the predictions as out-of-fold (OOF) predictions for fold \( k \).  

5. Repeat Step 4 until all folds have been processed and each sample has
   been used exactly once as validation data at the ensemble level.  

6. Aggregate fold-wise OOF predictions to compute evaluation metrics such as
   Accuracy, Precision, Recall, and F1-score.  

**End Algorithm**
