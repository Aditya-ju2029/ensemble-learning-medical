Algorithm 7: Overall Ensemble Learning Framework

Input:
    - Medical image dataset
    - Pre-trained CNN architectures (InceptionV3, ResNet50, VGG16)

Output:
    - Final ensemble evaluation results

Steps:
1. Train base CNN models using transfer learning and a fixed train–test split.
2. Perform model inference to obtain class probability predictions
   from each trained base CNN.
3. Construct probability-based feature representations by aggregating
   predictions from all base models.
4. Apply ensemble strategies using the probability features:
       a. Support Vector Machine (SVM) stacking
       b. Multi-Layer Perceptron (MLP) stacking
       c. Confidence-weighted inverted bell curve ensemble
5. Evaluate ensemble models using true 5-fold cross-validation with
   out-of-fold probability usage.
6. Report fold-wise performance metrics and aggregate results as
   mean ± standard deviation.
7. End.

The framework is modular and allows independent replacement of base
models or ensemble strategies.
