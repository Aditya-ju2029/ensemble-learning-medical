## Algorithm 2: Model Inference and Probability Extraction

### Input
- Trained CNN model file  
- Test image dataset organized in class-wise directories  
- Image input size  
- Number of classes \( C \)

### Output
- Predicted class labels  
- Class probability vectors for each test sample  
- CSV file containing predictions and probabilities  

### Steps

1. Initialize an image data generator with rescaling.  

2. Load the test dataset using a directory-based data loader:  
   a. Resize images to the required input size.  
   b. Disable data shuffling to preserve filename order.  
   c. Assign sparse integer labels to samples.  

3. Load the trained CNN model from disk.  

4. Perform model inference **without updating model parameters**:  
   a. Generate class probability predictions for all test samples.  
   b. Compute predicted class labels by selecting the class with the maximum probability.  

5. Retrieve ground-truth class labels from the data loader.  

6. Evaluate model performance:  
   a. Compute the confusion matrix using true and predicted labels.  
   b. Generate a classification report including precision, recall, and F1-score.  

7. Construct a results table containing:  
   a. Image filename  
   b. True class label  
   c. Predicted class label  
   d. Probability score for each class  

8. Save the results table as a CSV file for further analysis or ensemble learning.  

**Note:** Inference is performed using trained base models on a held-out test set or designated evaluation data.
