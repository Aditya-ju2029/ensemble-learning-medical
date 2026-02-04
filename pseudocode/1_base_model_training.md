Algorithm 1: Base CNN Training using Transfer Learning

Input:
    - Labeled image dataset D, organized into class-wise directories
    - Pre-trained CNN architecture A ∈ {InceptionV3, VGG16, ResNet50}
    - Image input size specific to architecture A
    - Number of classes C
    - Training hyperparameters (batch size, learning rate, epochs)

Output:
    - Trained CNN model saved to disk

Steps:
1. Initialize image data generators:
       a. Apply rescaling and data augmentation to training images.
       b. Apply rescaling only to validation/test images.

2. Load training and validation datasets using directory-based generators:
       a. Resize images to architecture-specific input size.
       b. Encode labels using categorical format.

3. Load pre-trained CNN base model A with ImageNet weights:
       a. Exclude the original classification head.
       b. Set input shape according to architecture requirements.

4. Freeze lower layers of the base model:
       a. Keep most convolutional layers non-trainable.
       b. Unfreeze top layers for fine-tuning.

5. Attach custom classification head:
       a. Global Average Pooling layer.
       b. Fully connected dense layer with ReLU activation.
       c. Dropout layer for regularization.
       d. Final dense layer with softmax activation for C classes.

6. Compile the model:
       a. Use Adam optimizer with a small learning rate.
       b. Use categorical cross-entropy loss.
       c. Monitor classification accuracy.

7. Train the model:
       a. Fit the model on augmented training data.
       b. Validate performance on validation/test data.
       c. Apply callbacks:
              - Model checkpointing to save best model
              - Early stopping to prevent overfitting
              - Learning rate reduction on plateau

8. Load the best-performing model checkpoint.

9. Evaluate the trained model:
       a. Generate class probability predictions on validation/test data.
       b. Compute predicted class labels.
       c. Calculate confusion matrix and classification metrics.

10. Compute and plot ROC curves for each class.

11. Save the final trained model for later inference or ensemble learning.
