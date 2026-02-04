import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

TEST_DIR = "Testing"
WEIGHTS_PATH = "inceptionv3_final_weights.keras"

IMG_SIZE = (299, 299)
BATCH_SIZE = 50
NUM_CLASSES = 4

test_gen = ImageDataGenerator(rescale=1 / 255)

test_loader = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

base_model = InceptionV3(
    weights="imagenet",
    include_top=False,
    input_shape=(299, 299, 3)
)

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.load_weights(WEIGHTS_PATH)

model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

y_prob = model.predict(test_loader)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_loader.classes

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

results = pd.DataFrame({
    "Filename": test_loader.filenames,
    "True_Class": y_true,
    "Predicted_Class": y_pred,
    "Prob_Class_0": y_prob[:, 0],
    "Prob_Class_1": y_prob[:, 1],
    "Prob_Class_2": y_prob[:, 2],
    "Prob_Class_3": y_prob[:, 3]
})

results.to_csv(
    "inceptionv3_predictions_with_probabilities.csv",
    index=False
)
