import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

IMG_SIZE = (224, 224)
BATCH_SIZE = 50
EPOCHS = 30
NUM_CLASSES = 4

train_gen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

test_gen = ImageDataGenerator(rescale=1 / 255)

train_loader = train_gen.flow_from_directory(
    "Training", IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

test_loader = test_gen.flow_from_directory(
    "Testing", IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base.layers[:-10]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base.input, out)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint("best_resnet50.keras", save_best_only=True),
    ReduceLROnPlateau(patience=3, factor=0.2, min_lr=1e-6)
]

model.fit(train_loader, epochs=EPOCHS, validation_data=test_loader, callbacks=callbacks)

best = load_model("best_resnet50.keras")
y_prob = best.predict(test_loader)
y_pred = np.argmax(y_prob, axis=1)
y_true = test_loader.classes
labels = list(test_loader.class_indices.keys())

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=labels))

model.save("resnet50_final.keras")
