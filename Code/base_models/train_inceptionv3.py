import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMG_SIZE = (299, 299)
BATCH_SIZE = 50
EPOCHS = 25
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

base = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))
for layer in base.layers[:-10]:
    layer.trainable = False

x = GlobalAveragePooling2D()(base.output)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(base.input, out)
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    ModelCheckpoint("best_inceptionv3.keras", save_best_only=True),
    EarlyStopping(patience=8, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.2, min_lr=1e-6)
]

model.fit(train_loader, epochs=EPOCHS, validation_data=test_loader, callbacks=callbacks)

best = load_model("best_inceptionv3.keras")
y_pred = np.argmax(best.predict(test_loader), axis=1)

print(confusion_matrix(test_loader.classes, y_pred))
print(classification_report(test_loader.classes, y_pred,
                            target_names=list(test_loader.class_indices.keys())))

model.save("inceptionv3_final.keras")
