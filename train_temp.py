# Script to train the model so the app works immediately for the user.
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import os
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Robust path detection
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
print(f"Script Directory: {SCRIPT_DIR}")
print(f"Base Directory: {BASE_DIR}")

DATASET_DIR = os.path.join(BASE_DIR, 'brain_mri_dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'Training')
TEST_DIR = os.path.join(DATASET_DIR, 'Testing')
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, 'cerebrascan_xception_model.h5')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10 # Short run to get file for user, Xception converges fast.

print(f"Dataset Path: {DATASET_DIR}")
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(f"Training directory not found at {TRAIN_DIR}")

print("Loading data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation", seed=SEED,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='categorical'
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

def build_model(num_classes):
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    # Use Rescaling layer instead of preprocess_input function to avoid "Unknown layer: TrueDivide" error
    # Xception expects inputs in [-1, 1]. x is [0, 255].
    # (x / 127.5) - 1  => scale=1/127.5, offset=-1
    x = layers.Rescaling(scale=1./127.5, offset=-1)(x) 
    
    base_model = applications.Xception(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )
    base_model.trainable = False 
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

model = build_model(4)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, 
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

print("Saving...")
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
print("Done.")
