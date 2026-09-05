#!/usr/bin/env python3
"""
Trains an image classifier on the Caltech101 dataset using transfer
learning with a two-phase (frozen head -> fine-tune) strategy.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 102  # 101 object classes + background
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 15
UNFREEZE_LAYERS = 40
SEED = 42

tf.random.set_seed(SEED)


def build_data_augmentation():
    """Common augmentation pipeline, seeded for reproducibility."""
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal", seed=SEED),
            keras.layers.RandomRotation(0.15, seed=SEED),
            keras.layers.RandomZoom(0.15, seed=SEED),
            keras.layers.RandomContrast(0.1, seed=SEED),
        ]
    )


def prepare_datasets():
    """Loads Caltech101 via tfds and builds train/val tf.data pipelines."""
    (train_raw, val_raw), info = tfds.load(
        "caltech101",
        split=["train", "test"],
        with_info=True,
        as_supervised=True,
    )

    preprocess_input = keras.applications.mobilenet_v2.preprocess_input

    def preprocess(image, label):
        image = tf.image.resize(image, IMG_SIZE)
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        return image, label

    train_ds = (
        train_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(2000, seed=SEED)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_raw.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


def build_model(data_augmentation):
    """Builds the full model: augmentation -> frozen backbone -> head."""
    base_model = keras.applications.MobileNetV2(
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_top=False,
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3, seed=SEED)(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base_model


def unfreeze_top_layers(base_model, num_layers):
    """Unfreezes the top num_layers layers, keeping BatchNorm frozen."""
    base_model.trainable = True
    freeze_until = len(base_model.layers) - num_layers

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def train_transfer_model():
    """
    Builds, trains, and saves an image classifier for Caltech101
    using transfer learning with MobileNetV2.

    Returns:
        The trained Keras Model.
    """
    train_ds, val_ds = prepare_datasets()
    data_augmentation = build_data_augmentation()
    model, base_model = build_model(data_augmentation)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            "caltech101_model.h5", monitor="val_accuracy", save_best_only=True
        ),
    ]

    # Phase 1: train classification head only
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    # Phase 2: unfreeze top layers and fine-tune with a lower LR
    unfreeze_top_layers(base_model, UNFREEZE_LAYERS)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks,
    )

    model.save("caltech101_model.h5")
    return model


if __name__ == "__main__":
    train_transfer_model()
