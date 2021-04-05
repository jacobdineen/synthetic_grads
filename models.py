from tensorflow import keras
from tensorflow.keras import layers


def get_4_layer(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64),
            layers.Dense(128),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def get_8_layer(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(256, kernel_size=(2, 2), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(512, kernel_size=(1, 1), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64),
            layers.Dense(128),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def get_VGG16(input_shape=(48, 48, 3), num_classes=10):
    from tensorflow.keras.applications import VGG16

    model = VGG16(include_top=False, input_shape=input_shape)
    return model

