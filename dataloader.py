import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img


def get_MNIST(num_classes=10, vgg_preprocess=False):
    """
    VGG16 requires special preprocessing. We need images to be size > 32
    and also need the channel dim to equal 3.

    No specification is made on whether or not the other two models follow this cleaning paradigm.
    """
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if vgg_preprocess:
        x_train = np.dstack([x_train] * 3)
        x_test = np.dstack([x_test] * 3)
        x_train = x_train.reshape(-1, 28, 28, 3)
        x_test = x_test.reshape(-1, 28, 28, 3)
        x_train = np.asarray(
            [
                img_to_array(array_to_img(im, scale=False).resize((48, 48)))
                for im in x_train
            ]
        )
        x_test = np.asarray(
            [
                img_to_array(array_to_img(im, scale=False).resize((48, 48)))
                for im in x_test
            ]
        )

    else:
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    return (x_train, y_train, x_test, y_test)
