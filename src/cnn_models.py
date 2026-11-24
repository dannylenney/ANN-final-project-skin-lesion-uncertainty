"""CNN classifiers for the image models (scratch CNN, #13)."""

import tensorflow as tf

SCRATCH_FILTERS = [32, 64, 128, 256, 256]


def build_scratch_cnn(input_size=224, n_classes=7):
    """Small CNN trained from scratch: 5 conv blocks, global average pooling, dropout, softmax.

    Each block is one 3x3 Conv (he_normal, ReLU) and a 2x2 MaxPool, so a 224 input ends
    on a 7x7 map. No BatchNormalization or other normalization layers, so MC dropout
    (training=True) only changes the dropout layer.
    """
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    x = inputs
    for filters in SCRATCH_FILTERS:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu",
                                   kernel_initializer="he_normal")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="scratch_cnn")
