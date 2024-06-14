# filter sizes 64 & 128 and thus the model was very elaborate to compute on local machine.
# This model was thus not fully built.

from keras import layers, models, regularizers


def build_image_enhancement_model(input_shape):
    inputs = layers.Input(shape=input_shape)  # (400, 600, 3)

    # Encoder
    conv1 = layers.Conv2D(
        64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(inputs)
    bn1 = layers.BatchNormalization()(conv1)
    relu1 = layers.ReLU()(bn1)
    drop1 = layers.Dropout(0.3)(relu1)

    conv2 = layers.Conv2D(
        128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(drop1)
    bn2 = layers.BatchNormalization()(conv2)
    relu2 = layers.ReLU()(bn2)
    drop2 = layers.Dropout(0.3)(relu2)

    # Decoder
    conv3 = layers.Conv2D(
        128, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(drop2)
    bn3 = layers.BatchNormalization()(conv3)
    relu3 = layers.ReLU()(bn3)
    drop3 = layers.Dropout(0.3)(relu3)

    conv4 = layers.Conv2D(
        64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(drop3)
    bn4 = layers.BatchNormalization()(conv4)
    relu4 = layers.ReLU()(bn4)

    conv5 = layers.Conv2D(
        3, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(relu4)

    # Skip Connection
    output = layers.Add()([inputs, conv5])
    model = models.Model(inputs, output)

    return model


mymodel = build_image_enhancement_model((400, 600, 3))
mymodel.compile(optimizer="adam", loss="mse")

mymodel.summary()
"""
 Total params: 300,291 (1.15 MB)
 Trainable params: 299,523 (1.14 MB)
 Non-trainable params: 768 (3.00 KB)
"""
