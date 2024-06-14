# An improvement in RAM utilisation from model_0, wherein filter sizes 32 and 64 are used.
# First model to be built on local machine.

from keras import layers, models, regularizers


def build_image_enhancement_model(input_shape):
    inputs = layers.Input(shape=input_shape)  # (400, 600, 3)

    # Encoder
    conv1 = layers.Conv2D(
        32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(inputs)
    bn1 = layers.BatchNormalization()(conv1)
    relu1 = layers.ReLU()(bn1)

    conv2 = layers.Conv2D(
        64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(relu1)
    bn2 = layers.BatchNormalization()(conv2)
    relu2 = layers.ReLU()(bn2)

    # Decoder
    conv3 = layers.Conv2D(
        64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(relu2)
    bn3 = layers.BatchNormalization()(conv3)
    relu3 = layers.ReLU()(bn3)

    conv4 = layers.Conv2D(
        32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(relu3)
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
 Total params: 76,419 (298.51 KB)
 Trainable params: 76,035 (297.01 KB)
 Non-trainable params: 384 (1.50 KB)
"""
