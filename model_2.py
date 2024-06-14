# This model contains additional additional dilation layer and a sigmoid activation. 

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

    # Dilated convolution for better context capturing
    conv3 = layers.Conv2D(
        64,
        (3, 3),
        dilation_rate=2,
        padding="same",
        kernel_regularizer=regularizers.l2(0.001),
    )(relu2)
    bn3 = layers.BatchNormalization()(conv3)
    relu3 = layers.ReLU()(bn3)

    # Decoder
    conv4 = layers.Conv2D(
        64, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(relu3)
    bn4 = layers.BatchNormalization()(conv4)
    relu4 = layers.ReLU()(bn4)

    conv5 = layers.Conv2D(
        32, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(relu4)
    bn5 = layers.BatchNormalization()(conv5)
    relu5 = layers.ReLU()(bn5)

    conv6 = layers.Conv2D(
        3, (3, 3), padding="same", kernel_regularizer=regularizers.l2(0.001)
    )(relu5)

    # Skip Connection
    output = layers.Add()([inputs, conv6])
    output = layers.Activation("sigmoid")(output)
    model = models.Model(inputs, output)

    return model


# mymodel = build_image_enhancement_model((400, 600, 3))
# mymodel.compile(optimizer="adam", loss="mse")

# mymodel.summary()


"""
 Total params: 113,603 (443.76 KB)
 Trainable params: 113,091 (441.76 KB)
 Non-trainable params: 512 (2.00 KB)
"""
