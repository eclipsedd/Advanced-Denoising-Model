"""
This is essentially the Google Colab implementation of model_2.py
The results were different as compared to the similar code implemented on local machine, owing to 
different version of tensorflow being used. 
Local Machine - 2.16.1
Google Colab - 2.15.0 {previous stable default version}

Complete code is presented here, which builds the model, trains it and finally saves it. 

"""

from keras import layers, models, regularizers
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import image


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


def psnr(y_true, y_pred):
    return image.psnr(y_true, y_pred, max_val=1.0)


def load_and_preprocess_image(file_path, target_size=(600, 400)):
    image = Image.open(file_path).resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1] range
    return image


def load_images(directory, target_size=(600, 400)):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(
            ".png"
        ):  # Add other image formats if needed
            img_path = os.path.join(directory, filename)
            img = load_and_preprocess_image(img_path, target_size)
            images.append(img)
    return np.array(images)


def show_performance_curve(training_result, metric, metric_label):
    train_perf = training_result.history[str(metric)]
    validation_perf = training_result.history["val_" + str(metric)]

    plt.plot(train_perf, label=metric_label)
    plt.plot(validation_perf, label="val_" + str(metric))

    if (
        metric in training_result.history
        and "val_" + str(metric) in training_result.history
    ):
        intersection_idx = np.argwhere(
            np.isclose(train_perf, validation_perf, atol=1e-2)
        ).flatten()
        if intersection_idx.size > 0:
            intersection_idx = intersection_idx[0]
            intersection_value = train_perf[intersection_idx]

            plt.axvline(
                x=intersection_idx, color="r", linestyle="--", label="Intersection"
            )

            plt.annotate(
                f"Optimal Value: {intersection_value:.4f}",
                xy=(intersection_idx, intersection_value),
                xycoords="data",
                fontsize=10,
                color="green",
            )

    plt.xlabel("Epoch")
    plt.ylabel(metric_label)
    plt.legend(loc="lower right")
    plt.title(f"Training and Validation {metric_label}")
    plt.show()


if __name__ == "__main__":
    # Load images
    dark_images = load_images("/content/drive/MyDrive/Img_Enhancement/Train/low")
    light_images = load_images("/content/drive/MyDrive/Img_Enhancement/Train/high")

    input_shape = (400, 600, 3)
    mymodel = build_image_enhancement_model(input_shape)

    METRICS = ["mse", "mae", psnr]

    mymodel.compile(optimizer="adam", loss="mse", metrics=METRICS)

    BATCH_SIZE = 16
    EPOCHS = 30

    training_history = mymodel.fit(
        dark_images,
        light_images,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
    )
    mymodel.save(
        "/content/drive/MyDrive/Img_Enhancement/low_light_enhancement_model3(colab).h5"
    )

    show_performance_curve(training_history, "mse", "Mean Squared Error")
    show_performance_curve(training_history, "mae", "Mean Absolute Error")
    show_performance_curve(training_history, "psnr", "Peak Signal-to-Noise Ratio")
