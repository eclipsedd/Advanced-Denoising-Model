"""
This is the common file I used while training different models on my local system.
It contains additional definitions than train.py to help visualise the performance
and efficiency of the model in terms of Mean Squared Error, Mean Absolute Error and 
Peak Signal-to-Noise Ratio, with an attempt to improve in newer model versions. 
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import image
from model_2 import (
    build_image_enhancement_model,
)  # chose the file from which model is to imported for building & testing.


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
    dark_images = load_images("Train/low")
    light_images = load_images("Train/high")

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
    mymodel.save("low_light_enhancement_model2.h5")

    show_performance_curve(training_history, "mse", "Mean Squared Error")
    show_performance_curve(training_history, "mae", "Mean Absolute Error")
    show_performance_curve(training_history, "psnr", "Peak Signal-to-Noise Ratio")
