"""
model 6 with more epochs and less small batch size. 
"""

from keras import layers, models, regularizers, optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import image
import keras


def build_image_enhancement_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoding layers
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

    # Dilated convolution
    conv3 = layers.Conv2D(
        64,
        (3, 3),
        dilation_rate=2,
        padding="same",
        kernel_regularizer=regularizers.l2(0.001),
    )(relu2)
    bn3 = layers.BatchNormalization()(conv3)
    relu3 = layers.ReLU()(bn3)

    # Decoding layers
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


def show_performance_curve(training_result, metric, metric_label):
    train_perf = training_result.history[str(metric)]
    val_perf = training_result.history["val_" + str(metric)]

    plt.plot(train_perf, label="train_" + metric_label)
    plt.plot(val_perf, label="val_" + metric_label)

    plt.xlabel("Epoch")
    plt.ylabel(metric_label)
    plt.legend(loc="lower right")
    plt.title(f"Training and Validation {metric_label}")
    plt.show()


if __name__ == "__main__":
    input_shape = (400, 600, 3)
    BATCH_SIZE = 8
    EPOCHS = 60

    # Data generator to avoid more RAM utilisation.
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.1
    )

    train_generator = datagen.flow_from_directory(
        directory="/content/drive/MyDrive/Img_Enhancement/Train",
        target_size=(400, 600),
        batch_size=BATCH_SIZE,
        class_mode="input",
        subset="training",
    )

    validation_generator = datagen.flow_from_directory(
        directory="/content/drive/MyDrive/Img_Enhancement/Train",
        target_size=(400, 600),
        batch_size=BATCH_SIZE,
        class_mode="input",
        subset="validation",
    )

    # Convert to tf.data.Dataset and apply repeat
    train_dataset = (
        tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 400, 600, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 400, 600, 3), dtype=tf.float32),
            ),
        )
        .shuffle(buffer_size=1000)
        .repeat()
    )

    steps_per_epoch = len(train_generator)

    mymodel = build_image_enhancement_model(input_shape)

    # Optimizer with learning rate scheduler
    initial_lr = 0.001
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=steps_per_epoch, decay_rate=0.99, staircase=True
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    METRICS = ["mse", "mae", psnr]

    mymodel.compile(optimizer=optimizer, loss="mse", metrics=METRICS)

    training_history = mymodel.fit(
        train_dataset,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=validation_generator,
    )

    # Save the model
    keras.saving.save_model(
        mymodel,
        "/content/drive/MyDrive/Img_Enhancement/low_light_enhancement_model7(colab).keras",
    )

    print("Validation Metrics per Epoch:")
    for i, (loss, mse, mae, psnr) in enumerate(
        zip(
            training_history.history["val_loss"],
            training_history.history["val_mse"],
            training_history.history["val_mae"],
            training_history.history["val_psnr"],
        )
    ):
        print(
            f"Epoch {i+1}: Loss - {loss:.4f}, MSE - {mse:.4f}, MAE - {mae:.4f}, PSNR - {psnr:.4f}"
        )

    # Plot training curves
    show_performance_curve(training_history, "mse", "Mean Squared Error")
    show_performance_curve(training_history, "mae", "Mean Absolute Error")
    show_performance_curve(training_history, "psnr", "Peak Signal-to-Noise Ratio")
