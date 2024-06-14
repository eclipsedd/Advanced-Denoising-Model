"""
This is the new version of the performance_colab.py 
Earlier, training data pairs were downloaded in the begining only which meant more RAM utilisation. 

Here the Image Data Generator is used instead so that datasets are downloaded in small batches. 

Another add-on was using the Adam Optimiser with a learning rate scheduler i.e. exponential decay 
scheduler which ensures-
- Better Convergence: gradually decreasing the learning rate helps to converge more smoothly. 
                       In initial stages, higher learning rate allows significant progress. 
                       While a lower learning rate in later stages helps fine-tune the model parameters. 
- Optimal Learning Rate: exponential decay helps in achieving near-optimal learning rates throughout the training process.

"""

from keras import optimizers, preprocessing
import matplotlib.pyplot as plt
from tensorflow import image
from model_2 import (
    build_image_enhancement_model,
)  # chose the file from which model is to imported for building & testing.


def psnr(y_true, y_pred):
    return image.psnr(y_true, y_pred, max_val=1.0)


def show_performance_curve(training_result, metric, metric_label):
    train_perf = training_result.history[str(metric)]

    plt.plot(train_perf, label=metric_label)

    plt.xlabel("Epoch")
    plt.ylabel(metric_label)
    plt.legend(loc="lower right")
    plt.title(f"Training {metric_label}")
    plt.show()


if __name__ == "__main__":
    input_shape = (400, 600, 3)
    BATCH_SIZE = 8
    EPOCHS = 60

    # Data generator to avoid more RAM utilisation.
    datagen = preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_generator = datagen.flow_from_directory(
        directory="/content/drive/MyDrive/vlg_project/Train",
        target_size=(400, 600),
        batch_size=BATCH_SIZE,
        class_mode="input",
        subset="training",
    )

    mymodel = build_image_enhancement_model(input_shape)

    # Optimizer with learning rate scheduler
    initial_lr = 0.001
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=100000, decay_rate=0.96, staircase=True
    )
    optimizer = optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

    METRICS = ["mse", "mae", psnr]

    mymodel.compile(optimizer=optimizer, loss="mse", metrics=METRICS)

    training_history = mymodel.fit(
        train_generator, steps_per_epoch=len(train_generator), epochs=EPOCHS
    )

    # Save the model
    mymodel.save("/content/drive/MyDrive/vlg_project/low_light_enhancement_model4.h5")
    mymodel.save(
        "/content/drive/MyDrive/vlg_project/low_light_enhancement_model4.keras"
    )

    # Plot training curves
    show_performance_curve(training_history, "mse", "Mean Squared Error")
    show_performance_curve(training_history, "mae", "Mean Absolute Error")
    show_performance_curve(training_history, "psnr", "Peak Signal-to-Noise Ratio")
