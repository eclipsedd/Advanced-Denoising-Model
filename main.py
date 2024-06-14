import numpy as np
from PIL import Image
import os
import tensorflow as tf
from train_old import load_images
from keras import losses


@tf.keras.utils.register_keras_serializable()
def save_enhanced_images(enhanced_images, filenames, output_dir="test/predicted"):
    os.makedirs(output_dir, exist_ok=True)
    for img, filename in zip(enhanced_images, filenames):
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img.save(os.path.join(output_dir, filename))


@tf.keras.utils.register_keras_serializable()
def psnr(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


if __name__ == "__main__":
    mymodel = tf.keras.models.load_model(
        "FINAL_low_light_enhancement_model10(colab).keras",
        #custom_objects={"mse": losses.MeanSquaredError()},
        custom_objects={"psnr": psnr},
    )
    print("Model loaded.")

    test_images, filenames = load_images("test/low", (600, 400))

    enhanced_images = mymodel.predict(test_images)
    print("Predictions completed.")

    save_enhanced_images(enhanced_images, filenames)
    print("Enhanced images saved in 'test/predicted' directory.")
