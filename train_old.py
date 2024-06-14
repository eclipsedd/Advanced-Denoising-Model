'''
This file contains code for training the model exclusively. It doesn't show any additional paramaters
which we might require to judge a model's performance. 
If a user wants to compare a model, he/she must use performance.py files instead.
'''

import os
import numpy as np
from PIL import Image
from model_0 import build_image_enhancement_model


def load_and_preprocess_image(file_path, target_size=(400, 600)):
    image = Image.open(file_path).resize((target_size))
    image = np.array(image) / 255.0  # Normalize to [0, 1] range
    return image


def load_images(directory, target_size=(400, 600)):
    images = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(
            ".png" or filename.endswith(".jpeg")
        ):  # Add other image formats if needed
            img_path = os.path.join(directory, filename)
            img = load_and_preprocess_image(img_path, target_size)
            images.append(img)
            filenames.append(filename)
    return np.array(images), filenames


if __name__ == "__main__":
    # Load images
    dark_images = load_images("Train/low", (400, 600))
    light_images = load_images("Train/high", (400, 600))

    input_shape = (400, 600, 3)
    mymodel = build_image_enhancement_model(input_shape)

    mymodel.compile(optimizer="adam", loss="mse")
    # mymodel.summary()

    history = mymodel.fit(dark_images, light_images, epochs=60, batch_size=8)

    mymodel.save("low_light_enhancement_model.h5")
    print("Model trained and saved as 'low_light_enhancement_model.h5'")
