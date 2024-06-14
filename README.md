# Advanced-Denoising-Model

INTRODUCTION

This file elucidates the architecture of an advanced image enhancement model developed using the Keras deep learning framework. Rather than employing the Sequential model, the Functional API of Keras was selected for its superior flexibility in layer connectivity, facilitating the construction of intricate architectures and the efficient implementation of skip connections. (discussed later). The model integrates multiple convolutional layers, batch normalization, ReLU activations, dropout layers, and skip connections to enhance its performance. Initial model prototypes were developed on a local machine. However, due to the high computational power and memory requirements, subsequent models were constructed using Google Collaboratory. Training these models locally proved impractical, with estimated completion times exceeding 24 hours.

The “performance{}.py” files encapsulate the common code necessary to construct various models while also computing additional statistics such as MAE (Mean Absolute Error), MSE (Mean Squared Error), and PSNR (Peak Signal-to-Noise Ratio). These metrics enable the user to compare model efficiency and provide insights into the behaviour of these scores depending on various model parameters and architectures. In contrast, the “train{}.py” files serve a similar purpose but do not calculate additional statistics. These files maybe utilized to construct models more efficiently when extra calculations are unnecessary.

Furthermore, files labelled as “_old” pertain to the code employed to train the initial models (1, 2, and 3). Subsequent models (4 onwards) incorporated Image Data Generators and exponentially decaying learning rates to enhance their efficiency. The corresponding code for these advanced models is contained in files labelled as “_latest”.
The “main.py” file is designed to download the pre-built and saved models, apply them to input images from the “./test/low” directory, and save the enhanced images in the “./test/predicted” directory.

Models 1, 2, 6, 7, 8, 9 and 10 have been built using the then latest version of TensorFlow(2.16.1) and Keras(3.3.3), 
while models 3, 4 & 5 have been built using TensorFlow(2.15.0) which was the previous stable version offered by default by Google Collaboratory then.

The Statistics folder mentions the PSNR, MAE and MSE values for each of the models along with their graphical representation.
