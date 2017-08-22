from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.misc import imsave, imread, imresize

from keras import backend as K

import os
""" LIST OF TODO """
"""
#TODO Have paths be stored in a class object that allows setable values
    (for border and path directories)
#TODO Use os.path to have path values that work for both MAC (/) and Windows (\)
#TODO Implement a function in setup.py that tiles an HD image with a given border size
#TODO Implement a setup.py main() function that initializes all the directories
#TODO Implement a function that takes images from input_images and divides them
    into train_images (for training and validation) and eval_images (for evaluation)
    --> Should be a variety of settings, e.g. Random, Sorted
"""

# Hard set to border = 2
tile_width, tile_height = 64, 64

batch_size = 16

input_path = "input_images/"
train_path = "train_images/"
weight_path = "weights/"
evaluation_path = "eval_images/"
predict_path = "predict_images/"

alpha = r"Alpha"
beta = r"Beta"
alphaset = alpha + "/" # Alpha is the input set for fitting a model
betaset = beta + "/" # Beta is the output set (for comparison)

#TODO Set this to your environment path
base_dataset_dir = "/Volumes/RBTRAIN/keras_workspace/SRCNN/"

training_output_path = base_dataset_dir + train_path + r"training/"
validation_output_path = base_dataset_dir + train_path + r"validation/"
evaluation_output_path = base_dataset_dir + evaluation_path

def image_count(imgs_path):
    return len([name for name in os.listdir(imgs_path)])

"""
    Following functions check the image count of important directories
    Asserts equality in image number for certain directories
"""
def train_images_count():
    a = image_count(training_output_path + alphaset)
    b = image_count(training_output_path + betaset)
    assert a == b, "WARNING: Alpha and Beta set have different image sizes.\n" \
                   "Check " + training_output_path
    return a

def val_images_count():
    a = image_count(validation_output_path + alphaset)
    b = image_count(validation_output_path + betaset)
    assert a == b, "WARNING: Alpha and Beta set have different image sizes.\n" \
                   "Check " + validation_output_path
    return a

def eval_images_count():
    a = image_count(evaluation_output_path + alphaset)
    b = image_count(evaluation_output_path + betaset)
    assert a == b, "WARNING: Alpha and Beta set have different image sizes.\n" \
                   "Check " + evaluation_output_path
    return a

def predict_images_count():
    return image_count(base_dataset_dir + predict_path + alphaset)


"""
    Code derived from Image-Super-Resolution/img_utils.py
    Image generator for fit_generator and evaluate_generator (Keras.model)
"""
def image_generator(directory, channels=3, shuffle=True, batch_size=16):
    if K.image_dim_ordering() == "th":
        image_shape = (channels, tile_width, tile_height)
        y_image_shape = image_shape
    else:
        image_shape = (tile_width, tile_height, channels)
        y_image_shape = image_shape

    file_names = [f for f in sorted(os.listdir(directory + alphaset))]
    X_filenames = [os.path.join(directory, alpha, f) for f in file_names]
    y_filenames = [os.path.join(directory, beta, f) for f in file_names]

    nb_images = len(file_names)
    print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch_x = np.zeros((current_batch_size,) + image_shape)
        batch_y = np.zeros((current_batch_size,) + y_image_shape)

        for i, j in enumerate(index_array):
            x_fn = X_filenames[j]
            img = imread(x_fn, mode='RGB')
            img = img.astype('float32') / 255.

            if K.image_dim_ordering() == "th":
                batch_x[i] = img.transpose((2, 0, 1))
            else:
                batch_x[i] = img

            y_fn = y_filenames[j]
            img = imread(y_fn, mode="RGB")
            img = img.astype('float32') / 255.

            if K.image_dim_ordering() == "th":
                batch_y[i] = img.transpose((2, 0, 1))
            else:
                batch_y[i] = img

        yield (batch_x, batch_y)


# Image generator for model.predict_generator function (Keras)
def predict_image_generator(directory, channels=3, shuffle=True, batch_size=16):
    if K.image_dim_ordering() == "th":
        image_shape = (channels, tile_width, tile_height)
    else:
        image_shape = (tile_width, tile_height, channels)

    file_names = [f for f in sorted(os.listdir(directory + alphaset))]
    filenames = [os.path.join(directory, alphaset, f) for f in file_names]

    nb_images = len(file_names)
    print("Found %d images." % nb_images)

    index_generator = _index_generator(nb_images, batch_size, shuffle)

    while 1:
        index_array, current_index, current_batch_size = next(index_generator)

        batch= np.zeros((current_batch_size,) + image_shape)

        for i, j in enumerate(index_array):
            fn = filenames[j]
            img = imread(fn, mode='RGB')
            img = img.astype('float32') / 255.

            if K.image_dim_ordering() == "th":
                batch[i] = img.transpose((2, 0, 1))
            else:
                batch[i] = img

        yield (batch)

# Helper to generate batch numbers
def _index_generator(N, batch_size=16, shuffle=True):
    batch_index = 0
    total_batches_seen = 0

    while 1:
        if batch_index == 0:
            index_array = np.arange(N)
            if shuffle:
                index_array = np.random.permutation(N)

        current_index = (batch_index * batch_size) % N

        if N >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
        total_batches_seen += 1

        yield (index_array[current_index: current_index + current_batch_size],
               current_index, current_batch_size)

"""
    Code derived from Image-Super-Resolution/img_utils.py
"""
