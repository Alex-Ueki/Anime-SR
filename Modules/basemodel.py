from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.misc import imsave, imread, imresize

from keras import backend as K

import os

import __main__

""" LIST OF TODO """
"""

#TODO Implement a function in setup.py that tiles an HD image with a given border size
#TODO Implement a setup.py main() function that initializes all the directories
#TODO Implement a function that takes images from input_images and divides them
    into train_images (for training and validation) and eval_images (for evaluation)
    --> Should be a variety of settings, e.g. Random, Sorted
"""

"""
    Path Manager class
    Handles all directories, image_size and batch_size
    Has functions to get image count
    Has image generators for training/evaluation/prediction
    Uses default paths in Data directory if a path is not specified
"""

class PathManager():

    def __init__(self, name, base_tile_width=60, base_tile_height=60, channels=3, border=2, batch_size=16, paths={}):

        self.tile_width, self.tile_height = base_tile_width + 2*border, base_tile_height + 2*border
        self.channels = channels

        # Getting the image size dimensions
        if K.image_dim_ordering() == 'th':
            self.image_shape = (self.channels, self.tile_width, self.tile_height)
        else:
            self.image_shape = (self.tile_width, self.tile_height, self.channels)

        self.batch_size = batch_size

        self.data_path = paths['data'] if 'data' in paths else 'Data'
        self.base_dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__main__.__file__)), self.data_path)

        # use specified or default paths
        self.input_path = paths['input'] if 'input' in paths else os.path.join(self.base_dataset_dir, 'input_images')
        self.training_path = paths['training'] if 'training' in paths else os.path.join(self.base_dataset_dir, 'train_images', 'training')
        self.validation_path = paths['validation'] if 'validation' in paths else os.path.join(self.base_dataset_dir, 'train_images', 'validation')
        self.evaluation_path = paths['evaluation'] if 'evaluation' in paths else os.path.join(self.base_dataset_dir, 'eval_images')
        self.predict_path = paths['predict'] if 'predict' in paths else os.path.join(self.base_dataset_dir, 'predict_images')
        self.history_path = paths['history'] if 'history' in paths else os.path.join(self.base_dataset_dir, 'weights', '%s-%d-%d-%d_history.h5' % (name, base_tile_width, base_tile_height, border))
        # weight_path is the path to weight.h5 file for this model
        self.weight_path = paths['weights'] if 'weights' in paths else os.path.join(self.base_dataset_dir, 'weights', '%s-%d-%d-%d.h5' % (name, base_tile_width, base_tile_height, border))

        self.alpha = 'Alpha'
        self.beta = 'Beta'

    """
        Following functions check the image count of important directories
        Asserts equality in image number for certain directories
        See _image_count for base helper function
    """
    def train_images_count(self):
        a = self._image_count(os.path.join(self.training_path, self.alpha))
        b = self._image_count(os.path.join(self.training_path, self.beta))
        assert a == b, 'WARNING: Alpha and Beta set have different image sizes.\n' \
                       'Check ' + self.training_path
        return a

    def val_images_count(self):
        a = self._image_count(os.path.join(self.validation_path, self.alpha))
        b = self._image_count(os.path.join(self.validation_path, self.beta))
        assert a == b, 'WARNING: Alpha and Beta set have different image sizes.\n' \
                       'Check ' + self.validation_path
        return a

    def eval_images_count(self):
        a = self._image_count(os.path.join(self.evaluation_path, self.alpha))
        b = self._image_count(os.path.join(self.evaluation_path, self.beta))
        assert a == b, 'WARNING: Alpha and Beta set have different image sizes.\n' \
                       'Check ' + self.evaluation_path
        return a

    def predict_images_count(self):
        return self._image_count(os.path.join(self.predict_path, self.alpha))

    def input_images_count(self):
        return self._image_count(os.path.join(self.input_path, self.alpha))

    """
        Convenience Functions for data generators
        See _image_generator, _index_generate, _predict_image_generator for base code
    """

    def training_data_generator(self, shuffle=True):
        return self._image_generator(self.training_path, shuffle)

    def validation_data_generator(self, shuffle=True):
        return self._image_generator(self.validation_path, shuffle)

    def evaluation_data_generator(self, shuffle=True):
        return self._image_generator(self.evaluation_path, shuffle)

    def prediction_data_generator(self, shuffle=False):
        return self._predict_image_generator(self.predict_path, shuffle)


    """
        Internal Helper functions
        START : Code derived from Image-Super-Resolution/img_utils.py for _image_generator, _index_generate
    """

    def _image_generator(self, directory, shuffle=True):

        file_names = [f for f in sorted(os.listdir(os.path.join(directory, self.alpha)))]
        X_filenames = [os.path.join(directory, self.alpha, f) for f in file_names]
        y_filenames = [os.path.join(directory, self.beta, f) for f in file_names]

        nb_images = len(file_names)
        print('Found %d images.' % nb_images)

        index_generator = self._index_generator(nb_images, self.batch_size, shuffle)

        while 1:
            index_array, current_index, current_batch_size = next(index_generator)

            batch_x = np.zeros((current_batch_size, ) + self.image_shape)
            batch_y = np.zeros((current_batch_size, ) + self.image_shape)

            for i, j in enumerate(index_array):
                x_fn = X_filenames[j]
                img = imread(x_fn, mode='RGB')
                img = img.astype('float32') / 255.

                if K.image_dim_ordering() == 'th':
                    batch_x[i] = img.transpose((2, 0, 1))
                else:
                    batch_x[i] = img

                y_fn = y_filenames[j]
                img = imread(y_fn, mode='RGB')
                img = img.astype('float32') / 255.

                if K.image_dim_ordering() == 'th':
                    batch_y[i] = img.transpose((2, 0, 1))
                else:
                    batch_y[i] = img

            yield (batch_x, batch_y)

    # Helper to generate batch numbers
    def _index_generator(self, N, batch_size, shuffle=True):
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
        END : Code derived from Image-Super-Resolution/img_utils.py
    """

    # Image generator for model.predict_generator function (Keras)
    # Returns a single value
    def _predict_image_generator(self, directory, shuffle=True):

        file_names = [f for f in sorted(os.listdir(os.path.join(directory, self.alpha)))]
        filenames = [os.path.join(directory, self.alpha, f) for f in file_names]

        nb_images = len(file_names)
        print('Found %d images.' % nb_images)

        index_generator = self._index_generator(nb_images, self.batch_size, shuffle)

        while 1:
            index_array, current_index, current_batch_size = next(index_generator)

            batch= np.zeros((current_batch_size, ) + self.image_shape)

            for i, j in enumerate(index_array):
                fn = filenames[j]
                img = imread(fn, mode='RGB')
                img = img.astype('float32') / 255.

                if K.image_dim_ordering() == 'th':
                    batch[i] = img.transpose((2, 0, 1))
                else:
                    batch[i] = img

            yield (batch)

    """
        Helper for getting image count
    """
    def _image_count(self, imgs_path):
        return len([name for name in os.listdir(imgs_path)])
