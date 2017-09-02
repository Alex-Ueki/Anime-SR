from __future__ import print_function, division, absolute_import
import __main__

import numpy as np
from scipy.misc import imsave, imread, imresize
from keras import backend as K
import Modules.frameops as frameops
import os

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

# Model parameter class


class PathManager():

    def __init__(self, name, base_tile_width=60, base_tile_height=60, channels=3, border=2, batch_size=16,
                 black_level=0.0, trim_top=0, trim_bottom=0, trim_left=0, trim_right=0, tiles_per_image=1, paths={}):

        self.base_tile_width, self.base_tile_height, self.border = base_tile_width, base_tile_height, border
        self.trim_left, self.trim_right, self.trim_top, self.trim_bottom = trim_left, trim_right, trim_top, trim_bottom
        self.tile_width, self.tile_height = base_tile_width + \
            2 * border, base_tile_height + 2 * border
        self.channels, self.tiles_per_image = channels, tiles_per_image
        self.black_level, self.batch_size = black_level, batch_size
        self.name = name
        self.paths = paths

        # Passed Parameter Paranoia

        print('')
        print('PathManager Initialization...')
        print('            Name : {}'.format(self.name))
        print(' base_tile_width : {}'.format(self.base_tile_width))
        print('base_tile_height : {}'.format(self.base_tile_height))
        print('          border : {}'.format(self.border))
        print('        channels : {}'.format(self.channels))
        print('      batch_size : {}'.format(self.batch_size))
        print('     black_level : {}'.format(self.black_level))
        print('       trim tblr : {} {} {} {}'.format(self.trim_top,
                                                      self.trim_bottom, self.trim_left, self.trim_right))
        print(' tiles_per_image : {}'.format(self.tiles_per_image))
        print('    path entries : {}'.format(self.paths.keys()))

        # Getting the image size dimensions
        if K.image_dim_ordering() == 'th':
            self.image_shape = (
                self.channels, self.tile_width, self.tile_height)
        else:
            self.image_shape = (
                self.tile_width, self.tile_height, self.channels)

        self.data_path = paths['data'] if 'data' in paths else 'Data'
        self.base_dataset_dir = os.path.join(os.path.dirname(
            os.path.abspath(__main__.__file__)), self.data_path)

        # use specified or default paths
        self.input_path = paths['input'] if 'input' in paths else os.path.join(
            self.base_dataset_dir, 'input_images')
        self.training_path = paths['training'] if 'training' in paths else os.path.join(
            self.base_dataset_dir, 'train_images', 'training')
        self.validation_path = paths['validation'] if 'validation' in paths else os.path.join(
            self.base_dataset_dir, 'train_images', 'validation')
        self.evaluation_path = paths['evaluation'] if 'evaluation' in paths else os.path.join(
            self.base_dataset_dir, 'eval_images')
        self.predict_path = paths['predict'] if 'predict' in paths else os.path.join(
            self.base_dataset_dir, 'predict_images')
        self.history_path = paths['history'] if 'history' in paths else os.path.join(
            self.base_dataset_dir, 'weights', '%s-%d-%d-%d_history.h5' % (name, base_tile_width, base_tile_height, border))
        self.weight_path = paths['weights'] if 'weights' in paths else os.path.join(
            self.base_dataset_dir, 'weights', '%s-%d-%d-%d.h5' % (name, base_tile_width, base_tile_height, border))

        self.alpha = 'Alpha'
        self.beta = 'Beta'

    # Check the image counts of important directories. Assumes error checking will be done at an outer level.

    def train_images_count(self):

        # We may have the list of files, otherwise go fetch it

        files = self.paths['training.alpha'] if 'training.alpha' in self.paths else frameops.image_files(
            os.path.join(self.training_path, self.alpha))

        # files will (hopefully) be a single element list containing a list of all the filenames.

        return self.tiles_per_image * len(files[0])

    def val_images_count(self):

        files = self.paths['validation.alpha'] if 'validation.alpha' in self.paths else frameops.image_files(
            os.path.join(self.validation_path, self.alpha))
        return self.tiles_per_image * len(files[0])

    def eval_images_count(self):
        files = self.paths['evaluation.alpha'] if 'evaluation.alpha' in self.paths else frameops.image_files(
            os.path.join(self.evaluation_path, self.alpha))
        return self.tiles_per_image * len(files[0])

    def predict_images_count(self):
        files = self.paths['predict.alpha'] if 'predict.alpha' in self.paths else frameops.image_files(
            os.path.join(self.predict_path, self.alpha))
        return self.tiles_per_image * len(files[0])

    def input_images_count(self):
        files = self.paths['input.alpha'] if 'input.alpha' in self.paths else frameops.image_files(
            os.path.join(self.input_path, self.alpha))
        return self.tiles_per_image * len(files[0])

    # Convenience Functions for data generators
    # See _image_generator, _index_generate, _predict_image_generator for base code

    def training_data_generator(self, shuffle=True, jitter=True, skip=True):
        return self._image_generator_frameops(self.training_path, jitter, shuffle, skip)

    def validation_data_generator(self, shuffle=True, jitter=True, skip=True):
        return self._image_generator_frameops(self.validation_path, jitter, shuffle, skip)

    def evaluation_data_generator(self, shuffle=True, jitter=True, skip=True):
        return self._image_generator_frameops(self.evaluation_path, jitter, shuffle, skip)

    def prediction_data_generator(self, shuffle=False, jitter=True, skip=True):
        return self._predict_image_generator_frameops(self.predict_path, jitter, shuffle, skip)

    # Frameops versions of image generators

    def _image_generator_frameops(self, directory, shuffle=True, jitter=True, skip=True):

        # frameops.image_files returns a list with an element for each image file type,
        # but at this point, we'll only ever have one...

        alpha_paths = frameops.image_files(
            os.path.join(directory, self.alpha), deep=True)[0]
        beta_paths = frameops.image_files(
            os.path.join(directory, self.beta), deep=True)[0]

        alpha_tiles = np.empty((self.batch_size, ) + self.image_shape)
        beta_tiles = np.empty((self.batch_size, ) + self.image_shape)
        batch_index = 0

        # Generate batches of tiles, repeating through the list as needed

        while True:
            for alpha_tile, beta_tile in frameops.tesselate_pair(
                    alpha_paths, beta_paths,
                    self.base_tile_width, self.base_tile_height, self.border,
                    black_level=self.black_level,
                    trim_left=self.trim_left, trim_right=self.trim_right,
                    trim_top=self.trim_top, trim_bottom=self.trim_bottom,
                    shuffle=shuffle, jitter=jitter, skip=skip):
                if K.image_dim_ordering() == 'th':
                    alpha_tile = alpha_tile.transpose((2, 0, 1))
                    beta_tile = beta_tile.transpose((2, 0, 1))
                alpha_tiles[batch_index] = alpha_tile
                beta_tiles[batch_index] = beta_tile
                batch_index += 1
                if batch_index >= self.batch_size:
                    batch_index = 0
                    yield (alpha_tiles, beta_tiles)

    def _predict_image_generator_frameops(self, directory, shuffle=True, jitter=False):

        alpha_paths = frameops.image_files(
            os.path.join(directory, self.alpha), deep=True)[0]

        alpha_tiles = np.empty((self.batch_size, ) + self.image_shape)
        batch_index = 0

        # Generate batches of tiles, repeating through list as needed

        while True:
            for alpha_tile in frameops.tesselate(
                    alpha_paths,
                    self.base_tile_width, self.base_tile_height, self.border,
                    trim_left=self.trim_left, trim_right=self.trim_right,
                    trim_top=self.trim_top, trim_bottom=self.trim_bottom,
                    shuffle=shuffle, jitter=jitter, skip=skip):
                if K.image_dim_ordering() == 'th':
                    alpha_tile = alpha_tile.transpose((2, 0, 1))

                alpha_tiles[batch_index] = alpha_tile
                batch_index += 1
                if batch_index >= self.batch_size:
                    batch_index = 0
                    yield (alpha_tiles)

    # Internal Helper functions
    # START : Code derived from Image-Super-Resolution/img_utils.py for _image_generator, _index_generate

    def _image_generator(self, directory, shuffle=True):

        file_names = [f for f in sorted(
            os.listdir(os.path.join(directory, self.alpha)))]
        X_filenames = [os.path.join(directory, self.alpha, f)
                       for f in file_names]
        y_filenames = [os.path.join(directory, self.beta, f)
                       for f in file_names]

        nb_images = len(file_names)
        print('Found %d images.' % nb_images)

        index_generator = self._index_generator(
            nb_images, self.batch_size, shuffle)

        while 1:
            index_array, current_index, current_batch_size = next(
                index_generator)

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

    # Helper to generate batch number

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

    # END : Code derived from Image-Super-Resolution/img_utils.py

    # Image generator for model.predict_generator function (Keras)
    # Returns a single value

    def _predict_image_generator(self, directory, shuffle=True):

        file_names = [f for f in sorted(
            os.listdir(os.path.join(directory, self.alpha)))]
        filenames = [os.path.join(directory, self.alpha, f)
                     for f in file_names]

        nb_images = len(file_names)
        print('Found %d images.' % nb_images)

        index_generator = self._index_generator(
            nb_images, self.batch_size, shuffle)

        while 1:
            index_array, current_index, current_batch_size = next(
                index_generator)

            batch = np.zeros((current_batch_size, ) + self.image_shape)

            for i, j in enumerate(index_array):
                fn = filenames[j]
                img = imread(fn, mode='RGB')
                img = img.astype('float32') / 255.

                if K.image_dim_ordering() == 'th':
                    batch[i] = img.transpose((2, 0, 1))
                else:
                    batch[i] = img

            yield (batch)

    # Helper for getting image count

    def _image_count(self, imgs_path):
        return len([name for name in os.listdir(imgs_path)])
