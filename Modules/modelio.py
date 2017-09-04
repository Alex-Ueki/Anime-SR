from __future__ import print_function, division, absolute_import
import __main__

import numpy as np
from scipy.misc import imsave, imread, imresize
from keras import backend as K
import Modules.frameops as frameops
import os

"""
    ModelIO class
    Handles all model IO (data generators, etc)
    Uses default paths in Data directory if a path is not specified
"""

# Model parameter class


class ModelIO():

    def __init__(self,
                 image_width=1920, image_height=1080,
                 base_tile_width=60, base_tile_height=60,
                 channels=3,
                 border=2,
                 batch_size=16,
                 black_level=0.0,
                 trim_top=0, trim_bottom=0, trim_left=0, trim_right=0,
                 jitter=True,
                 shuffle=True,
                 skip=True,
                 img_suffix='',
                 paths={}):

        self.base_tile_width, self.base_tile_height = base_tile_width, base_tile_height
        self.border = border
        self.trim_left, self.trim_right = trim_left, trim_right
        self.trim_top, self.trim_bottom = trim_top, trim_bottom
        self.channels, self.black_level = channels, black_level
        self.batch_size = batch_size
        self.jitter, self.shuffle, self.skip = jitter, shuffle, skip
        self.paths = paths

        # The actual internal tile size includes the overlap borders

        self.tile_width = base_tile_width + 2 * border
        self.tile_height = base_tile_height + 2 * border

        # How many tiles will we get out of each image? If we are jittering, then account for that

        trimmed_width = image_width - (trim_left + trim_right)
        trimmed_height = image_height - (trim_top + trim_bottom)
        tiles_across = trimmed_width // base_tile_width
        tiles_down = trimmed_height // base_tile_height
        self.tiles_per_image = tiles_across * tiles_down
        if jitter == 1:
            self.tiles_per_image += (tiles_across - 1) * (tiles_down - 1)

        # Passed Parameter Paranoia

        """
        print('')
        print('ModelIO Initialization...')
        print(' base_tile_width : {}'.format(self.base_tile_width))
        print('base_tile_height : {}'.format(self.base_tile_height))
        print('          border : {}'.format(self.border))
        print('        channels : {}'.format(self.channels))
        print('      batch_size : {}'.format(self.batch_size))
        print('     black_level : {}'.format(self.black_level))
        print('       trim tblr : {} {} {} {}'.format(self.trim_top,
                                                      self.trim_bottom, self.trim_left, self.trim_right))
        print(' tiles_per_image : {}'.format(self.tiles_per_image))

        print('          jitter : {}'.format(jitter == 1))
        print('         shuffle : {}'.format(shuffle == 1))
        print('            skip : {}'.format(skip == 1))
        print('    path entries : {}'.format(self.paths.keys()))
        """
        
        # Set image shape

        if K.image_dim_ordering() == 'th':
            self.image_shape = (
                self.channels, self.tile_width, self.tile_height)
        else:
            self.image_shape = (
                self.tile_width, self.tile_height, self.channels)

        # Set paths

        self.data_path = paths['data'] if 'data' in paths else 'Data'
        self.base_dataset_dir = os.path.join(os.path.dirname(
            os.path.abspath(__main__.__file__)), self.data_path)

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
            self.base_dataset_dir, 'weights',  '{}-{}-{}-{}-{}.h5'.format(model_type, tile_width, tile_height, tile_border, img_suffix))
        self.weight_path = paths['weights'] if 'weights' in paths else os.path.join(
            self.base_dataset_dir, 'weights',  '{}-{}-{}-{}-{}.h5'.format(model_type, tile_width, tile_height, tile_border, img_suffix))

        self.alpha = 'Alpha'
        self.beta = 'Beta'

    # Check the image counts of important directories. Adjusts for the number of tiles we will get

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

    # Data generators

    def training_data_generator(self):
        return self._image_generator_frameops(self.training_path, self.jitter, self.shuffle, self.skip)

    def validation_data_generator(self):
        return self._image_generator_frameops(self.validation_path, self.jitter, self.shuffle, self.skip)

    # Evaluation and Prediction generators will never shuffle, jitter or skip.

    def evaluation_data_generator(self):
        return self._image_generator_frameops(self.evaluation_path, False, False, False)

    def prediction_data_generator(self):
        return self._predict_image_generator_frameops(self.predict_path, False, False, False)

    # Frameops versions of image generators

    def _image_generator_frameops(self, directory, shuffle, jitter, skip):

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

    def _predict_image_generator_frameops(self, directory, shuffle, jitter, skip):

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
