# pylint: disable=C0301
# Line too long
"""
    ModelIO class
    Handles all model IO (data generators, etc)
    Uses default paths in Data directory if a path is not specified
"""

import os

import numpy as np

import __main__
import Modules.frameops as frameops



# Model parameter class


class ModelIO():
    """
        ModelIO class
        Handles all model IO (data generators, etc)
        Uses default paths in Data directory if a path is not specified
    """

    def __init__(self,
                 model_type='BasicSR',
                 image_width=1920, image_height=1080,
                 base_tile_width=60, base_tile_height=60,
                 channels=3,
                 border=2,
                 batch_size=16,
                 black_level=0.0,
                 border_mode='constant',
                 trim_top=0, trim_bottom=0, trim_left=0, trim_right=0,
                 jitter=True,
                 shuffle=True,
                 skip=True,
                 residual=False,
                 quality=1.0,
                 img_suffix='',
                 paths={},
                 epochs=10,
                 lr=0.001,
                ):

        self.model_type = model_type
        self.image_width, self.image_height = image_width, image_height
        self.base_tile_width, self.base_tile_height = base_tile_width, base_tile_height
        self.border, self.border_mode = border, border_mode
        self.trim_left, self.trim_right = trim_left, trim_right
        self.trim_top, self.trim_bottom = trim_top, trim_bottom
        self.channels, self.black_level = channels, black_level
        self.batch_size, self.img_suffix = batch_size, img_suffix
        self.jitter, self.shuffle, self.skip = jitter, shuffle, skip
        self.residual, self.quality, self.paths = residual, quality, paths

        # These values are just stashed in io for convenience

        self.epochs = epochs
        self.lr = lr

        from keras import backend as K
        self.theano = K.image_dim_ordering() == 'th'

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

        # Set image shape

        if self.theano:
            self.image_shape = (
                self.channels, self.tile_width, self.tile_height)
        else:
            self.image_shape = (
                self.tile_width, self.tile_height, self.channels)

        # Set paths

        self.data_path = paths['data'] if 'data' in paths else 'Data'
        self.base_dataset_dir = os.path.join(
            os.path.dirname(__main__.__file__), self.data_path)

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
        self.model_path = paths['model'] if 'model' in paths else os.path.join(
            self.base_dataset_dir, 'models', '{}-{}-{}-{}-{}.h5'.format(
                model_type, base_tile_width, base_tile_height, border, img_suffix))
        self.state_path = paths['state'] if 'state' in paths else os.path.join(
            self.base_dataset_dir, 'models', '{}-{}-{}-{}-{}_state.json'.format(
                model_type, base_tile_width, base_tile_height, border, img_suffix))

        self.alpha = 'Alpha'
        self.beta = 'Beta'


    def train_images_count(self):
        """ Check the image counts of important directories. Adjusts for quality parameter """

        # files will (hopefully) be a single element list containing a list of all the filenames.
        # Adjust count by quality fraction; we only apply quality to training images, not to
        # other image types.

        return int(self._images_count('training', self.training_path) * self.quality)

    def val_images_count(self):
        """ Count of files in validation alpha folder """

        return self._images_count('validation', self.validation_path)

    def eval_images_count(self):
        """ Count of files in evaluation alpha folder """

        return self._images_count('evaluation', self.evaluation_path)

    def predict_images_count(self):
        """ Count of files in predict alpha folder """

        return self._images_count('predict', self.predict_path)

    # PU: not currently used -- obsolete?

    def input_images_count(self):
        """ Count of files in images alpha folder """

        return self._images_count('input', self.input_path)

    def _images_count(self, path_code, path_name):
        """ Return number of image files in a path's alpha directory, checking for cached info """

        path_code += '.alpha'
        files = self.paths[path_code] if path_code in self.paths else frameops.image_files(
            os.path.join(path_name, self.alpha))
        return self.tiles_per_image * len(files[0])


    # Data generators generate matched pairs of tiles from the appropriate alpha and beta
    # folders.

    def training_data_generator(self):
        """ Training tile generator; use a subset if quality < 1.0 """

        return self._image_generator_frameops(self.training_path, self.jitter, self.shuffle, self.skip, self.quality)

    def validation_data_generator(self):
        """ Validation tile generator uses all tiles regardless of quality setting """

        return self._image_generator_frameops(self.validation_path, False, self.shuffle, self.skip, 1.0)

    def evaluation_data_generator(self):
        """ Generate tile pairs for evaluation; will not shuffle, jitter, skip or exclude tiles """

        return self._image_generator_frameops(self.evaluation_path, False, False, False, 1.0)


    def prediction_data_generator(self):
        """ Prediction tile generator generates single tiles, not tile pairs """

        return self._predict_image_generator_frameops(self.predict_path, False, False, False)

    # Frameops versions of image generators

    def _image_generator_frameops(self, directory, shuffle, jitter, skip, quality):
        """ Generate batches of pairs of tiles """

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
                    border_mode=self.border_mode,
                    trim_left=self.trim_left, trim_right=self.trim_right,
                    trim_top=self.trim_top, trim_bottom=self.trim_bottom,
                    shuffle=shuffle, jitter=jitter, skip=skip, quality=quality,
                    theano=self.theano, residual=self.residual):
                alpha_tiles[batch_index] = alpha_tile
                beta_tiles[batch_index] = beta_tile
                batch_index += 1
                if batch_index >= self.batch_size:
                    batch_index = 0
                    yield (alpha_tiles, beta_tiles)

    def _predict_image_generator_frameops(self, directory, shuffle, jitter, skip):
        """ Generate batches of tiles """

        alpha_paths = frameops.image_files(
            os.path.join(directory, self.alpha), deep=True)[0]

        alpha_tiles = np.empty((self.batch_size, ) + self.image_shape)
        batch_index = 0

        # Generate batches of tiles, repeating through list as needed

        while True:
            for alpha_tile in frameops.tesselate(
                    alpha_paths,
                    self.base_tile_width, self.base_tile_height, self.border,
                    black_level=self.black_level, border_mode=self.border_mode,
                    trim_left=self.trim_left, trim_right=self.trim_right,
                    trim_top=self.trim_top, trim_bottom=self.trim_bottom,
                    shuffle=shuffle, jitter=jitter, skip=skip, quality=1.0,
                    theano=self.theano):

                alpha_tiles[batch_index] = alpha_tile
                batch_index += 1
                if batch_index >= self.batch_size:
                    batch_index = 0
                    yield alpha_tiles

    #

    def asdict(self):
        """ Create dictionary of parameters """

        return {'image_width': self.image_width,
                'image_height': self.image_height,
                'base_tile_width': self.base_tile_width,
                'base_tile_height': self.base_tile_height,
                'channels': self.channels,
                'border': self.border,
                'border_mode': self.border_mode,
                'batch_size': self.batch_size,
                'black_level': self.black_level,
                'trim_top': self.trim_top,
                'trim_bottom': self.trim_bottom,
                'trim_left': self.trim_left,
                'trim_right': self.trim_right,
                'jitter': self.jitter,
                'shuffle': self.shuffle,
                'skip': self.skip,
                'quality': self.quality,
                'residual': self.residual,
                'img_suffix': self.img_suffix,
                'data_path': self.data_path,
                'training_path': self.training_path,
                'validation_path': self.validation_path,
                'evaluation_path': self.evaluation_path,
                'predict_path': self.predict_path,
                'model_path': self.model_path,
                'state_path': self.state_path,
                'model_type': self.model_type,
                'alpha': self.alpha,
                'beta': self.beta,
                'epochs': self.epochs,
                'lr': self.lr,
                'paths': self.paths
               }
