# pylint: disable=C0301
# Line too long
"""
    ModelIO class
    Handles all model IO (data generators, etc)
    Uses default paths in Data directory if a path is not specified
"""

import os

import numpy as np

import Modules.frameops as frameops


# Model parameter class


class ModelIO():
    """
        ModelIO class
        Handles all model IO (data generators, etc)
        Uses default paths in Data directory if a path is not specified
    """

    def __init__(self, config=None):

        from keras import backend as K

        # All of the configuration info is kept in the config dictionary.
        # A few of the really common items are also available as instance variables

        print(config)

        config = {} if config is None else config

        config.setdefault('model_type', 'BasicSR')
        config.setdefault('image_width', 1920)
        config.setdefault('image_height', 1080)
        config.setdefault('base_tile_width', 60)
        config.setdefault('base_tile_height', 60)
        config.setdefault('border', 2)
        config.setdefault('border_mode', 'constant')
        config.setdefault('trim_left', 240)
        config.setdefault('trim_right', 240)
        config.setdefault('trim_top', 0)
        config.setdefault('trim_bottom', 0)
        config.setdefault('channels', 3)
        config.setdefault('black_level', 0.0)
        config.setdefault('batch_size', 16)
        config.setdefault('img_suffix', 'dpx')
        config.setdefault('jitter', True)
        config.setdefault('shuffle', True)
        config.setdefault('skip', True)
        config.setdefault('residual', True)
        config.setdefault('quality', 1.0)
        config.setdefault('paths', {})
        config.setdefault('epochs', 10)
        config.setdefault('lr', 0.001)
        config.setdefault('theano', K.image_dim_ordering() == 'th')

        # The actual internal tile size includes the overlap borders

        config.setdefault('tile_width', config['base_tile_width'] + 2 * config['border'])
        config.setdefault('tile_height', config['base_tile_height'] + 2 * config['border'])

        # How many tiles will we get out of each image? If we are jittering, then account for that

        self.trimmed_width = config['image_width'] - (config['trim_left'] + config['trim_right'])
        self.trimmed_height = config['image_height'] - (config['trim_top'] + config['trim_bottom'])
        self.tiles_across = self.trimmed_width // config['base_tile_width']
        self.tiles_down = self.trimmed_height // config['base_tile_height']

        config.setdefault('trimmed_width', self.trimmed_width)
        config.setdefault('trimmed_height', self.trimmed_height)
        config.setdefault('tiles_across', self.tiles_across)
        config.setdefault('tiles_down', self.tiles_down)

        config.setdefault('tiles_per_image', self.tiles_across * self.tiles_down)
        if config['jitter'] == 1:
            config['tiles_per_image'] += (self.tiles_across - 1) * \
                (self.tiles_down - 1)

        # Set image shape

        if config['theano']:
            config['image_shape'] = (
                config['channels'], config['tile_width'], config['tile_height'])
        else:
            config['image_shape'] = (
                config['tile_width'], config['tile_height'], config['channels'])

        self.channels = config['channels']
        self.image_shape = config['image_shape']

        # Set paths

        self.paths = config['paths']

        self.paths.setdefault('data', 'Data')
        self.paths.setdefault('input', os.path.join(self.paths['data'], 'input_images'))
        self.paths.setdefault('training', os.path.join(self.paths['data'], 'train_images', 'training'))
        self.paths.setdefault('validation', os.path.join(self.paths['data'], 'train_images', 'validation'))
        self.paths.setdefault('evaluation', os.path.join(self.paths['data'], 'eval_images'))
        self.paths.setdefault('predict', os.path.join(self.paths['data'], 'predict_images'))

        self.paths.setdefault('model', os.path.join(
            self.paths['data'], 'models', '{}-{}-{}-{}-{}.h5'.format(
                config['model_type'],
                config['base_tile_width'],
                config['base_tile_height'],
                config['border'],
                config['img_suffix'])))

        self.paths.setdefault('state', os.path.join(
            self.paths['data'], 'models', '{}-{}-{}-{}-{}_state.json'.format(
                config['model_type'],
                config['base_tile_width'],
                config['base_tile_height'],
                config['border'],
                config['img_suffix'])))

        config['paths'] = self.paths

        self.alpha = 'Alpha'
        self.beta = 'Beta'

        self.config = config

    def train_images_count(self):
        """ Check the image counts of important directories. Adjusts for quality parameter """

        # files will (hopefully) be a single element list containing a list of all the filenames.
        # Adjust count by quality fraction; we only apply quality to training images, not to
        # other image types.

        return int(self._images_count('training', self.paths['training']) * self.config['quality'])

    def val_images_count(self):
        """ Count of files in validation alpha folder """

        return self._images_count('validation', self.paths['validation'])

    def eval_images_count(self):
        """ Count of files in evaluation alpha folder """

        return self._images_count('evaluation', self.paths['evaluation'])

    def predict_images_count(self):
        """ Count of files in predict alpha folder """

        return self._images_count('predict', self.paths['predict'])

    # PU: not currently used -- obsolete?

    def input_images_count(self):
        """ Count of files in images alpha folder """

        return self._images_count('input', self.paths['input'])

    def _images_count(self, path_code, path_name):
        """ Return number of image files in a path's alpha directory, checking for cached info """

        path_code += '.alpha'
        files = self.paths[path_code] if path_code in self.paths else frameops.image_files(
            os.path.join(path_name, self.alpha))
        return self.config['tiles_per_image'] * len(files[0])

    # Data generators generate matched pairs of tiles from the appropriate alpha and beta
    # folders.

    def training_data_generator(self):
        """ Training tile generator; use a subset if quality < 1.0 """

        return self._image_generator_frameops(self.paths['training'], self.config['jitter'], self.config['shuffle'], self.config['skip'], self.config['quality'])

    def validation_data_generator(self):
        """ Validation tile generator uses all tiles regardless of quality setting """

        return self._image_generator_frameops(self.paths['validation'], False, self.config['shuffle'], self.config['skip'], 1.0)

    def evaluation_data_generator(self):
        """ Generate tile pairs for evaluation; will not shuffle, jitter, skip or exclude tiles """

        return self._image_generator_frameops(self.paths['evaluation'], False, False, False, 1.0)

    def prediction_data_generator(self):
        """ Prediction tile generator generates single tiles, not tile pairs """

        return self._predict_image_generator_frameops(self.paths['preduct'], False, False, False)

    # Frameops versions of image generators

    def _image_generator_frameops(self, folder, shuffle, jitter, skip, quality):
        """ Generate batches of pairs of tiles """

        # frameops.image_files returns a list with an element for each image file type,
        # but at this point, we'll only ever have one...

        alpha_paths = frameops.image_files(
            os.path.join(folder, self.alpha), deep=True)[0]
        beta_paths = frameops.image_files(
            os.path.join(folder, self.beta), deep=True)[0]

        alpha_tiles = np.empty(
            (self.config['batch_size'], ) + self.config['image_shape'])
        beta_tiles = np.empty(
            (self.config['batch_size'], ) + self.config['image_shape'])
        batch_index = 0

        # Generate batches of tiles, repeating through the list as needed

        while True:
            for alpha_tile, beta_tile in frameops.tesselate_pair(
                    alpha_paths, beta_paths,
                    self.config['base_tile_width'], self.config['base_tile_height'], self.config['border'],
                    black_level=self.config['black_level'],
                    border_mode=self.config['border_mode'],
                    trim_left=self.config['trim_left'], trim_right=self.config['trim_right'],
                    trim_top=self.config['trim_top'], trim_bottom=self.config['trim_bottom'],
                    shuffle=shuffle, jitter=jitter, skip=skip, quality=quality,
                    theano=self.config['theano'], residual=self.config['residual']):
                alpha_tiles[batch_index] = alpha_tile
                beta_tiles[batch_index] = beta_tile
                batch_index += 1
                if batch_index >= self.config['batch_size']:
                    batch_index = 0
                    yield (alpha_tiles, beta_tiles)

    def _predict_image_generator_frameops(self, folder, shuffle, jitter, skip):
        """ Generate batches of tiles """

        alpha_paths = frameops.image_files(
            os.path.join(folder, self.alpha), deep=True)[0]

        alpha_tiles = np.empty(
            (self.config['batch_size'], ) + self.config['image_shape'])
        batch_index = 0

        # Generate batches of tiles, repeating through list as needed

        while True:
            for alpha_tile in frameops.tesselate(
                    alpha_paths,
                    self.config['base_tile_width'], self.config['base_tile_height'], self.config['border'],
                    black_level=self.config['black_level'], border_mode=self.config['border_mode'],
                    trim_left=self.config['trim_left'], trim_right=self.config['trim_right'],
                    trim_top=self.config['trim_top'], trim_bottom=self.config['trim_bottom'],
                    shuffle=shuffle, jitter=jitter, skip=skip, quality=1.0,
                    theano=self.config['theano']):

                alpha_tiles[batch_index] = alpha_tile
                batch_index += 1
                if batch_index >= self.config['batch_size']:
                    batch_index = 0
                    yield alpha_tiles

    def asdict(self):
        """ Create dictionary of parameters """

        return self.config
