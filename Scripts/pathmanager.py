from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.misc import imsave, imread, imresize

from keras import backend as K
import os

import dpxtile

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
    Functions to read and write .png, .dpx
"""

# Image data should be in range 0...1 inclusive as float32
def readPNGImageData(f, meta):
    img = imread(f, mode="RGB")
    img = img.astype('float32') / 255.
    return img

def writePNG(f, image, meta):
    # Scale and clip image
    output = np.clip(image*255., 0, 255).astype('uint8')
    imsave(f, output)
    return

# Returns empty dictionary because meta data is currently not needed
def readPNGMetaData(f):
    return {}

"""
Dictionary holds name, read, write, and meta data functions for each data-type in a list
Read should read from a file, using meta_data is needed, and scale to 0..1 number range
    ---> Output should be an numpy.array of type float32
    ---> Usage : Output = read(f, meta)
Write should take an numpy.array of type float32, scale to proper type, and output
    to a given filename
    ---> Usage : write(f, image, meta)
Meta should get metadata from an image, if needed. Otherwise it should return an
    empty dictionary
    ---> Usage : meta = meta(f)
"""
format_types = {
	0: ("DPX", dpxtile.readDPXImageData, dpxtile.writeDPX, dpxtile.readDPXMetaData),
	1: ("PNG", readPNGImageData, writePNG, readPNGMetaData)
}


class PathManager():

    def __init__(self, name, format_type=0, base_tile_size=60, channels=3, border=2, batch_size=16):

        self.tile_width, self.tile_height = base_tile_size + 2*border, base_tile_size + 2*border
        self.channels = channels

        self.format, self.img_read, self.img_write, self.img_meta = format_types[format_type]

        self.current_meta = None

        # Getting the image size dimensions
        if K.image_dim_ordering() == "th":
            self.image_shape = (self.channels, self.tile_width, self.tile_height)
        else:
            self.image_shape = (self.tile_width, self.tile_height, self.channels)

        self.batch_size = batch_size
        # self.predict_batch_size may be needed in the future

        self.base_dir, _ = os.path.split(os.path.dirname(os.path.abspath(__file__)))

        self.input_path = os.path.join(self.base_dir, "input_images", self.format)
        self.training_path = os.path.join(self.base_dir, "train_images", "training_" + self.format)
        self.validation_path = os.path.join(self.base_dir, "train_images", "validation_" + self.format)
        self.evaluation_path = os.path.join(self.base_dir, "eval_images", self.format)
        self.predict_path = os.path.join(self.base_dir, "predict_images", self.format)
        self.history_path = os.path.join(self.base_dir, "history")

        # weight_file is the path to weight.h5 file for this model
        self.weight_file = os.path.join(self.base_dir, "weights", "%s_%s_%d-PixelBorder.h5" % (name, self.format, border))

        self.alpha = "SD_" + self.format
        self.beta = "HD_" + self.format

    """
        Asserts that all directories/paths are created
    """
    def main(self):
        paths = [self.training_path, self.validation_path, self.evaluation_path]
        for p in paths:
            path_alpha = os.path.join(p, self.alpha)
            if not os.path.exists(path_alpha):
                os.makedirs(path_alpha)
            path_beta = os.path.join(p, self.beta)
            if not os.path.exists(path_beta):
                os.makedirs(path_beta)

        if not os.path.exists(os.path.join(self.predict_path, self.alpha)):
            os.makedirs(os.path.join(self.predict_path, self.alpha))

        if not os.path.exists(self.history_path):
            os.makedirs(self.history_path)
        #TODO Determine how input path will be needed

    """
        Following functions check the image count of important directories
        Asserts equality in image number for certain directories
        See _image_count for base helper function
    """

    def train_images_count(self):
        a = self._image_count(os.path.join(self.training_path, self.alpha))
        b = self._image_count(os.path.join(self.training_path, self.beta))
        assert a == b, "WARNING: Alpha and Beta set have different image sizes.\n" \
                       "Check " + self.training_path
        return a

    def val_images_count(self):
        a = self._image_count(os.path.join(self.validation_path, self.alpha))
        b = self._image_count(os.path.join(self.validation_path, self.beta))
        assert a == b, "WARNING: Alpha and Beta set have different image sizes.\n" \
                       "Check " + self.validation_path
        return a

    def eval_images_count(self):
        a = self._image_count(os.path.join(self.evaluation_path, self.alpha))
        b = self._image_count(os.path.join(self.evaluation_path, self.beta))
        assert a == b, "WARNING: Alpha and Beta set have different image sizes.\n" \
                       "Check " + self.evaluation_path
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

    def _image_generator(self, directory, shuffle=True):

        file_names = [f for f in sorted(os.listdir(os.path.join(directory, self.alpha)))]
        X_filenames = [os.path.join(directory, self.alpha, f) for f in file_names]
        y_filenames = [os.path.join(directory, self.beta, f) for f in file_names]

        nb_images = len(file_names)
        print("Found %d images." % nb_images)

        index_generator = self._index_generator(nb_images, self.batch_size, shuffle)

        while 1:
            index_array, current_index, current_batch_size = next(index_generator)

            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            batch_y = np.zeros((current_batch_size,) + self.image_shape)

            for i, j in enumerate(index_array):
                x_fn = X_filenames[j]
                if self.current_meta is None:
                    self.current_meta = self.img_meta(x_fn)
                    # Assume that all meta data in a directory is the same
                img = self.img_read(x_fn, self.current_meta)

                if K.image_dim_ordering() == "th":
                    batch_x[i] = img.transpose((2, 0, 1))
                else:
                    batch_x[i] = img

                y_fn = y_filenames[j]
                img = self.img_read(y_fn, self.current_meta)

                if K.image_dim_ordering() == "th":
                    batch_y[i] = img.transpose((2, 0, 1))
                else:
                    batch_y[i] = img

            yield (batch_x, batch_y)
    """
        END : Code derived from Image-Super-Resolution/img_utils.py
    """

    # Image generator for model.predict_generator function (Keras)
    # Returns a single value
    def _predict_image_generator(self, directory, shuffle=True):

        file_names = [f for f in sorted(os.listdir(os.path.join(directory, self.alpha)))]
        filenames = [os.path.join(directory, self.alpha, f) for f in file_names]

        nb_images = len(file_names)
        print("Found %d images." % nb_images)

        index_generator = self._index_generator(nb_images, self.batch_size, shuffle)

        while 1:
            index_array, current_index, current_batch_size = next(index_generator)

            batch= np.zeros((current_batch_size,) + self.image_shape)

            for i, j in enumerate(index_array):
                fn = filenames[j]
                # name, read, write, and meta data functions
                if self.current_meta is None:
                    self.current_meta = self.img_meta(fn)
                    # Assume that all meta data in a directory is the same

                img = self.img_read(fn, self.current_meta)

                if K.image_dim_ordering() == "th":
                    batch[i] = img.transpose((2, 0, 1))
                else:
                    batch[i] = img

            yield (batch)

    """
        Helper for getting image count
    """
    def _image_count(self, imgs_path):
        return len([name for name in os.listdir(imgs_path)])


if __name__ == "__main__":
    PathManager("", format_type=0).main()
    PathManager("", format_type=1).main()
