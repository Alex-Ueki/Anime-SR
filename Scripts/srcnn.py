from abc import ABCMeta, abstractmethod

import numpy as np
import os
import time

from keras.models import Sequential, Model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from pathmanager import PathManager
from advanced import HistoryCheckpoint

"""
    Intended HD image size is 1440 by 1080
    Tile Size is 60x60 with an added border of "b" pixels
        --> Tile Size is (60+2*b)x(60+2*b)
    Hard-Coded for 64, 64
"""

def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)

def PSNRLossBorder(border):
    def PeekSignaltoNoiseRatio(y_true, y_pred):
        if border == 0:
            return PSNRLoss(y_true, y_pred)
        else:
            if K.image_data_format() == 'channels_first':
                y_pred = y_pred[:, :, border:-border, border:-border]
                y_true = y_true[:, :, border:-border, border:-border]
            else:
                y_pred = y_pred[:, border:-border, border:-border, :]
                y_true = y_true[:, border:-border, border:-border, :]
            return PSNRLoss(y_true, y_pred)
    return PeekSignaltoNoiseRatio

class BaseSRCNNModel(object):

    __metaclass__ = ABCMeta

    def __init__(self, name, format_type=0, base_tile_size=60, border=2, channels=3, batch_size=16):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None
        self.name = name
        self.border = border

        """
        pm (PathManager) is a class that holds all the directory information
        Holds batch_size, image_shape, and all directory paths
        Includes functions for image generators and image counts
        See pathmanager.py for information
        """

        self.pm = PathManager(name, format_type=format_type, base_tile_size=base_tile_size, border=border, channels=channels, batch_size=batch_size)

        self.evaluation_function = PSNRLossBorder(border)

    @abstractmethod
    def create_model(self, load_weights=False):
        pass

    def fit(self, nb_epochs=80, save_history=True):
        """
        Standard method to train any of the models.
        Uses images in self.train_path for Training
        Uses images in self.validation_path for Validation
        """

        samples_per_epoch = self.pm.train_images_count()
        val_count = self.pm.val_images_count()
        if self.model == None: self.create_model()

        callback_list = [callbacks.ModelCheckpoint(self.pm.weight_file, monitor='val_PSNR', save_best_only=True,
                                                   mode='max', save_weights_only=True)]

        if save_history: callback_list.append(HistoryCheckpoint(self.name + "_history.txt"))

        print("Training model : %s" % (self.__class__.__name__))
        self.model.fit_generator(self.pm.training_data_generator(),
                                 steps_per_epoch=samples_per_epoch // self.pm.batch_size,
                                 epochs=nb_epochs,
                                 validation_data=self.pm.validation_data_generator(),
                                 validation_steps=val_count // self.pm.batch_size)

        return self.model

    def evaluate(self):
        """
            Evaluates the model on self.evaluation_path
        """
        print("Validating %s model" % self.name)
        if self.model == None: self.create_model(load_weights=True)

        results = self.model.evaluate_generator(self.pm.evaluation_data_generator(),
                                      steps = self.pm.eval_images_count() // self.pm.batch_size)
        print(self.name,results)

    def predict(self, verbose=True):
        """
            Runs predictions on images in predict_path
        """
        import os
        from scipy.misc import imread, imresize, imsave

        # Read images from predict_path, creating a generator

        model = self.create_model(load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict_generator(self.pm.prediction_data_generator(),
                                         steps = self.pm.predict_images_count() // self.pm.batch_size,
                                         verbose=verbose)

        if verbose: print("\nDe-processing images.")

         # Reordering image dimensions
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32)
        else:
            result = result.astype(np.float32)

        if verbose: print("Completed Reordering images.")

        (num, width, height, channels) = result.shape
        output_directory = os.path.join(self.pm.predict_path, self.name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if verbose: print(("Saving %d images into " % num) + output_directory)
        # Getting old file names
        file_names = [f for f in sorted(os.listdir(os.path.join(self.pm.predict_path,self.pm.alpha)))]
        for i in range(num):
            image = result[i, :, :, :]
            filename = os.path.join(output_directory, self.name + "_" + file_names[i])
            self.pm.img_write(filename, image, self.pm.current_meta)
        if verbose: print(("Saved %d images into " % num) + output_directory)

class BasicSR(BaseSRCNNModel):

    def __init__(self, format_type=0, base_tile_size=60, border=2, channels=3, batch_size=16):
        super(BasicSR, self).__init__("BasicSR",
                                      format_type=format_type,
                                      base_tile_size=base_tile_size,
                                      border=border, channels=channels,
                                      batch_size=batch_size)

    def create_model(self, channels=3, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        model = Sequential()

        model.add(Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=self.pm.image_shape))
        model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
        model.add(Conv2D(channels, (5, 5), padding='same'))

        adam = optimizers.Adam(lr=1e-3)

        model.compile(optimizer=adam, loss='mse', metrics=[self.evaluation_function])
        if load_weights: model.load_weights(self.pm.weight_file)

        self.model = model
        return model

class ExpansionSR(BaseSRCNNModel):

    def __init__(self, format_type=0, base_tile_size=60, border=2, channels=3, batch_size=16):
        super(ExpansionSR, self).__init__("ExpansionSR",
                                          format_type=format_type,
                                          base_tile_size=base_tile_size,
                                          border=border, channels=channels,
                                          batch_size=batch_size)

    def create_model(self, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """

        init = Input(shape=self.pm.image_shape)
        x = Conv2D(64, (9, 9), activation='relu', padding='same', name='level1')(init)

        x1 = Conv2D(32, (1, 1), activation='relu', padding='same', name='lavel1_1')(x)
        x2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='lavel1_2')(x)
        x3 = Conv2D(32, (5, 5), activation='relu', padding='same', name='lavel1_3')(x)

        x = Average()([x1, x2, x3])

        out = Conv2D(self.channels, (5, 5), activation='relu', padding='same', name='output')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)

        model.compile(optimizer=adam, loss='mse', metrics=[self.evaluation_function])
        if load_weights: model.load_weights(self.pm.weight_file)

        self.model = model
        return model

class DeepDenoiseSR(BaseSRCNNModel):

    def __init__(self, format_type=0, base_tile_size=60, border=2, channels=3, batch_size=16):
        super(DeepDenoiseSR, self).__init__("DeepDenoiseSR",
                                            format_type=format_type,
                                            base_tile_size=base_tile_size,
                                            border=border, channels=channels,
                                            batch_size=batch_size)

    def create_model(self, channels=3, load_weights=False):

        init = Input(shape=self.pm.image_shape)
        c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(init)
        c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        c2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)

        c1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(m1)
        c1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1_2)

        m2 = Add()([c1, c1_2])

        decoded = Conv2D(channels, (5, 5), activation='linear', padding='same')(m2)

        model = Model(init, decoded)

        adam = optimizers.Adam(lr=1e-3)

        model.compile(optimizer=adam, loss='mse', metrics=[self.evaluation_function])
        if load_weights: model.load_weights(self.pm.weight_file)

        self.model = model
        return model

# Very Deep Super Resolution
class VDSR(BaseSRCNNModel):

    def __init__(self, format_type=0, base_tile_size=60, border=2, channels=3, batch_size=16):
        super(VDSR, self).__init__("VDSR",
                                   format_type=format_type,
                                   base_tile_size=base_tile_size,
                                   border=border, channels=channels,
                                   batch_size=batch_size)

    def create_model(self, channels=3, load_weights=False):

        init = Input(shape=self.pm.image_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(init)

        for i in range(0,19):
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

        decode = Conv2D(channels, (3, 3), activation='linear', padding='same')(x)

        model = Model(init, decode)
        adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)

        model.compile(optimizer=adam, loss='mse', metrics=[self.evaluation_function])
        if load_weights: model.load_weights(self.pm.weight_file)

        self.model = model
        return model
