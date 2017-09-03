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

from Modules.basemodel import PathManager
from Modules.advanced import HistoryCheckpoint


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs)


def PSNRLoss(y_true, y_pred):

    # PSNR is Peak Signal to Noise Ratio, which is similar to mean squared error.
    #
    # It can be calculated as
    # PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    #
    # When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    # However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    # Thus we remove that component completely and only compute the remaining MSE component.

    return -10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


def PSNRLossBorder(border):

    # PU Note: GPU cannot spell "peak" correctly

    def PeakSignaltoNoiseRatio(y_true, y_pred):
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

    return PeakSignaltoNoiseRatio


class BaseSRCNNModel(object):

    __metaclass__ = ABCMeta

    # Base model to provide a standard interface of adding Super Resolution models

    def __init__(self, name, base_tile_width=60, base_tile_height=60, border=2, channels=3, batch_size=16,
                 black_level=0.0, trim_top=0, trim_bottom=0, trim_left=0, trim_right=0, tiles_per_image=1,
                 jitter=True, shuffle=True, skip=True, paths={}):

        self.model = None
        self.name = name
        # self.border = border

        # Parameter passing paranoia

        print('')
        print('Model Initialization...')
        print('            Name : {}'.format(name))
        print(' base_tile_width : {}'.format(base_tile_width))
        print('base_tile_height : {}'.format(base_tile_height))
        print('          border : {}'.format(border))
        print('        channels : {}'.format(channels))
        print('      batch_size : {}'.format(batch_size))
        print('     black_level : {}'.format(black_level))
        print('       trim tblr : {} {} {} {}'.format(
            trim_top, trim_bottom, trim_left, trim_right))
        print(' tiles_per_image : {}'.format(tiles_per_image))
        print('          jitter : {}'.format(jitter == 1)
        print('         shuffle : {}'.format(shuffle == 1)
        print('            skip : {}'.format(skip == 1)
        print('    path entries : {}'.format(paths.keys()))

        # pm (PathManager) is a class that holds all the directory information
        # Holds batch_size, image_shape, and all directory paths
        # Includes functions for image generators and image counts
        # See basemodel.py for information

        self.pm = PathManager(name, base_tile_width=base_tile_width, base_tile_height=base_tile_height,
                              border=border, channels=channels, batch_size=batch_size,
                              black_level=black_level, trim_top=trim_top, trim_bottom=trim_bottom,
                              trim_left=trim_left, trim_right=trim_right, tiles_per_image=tiles_per_image,
                              jitter=jitter, shuffle=shuffle, skip=skip, paths=paths)

        self.evaluation_function = PSNRLossBorder(border)

    @abstractmethod
    def create_model(self, load_weights=False):
        pass

    # Standard method to train any of the models.
    # Uses images in self.pm.training_path for Training
    # Uses images in self.pm.validation_path for Validation

    def fit(self, nb_epochs=80, save_history=True):

        samples_per_epoch = self.pm.train_images_count()
        val_count = self.pm.val_images_count()

        if self.model == None:
            self.create_model()

        loss_history = LossHistory()

        # PU Question: This was val_PeekSignaltoNoiseRatio. Is that a typo? Where is documentation on how to use monitor
        # field. PU is very confused.

        callback_list = [callbacks.ModelCheckpoint(self.pm.weight_path, monitor='val_PeakSignaltoNoiseRatio', save_best_only=True,
                                                   mode='max', save_weights_only=True),
                         loss_history]
        if save_history:
            callback_list.append(HistoryCheckpoint(self.pm.history_path))
        print('Training model : %s' % (self.__class__.__name__))

        self.model.fit_generator(self.pm.training_data_generator(),
                                 steps_per_epoch=samples_per_epoch // self.pm.batch_size,
                                 epochs=nb_epochs,
                                 callbacks=callback_list,
                                 validation_data=self.pm.validation_data_generator(),
                                 validation_steps=val_count // self.pm.batch_size)

        print('')
        print('          Training results for : %s' %
              (self.__class__.__name__))

        for key in ['loss', 'val_loss', 'PeakSignaltoNoiseRatio', 'val_PeakSignaltoNoiseRatio']:
            vals = [epoch[key] for epoch in loss_history.losses]
            min_val = min(vals)
            print('{0:>30} : {1:16.10f} @ epoch {2}'.format(
                key, min_val, 1 + vals.index(min_val)))
        print('')

        return self.model

    # Evaluate the model on self.evaluation_path

    def evaluate(self):

        print('Validating %s model' % self.name)
        if self.model == None:
            self.create_model(load_weights=True)

        results = self.model.evaluate_generator(self.pm.evaluation_data_generator(),
                                                steps=self.pm.eval_images_count() // self.pm.batch_size)
        print(self.name, results)

    # Run predictions on images in self.pm.predict_path

    def predict(self, verbose=True):

        import os
        from Modules.frameops import imsave

        # Read images from predict_path, creating a generator

        model = self.create_model(load_weights=True)
        if verbose:
            print('Model loaded.')

        # Create prediction for image patches
        result = model.predict_generator(self.pm.prediction_data_generator(),
                                         steps=self.pm.predict_images_count() // self.pm.batch_size,
                                         verbose=verbose)

        if verbose:
            print('\nDe-processing images.')

        # Deprocess patches
        if K.image_dim_ordering() == 'th':
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.

        if verbose:
            print('Completed De-processing image.')

        (num, width, height, channels) = result.shape
        output_directory = os.path.join(self.pm.predict_path, self.name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if verbose:
            print(('Saving %d images into ' % num) + output_directory)
        # Getting old file names
        file_names = [f for f in sorted(os.listdir(
            os.path.join(self.pm.predict_path, self.pm.alpha)))]
        for i in range(num):
            output = np.clip(result[i, :, :, :], 0, 255).astype(
                'uint8')  # TODO Change this for new image type
            filename = output_directory + (self.name + '_' + file_names[i])
            imsave(filename, output)
        if verbose:
            print(('Save %d images into ' % num) + output_directory)

    # Save the model to a weights file

    def save(self, path=None):

        self.model.save(self.pm.weight_path if path == None else path)


class BasicSR(BaseSRCNNModel):

    def __init__(self, base_tile_width=60, base_tile_height=60, border=2, channels=3, batch_size=16,
                 black_level=0.0, trim_top=0, trim_bottom=0, trim_left=0, trim_right=0,
                 tiles_per_image=1, paths={}):
        super(BasicSR, self).__init__('BasicSR', base_tile_width=base_tile_width, base_tile_height=base_tile_height,
                                      border=border, channels=channels, batch_size=batch_size, black_level=black_level,
                                      trim_top=trim_top, trim_bottom=trim_bottom, trim_left=trim_left, trim_right=trim_right,
                                      tiles_per_image=tiles_per_image, paths=paths)

    # Create a model to be used to scale images of specific height and width.

    def create_model(self, load_weights=False):
        model = Sequential()

        model.add(Conv2D(64, (9, 9), activation='relu',
                         padding='same', input_shape=self.pm.image_shape))
        model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
        model.add(Conv2D(self.pm.channels, (5, 5), padding='same'))

        adam = optimizers.Adam(lr=1e-3)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])
        if load_weights:
            model.load_weights(self.pm.weight_path)

        self.model = model
        return model

# PU Note: I've only run BasicSR


class ExpansionSR(BaseSRCNNModel):

    def __init__(self, base_tile_width=60, base_tile_height=60, border=2, channels=3, batch_size=16,
                 black_level=0.0, trim_top=0, trim_bottom=0, trim_left=0, trim_right=0,
                 tiles_per_image=1, paths={}):

        super(ExpansionSR, self).__init__('ExpansionSR', base_tile_width=base_tile_width, base_tile_height=base_tile_height,
                                          border=border, channels=channels, batch_size=batch_size, black_level=black_level,
                                          trim_top=trim_top, trim_bottom=trim_bottom, trim_left=trim_left, trim_right=trim_right,
                                          tiles_per_image=tiles_per_image, paths=paths)

    # Create a model to be used to scale images of specific height and width.

    def create_model(self, load_weights=False):

        init = Input(shape=self.pm.image_shape)
        x = Conv2D(64, (9, 9), activation='relu',
                   padding='same', name='level1')(init)

        x1 = Conv2D(32, (1, 1), activation='relu',
                    padding='same', name='lavel1_1')(x)
        x2 = Conv2D(32, (3, 3), activation='relu',
                    padding='same', name='lavel1_2')(x)
        x3 = Conv2D(32, (5, 5), activation='relu',
                    padding='same', name='lavel1_3')(x)

        x = Average()([x1, x2, x3])

        out = Conv2D(self.pm.channels, (5, 5), activation='relu',
                     padding='same', name='output')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=1e-3)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])
        if load_weights:
            model.load_weights(self.pm.weight_path)

        self.model = model
        return model


class DeepDenoiseSR(BaseSRCNNModel):

    def __init__(self, base_tile_width=60, base_tile_height=60, border=2, channels=3, batch_size=16,
                 black_level=0.0, trim_top=0, trim_bottom=0, trim_left=0, trim_right=0,
                 tiles_per_image=1, paths={}):

        super(DeepDenoiseSR, self).__init__('DeepDenoiseSR', base_tile_width=base_tile_width, base_tile_height=base_tile_height,
                                            border=border, channels=channels, batch_size=batch_size, black_level=black_level,
                                            trim_top=trim_top, trim_bottom=trim_bottom, trim_left=trim_left, trim_right=trim_right,
                                            tiles_per_image=tiles_per_image, paths=paths)

    def create_model(self, load_weights=False):

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

        decoded = Conv2D(self.pm.channels, (5, 5), activation='linear',
                         border_mode='same')(m2)

        model = Model(init, decoded)

        adam = optimizers.Adam(lr=1e-3)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])
        if load_weights:
            model.load_weights(self.pm.weight_path)

        self.model = model
        return model

# Very Deep Super Resolution


class VDSR(BaseSRCNNModel):

    def __init__(self, base_tile_width=60, base_tile_height=60, border=2, channels=3, batch_size=16,
                 black_level=0.0, trim_top=0, trim_bottom=0, trim_left=0, trim_right=0,
                 tiles_per_image=1, paths={}):

        super(VDSR, self).__init__('VDSR', base_tile_width=base_tile_width, base_tile_height=base_tile_height,
                                   border=border, channels=channels, batch_size=batch_size, black_level=black_level,
                                   trim_top=trim_top, trim_bottom=trim_bottom, trim_left=trim_left, trim_right=trim_right,
                                   tiles_per_image=tiles_per_image, paths=paths)

    def create_model(self, load_weights=False):

        init = Input(shape=self.pm.image_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(init)

        for i in range(0, 19):
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

        decode = Conv2D(self.pm.channels, (3, 3), activation='linear',
                        border_mode='same')(x)

        model = Model(init, decode)
        adam = optimizers.Adam(lr=0.01, beta_1=0.9,
                               beta_2=0.999, epsilon=0.01, decay=0.0)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])
        if load_weights:
            model.load_weights(self.pm.weight_path)

        self.model = model
        return model

# Handy dictionary of all the model classes


models = {'BasicSR': BasicSR,
          'VDSR': VDSR,
          'DeepDenoiseSR': DeepDenoiseSR,
          'ExpansionSR': ExpansionSR
          }
