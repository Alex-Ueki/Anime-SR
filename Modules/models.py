from abc import ABCMeta, abstractmethod

import numpy as np
import os
import time
import json

from keras.models import Sequential, Model, load_model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from Modules.modelio import ModelIO

# State monitor callback. Tracks how well we are doing and writes
# some state to a json file. This lets us resume training seamlessly.
#
# ModelState.state is:
#
# { "epoch_count": nnnn,
#   "best_values": { dictionary of logs values },
#   "best_epoch": { dictionary of logs values }
# }
#
# PU: Add history when time permits

class ModelState(callbacks.Callback):

    def __init__(self, io):

        self.io = io

        if os.path.isfile(io.state_path):
            print('Loading existing .json state')
            with open(io.state_path, 'r') as f:
                self.state = json.load(f)

            # Update state with current run information

            self.state['io'] = io.asdict()
        else:
            self.state = { 'epoch_count': 0,
                           'best_values': {},
                           'best_epoch': {},
                           'io': io.asdict()
                         }

    def on_train_begin(self, logs={}):

        print('Training commences...')

    def on_epoch_end(self, batch, logs={}):

        # Currently, for everything we track, lower is better

        for k in logs:
            if k not in self.state['best_values'] or logs[k] < self.state['best_values'][k]:
                self.state['best_values'][k] = float(logs[k])
                self.state['best_epoch'][k] = self.state['epoch_count']

        with open(self.io.state_path, 'w') as f:
            json.dump(self.state, f, indent=4)
        print('Completed epoch', self.state['epoch_count'])

        self.state['epoch_count'] += 1


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

# Dictionary of loss functions (currently only one). All must take border as
# a parameter and return a curried loss function.

loss_functions = { 'PeakSignaltoNoiseRatio': PSNRLossBorder
                 }

# Base superresolution model; subclass for each individual model.


class BaseSRCNNModel(object):

    __metaclass__ = ABCMeta

    # io is a modelio.ModelIO handler class; it deals with all the file io,
    # tiling, image generators, etc. lf is the name of the
    # loss function to use.

    def __init__(self, name, io, lf='PeakSignaltoNoiseRatio'):

        self.name = name
        self.io = io
        self.lf = lf
        self.val_lf = 'val_' + lf
        self.evaluation_function = loss_functions[lf](io.border)

        if os.path.isfile(io.model_path):
            print('Loading existing .h5 model')
            self.model = load_model(io.model_path, custom_objects={ lf: self.evaluation_function })
        else:
            print('Creating new untrained model')
            self.model = self.create_model(load_weights=False)


    @abstractmethod
    def create_model(self, load_weights):
        pass

    # Standard method to train any of the models.
    # Uses images in self.io.training_path for Training
    # Uses images in self.io.validation_path for Validation

    def fit(self, epochs=255, save_history=True):

        samples_per_epoch = self.io.train_images_count()
        val_count = self.io.val_images_count()

        learning_rate = callbacks.ReduceLROnPlateau(monitor=self.val_lf,
                                                    mode='min',
                                                    factor=0.9,
                                                    min_lr=0.0002,
                                                    patience=10,
                                                    verbose=1)

        # GPU. mode was 'max', but since we want to minimize the PSNR (better = more
        # negative) shouldn't it be 'min'?

        model_checkpoint = callbacks.ModelCheckpoint(self.io.model_path,
                                                     monitor=self.val_lf,
                                                     save_best_only=True,
                                                     verbose=1,
                                                     mode='min',
                                                     save_weights_only=False)

        # Set up the model state, reading in prior results info if available

        model_state = ModelState(self.io)

        # If we have trained previously, set up the model checkpoint so it won't save
        # until it finds something better. Otherwise, it would always save the results
        # of the first epoch.

        if 'best_values' in model_state.state and self.val_lf in model_state.state['best_values']:
            model_checkpoint.best = model_state.state['best_values'][self.val_lf]

        print('Best {} found so far: {}'.format(self.val_lf,model_checkpoint.best))

        callback_list = [model_checkpoint,
                         learning_rate,
                         model_state]

        print('Training model : %s' % (self.__class__.__name__))

        # Offset epoch counts if we are resuming training

        initial_epoch = model_state.state['epoch_count']
        if initial_epoch > 0:
            initial_epoch += 1
            epochs += initial_epoch

        self.model.fit_generator(self.io.training_data_generator(),
                                 steps_per_epoch=samples_per_epoch // self.io.batch_size,
                                 epochs=epochs,
                                 callbacks=callback_list,
                                 verbose=0,
                                 validation_data=self.io.validation_data_generator(),
                                 validation_steps=val_count // self.io.batch_size,
                                 initial_epoch=initial_epoch)

        print('')
        print('          Training results for : %s' %
              (self.__class__.__name__))

        for key in ['loss', self.lf]:
            if key in model_state['best_values']:
                print('{0:>30} : {1:16.10f} @ epoch {2}'.format(
                    key, model_state.state['best_values'][key], model_state.state['best_epoch'][key]))
                vkey = 'val_' + key
                print('{0:>30} : {1:16.10f} @ epoch {2}'.format(
                    vkey, model_state.state['best_values'][vkey], model_state.state['best_epoch'][vkey]))
        print('')

        return self.model

    # Predict a sequence of tiles. This can later be expanded to do multiprocessing

    def predict_tiles(self, tile_generator, batches, verbose=True):

        print('predict tiles')

        result = self.model.predict_generator(generator=tile_generator,
                                              steps=batches,
                                              verbose=verbose)

        # Deprocess patches
        if K.image_dim_ordering() == 'th':
            result = result.transpose((0, 2, 3, 1))

        return result

    """
    # Evaluate the model on self.evaluation_path

    def evaluate(self):

        print('Validating %s model' % self.name)

        results = self.model.evaluate_generator(self.io.evaluation_data_generator(),
                                                steps=self.io.eval_images_count() // self.io.batch_size)
        print(self.name, results)

    # Run predictions on images in self.io.predict_path

    def predict(self, verbose=True):

        import os
        from Modules.frameops import imsave

        # Create prediction for image patches

        result = self.model.predict_generator(self.io.prediction_data_generator(),
                                              steps=self.io.predict_images_count() // self.io.batch_size,
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
        output_directory = os.path.join(self.io.predict_path, self.name)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if verbose:
            print(('Saving %d images into ' % num) + output_directory)
        # Getting old file names
        file_names = [f for f in sorted(os.listdir(
            os.path.join(self.io.predict_path, self.io.alpha)))]
        for i in range(num):
            output = np.clip(result[i, :, :, :], 0, 255).astype(
                'uint8')  # TODO Change this for new image type
            filename = output_directory + (self.name + '_' + file_names[i])
            imsave(filename, output)
        if verbose:
            print(('Save %d images into ' % num) + output_directory)
"""

    # Save the model to a .h5 file

    def save(self, path=None):

        self.model.save(self.io.model_path if path == None else path)


class BasicSR(BaseSRCNNModel):

    def __init__(self, io):

        super(BasicSR, self).__init__('BasicSR', io)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):
        model = Sequential()

        model.add(Conv2D(64, (9, 9), activation='relu',
                         padding='same', input_shape=self.io.image_shape))
        model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
        model.add(Conv2D(self.io.channels, (5, 5), padding='same'))

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.model_path)

        self.model = model
        return model

class ExpansionSR(BaseSRCNNModel):

    def __init__(self, io):

        super(ExpansionSR, self).__init__('ExpansionSR', io)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):

        init = Input(shape=self.io.image_shape)
        x = Conv2D(64, (9, 9), activation='relu',
                   padding='same', name='level1')(init)

        x1 = Conv2D(32, (1, 1), activation='relu',
                    padding='same', name='lavel1_1')(x)
        x2 = Conv2D(32, (3, 3), activation='relu',
                    padding='same', name='lavel1_2')(x)
        x3 = Conv2D(32, (5, 5), activation='relu',
                    padding='same', name='lavel1_3')(x)

        x = Average()([x1, x2, x3])

        out = Conv2D(self.io.channels, (5, 5), activation='relu',
                     padding='same', name='output')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.model_path)

        self.model = model
        return model


class DeepDenoiseSR(BaseSRCNNModel):

    def __init__(self, io):

        super(DeepDenoiseSR, self).__init__('DeepDenoiseSR', io)

    def create_model(self, load_weights):

        init = Input(shape=self.io.image_shape)
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

        decoded = Conv2D(self.io.channels, (5, 5), activation='linear',
                         border_mode='same')(m2)

        model = Model(init, decoded)

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.model_path)

        self.model = model
        return model

# Very Deep Super Resolution


class VDSR(BaseSRCNNModel):

    def __init__(self, io):

        super(VDSR, self).__init__('VDSR', io)

    def create_model(self, load_weights):

        init = Input(shape=self.io.image_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(init)

        for i in range(0, 19):
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

        decode = Conv2D(self.io.channels, (3, 3), activation='linear',
                        border_mode='same')(x)

        model = Model(init, decode)
        adam = optimizers.Adam(lr=0.01, beta_1=0.9,
                               beta_2=0.999, epsilon=0.01, decay=0.0)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.model_path)

        self.model = model
        return model

# Dictionary of all the model classes


models = {'BasicSR': BasicSR,
          'VDSR': VDSR,
          'DeepDenoiseSR': DeepDenoiseSR,
          'ExpansionSR': ExpansionSR
          }
