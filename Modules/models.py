# pylint: disable=C0301
# Line too long
"""
Anime-SR core models
"""

import os
import json
from abc import ABCMeta, abstractmethod

from keras.models import Sequential, Model, load_model
from keras.layers import Add, Average, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras import backend as K
import keras.callbacks as callbacks
import keras.optimizers as optimizers


class ModelState(callbacks.Callback):
    """ State monitor callback. Tracks how well we are doing and writes
        some state to a json file. This lets us resume training seamlessly.

        ModelState.state is:

        { "epoch_count": nnnn,
          "best_values": { dictionary of logs values },
          "best_epoch": { dictionary of logs values },
          "io" : { dictionary of io class state values }
        }
    """
    def __init__(self, io, verbose=True):

        super().__init__()

        self.io = io
        self.verbose = verbose

        if os.path.isfile(io.paths['state']):
            if self.verbose:
                print('Loading existing .json state')
            with open(io.paths['state'], 'r') as jsonfile:
                self.state = json.load(jsonfile)

            # Update state with current run information

            self.state['io'] = io.config
        else:
            self.state = {'epoch_count': 0,
                          'best_values': {},
                          'best_epoch': {},
                          'io': io.config
                         }

    def on_train_begin(self, logs=None):

        if self.verbose:
            print('Training commences...')

    def on_epoch_end(self, epoch, logs=None):

        self.state['epoch_count'] += 1

        # Currently, for everything we track, lower is better.

        for k in logs:
            if k not in self.state['best_values'] or logs[k] < self.state['best_values'][k]:
                self.state['best_values'][k] = float(logs[k])
                self.state['best_epoch'][k] = self.state['epoch_count']

        with open(self.io.paths['state'], 'w') as jsonfile:
            json.dump(self.state, jsonfile, indent=4)

        if self.verbose:
            print('Completed epoch', self.state['epoch_count'])

# GPU : Untested, but may be needed for VDSR

class AdjustableGradient(callbacks.Callback):
    """ Untested callback written by GPU """

    def __init__(self, optimizer, theta=1.0, verbose=True):

        super().__init__()

        self.optimizer = optimizer
        self.lr = optimizer.lr.get_value()
        self.theta = theta
        self.verbose = verbose

    def on_train_begin(self, logs=None):

        if self.verbose:
            print('Starting Gradient Clipping Value: %f' % (self.theta/self.lr))
        self.optimizer.clipvalue.set_value(self.theta/self.lr)

    def on_epoch_end(self, epoch, logs=None):

        # Get current LR
        if self.lr != self.optimizer.lr.get_value():
            self.lr = self.optimizer.lr.get_value()
            self.optimizer.clipvalue.set_value(self.theta/self.lr)
            if self.verbose:
                print('Changed Gradient Clipping Value: %f' % (self.theta/self.lr))



def psnr_loss(y_true, y_pred):
    """ PSNR is Peak Signal to Noise Ratio, which is similar to mean squared error.

        It can be calculated as
        PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

        When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
        However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
        Thus we remove that component completely and only compute the remaining MSE component.
    """

    return -10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


def psnr_loss_border(border):
    """ PSNR excluding the border pixels (returns function) """

    def psnr_loss_border_func(y_true, y_pred):
        """ Curried psnr function that accounts for border pixels """

        if border == 0:
            return psnr_loss(y_true, y_pred)
        else:
            if K.image_data_format() == 'channels_first':
                y_pred = y_pred[:, :, border:-border, border:-border]
                y_true = y_true[:, :, border:-border, border:-border]
            else:
                y_pred = y_pred[:, border:-border, border:-border, :]
                y_true = y_true[:, border:-border, border:-border, :]
            return psnr_loss(y_true, y_pred)

    func = psnr_loss_border_func
    func.__name__ = 'PeakSignaltoNoiseRatio'
    return func

# Dictionary of loss functions (currently only one). All must take border as
# a parameter and return a curried loss function.

LOSS_FUNCTIONS = {'PeakSignaltoNoiseRatio': psnr_loss_border
                 }

#


class BaseSRCNNModel(object):
    """ Base superresolution model; subclass for each individual model. """

    __metaclass__ = ABCMeta

    # io is a modelio.ModelIO handler class; it deals with all the file io,
    # tiling, image generators, etc. lf is the name of the
    # loss function to use.

    def __init__(self, name, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        self.name = name
        self.io = io
        self.lf = lf
        self.val_lf = 'val_' + lf
        self.evaluation_function = LOSS_FUNCTIONS[lf](io.config['border'])
        self.verbose = verbose
        self.bargraph = bargraph

        if os.path.isfile(io.paths['model']):
            if self.verbose:
                print('Loading existing .h5 model')
            self.model = load_model(io.paths['model'], custom_objects={lf: self.evaluation_function})
        else:
            if self.verbose:
                print('Creating new untrained model')
            self.model = self.create_model(load_weights=False)

    # Config will be a dictionary with contents similar to this:
    # {'beta_2': 0.9990000128746033, 'beta_1': 0.8999999761581421, 'decay': 0.0, 'lr': 0.0008100000559352338, 'epsilon': 1e-08}

    def get_config(self):
        """ Config will be a dictionary with contents similar to this:
            {'beta_2': 0.9990000128746033, 'beta_1': 0.8999999761581421, 'decay': 0.0, 'lr': 0.0008100000559352338, 'epsilon': 1e-08}
        """

        return self.model.optimizer.get_config()

    def set_lr(self, new_lr):
        """ Learning rate setter """

        self.model.optimizer.lr = K.variable(new_lr, name='lr')

    @abstractmethod
    def create_model(self, load_weights):
        """ create_model is handled by subclasses """

        pass



    def fit(self, max_epochs=255, run_epochs=0):
        """ Train a model.
            Uses images in self.io.paths['training'] for Training
            Uses images in self.io.paths['validation'] for Validation
        """

        samples_per_epoch = self.io.train_images_count()
        val_count = self.io.val_images_count()

        learning_rate = callbacks.ReduceLROnPlateau(monitor=self.val_lf,
                                                    mode='min',
                                                    factor=0.9,
                                                    min_lr=0.0002,
                                                    patience=10,
                                                    verbose=self.verbose)

        # GPU. mode was 'max', but since we want to minimize the PSNR (better = more
        # negative) shouldn't it be 'min'?

        model_checkpoint = callbacks.ModelCheckpoint(self.io.paths['model'],
                                                     monitor=self.val_lf,
                                                     save_best_only=True,
                                                     verbose=self.verbose,
                                                     mode='min',
                                                     save_weights_only=False)

        # Set up the model state. Can potentially load saved state.

        model_state = ModelState(self.io, verbose=self.verbose)

        # If we have trained previously, set up the model checkpoint so it won't save
        # until it finds something better. Otherwise, it would always save the results
        # of the first epoch.

        if 'best_values' in model_state.state and self.val_lf in model_state.state['best_values']:
            model_checkpoint.best = model_state.state['best_values'][self.val_lf]

        if self.verbose:
            print('Best {} found so far: {}'.format(self.val_lf, model_checkpoint.best))

        callback_list = [model_checkpoint,
                         learning_rate,
                         model_state]

        if self.verbose:
            print('Training model : {}'.format(self.io.config['model']))

        # Offset epoch counts if we are resuming training.

        initial_epoch = model_state.state['epoch_count']

        epochs = max_epochs if run_epochs <= 0 else initial_epoch + run_epochs

        # PU: There is an inconsistency when Keras prints that it has saved an improved
        # model. It reports that it happened in the previous epoch.

        self.model.fit_generator(self.io.training_data_generator(),
                                 steps_per_epoch=samples_per_epoch // self.io.config['batch_size'],
                                 epochs=epochs,
                                 callbacks=callback_list,
                                 verbose=self.bargraph,
                                 validation_data=self.io.validation_data_generator(),
                                 validation_steps=val_count // self.io.config['batch_size'],
                                 initial_epoch=initial_epoch)

        if self.verbose:
            if self.bargraph:
                print('')
            print('          Training results for : {}'.format(self.name))

            for key in ['loss', self.lf]:
                if key in model_state.state['best_values']:
                    print('{0:>30} : {1:16.10f} @ epoch {2}'.format(
                        key, model_state.state['best_values'][key], model_state.state['best_epoch'][key]))
                    vkey = 'val_' + key
                    print('{0:>30} : {1:16.10f} @ epoch {2}'.format(
                        vkey, model_state.state['best_values'][vkey], model_state.state['best_epoch'][vkey]))
            print('')

        # PU: Changed to return the best validation results

        return model_state.state['best_values']['val_' + self.lf]



    def predict_tiles(self, tile_generator, batches):
        """ Predict a sequence of tiles. This can later be expanded to do multiprocessing """

        print('predict tiles')

        result = self.model.predict_generator(generator=tile_generator,
                                              steps=batches,
                                              verbose=self.verbose)

        # Deprocess patches
        if K.image_dim_ordering() == 'th':
            result = result.transpose((0, 2, 3, 1))

        return result



    def evaluate(self):
        """ Evaluate the model on self.evaluation_path """

        print('Validating %s model' % self.name)

        results = self.model.evaluate_generator(self.io.evaluation_data_generator(),
                                                steps=self.io.eval_images_count() // self.io.config['batch_size'])
        print("Loss = %.5f, PeekSignalToNoiseRatio = %.5f" % (results[0], results[1]))


    #

    def save(self, path=None):
        """ Save the model to a .h5 file """

        self.model.save(self.io.paths['model'] if path is None else path)

#----------------------------------------------------------------------------------
# BaseSRCNNModel Subclasses (add your custom models here)
#----------------------------------------------------------------------------------

class BasicSR(BaseSRCNNModel):
    """ Basic SuperResolution """

    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(BasicSR, self).__init__('BasicSR', io, lf, verbose, bargraph)

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
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model

class ExpansionSR(BaseSRCNNModel):
    """ Expansion SuperResolution """

    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(ExpansionSR, self).__init__('ExpansionSR', io, lf, verbose, bargraph)

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
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model


class DeepDenoiseSR(BaseSRCNNModel):
    """ Deep Noise Reduction """

    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(DeepDenoiseSR, self).__init__('DeepDenoiseSR', io, lf, verbose, bargraph)

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
                         padding='same')(m2)

        model = Model(init, decoded)

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model


class VDSR(BaseSRCNNModel):

    """ Very Deep Super Resolution """

    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(VDSR, self).__init__('VDSR', io, lf, verbose, bargraph)

    def create_model(self, load_weights):

        init = Input(shape=self.io.image_shape)

        x = Conv2D(64, (3, 3), activation='relu', padding='same')(init)

        for _ in range(0, 19):
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

        decode = Conv2D(self.io.channels, (3, 3), activation='linear', padding='same')(x)

        model = Model(init, decode)
        adam = optimizers.Adam(lr=0.01, beta_1=0.9,
                               beta_2=0.999, epsilon=0.01, decay=0.0)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model


class PUPSR(BaseSRCNNModel):
    """ Parental Unit Pathetic Super-Resolution Model (elu vs. relu test) """

    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(PUPSR, self).__init__('PUPSR', io, lf, verbose, bargraph)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):

        a = Input(shape=self.io.image_shape)
        b = Conv2D(64, (9, 9), activation='elu', padding='same')(a)
        c = Conv2D(32, (1, 1), activation='elu', padding='same')(b)
        d = Conv2D(self.io.channels, (5, 5), padding='same')(c)
        model = Model(a, d)

        adam = optimizers.Adam(lr=.001, clipvalue=(1.0/.001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model


class GPUSR(BaseSRCNNModel):
    """ Gene-Perpetuation Unit Super-Resolution Model
        Tackles two problems with VDSR, Gradients and 'dead' neurons
        Using VDSR with Exponential Linear Unit (elu), which allows negative values
        I think the issue with VDSR is the ReLu creating "dead" neurons
        Also added gradient clipping and an epsilon
        TO-DO Residuals
    """

    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(GPUSR, self).__init__('GPUSR', io, lf, verbose, bargraph)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):

        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation='elu',
                         padding='same', input_shape=self.io.image_shape))
        for _ in range(19):
            model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))

        model.add(Conv2D(self.io.channels, (3, 3), padding='same'))
        adam = optimizers.Adam(lr=.001, clipvalue=(1.0/.001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model


MODELS = {'BasicSR': BasicSR,
          'VDSR': VDSR,
          'DeepDenoiseSR': DeepDenoiseSR,
          'ExpansionSR': ExpansionSR,
          'PUPSR': PUPSR,
          'GPUSR': GPUSR
         }

"""
    TestModels section
    Define all experimental models
    train.py can train this models using type=test
"""

"""
    Evaluation results
       BasicSR : Loss = 0.00058 | PeekSignalToNoiseRatio = -39.41335
    ELUBasicSR : Loss = 0.00032 | PeekSignalToNoiseRatio = -44.32544
"""
class ELUBasicSR(BaseSRCNNModel):
    """ Test model """
    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(ELUBasicSR, self).__init__('BasicSR', io, lf, verbose, bargraph)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):
        model = Sequential()

        model.add(Conv2D(64, (9, 9), activation='elu',
                         padding='same', input_shape=self.io.image_shape))
        model.add(Conv2D(32, (1, 1), activation='elu', padding='same'))
        model.add(Conv2D(self.io.channels, (5, 5), padding='same'))

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model

"""
    Evaluation results
    ELUExpansionSR : Loss = 0.00033 | PeekSignalToNoiseRatio = -45.90310
       ExpansionSR : Loss = 0.00043 | PeekSignalToNoiseRatio = -41.49931
"""
class ELUExpansionSR(BaseSRCNNModel):
    """ Test Model """

    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(ELUExpansionSR, self).__init__('ExpansionSR', io, lf, verbose, bargraph)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):

        init = Input(shape=self.io.image_shape)
        x = Conv2D(64, (9, 9), activation='elu',
                   padding='same', name='level1')(init)

        x1 = Conv2D(32, (1, 1), activation='elu',
                    padding='same', name='level1_1')(x)
        x2 = Conv2D(32, (3, 3), activation='elu',
                    padding='same', name='level1_2')(x)
        x3 = Conv2D(32, (5, 5), activation='elu',
                    padding='same', name='level1_3')(x)

        x = Average()([x1, x2, x3])

        out = Conv2D(self.io.channels, (5, 5), activation='elu',
                     padding='same', name='output')(x)

        model = Model(init, out)

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model

"""
    Evaluation results
    ELUDeepDenoiseSR : Loss = 0.00065 | PeekSignalToNoiseRatio = -40.46243
       DeepDenoiseSR : Loss = 0.00044 | PeekSignalToNoiseRatio = -42.58665
"""
class ELUDeepDenoiseSR(BaseSRCNNModel):
    """ Test Model """
    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(ELUDeepDenoiseSR, self).__init__('DeepDenoiseSR', io, lf, verbose, bargraph)

    def create_model(self, load_weights):

        init = Input(shape=self.io.image_shape)
        c1 = Conv2D(64, (3, 3), activation='elu', padding='same')(init)
        c1 = Conv2D(64, (3, 3), activation='elu', padding='same')(c1)

        x = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(128, (3, 3), activation='elu', padding='same')(x)
        c2 = Conv2D(128, (3, 3), activation='elu', padding='same')(c2)

        x = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(256, (3, 3), activation='elu', padding='same')(x)

        x = UpSampling2D()(c3)

        c2_2 = Conv2D(128, (3, 3), activation='elu', padding='same')(x)
        c2_2 = Conv2D(128, (3, 3), activation='elu', padding='same')(c2_2)

        m1 = Add()([c2, c2_2])
        m1 = UpSampling2D()(m1)

        c1_2 = Conv2D(64, (3, 3), activation='elu', padding='same')(m1)
        c1_2 = Conv2D(64, (3, 3), activation='elu', padding='same')(c1_2)

        m2 = Add()([c1, c1_2])

        decoded = Conv2D(self.io.channels, (5, 5), activation='linear',
                         padding='same')(m2)

        model = Model(init, decoded)

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model

"""
    Evaluation results
    ELUVDSR : Loss = 0.00027 | PeekSignalToNoiseRatio = -45.43717
       VDSR : Loss = 0.01211 | PeekSignalToNoiseRatio = -19.73214
"""
class ELUVDSR(BaseSRCNNModel):
    """ Test Model """

    def __init__(self, io, lf='PeakSignaltoNoiseRatio', verbose=True, bargraph=True):

        super(ELUVDSR, self).__init__('VDSR', io, lf, verbose, bargraph)

    def create_model(self, load_weights):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation='elu',
                         padding='same', input_shape=self.io.image_shape))
        for _ in range(19):
            model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))

        model.add(Conv2D(self.io.channels, (3, 3), padding='same'))
        adam = optimizers.Adam(lr=.001, clipvalue=(1.0/.001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.io.paths['model'])

        self.model = model
        return model


TESTMODELS = {'ELUBasicSR': ELUBasicSR,
              'ELUVDSR': ELUVDSR,
              'ELUDeepDenoiseSR': ELUDeepDenoiseSR,
              'ELUExpansionSR': ELUExpansionSR
             }

MODELS.update(TESTMODELS) # Adds test models in all

"""
     TODO : Genetic Models section
"""
