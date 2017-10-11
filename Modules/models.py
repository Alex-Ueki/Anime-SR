# pylint: disable=C0301
# Line too long
"""
Anime-SR core models
"""

import os
import json
import copy
from abc import ABCMeta, abstractmethod

from keras.models import Sequential, Model, load_model
from keras.layers import Add, Average, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from Modules.misc import printlog

class ModelState(callbacks.Callback):
    """ State monitor callback. Tracks how well we are doing and writes
        some state to a json file. This lets us resume training seamlessly.

        ModelState.state is:

        { "epoch_count": nnnn,
          "best_values": { dictionary of logs values },
          "best_epoch": { dictionary of logs values },
          "config" : { dictionary of ModelIO state values }
        }
    """
    def __init__(self, config):

        super().__init__()

        self.verbose = config.verbose
        self.path = config.paths['state']

        if os.path.isfile(self.path):
            if config.verbose:
                printlog('Loading existing .json state: ' + self.path)
            with open(self.path, 'r') as jsonfile:
                self.state = json.load(jsonfile)

            # Update state with current run information

            self.state['config'] = config.config
        else:
            printlog('Initializing new model')
            self.state = {'epoch_count': 0,
                          'best_values': {},
                          'best_epoch': {},
                          'config': config.config
                         }

    def on_train_begin(self, logs=None):

        if self.verbose:
            printlog('Training commences...')

    def on_epoch_end(self, epoch, logs=None):

        self.state['epoch_count'] += 1

        # Currently, for everything we track, lower is better.

        for k in logs:
            if k not in self.state['best_values'] or logs[k] < self.state['best_values'][k]:
                self.state['best_values'][k] = float(logs[k])
                self.state['best_epoch'][k] = self.state['epoch_count']

        with open(self.path, 'w') as jsonfile:
            json.dump(self.state, jsonfile, indent=4)

        if self.verbose:
            printlog('Completed epoch', self.state['epoch_count'])

# GPU : Untested, but may be needed for VDSR

class AdjustableGradient(callbacks.Callback):
    """ Untested callback written by GPU """

    def __init__(self, optimizer, theta=1.0, verbose=True):

        super().__init__()

        self.optimizer = optimizer
        self.learning_rate = optimizer.lr.get_value()
        self.theta = theta
        self.verbose = verbose

    def on_train_begin(self, logs=None):

        if self.verbose:
            printlog('Starting Gradient Clipping Value: %f' % (self.theta/self.learning_rate))
        self.optimizer.clipvalue.set_value(self.theta/self.learning_rate)

    def on_epoch_end(self, epoch, logs=None):

        # Get current LR
        if self.learning_rate != self.optimizer.lr.get_value():
            self.learning_rate = self.optimizer.lr.get_value()
            self.optimizer.clipvalue.set_value(self.theta/self.learning_rate)
            if self.verbose:
                printlog('Changed Gradient Clipping Value: %f' % (self.theta/self.learning_rate))



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


class BaseSRCNNModel(object):
    """ Base superresolution model; subclass for each individual model. """

    __metaclass__ = ABCMeta

    # io is a modelio.ModelIO handler class; it deals with all the file io,
    # tiling, image generators, etc.

    def __init__(self, name, config, loss_function='PeakSignaltoNoiseRatio'):

        self.name = name
        self.config = copy.deepcopy(config)
        self.loss_function = loss_function
        self.val_loss_function = 'val_' + loss_function
        self.evaluation_function = LOSS_FUNCTIONS[loss_function](config.border)

        if os.path.isfile(config.paths['model']):
            if self.config.verbose:
                printlog('Loading existing .h5 model: ' + config.paths['model'])
            self.model = load_model(config.paths['model'], custom_objects={loss_function: self.evaluation_function})
        else:
            if self.config.verbose:
                printlog('Creating new untrained model')
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
            Uses images in self.config.paths['training'] for Training
            Uses images in self.config.paths['validation'] for Validation
        """

        """
        printlog('fit')
        for key in self.config.config:
            if key != 'paths':
                printlog(key, self.config.config[key])
        """

        samples_per_epoch = self.config.train_images_count()
        val_count = self.config.val_images_count()

        learning_rate = callbacks.ReduceLROnPlateau(monitor=self.val_loss_function,
                                                    mode='min',
                                                    factor=0.9,
                                                    min_lr=0.0002,
                                                    patience=10,
                                                    verbose=self.config.verbose)

        # GPU. mode was 'max', but since we want to minimize the PSNR (better = more
        # negative) shouldn't it be 'min'?

        model_checkpoint = callbacks.ModelCheckpoint(self.config.paths['model'],
                                                     monitor=self.val_loss_function,
                                                     save_best_only=True,
                                                     verbose=self.config.verbose,
                                                     mode='min',
                                                     save_weights_only=False)

        # Set up the model state. Can potentially load saved state.

        model_state = ModelState(self.config)

        # If we have trained previously, set up the model checkpoint so it won't save
        # until it finds something better. Otherwise, it would always save the results
        # of the first epoch.

        if 'best_values' in model_state.state and self.val_loss_function in model_state.state['best_values']:
            model_checkpoint.best = model_state.state['best_values'][self.val_loss_function]

        if self.config.verbose:
            printlog('Best {} found so far: {}'.format(self.val_loss_function, model_checkpoint.best))

        callback_list = [model_checkpoint,
                         learning_rate,
                         model_state]

        if self.config.verbose:
            printlog('Training model : {}'.format(self.config.config['model_type']))

        # Offset epoch counts if we are resuming training.

        initial_epoch = model_state.state['epoch_count']

        epochs = max_epochs if run_epochs <= 0 else initial_epoch + run_epochs

        # PU: There is an inconsistency when Keras prints that it has saved an improved
        # model. It reports that it happened in the previous epoch.

        self.model.fit_generator(self.config.training_data_generator(),
                                 steps_per_epoch=samples_per_epoch // self.config.batch_size,
                                 epochs=epochs,
                                 callbacks=callback_list,
                                 verbose=self.config.bargraph,
                                 validation_data=self.config.validation_data_generator(),
                                 validation_steps=val_count // self.config.batch_size,
                                 initial_epoch=initial_epoch)

        if self.config.verbose:
            if self.config.bargraph:
                print('')
            print('          Training results for : {}'.format(self.name))

            for key in ['loss', self.loss_function]:
                if key in model_state.state['best_values']:
                    print('{0:>30} : {1:16.10f} @ epoch {2}'.format(
                        key, model_state.state['best_values'][key], model_state.state['best_epoch'][key]))
                    vkey = 'val_' + key
                    print('{0:>30} : {1:16.10f} @ epoch {2}'.format(
                        vkey, model_state.state['best_values'][vkey], model_state.state['best_epoch'][vkey]))
            print('')

        # PU: Changed to return the best validation results

        return model_state.state['best_values']['val_' + self.loss_function]



    def predict_tiles(self, tile_generator, batches):
        """ Predict a sequence of tiles. This can later be expanded to do multiprocessing """

        result = self.model.predict_generator(generator=tile_generator,
                                              steps=batches,
                                              verbose=self.config.verbose)

        # Deprocess patches
        if K.image_dim_ordering() == 'th':
            result = result.transpose((0, 2, 3, 1))

        return result



    def evaluate(self):
        """ Evaluate the model on self.evaluation_path """

        printlog('Validating %s model' % self.name)

        results = self.model.evaluate_generator(self.config.evaluation_data_generator(),
                                                steps=self.config.eval_images_count() // self.config.batch_size)
        print("Loss = %.5f, PeekSignalToNoiseRatio = %.5f" % (results[0], results[1]))




    def save(self, path=None):
        """ Save the model to a .h5 file """

        self.model.save(self.config.paths['model'] if path is None else path)

#----------------------------------------------------------------------------------
# BaseSRCNNModel Subclasses (add your custom models here)
#----------------------------------------------------------------------------------

class BasicSR(BaseSRCNNModel):
    """ Basic SuperResolution """

    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(BasicSR, self).__init__('BasicSR', config, loss_function)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):
        model = Sequential()

        model.add(Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=self.config.image_shape))
        model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
        model.add(Conv2D(self.config.channels, (5, 5), padding='same'))

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse', metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

        self.model = model

        return model

class ExpansionSR(BaseSRCNNModel):
    """ Expansion SuperResolution """

    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(ExpansionSR, self).__init__('ExpansionSR', config, loss_function)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):

        init = Input(shape=self.config.image_shape, dtype='float32')
        layer1 = Conv2D(64, (9, 9), activation='relu', padding='same', name='level1')(init)

        layer2a = Conv2D(32, (1, 1), activation='relu', padding='same', name='lavel1_1')(layer1)
        layer2b = Conv2D(32, (3, 3), activation='relu', padding='same', name='lavel1_2')(layer1)
        layer2c = Conv2D(32, (5, 5), activation='relu', padding='same', name='lavel1_3')(layer1)

        layer3 = Average()([layer2a, layer2b, layer2c])

        out = Conv2D(self.config.channels, (5, 5), activation='relu', padding='same', name='output')(layer3)

        model = Model(init, out)

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

        self.model = model

        return model


class DeepDenoiseSR(BaseSRCNNModel):
    """ Deep Noise Reduction """

    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(DeepDenoiseSR, self).__init__('DeepDenoiseSR', config, loss_function)

    def create_model(self, load_weights):

        init = Input(shape=self.config.image_shape, dtype='float32')
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(init)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)

        mix = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(mix)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)

        mix = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(mix)

        mix = UpSampling2D()(conv3)

        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(mix)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_2)

        mix1 = Add()([conv2, conv2_2])
        mix1 = UpSampling2D()(mix1)

        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(mix1)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_2)

        mix2 = Add()([conv1, conv1_2])

        decoded = Conv2D(self.config.channels, (5, 5), activation='linear', padding='same')(mix2)

        model = Model(init, decoded)

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

        self.model = model

        return model


class VDSR(BaseSRCNNModel):

    """ Very Deep Super Resolution """

    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(VDSR, self).__init__('VDSR', config, loss_function)

    def create_model(self, load_weights):

        init = Input(shape=self.config.image_shape, dtype='float32')

        mix = Conv2D(64, (3, 3), activation='relu', padding='same')(init)

        for _ in range(0, 19):
            mix = Conv2D(64, (3, 3), activation='relu', padding='same')(mix)

        decode = Conv2D(self.config.channels, (3, 3), activation='linear', padding='same')(mix)

        model = Model(init, decode)
        adam = optimizers.Adam(lr=0.01, beta_1=0.9,
                               beta_2=0.999, epsilon=0.01, decay=0.0)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

        self.model = model

        return model


class PUPSR(BaseSRCNNModel):
    """ Parental Unit Pathetic Super-Resolution Model (batch normalization, residual, sequential)

        BasicSR (20 Epochs, relu) : -39.7536911815 @ epoch 15
        PUPSR (20 Epochs, relu)   : -40.8934259724 @ epoch 20
        PUPSR (20 Epochs, elu)    : -40.7926285811 @ epoch 15
        PUPSR (20 Epochs, leveld) : -40.8624241233 @ epoch 16
    """

    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(PUPSR, self).__init__('PUPSR', config, loss_function)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):

        model = Sequential()
        model.add(Conv2D(64, (9, 9), padding='same', activation='elu', input_shape=self.config.image_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (1, 1), padding='same', activation='elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(self.config.channels, (5, 5), padding='same'))

        adam = optimizers.Adam(lr=.001, clipvalue=(1.0/.001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

        self.model = model

        return model

class PUPSR2(BaseSRCNNModel):
    """ Parental Unit Pathetic Super-Resolution Model V2 - API generated
    """

    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(PUPSR2, self).__init__('PUPSR2', config, loss_function)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):

        layer0 = Input(shape=self.config.image_shape, dtype='float32')
        layer1 = Conv2D(64, (9, 9), padding='same', activation='elu')(layer0)
        layer1 = BatchNormalization()(layer1)
        layer1 = Conv2D(32, (1, 1), padding='same', activation='elu')(layer1)
        layer1 = BatchNormalization()(layer1)
        layer1 = Conv2D(self.config.channels, (5, 5), padding='same')(layer1)

        model = Model(layer0, layer1)

        adam = optimizers.Adam(lr=.001, clipvalue=(1.0/.001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

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

    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(GPUSR, self).__init__('GPUSR', config, loss_function)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):

        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation='elu',
                         padding='same', input_shape=self.config.image_shape))
        for _ in range(19):
            model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))

        model.add(Conv2D(self.config.channels, (3, 3), padding='same'))
        adam = optimizers.Adam(lr=.001, clipvalue=(1.0/.001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

        self.model = model

        return model


MODELS = {'BasicSR': BasicSR,
          'VDSR': VDSR,
          'DeepDenoiseSR': DeepDenoiseSR,
          'ExpansionSR': ExpansionSR,
          'PUPSR': PUPSR,
          'PUPSR2': PUPSR2,
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
    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(ELUBasicSR, self).__init__('BasicSR', config, loss_function)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):
        model = Sequential()

        model.add(Conv2D(64, (9, 9), activation='elu',
                         padding='same', input_shape=self.config.image_shape))
        model.add(Conv2D(32, (1, 1), activation='elu', padding='same'))
        model.add(Conv2D(self.config.channels, (5, 5), padding='same'))

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

        self.model = model

        return model

"""
    Evaluation results
    ELUExpansionSR : Loss = 0.00033 | PeekSignalToNoiseRatio = -45.90310
       ExpansionSR : Loss = 0.00043 | PeekSignalToNoiseRatio = -41.49931
"""
class ELUExpansionSR(BaseSRCNNModel):
    """ Test Model """

    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(ELUExpansionSR, self).__init__('ExpansionSR', config, loss_function)

    # Create a model to be used to sharpen images of specific height and width.

    def create_model(self, load_weights):

        init = Input(shape=self.config.image_shape, dtype='float32')
        mix = Conv2D(64, (9, 9), activation='elu', padding='same', name='level1')(init)

        conv1 = Conv2D(32, (1, 1), activation='elu', padding='same', name='level1_1')(mix)
        conv2 = Conv2D(32, (3, 3), activation='elu', padding='same', name='level1_2')(mix)
        conv3 = Conv2D(32, (5, 5), activation='elu', padding='same', name='level1_3')(mix)

        mix = Average()([conv1, conv2, conv3])

        out = Conv2D(self.config.channels, (5, 5), activation='elu', padding='same', name='output')(mix)

        model = Model(init, out)

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

        self.model = model

        return model

"""
    Evaluation results
    ELUDeepDenoiseSR : Loss = 0.00065 | PeekSignalToNoiseRatio = -40.46243
       DeepDenoiseSR : Loss = 0.00044 | PeekSignalToNoiseRatio = -42.58665
"""
class ELUDeepDenoiseSR(BaseSRCNNModel):
    """ Test Model """
    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(ELUDeepDenoiseSR, self).__init__('DeepDenoiseSR', config, loss_function)

    def create_model(self, load_weights):

        init = Input(shape=self.config.image_shape, dtype='float32')
        conv1 = Conv2D(64, (3, 3), activation='elu', padding='same')(init)
        conv1 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv1)

        mix = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='elu', padding='same')(mix)
        conv2 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv2)

        mix = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='elu', padding='same')(mix)

        mix = UpSampling2D()(conv3)

        conv2_2 = Conv2D(128, (3, 3), activation='elu', padding='same')(mix)
        conv2_2 = Conv2D(128, (3, 3), activation='elu', padding='same')(conv2_2)

        mix1 = Add()([conv2, conv2_2])
        mix1 = UpSampling2D()(mix1)

        conv1_2 = Conv2D(64, (3, 3), activation='elu', padding='same')(mix1)
        conv1_2 = Conv2D(64, (3, 3), activation='elu', padding='same')(conv1_2)

        mix2 = Add()([conv1, conv1_2])

        decoded = Conv2D(self.config.channels, (5, 5), activation='linear',
                         padding='same')(mix2)

        model = Model(init, decoded)

        adam = optimizers.Adam(lr=.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

        self.model = model

        return model

"""
    Evaluation results
    ELUVDSR : Loss = 0.00027 | PeekSignalToNoiseRatio = -45.43717
       VDSR : Loss = 0.01211 | PeekSignalToNoiseRatio = -19.73214
"""
class ELUVDSR(BaseSRCNNModel):
    """ Test Model """

    def __init__(self, config, loss_function='PeakSignaltoNoiseRatio'):

        super(ELUVDSR, self).__init__('VDSR', config, loss_function)

    def create_model(self, load_weights):
        model = Sequential()

        model.add(Conv2D(64, (3, 3), activation='elu',
                         padding='same', input_shape=self.config.image_shape))
        for _ in range(19):
            model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))

        model.add(Conv2D(self.config.channels, (3, 3), padding='same'))
        adam = optimizers.Adam(lr=.001, clipvalue=(1.0/.001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse',
                      metrics=[self.evaluation_function])

        if load_weights:
            model.load_weights(self.config.paths['model'])

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
