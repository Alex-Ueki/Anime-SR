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

import setup
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

    def __init__(self, name, border = 2, channels = 3):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None
        self.name = name
        self.border = border

        # This will be replaced with a class from "setup.py" in the future
        """
            Declare all your paths and constants in setup.py
        """
        self.batch_size = setup.batch_size
        self.weight_path = setup.weight_path + self.name + ("%d-PixelBorder.h5" % border)
        self.train_path = setup.training_output_path
        self.validation_path = setup.validation_output_path
        self.evaluation_path = setup.evaluation_output_path
        self.predict_path = setup.predict_path

        self.evaluation_function = PSNRLossBorder(border)

        if K.image_data_format() == 'channels_first':
            self.input_shape = (channels, 60 + 2*border, 60 + 2*border)
        else:
            self.input_shape = (60 + 2*border, 60 + 2*border, channels)

    @abstractmethod
    def create_model(self, channels=3, load_weights=False):
        pass

    def fit(self, nb_epochs=80, save_history=True):
        """
        Standard method to train any of the models.
        Uses images in self.train_path for Training
        Uses images in self.validation_path for Validation
        """

        samples_per_epoch = setup.train_images_count()
        val_count = setup.val_images_count()
        if self.model == None: self.create_model(batch_size=self.batch_size)

        callback_list = [callbacks.ModelCheckpoint(self.weight_path, monitor='val_PSNR', save_best_only=True,
                                                   mode='max', save_weights_only=True)]
        if save_history: callback_list.append(HistoryCheckpoint(self.name + "_history.txt"))

        print("Training model : %s" % (self.__class__.__name__))
        self.model.fit_generator(setup.image_generator(self.train_path, batch_size=self.batch_size),
                                 steps_per_epoch=samples_per_epoch // self.batch_size,
                                 epochs=nb_epochs,
                                 validation_data=setup.image_generator(self.validation_path, batch_size=self.batch_size),
                                 validation_steps=val_count // self.batch_size)

        return self.model

    def evaluate(self):
        """
            Evaluates the model on self.evaluation_path
        """
        print("Validating %s model" % self.name)
        if self.model == None: self.create_model(load_weights=True)

        results = self.model.evaluate_generator(setup.image_generator(self.evaluation_path, batch_size=self.batch_size),
                                      steps = setup.eval_images_count() // self.batch_size)
        print(self.name,results)

    def predict(self, verbose=True):
        """
            Runs predictions on images in predict_path
        """
        import os
        from scipy.misc import imread, imresize, imsave

        # Read images from predict_path, creating a generator
        imagen = setup.predict_image_generator(self.predict_path, shuffle=False, batch_size=self.batch_size)

        model = self.create_model(load_weights=True)
        if verbose: print("Model loaded.")

        # Create prediction for image patches
        result = model.predict_generator(imagen, steps = setup.predict_images_count() // self.batch_size, verbose=verbose)

        if verbose: print("\nDe-processing images.")

         # Deprocess patches
        if K.image_dim_ordering() == "th":
            result = result.transpose((0, 2, 3, 1)).astype(np.float32) * 255.
        else:
            result = result.astype(np.float32) * 255.

        if verbose: print("Completed De-processing image.")

        (num, width, height, channels) = result.shape
        output_directory = self.predict_path + self.name + "/" # Will need to be changed
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if verbose: print(("Saving %d images into " % num) + output_directory)
        # Getting old file names
        file_names = [f for f in sorted(os.listdir(self.predict_path + setup.alphaset))]
        for i in range(num):
            output = np.clip(result[i, :, :, :], 0, 255).astype('uint8') # TODO Change this for new image type
            filename = output_directory + (self.name + "_" + file_names[i])
            imsave(filename, output)
        if verbose: print(("Save %d images into " % num) + output_directory)

class BasicSR(BaseSRCNNModel):

    def __init__(self):
        super(BasicSR, self).__init__("BasicSR", border = 2, channels = 3)

    def create_model(self, channels=3, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """
        model = Sequential()

        model.add(Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=self.input_shape))
        model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
        model.add(Conv2D(channels, (5, 5), padding='same'))

        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[self.evaluation_function])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

class ExpansionSR(BaseSRCNNModel):

    def __init__(self, border = 2, channels = 3):
        super(ExpansionSR, self).__init__("ExpansionSR", border = 2, channels = 3)

    def create_model(self, channels=3, load_weights=False):
        """
            Creates a model to be used to scale images of specific height and width.
        """

        init = Input(shape=self.input_shape)
        x = Conv2D(64, (9, 9), activation='relu', padding='same', name='level1')(init)

        x1 = Conv2D(32, (1, 1), activation='relu', padding='same', name='lavel1_1')(x)
        x2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='lavel1_2')(x)
        x3 = Conv2D(32, (5, 5), activation='relu', padding='same', name='lavel1_3')(x)

        x = Average()([x1, x2, x3])

        out = Conv2D(channels, (5, 5), activation='relu', padding='same', name='output')(x)

        model = Model(init, out)
        adam = optimizers.Adam(lr=1e-3)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

class DeepDenoiseSR(BaseSRCNNModel):

    def __init__(self, border = 2, channels = 3):
        super(DeepDenoiseSR, self).__init__("DeepDenoiseSR", border = 2, channels = 3)

    def create_model(self, channels=3, load_weights=False):

        init = Input(shape=self.input_shape)
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

        decoded = Conv2D(channels, (5, 5), activation='linear', border_mode='same')(m2)

        model = Model(init, decoded)
        adam = optimizers.Adam(lr=1e-3)
        PSNRLoss = PSNRLossBorder(self.border)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model

# Very Deep Super Resolution
class VDSR(BaseSRCNNModel):

    def __init__(self, border = 2, channels = 3):
        super(VDSR, self).__init__("VDSR", border = 2, channels = 3)

    def create_model(self, channels=3, load_weights=False):

        init = Input(shape=self.input_shape)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(init)

        for i in range(0,19):
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

        decode = Conv2D(channels, (3, 3), activation='linear', border_mode='same')(x)

        model = Model(init, decode)
        adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.01, decay=0.0)
        PSNRLoss = PSNRLossBorder(self.border)
        model.compile(optimizer=adam, loss='mse', metrics=[PSNRLoss])
        if load_weights: model.load_weights(self.weight_path)

        self.model = model
        return model
