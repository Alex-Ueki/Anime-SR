"""
Toolkit for evolving Keras models - WORK IN PROGRESS
"""

import random

from keras.models import Sequential, Model, load_model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from Modules.models import BaseSRCNNModel

DEBUG = True
DEBUG_FLAG = False

# Codons are the individual model layers that the evolver supports. Each codon is
# a tuple consisting of a layer generator and the layer offset(s) that feed it.

# Convolutions are the most common unit

convolutions = {

    "c321" : [Conv2D(32, (1, 1), activation='elu', padding='same'), -1],
    "c323" : [Conv2D(32, (3, 3), activation='elu', padding='same'), -1],
    "c325" : [Conv2D(32, (5, 5), activation='elu', padding='same'), -1],
    "c327" : [Conv2D(32, (7, 7), activation='elu', padding='same'), -1],
    "c329" : [Conv2D(32, (9, 9), activation='elu', padding='same'), -1],

    "c641" : [Conv2D(64, (1, 1), activation='elu', padding='same'), -1],
    "c643" : [Conv2D(64, (3, 3), activation='elu', padding='same'), -1],
    "c645" : [Conv2D(64, (5, 5), activation='elu', padding='same'), -1],
    "c647" : [Conv2D(64, (7, 7), activation='elu', padding='same'), -1],
    "c649" : [Conv2D(64, (9, 9), activation='elu', padding='same'), -1],

    "c1281" : [Conv2D(128, (1, 1), activation='elu', padding='same'), -1],
    "c1283" : [Conv2D(128, (3, 3), activation='elu', padding='same'), -1],
    "c1285" : [Conv2D(128, (5, 5), activation='elu', padding='same'), -1],
    "c1287" : [Conv2D(128, (7, 7), activation='elu', padding='same'), -1],
    "c1289" : [Conv2D(128, (9, 9), activation='elu', padding='same'), -1],

    "c2561" : [Conv2D(256, (1, 1), activation='elu', padding='same'), -1],
    "c2563" : [Conv2D(256, (3, 3), activation='elu', padding='same'), -1],
    "c2565" : [Conv2D(256, (5, 5), activation='elu', padding='same'), -1],
    "c2567" : [Conv2D(256, (7, 7), activation='elu', padding='same'), -1],
    "c2569" : [Conv2D(256, (9, 9), activation='elu', padding='same'), -1],

    "c643b" : [Conv2D(64, (3, 3), activation='elu', padding='same'), -1],

}

if __name__ == "__main__":

    derf = Conv2D(32, (1, 1), activation='elu', padding='same')
    dork = Conv2D(32, (1, 1), activation='elu', padding='same')
    print(derf)
    print(dork)
    dlist = [derf, deepcopy(derf), deepcopy(derf)]
    print(dlist)
