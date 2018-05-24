"""
    @author: aru5

    Implementation of a basic dense convolutional layer block
"""

from keras.layers import Conv2D, concatenate
from keras.models import Model

def dense_block(input, layers=5, k=4, window=(3,3), activation='relu'):
    """
        Creates a densely connected block of convolutional layers

        args:
            input : a pre-initialized input layer
            layers : number of convolutional layers for the dense block
            k : growth rate, number of filters in each layer
            window : convolutional window size, default (3,3)
            activation : activation function on convolutional layers

        outputs a set of keras layers representing the dense block
    """
    block = input
    for layer in range(layers):
        conv_layer = Conv2D(k, window, padding='same', activation='relu')(block)
        block = concatenate([block, conv_layer], axis=-1)
    return block
