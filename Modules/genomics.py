"""
Toolkit for evolving Keras models - WORK IN PROGRESS
"""

import random
import sys
import datetime
from copy import deepcopy

from keras.models import Sequential, Model, load_model
from keras.layers import Concatenate, Add, Average, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from Modules.models import BaseSRCNNModel
from Modules.misc import printlog

DEBUG = True
DEBUG_FLAG = False

# Codons are the individual model layers that the evolver supports. Each codon is
# a tuple consisting of a layer generator and the layer offset(s) that feed it.
#

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

}

# Combinations combine multiple layers. If a layer is the source for a combiner,
# the next layer actually gets its input from the layer that feeds the source
# layer. This permits the gene to have multiple paths.

combinations = {

    "add12"  : [Add(), -1, -2],
    "add13"  : [Add(), -1, -3],
    "add14"  : [Add(), -1, -4],
    "add15"  : [Add(), -1, -5],
    "add16"  : [Add(), -1, -6],
    "add17"  : [Add(), -1, -7],
    "add18"  : [Add(), -1, -8],
    "add19"  : [Add(), -1, -9],
    "add110"  : [Add(), -1, -10],
    "add111"  : [Add(), -1, -11],
    "add112"  : [Add(), -1, -12],
    "avg12"  : [Average() , -1, -2],
    "avg123" : [Average() , -1, -2, -3],
    "addbad" : [Add(), -2, -4],

}

# Introns are intermediate layers.

introns = {

    "pool" : [MaxPooling2D((2, 2)), -1],
    "usam" : [UpSampling2D(), -1],

}

# The output layer is always a convolution layer that generates 3 channels

outputs = {

    "out1" : [Conv2D(3, (1, 1), padding='same'), -1],
    "out3" : [Conv2D(3, (3, 3), padding='same'), -1],
    "out5" : [Conv2D(3, (5, 5), padding='same'), -1],
    "out7" : [Conv2D(3, (7, 7), padding='same'), -1],
    "out9" : [Conv2D(3, (9, 9), padding='same'), -1],

}

# For convenience, make a dictionary of all the possible codons that can be expressed.
# Python 3.5 black magic!

all_codons = {**convolutions, **combinations, **introns, **outputs}

# Mutation selection list (of dicts. Used to select a random codon for mutation.

mutable_codons = [convolutions, combinations, introns]

# A list of codons names is a genome. An expressed genome is one where the
# codon name is replaced by the values in the codon (the function and connections),
# and the relative connections have been replaced by absolute connection indexes.
# Also, a dummy input layer is spliced on. Expects a list but will convert a string
# if provided.

def expressed(sequence):

    if type(sequence) is not list:
        sequence = sequence.split('-')

    # Get the wiring hookups for the sequence

    expression = [list(all_codons[gene]) for gene in sequence]

    # Add a dummy input layer

    expression.insert(0, [None, 0])

    # Convert wiring from offsets to absolute positions

    expression = [ [v if i==0 else v+n for i,v in enumerate(gene)] for n,gene in enumerate(expression)]

    return expression

# Build and compile a keras model from an expressed sequence of codons.
# Returns the model or None if the compile failed in some way
#
# genome    list of codon names (if string, will be converted)
# layers    expressed(genome). Will be computed if not provided, but caller usually has it
# shape     shape of model input
# lr        initial learning rate
# metrics   callbacks

def build_model(genome, layers=None, shape=(64,64,3), lr=0.001, metrics=[]):

    if type(genome) is not list:
        genome = genome.split('-')

    if layers == None:
        layers = expressed(genome)

    # Wire the layers. As we generate each layer, we update the layers[] list
    # with the constructed layer, so that subsequent layers can be created that
    # link back to them.

    try:

        for i,layer in enumerate(layers):
            if i == 0:
                # Set up the input layer (which expressed() set up as a dummy layer).
                layers[0] = Input(shape=shape)
            else:
                # If we do not deep copy the layer object, then if the model reuses
                # the same type layer type, keras bombs.

                layer_function = deepcopy(layer[0])

                # Also, we can't have two layers with the same name, so make them
                # unique. Use the genome code to make it more clear (keep in mind
                # it does not have a dummy input layer, so -1 offset)

                layer_function.name = genome[i-1] + '_' + str(i)

                # Our inputs are either a single layer or a list of layers

                layer_inputs = [layers[n] for n in layer[1:]]

                if len(layer_inputs) == 1:
                    layer_inputs = layer_inputs[0]

                layers[i] = layer_function(layer_inputs)

        # Create and compile the model

        model = Model(layers[0],layers[-1])

        adam = optimizers.Adam(lr=lr, clipvalue=(1.0/.001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse', metrics=metrics)

        return model

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot compile: {}'.format(sys.exc_info()[1]))
        return None

# Determine if a genome is viable (in other words, that the model makes some
# sort of sense, does not contain useless layers, etc.). You can write your
# own viability function.

def basic_viability(genome):

    if type(genome) is not list:
        genome = genome.split('-')

    printlog('Checking viability of {}'.format('-'.join(genome)))

    expression = expressed(genome)

    # What are the source layers for each layer? It's just handy not to have
    # the layer functions.

    sources = [gene[1:] for gene in expression]

    # A sequence is not viable if it has a layer that wants to connect to
    # a layer prior to the base layer.

    if min([min(s) for s in sources]) < 0:
        return False

    # A sequence is not viable unless all layers are the source for at least
    # one other layer (except for the final output layer, of course).

    used = {link for layer in sources for link in layer}

    if len(used) != len(genome):
        return False

    # Finally, just try and build the model and if it fails, we know we
    # had a problem.

    return build_model(genome, expression) != None

# Choose a random codon. If items is None, select a random codon. Otherwise
# Items must be a dict or a list of dicts.

def random_codon(items=None):

    if items == None:
        items = mutable_codons

    while type(items) in (list, tuple):
        items = random.choice(items)

    if type(items) is dict:
        return random.choice(list(items.keys()))
    else:
        return items

# Mutate a model. There are 4 possible mutations that can occur; transposition
# between two models, addition of a codon, deletion of a codon, or mutation of
# a codon. Parameters are:
#
# mother    parental genome (list of codons; if string will be converted)
# father    parental genome (only used for transposition)
# min_len   minimum length of the resulting genome
# max_len   maximum length of the resulting genome
# odds      list of relative odds of transpositon, addition, deletion, mutate
# viable    viability function; takes a codon list, returns true if it is acceptable

def mutate(mother, father, min_len=3, max_len=30, odds=[3,5,5,8], viable=basic_viability):

    if type(mother) is not list:
        mother = mother.split('-')

    if type(father) is not list:
        father = father.split('-')

    mother_len = len(mother)
    child = None

    while child == None or not viable(child):

        choice = random.randint(1,sum(odds))

        # transpose codons from father to mother. Always at least 1 codon
        # from mother.

        if choice <= odds[0]:
            splice = random.randint(1,mother_len-1)
            child = mother[:-splice] + father[-splice:]
            if child == mother or child == father:
                child = None
            continue

        choice -= odds[0]

        child = mother[:]
        splice = random.randint(0,mother_len-1)


        # add codon; fail if it would make genome too long

        if choice <= odds[1]:
            if mother_len == max_len :
                child = None
            else:
                child.insert(splice,random_codon())
            continue

        choice -= odds[1]

        last_codon = splice == (mother_len - 1)

        # delete codon (except we never delete last codon, which is always an output codon)

        if choice <= odds[2]:
            if mother_len == min_len or last_codon:
                child = None
            else:
                del child[splice]
            continue

        # mutate codon

        child[splice] = random_codon(outputs if last_codon else None)
        if child == mother or child == father:
            child == None

    return child

# Testing...

import types

# Determine the fitness of an organism by creating its model and running it.
#
# organism      string or list with genetic code to be tested
# io            ModelIO parameter record
# fail_first    fail after first epoch if fitness greater than this value.
# fail_halfway  fail after midpoint in training if fitness greater than this value.
#               (None == don't check)

def fitness(genome, io, fail_first=None, fail_halfway=None):

    if type(genome) is not list:
        genome = genome.split('-')

    organism = '-'.join(genome)

    printlog('Testing fitness of {}'.format(organism))

    io.model_type = organism

    m = BaseSRCNNModel(organism, io, verbose=False, bargraph=False)

    model = build_model(genome, shape=io.image_shape, lr=io.lr, metrics=[m.evaluation_function])

    if model == None:
        return 999999.0

    m.model = model

    # Now we have a compiled model, execute it - or at least try to, there are still some
    # models that may bomb out

    try:

        stime = datetime.datetime.now()
        results = m.fit(max_epochs=1)
        etime = datetime.datetime.now()

        halfway = io.epochs // 2

        if fail_first != None and results > fail_first:
            printlog('Fail_first triggered - fitness={}'.format(results))
            return results
        else:
            eta = etime + (etime - stime) * (halfway - 1)
            printlog('After 1 epoch: fitness={}, halfway ETA={:%I:%M:%S %p}'.format(results, eta))

        stime = datetime.datetime.now()
        results = m.fit(max_epochs=halfway)
        etime = datetime.datetime.now()

        if fail_halfway != None and results > fail_halfway:
            printlog('Fail_halfway triggered - fitness={}'.format(results))
            return results
        else:
            eta = etime + (etime - stime) * halfway / (halfway - 1)
            printlog('After {} epochs: fitness={}, completion ETA={:%I:%M:%S %p}'.format(halfway, results, eta))

        results = m.fit(max_epochs=io.epochs)
        printlog('After {} epochs: fitness={}'.format(io.epochs, results))

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot fit: {}'.format(sys.exc_info()[1]))
        results = 999999.0

    printlog('Fitness: {}'.format(results))
    return results
