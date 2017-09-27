"""
Toolkit for evolving Keras models
"""

import random
import sys
import datetime
import itertools
from copy import deepcopy

from keras.models import Sequential, Model, load_model
from keras.layers import Concatenate, Add, Average, Multiply, Maximum, Input, Dense, Flatten, BatchNormalization, Activation, LeakyReLU
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.utils.np_utils import to_categorical
import keras.callbacks as callbacks
import keras.optimizers as optimizers

from Modules.models import BaseSRCNNModel
from Modules.misc import printlog

_DEBUG = True

# Codons are the individual model layers that the evolver supports. A genetic
# code consists of a sequence of codons, each of which gets input from the
# previous one and feeds it to the subsequent one. They are expressed as a
# sequence of codon names, separated by - characters.
#
# Composite codons consist of a set of codons separated by : characters. In
# a composite codon containing N codons, the first N-1 codons get the same
# input, and then feed their outputs to a final merge codon.

# Convolutions are the most common unit. Autogenerate codons with a variety of
# filter sizes, kernel ranges, and activation functions. The parameters to the
# convolution are embedded in the codon name so that they can be tweaked easily.
# Codon names always start with a type code and end with an activation code.

filters = [32, 64, 128, 256]
kernels = [1, 3, 5, 7, 9]
acts = ['linear', 'elu', 'tanh', 'softsign']
depths = [2, 3, 4]

convolutions = {'conv_f{}_k{}_{}'.format(f, k, a): Conv2D(f, (k, k), activation=a, padding='same')
                for a in acts for k in kernels for f in filters}

# Merge codons combine multiple layers. They are only expressed inside composite
# codons

mergers = {'add': Add(), 'avg': Average(), 'mult': Multiply(), 'max': Maximum()}

# Autogenerate composite codons. We have some restrictions; for example, the filter
# size of all the convolutions must be the same.

composites = {}
for m in mergers:
    for f in filters:
        for a in acts:
            for d in depths:
                for k in itertools.combinations(kernels, d):
                    cname = '{}_f{}_k{}_d{}_{}'.format(m, f, ''.join([str(x) for x in k]), d, a)
                    flist = [convolutions['conv_f{}_k{}_{}'.format(f, n, a)] for n in k] + [mergers[m]]
                    composites[cname] = flist


# The output layer is always a convolution layer that generates 3 channels

outputs = {'out_k{}_{}'.format(k, a): Conv2D(3, (k,k), padding='same') for k in kernels for a in acts}

# For convenience, make a dictionary of all the possible codons. Python 3.5 black magic!

all_codons = {**convolutions, **composites, **outputs, **mergers}

# Mutation selection list (of dicts. Used to select a random codon for mutation.

mutable_codons = [convolutions, composites]

# Build and compile a keras model from an expressed sequence of codons.
# Returns the model or None if the compile failed in some way
#
# genome    list of codon names (if string, will be converted)
# shape     shape of model input
# lr        initial learning rate
# metrics   callbacks


def build_model(genome, shape=(64, 64, 3), lr=0.001, metrics=[]):

    if type(genome) is not list:
        genome = genome.split('-')

    # Get the wiring hookups for the sequence, and preface them with an
    # input layer. This means genome[i-1] is the code for codon[i].

    codons = [Input(shape=shape)] + [all_codons[gene] for gene in genome]

    # Wire the Codons. As we generate each layer, we update the codons[] list
    # with the constructed layer, so that subsequent codons can be connected
    # to them. We don't have to create the first (input) codon. All Keras layer
    # functions have to be deep-copied so that they are unique objects.

    if _DEBUG:
        all_layers = [codons[0]]

    try:

        for i, layer in enumerate(codons):
            if i > 0:
                layer = deepcopy(layer)
                if type(layer) is list:
                    # Composite multi-layer codon with a merge as the last element.
                    # Wire all the input layers to previous codon, then
                    # wire their outputs to the merge layer
                    for j in range(len(layer)):
                        layer[j] = deepcopy(layer[j])
                        layer[j].name = genome[i-1] + '_{}_{}'.format(i,j)
                        if j < len(layer) - 1:
                            layer[j] = layer[j](codons[i-1])
                        else:
                            layer[j] = layer[j](layer[:-1])
                    # Update the layer to point to the output layer
                    codons[i] = layer[-1]
                    if _DEBUG:
                        all_layers.extend(layer)
                else:
                    # Simple 1-layer codon
                    layer.name = genome[i-1] + '_{}'.format(i)
                    codons[i] = layer(codons[i-1])
                    if _DEBUG:
                        all_layers.append(codons[i])

        if _DEBUG:
            for i,layer in enumerate(all_layers):
                print('Layer',i, layer.name, layer, layer._consumers)
            print('')

        # Create and compile the model

        model = Model(codons[0], codons[-1])

        adam = optimizers.Adam(lr=lr, clipvalue=(1.0 / .001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse', metrics=metrics)

        if _DEBUG:
            print('Compiled model: shape={}, lr={}, metrics={}'.format(shape, lr, metrics))
        return model

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot compile: {}'.format(sys.exc_info()[1]))
        raise
        return None

# Generate a random sorted kernel sequence string

def kernel_sequence(number):

    return ''.join(sorted([str(n) for n in random.sample(kernels, number)]))

# Mutate a model. There are 5 possible mutations that can occur:
#
# point     point mutation of a parameter of a codon
# insert    insert a new codon
# delete    delete a codon
# transpose move a codon somewhere else in the genome
# conjugate replace codons with codons in another genome
#
# Parameters:
#
# mother        parental genome (list of codons; if string will be converted)
# father        parental genome (only used for transposition)
# min_len       minimum length of the resulting genome
# max_len       maximum length of the resulting genome
# odds          list of relative odds of point, insert, delete, transpose, conjugate
# best_fitness  the fitness of the best genome found so far.
# statistics    dictionary of codon statistics for guiding evolotion
# viable        viability function; takes a codon list, returns true if it is acceptable

def mutate(mother, father, min_len=3, max_len=30, odds=(3, 3, 1, 1, 1), best_fitness=0.0, statistics={}, viable=None):

    # a codon is "fit enough" if it is *not* in the statistics dictionary or
    # if it passes a dice roll based on how close its mean fitness is to the
    # best fitness of all the genomes. This will give a slight preference to
    # the better codons, but still allow for variety

    def selection_pressure(codon):

        if best_fitness >= 0 or codon not in statistics:
            return True

        codon_fitness = statistics[codon][1]

        return codon_fitness >= 0 or random.random() >= codon_fitness / best_fitness

    # Choose a random codon from mutable_codons


    def random_codon():

        items = mutable_codons

        while type(items) in (list, tuple):
            items = random.choice(items)

        if type(items) is dict:
            return random.choice(list(items.keys()))
        else:
            return items

    # Make sure inputs are lists

    if type(mother) is not list:
        mother = mother.split('-')

    if type(father) is not list:
        father = father.split('-')

    mother_len = len(mother)
    child = None

    while child == None or child == mother or viable != None and not viable(child):
        child = mother[:]
        choice = random.randint(1, sum(odds))

        # make a point mutation in a codon. Codon will always be in the format
        # type_parameter_parameter_... _activation and parameter will always
        # be in format letter-code[digits]. The type is never changed, and we
        # do special handling for the activation function

        if choice <= odds[0]:
            locus = random.randrange(mother_len)
            codons = mother[locus].split('_')
            basepair = random.randrange(len(codons))
            while True:
                if basepair == len(codons) - 1:
                    # choose new activation function
                    new_codon = random.choice(acts)
                elif basepair == 0:
                    # choose new codon type (only if a merge-type codon)
                    if codons[0] in mergers:
                        new_codon = random.choice(mergers.keys) + codons[0][1:]
                    else:
                        new_codon = codons[0]
                else:
                    # tweak a codon parameter
                    base = codons[basepair][0]
                    param = codons[basepair][1:]
                    if base == 'k':
                        # If the codon has a depth parameter we need a sequence of kernel sizes, but we
                        # can deduce this by the length of the current k parameter, since currently
                        # they are all single-digit.
                        param = kernel_sequence(len(param))
                    elif base == 'd':
                        # If we change the depth we have to also change the k parameter
                        param = random.choice(depths)
                        codons = [c if c[0] != 'k' else 'k' + kernel_sequence(param) for c in codons]
                    elif base == 'f':
                        param = random.choice(filters)
                    else:
                        printlog('Unknown parameter base {} in {}'.format(base, mother[locus]))
                        param = 'XXX'
                    new_codon = base + str(param)
                if selection_pressure(new_codon):
                    break
            codons[basepair] = new_codon
            child[locus] = '_'.join(codons)
            if _DEBUG:
                print('Point mutation: {} -> {}'.format(mother[locus], child[locus]))
            continue

        choice -= odds[0]
        splice = random.randrange(mother_len)

        # Insert codon; fail if it would make genome too long

        if choice <= odds[1]:
            if mother_len >= max_len:
                child = None
            else:
                while True:
                    codon = random_codon()
                    if selection_pressure(codon):
                        break
                child.insert(splice, codon)
                if _DEBUG:
                    print('Insertion: {}'.format(codon))
            continue

        choice -= odds[1]

        # Delete codon (except we never delete last codon, which is always an output codon)

        if choice <= odds[2]:
            if mother_len <= min_len or splice == (mother_len - 1):
                child = None
            else:
                if _DEBUG:
                    print('Deletion: {}'.format(child[splice]))
                del child[splice]
            continue


        # Transpose a codon -- but never the last one, and never move after
        # the last one.

        choice -= odds[2]

        if choice <= odds[3]:
            if splice == (mother_len - 1):
                child = None
            else:
                codon = child[splice]
                del child[splice]
                splice = random.randrange(len(child)-1)
                child.insert(splice, codon)
                if _DEBUG:
                    print('Transposition: {}'.format(codon))
            continue


        # Conjugate father and mother.

        splice = random.randrange(1, mother_len)
        child = mother[:-splice] + father[-splice:]
        if child == mother or child == father:
            child = None
        else:
            if _DEBUG:
                print('Conjugation')

        # Loop around until we have a useful child.

    return child

# Slice and dice a genome and generate statistics for analysis. Statistics
# is a dictionary; for each genetic fragment, it contains a (best, mean,
# worst, count) tuple


def ligate(statistics, genome, fitness):

    if type(genome) is not list:
        genome = genome.split('-')

    # compute the list of codons and subparts taken 1 to n at a time, where
    # n is the size of the codon.

    fragments = []
    for codon in genome:
        bases = codon.split('_')
        for i in range(len(bases)):
            fragments.extend(itertools.combinations(bases, i + 1))

    # update the statistics tuple for each fragment. The tuples are
    # (best, mean, worst, count), and in this case best is the most
    # negative.

    for f in fragments:
        f = '_'.join(f)
        if f not in statistics:
            statistics[f] = (fitness, fitness, fitness, 1)
        else:
            best, mean, worst, count = statistics[f]

            best = min(best, fitness)
            worst = max(worst, fitness)
            mean = (mean * count + fitness) / (count + 1)
            count += 1

            statistics[f] = (best, mean, worst, count)

    return statistics



# Determine the fitness of an organism by creating its model and running it.
#
# organism      string or list with genetic code to be tested
# io            ModelIO parameter record
# apoptosis     Function that returns True if we should quit early


def fitness(genome, io, apoptosis=None):

    if type(genome) is not list:
        genome = genome.split('-')

    organism = '-'.join(genome)

    printlog('Testing fitness of {}'.format(organism))

    io.model_type = organism

    m = BaseSRCNNModel(organism, io, verbose=False, bargraph=False)

    model = build_model(genome, shape=io.image_shape,
                        lr=io.lr, metrics=[m.evaluation_function])

    if model == None:
        return 0.0

    m.model = model

    # Now we have a compiled model, execute it - or at least try to, there are still some
    # models that may bomb out.

    try:

        stime = datetime.datetime.now()
        results = m.fit(max_epochs=1)
        etime = datetime.datetime.now()

        halfway = io.epochs // 2

        eta = etime + (etime - stime) * (halfway - 1)
        printlog(
            'After 1 epoch: fitness={}, will be halfway @ {:%I:%M:%S %p}'.format(results, eta))

        if apoptosis != None and apoptosis(results, 1, io.epochs):
            printlog('Apoptosis triggered!')
            return results

        prev_results = results
        stime = datetime.datetime.now()
        results = m.fit(max_epochs=halfway)
        etime = datetime.datetime.now()

        eta = etime + (etime - stime) * halfway / (halfway - 1)
        printlog('After {} epochs: fitness={}, will complete @ {:%I:%M:%S %p}'.format(
            halfway, results, eta))

        if apoptosis != None and apoptosis(results, halfway, io.epochs, last_fitness=prev_results):
            printlog('Apoptosis triggered!')
            return results

        results = m.fit(max_epochs=io.epochs)
        printlog('After {} epochs: fitness={}'.format(io.epochs, results))

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot fit: {}'.format(sys.exc_info()[1]))
        raise
        results = 0.0

    printlog('Fitness: {}'.format(results))
    return results
