# pylint: disable=C0301
# (line_too_long disabled)

"""
Toolkit for evolving Keras models
"""

import random
import sys
import itertools

from keras.models import Model
from keras.layers import Add, Average, Multiply, Maximum, Input
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
import keras.optimizers as optimizers
import keras.backend as K

from Modules.models import BaseSRCNNModel
from Modules.misc import printlog

# Batch normalization axis is -1 for tensorflow and 1 for theano

_BN_AXIS = 1 if K.image_dim_ordering() == 'th' else -1

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
#
# Codons can have a *{number} suffix that means "repeat the layer {number) times"

# Define the complete range of our variable parameters

ALL_FILTERS = [32, 64, 128, 256]
ALL_KERNELS = [3, 5, 7, 9]
ALL_ACTS = ['linear', 'elu', 'tanh', 'softsign', 'relu']
ALL_DEPTHS = [2, 3, 4]
ALL_MULTIPLIERS = [1, 2, 4, 8, 16]


# Merge codons combine multiple layers. They are internal codons that are the
# final layer of composite codons

ALL_MERGERS = {
    'add': Add(),
    'avg': Average(),
    'mult': Multiply(),
    'max': Maximum()
}

# The subset of the complete range that we will use in mutation. The reason for
# the difference is that it lets us evolve using older models with a different
# subset, and change the parameter space we are exploring as needed.
#
# Currently we are only using elu activation layers, the add and max merge types,
# and a restricted number of filter and kernel sizes. Once we find a good model
# topology, we can explore variants of that topology.

FILTERS = [32, 64]
KERNELS = [3, 5, 7, 9]
ACTS = ['elu']
DEPTHS = [2, 3]
MULTIPLIERS = [1, 2, 4, 8]

MERGERS = {
    'add': Add(),
    'max': Maximum()
}

# List of layers created by a build_model operation (useful for debugging/testing)

_LAYERS = []

# Batch-normalized convolution closure

def _bn_conv2d(filters, kernels, activation, padding):
    """ Return a closure that returns a stack of batch-normalized convolution layers """

    def _closure(inputs, depth=1):
        """ Closure """

        for _ in range(depth):
            inputs = Conv2D(filters, (kernels, kernels), padding=padding, activation=activation)(inputs)
            _LAYERS.append(inputs)
            inputs = BatchNormalization(axis=_BN_AXIS)(inputs)
            _LAYERS.append(inputs)

        return inputs

    return _closure


# Generate all the possible codons; replaced Conv2D with bn_conv2d

CONVOLUTIONS = {'conv_f{}_k{}_{}'.format(f, k, a): _bn_conv2d(f, k, activation=a, padding='same')
                for a in ALL_ACTS for k in ALL_KERNELS for f in ALL_FILTERS}

# Also a dictionary of all the convolutions we may generate via mutation

MUTABLE_CONVOLUTIONS = {'conv_f{}_k{}_{}'.format(f, k, a): _bn_conv2d(f, k, activation=a, padding='same')
                        for a in ACTS for k in KERNELS for f in FILTERS}

def _make_composites(mergers, filters, acts, depths, kernels):
    """ Make composite codons. """

    def _bn_merge(mrg, flt, knl, act):
        """ Return a closure that returns a merger of stacks of batch-normalized convolution layers """

        def _closure(inputs, depth=1):
            """" Closure """
            inputs = [_bn_conv2d(flt, k, act, 'same')(inputs, depth) for k in knl]
            merge = ALL_MERGERS[mrg](inputs)

            _LAYERS.append(merge)

            return merge

        return _closure

    codons = {}

    for mrg in mergers:
        for flt in filters:
            for act in acts:
                for dep in depths:
                    for knl in itertools.combinations(kernels, dep):
                        cname = '{}_f{}_k{}_d{}_{}'.format(
                            mrg, flt, '.'.join([str(k) for k in knl]), dep, act)
                        codons[cname] = _bn_merge(mrg, flt, knl, act)

    return codons

COMPOSITES = _make_composites(ALL_MERGERS, ALL_FILTERS, ALL_ACTS, ALL_DEPTHS, ALL_KERNELS)
MUTABLE_COMPOSITES = _make_composites(MERGERS, FILTERS, ACTS, DEPTHS, KERNELS)

MODIFIERS = {'mod_r' + str(n): n for n in ALL_MULTIPLIERS}
MUTABLE_MODIFIERS = {'mod_r' + str(n): n for n in MULTIPLIERS}

# The output layer is always an un-normalized convolution layer that generates 3 channels.
# Input layers are the same, but normalized

SIMPLE_KERNELS = [1, 3, 5, 7, 9, 11, 13, 15]

# Simple output layer closure

def _out(kernels, padding):
    """ Return a closure that returns an output layer. Has the same parameters as all
        the other closures but ignores depth.
    """

    def _closure(inputs, _):
        """ Closure """

        output = Conv2D(3, (kernels, kernels), padding=padding)(inputs)

        _LAYERS.append(output)

        return output

    return _closure

# Simple input layer closure

def _in(kernels, padding):
    """ Return a closure that returns an input layer. Has the same parameters as all
        the other closures but ignores depth.
    """

    def _closure(inputs, _):
        """ Closure """

        inputs = Conv2D(3, (kernels, kernels), padding=padding)(inputs)
        _LAYERS.append(inputs)
        inputs = BatchNormalization(axis=_BN_AXIS)(inputs)
        _LAYERS.append(inputs)

        return inputs

    return _closure

# Output and input codons all generate (x,y,3) shape results. Outputs can
# only be at end of genome, inputs only at the start. Input layers are
# effectively naive sharpeners.

OUTPUTS = {'out_k{}'.format(k): _out(k, 'same') for k in SIMPLE_KERNELS}
INPUTS = {'in_k{}'.format(k): _in(k, 'same') for k in SIMPLE_KERNELS}

# For convenience, make a dictionary of all the possible codons. Python 3.5 black magic!

ALL_CODONS = {**CONVOLUTIONS, **COMPOSITES, **OUTPUTS, **INPUTS, **MODIFIERS}

# Also a dictionary of call codons with activation functions

HAS_ACTIVATION = {**CONVOLUTIONS, **COMPOSITES}

# What codons are available in the primordial soup for random inclusion?

PRIMORDIAL_CODONS = [k for k in {**MUTABLE_CONVOLUTIONS, **MUTABLE_COMPOSITES, **MUTABLE_MODIFIERS}]

# What codons are conserved -- and thus will not be point-mutated once they appear
# in a genome.

CONSERVED_CODONS = [] # [k for k in {**INPUTS, **OUTPUTS}]

def build_model(genome, shape=(64, 64, 3), learning_rate=0.001, metrics=None):
    """ Build and compile a keras model from an expressed sequence of codons.
        Returns (model, layer_count) tuple or None if the compile failed in some way

        genome          list of codon names (if string, will be converted)
        shape           shape of model input
        learning_rate   initial learning rate
        metrics         callbacks
    """

    # Remove any old layers from global

    while _LAYERS:
        _LAYERS.pop()

    if not isinstance(genome, list):
        genome = genome.split('-')

    if _DEBUG:
        printlog('Compiling', genome)

    try:

        # Initial layer stacking depth, will be adjusted by modifier codons

        depth = 1

        # Initial model state, just an input layer

        first_layer = Input(shape=shape, dtype='float32')
        last_layer = first_layer

        _LAYERS.append(first_layer)

        # Build the layers of the model.

        for i, codon in enumerate(genome):

            # If a modifier codon, adjust the depth of the model. We may encounter
            # several modifier codons in a row, they are additive. If genome tries
            # to modify an output codon, abort with a failure.

            if codon in MODIFIERS:
                if genome[i + 1] not in OUTPUTS:
                    depth += ALL_CODONS[codon] - 1
                else:
                    printlog('Cannot compile: Modifier codon trying to modify output layer.')
                    return None, 0
            else:

                # Add the new layer or layer stack and reset the depth

                last_layer = ALL_CODONS[codon](last_layer, depth)
                depth = 1

        # This debug print code assumes Tensorflow is being used.

        if _DEBUG:
            layer_outputs = {}
            for layer in _LAYERS:
                lname = layer.name
                layer_outputs[lname] = ' + '.join([l.name for l in layer.consumers()])
            nwidth = max([len(l) for l in layer_outputs])
            fstr = 'Layer {:>3d}: {:>' + str(nwidth) + 's} -> {}'
            for i, layer in enumerate(_LAYERS):
                lname = layer.name
                printlog(fstr.format(i, lname, layer_outputs[lname]))

        # Create and compile the model

        model = Model(first_layer, last_layer)

        adam = optimizers.Adam(lr=learning_rate, clipvalue=(1.0 / .001), epsilon=0.001)

        model.compile(optimizer=adam, loss='mse', metrics={} if metrics is None else metrics)

        if _DEBUG:
            printlog('Compiled model: shape={}, learning rate={}, metrics={}'.format(
                shape, learning_rate, metrics))

        return (model, len(_LAYERS))

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot compile: {}'.format(sys.exc_info()[1]))
        raise
        # return None, 0

# Support functions for mutate()


def fit_enough(best_fitness, statistics):
    """ A codon is "fit enough" if it is *not* in the statistics dictionary or
        if it passes a dice roll based on how close its mean fitness is to the
        best fitness of all the genomes. This will give a slight preference to
        the better codons, but still allow for variety.

        Returns a curried function
    """

    def fitness(codon):
        """ Is the codon of acceptable fitness? """

        if best_fitness >= 0.0 or codon not in statistics:
            return True

        codon_fitness = statistics[codon][1]

        return codon_fitness >= 0 or random.random() >= codon_fitness / best_fitness

    return fitness


def random_codon(acceptable_fitness):
    """ Choose a random codon from PRIMORDIAL_CODONS, with suitable fitness.
        Will always return because of the random diceroll.
    """

    # Try to get a codon of acceptable fitness but give up if we have been
    # looking too long.

    for _ in range(0, 100):
        codon = random.choice(PRIMORDIAL_CODONS)
        if acceptable_fitness(codon):
            if _DEBUG:
                printlog('Random Codon =', codon)
            break

    return codon

def kernel_sequence(number):
    """ Generate a random sorted kernel sequence string """

    return '.'.join(sorted([str(n) for n in random.sample(KERNELS, number)]))


def point_mutation(child, _, acceptable_fitness):
    """ Make a point mutation in a codon. Codon will always be in the format
        type_parameter_parameter_... {_activation} and parameter will always
        be in format letter-code[digits]. The type is never changed, and we
        do special handling for the activation function if it is present.
        Currently modifier codons do not have activation functions.
    """

    if _DEBUG:
        printlog('Point Mutation <', child)

    locus = random.randrange(len(child))
    original_locus = child[locus]
    codons = original_locus.split('_')
    basepair = random.randrange(len(codons))
    has_depth = any([c[0] == 'd' for c in codons])

    # If the chosen codon is highly conserved, do not mess with it.

    if original_locus in CONSERVED_CODONS:
        if _DEBUG:
            printlog('Conserved Codon =', original_locus)
        return child

    while True:
        if basepair == len(codons) - 1 and original_locus in HAS_ACTIVATION:
            # choose new activation function
            new_codon = random.choice(ACTS)
        elif basepair == 0:
            # choose new codon type (only if a merge-type codon)
            new_codon = random.choice(list(MERGERS.keys())) if codons[0] in MERGERS else codons[0]
        else:
            # tweak a codon parameter
            base = codons[basepair][0]
            param = codons[basepair][1:]

            # possible base codes are:
            # k kernel size
            # d depth of merger codon
            # f number of filters
            # r replication number of modifier codon

            if base == 'k':
                # If the codon has a depth parameter we need a sequence of kernel sizes.
                # If we are tweaking a codon with no activation, it's an input/output codon
                # so we have a bigger range.
                ktype = KERNELS if original_locus in HAS_ACTIVATION else SIMPLE_KERNELS
                param = kernel_sequence(len(param)) if has_depth else random.choice(ktype)
            elif base == 'd':
                # If we change the depth we have to also change the k parameter
                param = random.choice(DEPTHS)
                codons = [c if c[0] != 'k' else 'k' + kernel_sequence(param) for c in codons]
            elif base == 'f':
                param = random.choice(FILTERS)
            elif base == 'r':
                param = random.choice(MULTIPLIERS)
            else:
                printlog('Unknown parameter base {} in {}'.format(base, original_locus))
                param = 'XXX'
            new_codon = base + str(param)
        if acceptable_fitness(new_codon):
            break

    codons[basepair] = new_codon
    child[locus] = '_'.join(codons)

    if _DEBUG:
        printlog('Point Mutation >', child)

    return child


def insertion(child, _, acceptable_fitness):
    """ Insertion mutation - never insert after output codon """

    if _DEBUG:
        printlog('Insertion <', child)

    child_len = len(child)

    if child_len == 0:
        return child

    # If first codon is an input codon, never insert before it (since this will
    # almost always create an invalid genome).

    start_offset = 1 if child[0] in INPUTS else 0
    insert_position = start_offset if child_len == 1 else random.randrange(start_offset, child_len)

    codon = random_codon(acceptable_fitness)
    child.insert(insert_position, codon)

    if _DEBUG:
        printlog('Insertion >', child)

    return child


def deletion(child, _, min_len):
    """ Deletion mutation - never delete output codon! """

    if _DEBUG:
        printlog('Deletion <', child)

    child_len = len(child)

    if child_len <= min_len:
        return child

    del child[random.randrange(child_len - 1)]

    if _DEBUG:
        printlog('Deletion <', child)

    return child


def conjugation(child, conjugate, _):
    """ Conjugate two genomes - always at least one codon from each contributor """

    if _DEBUG:
        printlog('Conjugation <', child)

    splice = random.randrange(1, len(child))
    child = child[:-splice] + conjugate[-splice:]

    if _DEBUG:
        printlog('Conjugation >', child)

    return child


def transposition(child, _, __):
    """ Transposition mutation - never transpose output codon """

    if _DEBUG:
        printlog('Transposition <', child)

    # Cannot transpose if 2 or fewer elements

    if len(child) <= 2:
        return child

    splice = random.randrange(len(child) - 1)
    codon = child[splice]
    del child[splice]

    splice = random.randrange(len(child) - 1)
    child.insert(splice, codon)

    if _DEBUG:
        printlog('   Transposed :', codon)
        printlog('Transposition >', child)

    return child


def inversion(child, _, __):
    """ Invert two adjacent codons """

    # Cannot invert if 3 or fewer elements

    if _DEBUG:
        printlog('Inversion <', child)

    if len(child) <= 2:
        return child

    locus = 0 if len(child) == 2 else random.randrange(len(child) - 2)

    child[locus], child[locus + 1] = child[locus + 1], child[locus]

    if _DEBUG:
        printlog('Inversion >', child)

    return child


def regularize(child):
    """ Rearrange modifier codons into consistent order """

    def crispr(codon):
        """ cut up a codon for sort purposes """

        codon = codon.split('_')[1:]
        codon = [(p[0], int(p[1:])) for p in reversed(codon)]

        return codon

    shuffled = True

    if _DEBUG:
        printlog('Regularize <', child)

    while shuffled:

        shuffled = False

        for i in range(len(child) - 1):
            if child[i] in MODIFIERS and child[i + 1] in MODIFIERS:
                # Currently depends on the modifier parameter codes being
                # in string sort order, but since we only have one of them
                # (r) this is not an issue.
                if crispr(child[i]) > crispr(child[i + 1]):
                    child[i], child[i + 1] = child[i + 1], child[i]
                    shuffled = True

    if _DEBUG:
        printlog('Regularize >', child)

    return child

def valid(child):
    """ Return True if child is valid """

    # Nonexistent or empty children are invalid

    if child is None or not child:
        return False

    # One or more leading input layers are OK

    while child and child[0] in INPUTS:
        child = child[1:]

    # One or more trailing output layers are OK

    while child and child[-1] in OUTPUTS:
        child = child[:-1]

    # Only input and output layers also OK

    if not child:
        return True

    # If the first layer is a modifier, then if it's a x1 multiplier
    # then the subsequent codon must also be a multiplier.

    if child[0] in MODIFIERS:
        if len(child) == 1 or (child[0] == 'mod-r1' and child[1] not in MODIFIERS):
            return False
        return valid(child[1:])

    # Interior input or output layers are invalid

    for codon in child:
        if codon in INPUTS or codon in OUTPUTS:
            return False

    return True

def mutate(parent, conjugate, tried, min_len=2, max_len=90, odds=(4, 8, 9, 10, 11, 12), best_fitness=0.0, statistics=None, viable=None):
    """ Mutate a model. There are 6 possible mutations that can occur:

        point     point mutation of a parameter of a codon
        insert    insert a new codon
        delete    delete a codon
        invert    two adjacent codons are flipped
        transpose move a codon somewhere else in the genome
        conjugate replace codons with codons in another genome

        Parameters:

        parent        parental genome (list of codons; if string will be converted)
        conjugate     parental genome (only used for conjugation)
        tried         genomes we have already tried (for quick rejection; = parents + graveyard)
        min_len       minimum length of the resulting genome
        max_len       maximum length of the resulting genome
        odds          list of *cumulative* odds of point, insert, delete, transpose, conjugate
        best_fitness  the fitness of the best genome found so far.
        statistics    dictionary of codon statistics for guiding evolotion
        viable        viability function; takes a codon list, returns true if it is acceptable
    """

    if not isinstance(parent, list):
        parent = parent.split('-')

    if not isinstance(conjugate, list):
        conjugate = conjugate.split('-')

    statistics = {} if statistics is None else statistics

    if _DEBUG:
        printlog('   Mutation parent', '-'.join(parent))
        printlog('Mutation conjugate', '-'.join(conjugate))

    # The mutations and their optional parameter, if any

    acceptable = fit_enough(best_fitness, statistics)

    operations = [point_mutation, insertion, deletion, inversion, transposition, conjugation]
    parameters = [acceptable, acceptable, min_len, None, None, None]

    child = None

    # Repeat until we get a useful mutation

    while child is None or child == parent or child == conjugate or '-'.join(child) in tried:

        # Deep copy the parent into child, choose a mutation type, and
        # call the appropriate mutation function

        child = parent[:]

        # I admit, this is a bit tricky! Will generate 5 for the first
        # possible choice, 4 for the next, etc. Then use negative indexing
        # to choose the right function!

        todo = len([i for i in odds if random.randrange(sum(odds)) < i])
        child = operations[-todo](child, conjugate, parameters[-todo])

        # Rearrange any adjacent modifier codons into a consistent order so we don't end up training
        # two effectively identical genomes.

        child = regularize(child)

        # Quick retry if known to be dead

        if child == parent or child == conjugate or '-'.join(child) in tried:
            continue

        # Check for invalid and nonviable children

        if not valid(child) or (viable != None and not viable(child)):
            child = None
        else:
            model, layer_count = build_model(child)
            if model is None or layer_count > max_len:
                child = None


    if _DEBUG:
        printlog('New child', '-'.join(child))

    return child


def ligate(statistics, genome, new_fitness):
    """ Slice and dice a genome and generate statistics for analysis. Statistics
        is a dictionary; for each genetic fragment, it contains a (best, mean,
        worst, count) tuple
    """

    if not isinstance(genome, list):
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

    for frag in fragments:
        frag = '_'.join(frag)
        if frag not in statistics:
            statistics[frag] = (new_fitness, new_fitness, new_fitness, 1)
        else:
            best, mean, worst, count = statistics[frag]

            best = min(best, new_fitness)
            worst = max(worst, new_fitness)
            mean = (mean * count + new_fitness) / (count + 1)
            count += 1

            statistics[frag] = (best, mean, worst, count)

    return statistics

class Organism():
    """ Holds information about an organism and its state """

    def __init__(self, item):

        if isinstance(item, list):
            self.genome = item[0]
            self.fitness = item[1] if len(item) > 1 else 0.0
            self.epoch = item[2] if len(item) > 2 else 0
            self.improved = item[3] if len(item) > 3 else True
        elif isinstance(item, str):
            self.genome = item
            self.fitness, self.epoch, self.improved = 0.0, 0, True

    def __iter__(self):
        yield self.genome
        yield self.fitness
        yield self.epoch
        yield self.improved

    def __repr__(self):
        return str(list(self))

    def __str__(self):
        return "Genome={}, Fitness={}, Epoch={}, Improved={}".format(self.genome, self.fitness, self.epoch, self.improved)

def train(org, config, epochs=1):
    """ Train an organism for 1 or more epochs

        org           Organism class instance [genome, fitness, epochs, boolean]
        config        ModelIO configuration
        epochs        How many epochs to run

        Returns updated organism
    """

    genome = org.genome

    if not isinstance(genome, list):
        genome = genome.split('-')

    printlog('Training for {} epoch{} : {}'.format(epochs, 's' if epochs > 0 else '', '-'.join(genome)))

    cell = BaseSRCNNModel(org.genome, config)

    if cell.model is None:
        printlog("Compiling model")
        model, _ = build_model(genome, shape=config.image_shape, learning_rate=config.learning_rate, metrics=[cell.evaluation_function])

        if model is None:
            return Organism([org.genome, 0.0, 0, False])

        cell.model = model
    else:
        printlog("Using loaded model...")

    # Now we have a compiled model, execute it - or at least try to, there are still some
    # models that may bomb out.

    try:

        results = cell.fit(run_epochs=epochs)

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot train: {}'.format(sys.exc_info()[1]))
        raise

    printlog('Fitness: {}'.format(results))

    return Organism([org.genome, results, org.epoch + epochs, org.improved and results < org.fitness])
