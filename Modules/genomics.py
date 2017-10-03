# pylint: disable=C0301
# (line_too_long disabled)

"""
Toolkit for evolving Keras models
"""

import random
import sys
import itertools
from copy import deepcopy

from keras.models import Model
from keras.layers import Add, Average, Multiply, Maximum, Input
from keras.layers.convolutional import Conv2D
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
#
# Codons can have a *{number} suffix that means "repeat the layer {number) times"

FILTERS = [32, 64, 128, 256]
KERNELS = [1, 3, 5, 7, 9]
ACTS = ['linear', 'elu', 'tanh', 'softsign']
DEPTHS = [2, 3, 4]

CONVOLUTIONS = {'conv_f{}_k{}_{}'.format(f, k, a): Conv2D(f, (k, k), activation=a, padding='same')
                for a in ACTS for k in KERNELS for f in FILTERS}

# Merge codons combine multiple layers. They are internal codons that are the
# final layer of composite codons

MERGERS = {
    'add': Add(),
    'avg': Average(),
    'mult': Multiply(),
    'max': Maximum()
}

# Autogenerate composite codons. We have some restrictions; for example, the filter
# size of all the convolutions must be the same.


def _make_composites():
    """ Make composite codons """

    codons = {}

    for mrg in MERGERS:
        for flt in FILTERS:
            for act in ACTS:
                for dep in DEPTHS:
                    for knl in itertools.combinations(KERNELS, dep):
                        cname = '{}_f{}_k{}_d{}_{}'.format(
                            mrg, flt, ''.join([str(k) for k in knl]), dep, act)
                        flist = [CONVOLUTIONS['conv_f{}_k{}_{}'.format(
                            flt, k, act)] for k in knl] + [MERGERS[mrg]]
                        codons[cname] = flist

    return codons


COMPOSITES = _make_composites()

# Modifier codons modify the effect of the next codon. Currently, the only one
# is a dup_nnn codon that duplicates the next codon nnn times. Modifiers are
# additive, not cumulative, and applied in reverse order. So for example,
# dup_2-dup_3-CODON will first duplicate CODON 3 times, then duplicate it
# again 2 times, resulting in *6* CODON codons.

def mod_duplicate(codon, layers, inputs, levels):
    """ mod_rnnn: Duplicate the current front codon nnn times """

    # We want to duplicate all layers at the same level as the last layer
    # added (so we can duplicate merge layers). Reverse the layers so we
    # add the last ones first, which will cause the order to be maintained.

    level = levels[0]
    codon_layers = [layer for i, layer in enumerate(layers) if levels[i] == level]
    codon_inputs = [inputs for i, inputs in enumerate(inputs) if levels[i] == level]
    codon_layers.reverse()
    codon_inputs.reverse()

    for i in range(int(codon.split('_')[1][1:])):
        level -= 1
        for j, _ in enumerate(codon_layers):
            layer = deepcopy(codon_layers[j])
            layer.name = layer.name + '_r{}'.format(i)
            layers.insert(0, layer)
            levels.insert(0, level)
            inputs.insert(0, codon_inputs[j])

    return layers, inputs, levels

PRIMES = [1, 2, 3, 5, 7, 11, 13]

MODIFIERS = {'mod_r' + str(n): mod_duplicate for n in PRIMES}

MODIFIER_FUNCTIONS = [mod_duplicate]

# The output layer is always a convolution layer that generates 3 channels

OUTPUTS = {'out_k{}_{}'.format(k, a): Conv2D(3, (k, k), padding='same')
           for k in KERNELS for a in ACTS}

# For convenience, make a dictionary of all the possible codons. Python 3.5 black magic!

ALL_CODONS = {**CONVOLUTIONS, **COMPOSITES, **OUTPUTS, **MERGERS, **MODIFIERS}

# Mutation selection list (of dicts. Used to select a random codon for mutation.

MUTABLE_CODONS = [CONVOLUTIONS, COMPOSITES, MODIFIERS]


def build_model(genome, shape=(64, 64, 3), learning_rate=0.001, metrics=None):
    """ Build and compile a keras model from an expressed sequence of codons.
        Returns (model, layer_count) tuple or None if the compile failed in some way

        genome          list of codon names (if string, will be converted)
        shape           shape of model input
        learning_rate   initial learning rate
        metrics         callbacks
    """

    if not isinstance(genome, list):
        genome = genome.split('-')

    if _DEBUG:
        print(genome)

    try:

        # layers are the actual model layers. inputs are how they have
        # to be wired up (a relative reference, always negative). For
        # the final layer of a composite codon, this will be a list.

        layers = []
        inputs = []
        levels = []

        level = 0

        # Build the layers of the model in reverse, from output to input.
        # Note that here we use reversed(list(enumerate())) so while we
        # process the codons in reverse order, i still links them to the
        # genome (i will count down, in other words)

        for i, codon in reversed(list(enumerate(genome))):

            # Keep track of layer levels so we can figure out what layers
            # were generated by the same codon.

            level += 1

            # If the codon is a modifier, use it's function to modify the model
            # as built so far. Do not modify an output!

            if ALL_CODONS[codon] in MODIFIER_FUNCTIONS:
                if genome[i + 1] not in OUTPUTS:
                    layers, inputs, levels = ALL_CODONS[codon](codon, layers, inputs, levels)
                    level = levels[0]
                else:
                    print('Cannot compile: Modifier codon trying to modify output layer.')
                    return None, 0
            else:
                # Make a deep copy of the layer to generate. If the result is
                # a list, then it is a composite codon.

                layer = deepcopy(ALL_CODONS[codon])
                if isinstance(layer, list):
                    # Composite multi-layer codon with a merge as the last element.
                    # Wire all the input layers to previous codon, then
                    # wire their outputs to the merge layer. Note that here we
                    # use enumerate(reversed()), so that j will count up (and
                    # will be 0 when we do the merge layer)

                    for j, sublayer in enumerate(reversed(layer)):
                        sublayer = deepcopy(sublayer)
                        sublayer.name = genome[i] + '_{}_{}'.format(i, j)
                        layers.insert(0, sublayer)
                        levels.insert(0, level)
                        if j:
                            # Input layers are wired to the previous codon output
                            inputs.insert(0, j - len(layer))
                        else:
                            # Merge layer is wired to all the input layers
                            inputs.insert(0, [-1 * n for n in range(1, len(layer))])
                else:
                    # Simple 1-layer codon.
                    layer.name = codon + '_{}'.format(i)
                    layers.insert(0, layer)
                    levels.insert(0, level)
                    inputs.insert(0, -1)

        # Add the input layer

        layers.insert(0, Input(shape=shape))
        inputs.insert(0, None)

        # Wire the layers

        for i, wires in enumerate(inputs):
            if wires:
                if isinstance(wires, list):
                    layers[i] = layers[i]([layers[i + j] for j in wires])
                else:
                    layers[i] = layers[i](layers[i + wires])

        if _DEBUG:
            for i, layer in enumerate(layers):
                print('Layer', i, layer.name, layer, layer._consumers)
            print('')

        # Create and compile the model

        model = Model(layers[0], layers[-1])

        adam = optimizers.Adam(lr=learning_rate, clipvalue=(1.0 / .001), epsilon=0.001)

        metrics = {} if metrics is None else metrics

        model.compile(optimizer=adam, loss='mse', metrics=metrics)

        if _DEBUG:
            print('Compiled model: shape={}, learning rate={}, metrics={}'.format(
                shape, learning_rate, metrics))
        return (model, len(layers))

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot compile: {}'.format(sys.exc_info()[1]))
        raise
        return None, 0


def mutate(parent, conjugate, min_len=3, max_len=30, odds=(3, 6, 7, 9, 10, 11), best_fitness=0.0, statistics=None, viable=None):
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
        min_len       minimum length of the resulting genome
        max_len       maximum length of the resulting genome
        odds          list of *cumulative* odds of point, insert, delete, transpose, conjugate
        best_fitness  the fitness of the best genome found so far.
        statistics    dictionary of codon statistics for guiding evolotion
        viable        viability function; takes a codon list, returns true if it is acceptable
    """

    def fit_enough(codon):
        """ A codon is "fit enough" if it is *not* in the statistics dictionary or
            if it passes a dice roll based on how close its mean fitness is to the
            best fitness of all the genomes. This will give a slight preference to
            the better codons, but still allow for variety
        """

        if best_fitness >= 0 or codon not in statistics:
            return True

        codon_fitness = statistics[codon][1]

        return codon_fitness >= 0 or random.random() >= codon_fitness / best_fitness


    def random_codon():
        """ Choose a random codon from mutable_codons, with suitable fitness """

        while True:
            codon = MUTABLE_CODONS

            while isinstance(codon, (list, tuple)):
                codon = random.choice(codon)

            if isinstance(codon, dict):
                return random.choice(list(codon.keys()))

            if fit_enough(codon):
                break

        return codon

    def point_mutation(child, _):
        """ Make a point mutation in a codon. Codon will always be in the format
            type_parameter_parameter_... {_activation} and parameter will always
            be in format letter-code[digits]. The type is never changed, and we
            do special handling for the activation function if it is present.
            Currently modifier codons do not have activation functions.
        """

        def kernel_sequence(number):
            """ Generate a random sorted kernel sequence string """

            return ''.join(sorted([str(n) for n in random.sample(KERNELS, number)]))

        locus = random.randrange(len(child))
        original_locus = child[locus]
        codons = original_locus.split('_')
        basepair = random.randrange(len(codons))

        while True:
            if basepair == len(codons) - 1 and original_locus not in MODIFIERS:
                # choose new activation function
                new_codon = random.choice(ACTS)
            elif basepair == 0:
                # choose new codon type (only if a merge-type codon)
                if codons[0] in MERGERS:
                    new_codon = random.choice(list(MERGERS.keys()))
                else:
                    new_codon = codons[0]
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
                    # If the codon has a depth parameter we need a sequence of kernel sizes, but we
                    # can deduce this by the length of the current k parameter, since currently
                    # they are all single-digit.
                    param = kernel_sequence(len(param))
                elif base == 'd':
                    # If we change the depth we have to also change the k parameter
                    param = random.choice(DEPTHS)
                    codons = [c if c[0] != 'k' else 'k' +
                              kernel_sequence(param) for c in codons]
                elif base == 'f':
                    param = random.choice(FILTERS)
                elif base == 'r':
                    param = random.choice(PRIMES)
                else:
                    printlog('Unknown parameter base {} in {}'.format(
                        base, original_locus))
                    param = 'XXX'
                new_codon = base + str(param)
            if fit_enough(new_codon):
                break

        codons[basepair] = new_codon
        child[locus] = '_'.join(codons)

        if _DEBUG:
            print('    Point mutation {} -> {}'.format(original_locus, child[locus]))

        return child

    def insertion(child, _):
        """ Insertion mutation - never insert after output codon """

        child_len = len(child)

        # This may cause the layer count to become too high, but we can't check for that
        # until later, when we do a test-build of the model.

        codon = random_codon()

        child.insert(random.randrange(child_len), codon)

        if _DEBUG:
            print('         Insertion', codon)

        return child

    def deletion(child, _):
        """ Deletion mutation - never delete output codon! """

        child_len = len(child)

        if child_len <= min_len:
            return None

        del child[random.randrange(child_len - 1)]

        if _DEBUG:
            print('          Deletion')

        return child

    def conjugation(child, conjugate):
        """ Conjugate two genomes - always at least one codon from each contributor """

        splice = random.randrange(1, len(child))
        child = child[:-splice] + conjugate[-splice:]

        if _DEBUG:
            print('       Conjugation')

        return child

    def transposition(child, _):
        """ Transposition mutation - never transpose output codon """

        splice = random.randrange(len(child) - 1)
        codon = child[splice]
        del child[splice]

        splice = random.randrange(len(child) - 1)
        child.insert(splice, codon)

        if _DEBUG:
            print('     Transposition {}'.format(codon))

        return child

    def inversion(child, _):
        """ Invert two adjacent codons """

        locus = random.randrange(len(child) - 2)

        child[locus], child[locus + 1] = child[locus + 1], child[locus]

        if _DEBUG:
            print('       Inversion @ {}'.format(locus))

        return child

    def regularize(child):
        """ Rearrange modifier codons into consistent order """

        shuffled = True

        while shuffled:

            shuffled = False

            for i in range(len(child)-1):
                if child[i] in MODIFIERS and child[i + 1] in MODIFIERS:
                    # Currently depends on the modifier parameter codes being
                    # in string sort order, but since we only have one of them
                    # (*) this is not an issue.
                    codes = [(p[0], int(p[1:])) for p in child[i:i + 2]]
                    if codes[0] > codes[1]:
                        child[i], child[i + 1] = child[i + 1], child[i]
                        shuffled = True

        return child

    # -------------------------------------------------
    # MAIN BODY OF evolve(). Make sure inputs are lists
    # -------------------------------------------------

    if not isinstance(parent, list):
        parent = parent.split('-')

    if not isinstance(conjugate, list):
        conjugate = conjugate.split('-')

    statistics = {} if statistics is None else statistics

    if _DEBUG:
        print('   Mutation parent', '-'.join(parent))
        print('Mutation conjugate', '-'.join(conjugate))

    operations = [point_mutation, insertion, deletion, inversion, transposition, conjugation]

    child = None

    # Repeat until we get a useful mutation

    while child is None or child == parent or child == conjugate:

        # Deep copy the parent into child, choose a mutation type, and
        # call the appropriate mutation function

        child = parent[:]
        choice = random.randrange(sum(odds))

        # I admit, this is a bit tricky! Will generate 5 for the first
        # possible choice, 4 for the next, etc. Then use negative indexing
        # to choose the right function!

        todo = len([i for i in odds if choice < i])
        child = operations[-todo](child, conjugate)

        # Check for invalid children

        if viable != None and not viable(child):
            child = None
        else:
            model, layer_count = build_model(child)
            if model is None or layer_count > max_len:
                child = None
    # Rearrange any adjacent modifier codons into a consistent order so we don't end up training
    # two effectively identical genomes.

    child = regularize(child)

    if _DEBUG:
        print('   Resulting child', '-'.join(child))
        print('')

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


def train(genome, config, epochs=1):
    """ Train a model for 1 or more epochs

        organism      string or list with genetic code to be tested
        config        ModelIO configuration
        epochs        How many epochs to run
    """

    if not isinstance(genome, list):
        genome = genome.split('-')

    organism = '-'.join(genome)

    printlog('Training for {} epoch{} : {}'.format(epochs, 's' if epochs > 0 else '', organism))

    cell = BaseSRCNNModel(organism, config)

    if cell.model is None:
        print("Compiling model")
        model, _ = build_model(genome, shape=config.image_shape, learning_rate=config.learning_rate, metrics=[cell.evaluation_function])

        if model is None:
            return 0.0

        cell.model = model
    else:
        print("Using loaded model...")

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
    return results
