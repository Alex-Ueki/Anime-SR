# pylint: disable=C0301
# (line_too_long disabled)

"""
Toolkit for evolving Keras models
"""

import random
import sys
import datetime
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

FILTERS = [32, 64, 128, 256]
KERNELS = [1, 3, 5, 7, 9]
ACTS = ['linear', 'elu', 'tanh', 'softsign']
DEPTHS = [2, 3, 4]

CONVOLUTIONS = {'conv_f{}_k{}_{}'.format(f, k, a): Conv2D(f, (k, k), activation=a, padding='same')
                for a in ACTS for k in KERNELS for f in FILTERS}

# Merge codons combine multiple layers. They are only expressed inside composite
# codons

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

# The output layer is always a convolution layer that generates 3 channels

OUTPUTS = {'out_k{}_{}'.format(k, a): Conv2D(3, (k, k), padding='same')
           for k in KERNELS for a in ACTS}

# For convenience, make a dictionary of all the possible codons. Python 3.5 black magic!

ALL_CODONS = {**CONVOLUTIONS, **COMPOSITES, **OUTPUTS, **MERGERS}

# Mutation selection list (of dicts. Used to select a random codon for mutation.

MUTABLE_CODONS = [CONVOLUTIONS, COMPOSITES]


def build_model(genome, shape=(64, 64, 3), learning_rate=0.001, metrics=None):
    """ Build and compile a keras model from an expressed sequence of codons.
        Returns the model or None if the compile failed in some way

        genome      list of codon names (if string, will be converted)
        shape       shape of model input
        learning_rate  initial learning rate
        metrics     callbacks
    """

    if not isinstance(genome, list):
        genome = genome.split('-')

    # Get the wiring hookups for the sequence, and preface them with an
    # input layer. This means genome[i-1] is the code for codon[i].

    codons = [Input(shape=shape)] + [ALL_CODONS[gene] for gene in genome]

    # Wire the Codons. As we generate each layer, we update the codons[] list
    # with the constructed layer, so that subsequent codons can be connected
    # to them. We don't have to create the first (input) codon. All Keras layer
    # functions have to be deep-copied so that they are unique objects.

    if _DEBUG:
        all_layers = [codons[0]]

    try:

        # We don't wire up the first (input) codon, so because of the slice
        # in enumerate(), i = codon number - 1. Keep this in mind when
        # reading references to i inside this loop.

        for i, layer in enumerate(codons[1:]):
            layer = deepcopy(layer)
            if isinstance(layer, list):
                # Composite multi-layer codon with a merge as the last element.
                # Wire all the input layers to previous codon, then
                # wire their outputs to the merge layer
                for j, _ in enumerate(layer):
                    layer[j] = deepcopy(layer[j])
                    layer[j].name = genome[i] + '_{}_{}'.format(i, j)
                    layer[j] = layer[j](codons[i] if j < len(layer) - 1 else layer[:-1])
                # Update the layer to point to the output layer
                codons[i + 1] = layer[-1]
                if _DEBUG:
                    all_layers.extend(layer)
            else:
                # Simple 1-layer codon
                layer.name = genome[i] + '_{}'.format(i)
                codons[i + 1] = layer(codons[i])
                if _DEBUG:
                    all_layers.append(codons[i + 1])

        if _DEBUG:
            for i, layer in enumerate(all_layers):
                print('Layer', i, layer.name, layer, layer._consumers)
            print('')

        # Create and compile the model

        model = Model(codons[0], codons[-1])

        adam = optimizers.Adam(lr=learning_rate, clipvalue=(1.0 / .001), epsilon=0.001)

        metrics = {} if metrics is None else metrics

        model.compile(optimizer=adam, loss='mse', metrics=metrics)

        if _DEBUG:
            print('Compiled model: shape={}, learning rate={}, metrics={}'.format(
                shape, learning_rate, metrics))
        return model

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot compile: {}'.format(sys.exc_info()[1]))
        raise


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
            type_parameter_parameter_... _activation and parameter will always
            be in format letter-code[digits]. The type is never changed, and we
            do special handling for the activation function
        """

        def kernel_sequence(number):
            """ Generate a random sorted kernel sequence string """

            return ''.join(sorted([str(n) for n in random.sample(KERNELS, number)]))

        locus = random.randrange(len(child))
        original_locus = child[locus]
        codons = original_locus.split('_')
        basepair = random.randrange(len(codons))

        while True:
            if basepair == len(codons) - 1:
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

        if child_len >= max_len:
            return None

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

    while child is None or child == parent or child == conjugate or viable != None and not viable(child):

        # Deep copy the parent into child, choose a mutation type, and
        # call the appropriate mutation function

        child = parent[:]
        choice = random.randrange(sum(odds))

        # I admit, this is a bit tricky! Will generate 5 for the first
        # possible choice, 4 for the next, etc. Then use negative indexing
        # to choose the right function!

        todo = len([i for i in odds if choice < i])
        child = operations[-todo](child, conjugate)

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


def fitness(genome, config, best_fitness=0.0, apoptosis=None):
    """ Determine the fitness of an organism by creating its model and running it.

        organism      string or list with genetic code to be tested
        config        ModelIO configuration
        best_fitness  Best fitness in the gene pool
        apoptosis     Function that returns True if we should quit early
    """

    if not isinstance(genome, list):
        genome = genome.split('-')

    organism = '-'.join(genome)

    printlog('Testing fitness of {}'.format(organism))

    config.config['model'] = organism

    cell = BaseSRCNNModel(organism, config)

    model = build_model(genome, shape=config.image_shape, learning_rate=config.learning_rate, metrics=[cell.evaluation_function])

    if model is None:
        return 0.0

    cell.model = model

    # Now we have a compiled model, execute it - or at least try to, there are still some
    # models that may bomb out.

    try:

        stime = datetime.datetime.now()
        results = cell.fit(max_epochs=1)
        etime = datetime.datetime.now()

        epochs = config.epochs
        halfway = epochs // 2

        eta = etime + (etime - stime) * (halfway - 1)
        printlog(
            'After 1 epoch: fitness={}, will be halfway @ {:%I:%M:%S %p}'.format(results, eta))

        if apoptosis != None and apoptosis(results, 1, epochs, best_fitness=best_fitness, last_fitness=None):
            printlog('Apoptosis triggered!')
            return results

        prev_results = results
        stime = datetime.datetime.now()
        results = cell.fit(max_epochs=halfway)
        etime = datetime.datetime.now()

        eta = etime + (etime - stime) * halfway / (halfway - 1)
        printlog('After {} epochs: fitness={}, will complete @ {:%I:%M:%S %p}'.format(
            halfway, results, eta))

        if apoptosis != None and apoptosis(results, halfway, epochs, best_fitness=best_fitness, last_fitness=prev_results):
            printlog('Apoptosis triggered!')
            return results

        results = cell.fit(max_epochs=epochs)
        printlog('After {} epochs: fitness={}'.format(epochs, results))

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot fit: {}'.format(sys.exc_info()[1]))
        raise

    printlog('Fitness: {}'.format(results))
    return results

def one_epoch(genome, config):
    """ Train a model by 1 epoch

        organism      string or list with genetic code to be tested
        config        ModelIO configuration
        best_fitness  Best fitness in the gene pool
        apoptosis     Function that returns True if we should quit early
    """

    if not isinstance(genome, list):
        genome = genome.split('-')

    organism = '-'.join(genome)

    printlog('Training for one epoch : {}'.format(organism))

    cell = BaseSRCNNModel(organism, config)

    if cell.model is None:
        print("Compiling model")
        model = build_model(genome, shape=config.image_shape, learning_rate=config.learning_rate, metrics=[cell.evaluation_function])

        if model is None:
            return 0.0

        cell.model = model
    else:
        print("Using loaded model - x fingers")

    # Now we have a compiled model, execute it - or at least try to, there are still some
    # models that may bomb out.

    try:

        results = cell.fit(max_epochs=999, run_epochs=1)

    except KeyboardInterrupt:

        raise

    except:
        printlog('Cannot train: {}'.format(sys.exc_info()[1]))
        raise

    printlog('Fitness: {}'.format(results))
    return results
