# pylint: disable=C0301
# Line too long
"""
Usage: evolve.py [option(s)] ...

    Evolves (hopefully) improved models. In each generation, the 5 best models
    are retained, and then 15 new models are evolved from them. Progressively
    trains the new models one epoch at a time for 10 epochs, discarding the
    worst performer in each iteration.

Options are:

    genepool=path       path to genepool file, default is {Data}/genepool.json
    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    lr=.nnn             set initial learning rate. Should be 0.001 or less. Default = 0.001
    quality=.nnn        fraction of the "best" tiles used in training (but not validation). Default is 1.0 (use all)
    residual=1|0|T|F    have the model train using residual images. Default=True.
    black=auto|nnn      black level (0..1) for image border pixels, default=auto (use blackest pixel in first image)
    trimleft=nnn        pixels to trim on image left edge, default = 240; can also use left=nnn
    trimright=nnn       pixels to trim on image right edge, default = 240; can also use right=nnn
    trimtop=nnn         pixels to trim on image top edge, default = 0; can also use top=nnn
    trimbottom=nnn      pixels to trim on image bottom edge, default = 0; can also use bottom=nnn
    jitter=1|0|T|F      include tiles offset by half a tile across&down when training (but not validation); default=True
    skip=1|0|T|F        randomly skip 0-3 tiles between tiles when training; default=True
    shuffle=1|0|T|F     shuffle tiles into random order when training; default=True
    data=path           path to the main data folder, default = Data
    training=path       path to training folder, default = {Data}/train_images/training
    validation=path     path to validation folder, default = {Data}/train_images/validation

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK).

    Options are overridden by the contents of genepool.json, if any. Thus they are typically only specified on
    the first run. If t genepool.json file does not exist, it will be created with an initial population similar
    to some of the models in models.py

    See Modules/genomics.py for details on the structure of genes, codons and other genetic elements.

"""

import os
import json
import random

import numpy as np

from Modules.misc import oops, terminate, set_docstring, printlog, parse_options
from Modules.modelio import ModelIO


set_docstring(__doc__)

# Evolutionary constants (not saved in the JSON file)

MAX_POPULATION = 25         # Maximum size of population
MIN_POPULATION = 5          # Minimum size of population
EPOCHS = 10                 # Number of epochs to train


# Checkpoint state to genepool.json file


def checkpoint(json_path, population, graveyard, statistics, config):
    """ Checkpoint the current evolutionary state to json file """

    state = {
        'population': population,
        'graveyard': graveyard,
        'statistics': statistics,
        'config': config.config
    }

    with open(json_path, 'w') as jsonfile:
        json.dump(state, jsonfile, indent=4)



def setup(options):
    """Set up configuration """

    # set up our initial state

    errors = False
    genepool = {}
    options['paths'].setdefault('genepool', os.path.join('Data', 'genepool.json'))
    poolpath = options['paths']['genepool']

    if os.path.exists(poolpath):
        if os.path.isfile(poolpath):
            print('Loading existing genepool')
            try:
                with open(poolpath, 'r') as jsonfile:
                    genepool = json.load(jsonfile)

                # PU: Temp hack to change 'io' key to 'config'

                if 'io' in genepool:
                    genepool['config'] = genepool['io']
                    del genepool['io']

            except json.decoder.JSONDecodeError:
                print(
                    'Could not parse json. Did you edit "population" and forget to delete the trailing comma?')
                errors = True
        else:
            errors = oops(
                errors, True, 'Genepool path is not a reference to a file ({})', poolpath)
    else:
        errors = oops(errors, not os.access(os.path.dirname(
            poolpath), os.W_OK), 'Genepool folder is not writeable ({})', poolpath)

    terminate(errors, False)

    # Genepool settings override config, so we need to update them

    for setting in genepool['config']:
        if setting not in options or options[setting] != genepool['config'][setting]:
            options[setting] = genepool['config'][setting]

    # Reload config with possibly changed settings

    config = ModelIO(options)

    # Validation and error checking

    import Modules.frameops as frameops

    image_paths = ['training', 'validation']
    sub_folders = ['Alpha', 'Beta']
    image_info = [[[], []], [[], []]]

    for fcnt, fpath in enumerate(image_paths):
        for scnt, _ in enumerate(sub_folders):
            image_info[fcnt][scnt] = frameops.image_files(
                os.path.join(config.paths[fpath], sub_folders[scnt]), True)

    for fcnt in [0, 1]:
        for scnt in [0, 1]:
            errors = oops(
                errors, image_info[fcnt][scnt] is None, '{} images folder does not exist', image_paths[fcnt] + '/' + sub_folders[scnt])

    terminate(errors, False)

    for fcnt in [0, 1]:
        for scnt in [0, 1]:
            errors = oops(errors, len(
                image_info[fcnt][scnt]) == 0, '{} images folder does not contain any images', image_paths[fcnt] + '/' + sub_folders[scnt])
            errors = oops(errors, len(
                image_info[fcnt][scnt]) > 1, '{} images folder contains more than one type of image', image_paths[fcnt] + '/' + sub_folders[scnt])

    terminate(errors, False)

    for fcnt in [0, 1]:
        errors = oops(errors, len(image_info[fcnt][0][0]) != len(
            image_info[fcnt][1][0]), '{} images folders have different numbers of images', image_paths[fcnt])

    terminate(errors, False)

    for fcnt in [0, 1]:
        for path1, path2 in zip(image_info[fcnt][0][0], image_info[fcnt][1][0]):
            path1, path2 = os.path.basename(path1), os.path.basename(path2)
            errors = oops(
                errors, path1 != path2, '{} images folders do not have identical image filenames ({} vs {})', (image_paths[fcnt], path1, path2))
            terminate(errors, False)

    # test_files = [[image_info[f][g][0][0] for g in [0, 1]] for f in [0, 1]]

    test_images = [[frameops.imread(image_info[f][g][0][0])
                    for g in [0, 1]] for f in [0, 1]]

    # What kind of file is it? Do I win an award for the most brackets?

    # img_suffix = os.path.splitext(image_info[0][0][0][0])[1][1:]

    # Check that the Beta tiles are the same size.

    size1, size2 = np.shape(test_images[0][1]), np.shape(test_images[1][1])
    errors = oops(errors, size1 != size2, 'Beta training and evaluation images do not have identical size ({} vs {})',
                  (size1, size2))

    # Warn if we do have some differences between Alpha and Beta sizes

    for fcnt in [0, 1]:
        size1, size2 = np.shape(test_images[fcnt][0]), np.shape(
            test_images[fcnt][1])
        if size1 != size2:
            print('Warning: {} Alpha and Beta images are not the same size ({} vs {}). Will attempt to scale Alpha images.'.format(
                image_paths[fcnt].title(), size1, size2))

    terminate(errors, False)

    # Only check the size of the Beta output for proper configuration, since Alpha tiles will
    # be scaled as needed.

    errors = oops(errors, len(size2) != 3 or size2[2] != 3, 'Images have improper shape ({0})', str(size2))

    terminate(errors, False)

    image_width, image_height = size2[1], size2[0]
    trimmed_width = image_width - (config.trim_left + config.trim_right)
    trimmed_height = image_height - (config.trim_top + config.trim_bottom)

    errors = oops(errors, trimmed_width <= 0,
                  'Trimmed images have invalid width ({} - ({} + {}) <= 0)', (size1[0], config.trim_left, config.trim_right))
    errors = oops(errors, trimmed_width <= 0,
                  'Trimmed images have invalid height ({} - ({} + {}) <= 0)', (size1[1], config.trim_top, config.trim_bottom))

    terminate(errors, False)

    errors = oops(errors, (trimmed_width % config.base_tile_width) != 0,
                  'Trimmed images do not evenly tile horizontally ({} % {} != 0)', (trimmed_width, config.tile_width))
    errors = oops(errors, (trimmed_height % config.base_tile_height) != 0,
                  'Trimmed images do not evenly tile vertically ({} % {} != 0)', (trimmed_height, config.tile_height))

    terminate(errors, False)

    # Attempt to automatically figure out the border color black level, by finding the minimum pixel value in one of our
    # sample images. This will definitely work if we are processing 1440x1080 4:3 embedded in 1920x1080 16:19 images.
    # Write back any change into config.

    if config.black_level < 0:
        config.black_level = np.min(test_images[0][0])
        config.config['black_level'] = config.black_level

    return (config, genepool, image_info)

def evolve(config, genepool, image_info):
    """ Evolve the genepool """

    # Initialize missing values in genepool

    if 'population' in genepool:
        population = genepool['population']
    else:
        print('Initializing population...')
        population = [
            ["conv_f64_k9_elu-conv_f32_k1_elu-out_k5_elu", 0.0, 0],
            ["conv_f64_k9_elu-conv_f32_k1_elu-avg_f32_k135_d3_elu-out_k5_elu", 0.0, 0]
        ]

    if 'graveyard' in genepool:
        graveyard = genepool['graveyard']
    else:
        print('Initializing graveyard...')
        graveyard = []

    if 'statistics' in genepool:
        statistics = genepool['statistics']
    else:
        print('Initializing statistics...')
        statistics = {}

    poolpath = config.paths['genepool']

    # Remind user what we're about to do.

    print('          Genepool : {}'.format(config.paths['genepool']))
    print('        Tile Width : {}'.format(config.base_tile_width))
    print('       Tile Height : {}'.format(config.base_tile_height))
    print('       Tile Border : {}'.format(config.border))
    print('    Min Population : {}'.format(MIN_POPULATION))
    print('    Max Population : {}'.format(MAX_POPULATION))
    print('   Epochs to train : {}'.format(config.epochs))
    print('    Data root path : {}'.format(config.paths['data']))
    print('   Training Images : {}'.format(config.paths['training']))
    print(' Validation Images : {}'.format(config.paths['validation']))
    print('  Input Image Size : {} x {}'.format(config.image_width, config.image_height))
    print('          Trimming : Top={}, Bottom={}, Left={}, Right={}'.format(
        config.trim_top, config.trim_bottom, config.trim_left, config.trim_right))
    print(' Output Image Size : {} x {}'.format(
        config.trimmed_width, config.trimmed_height))
    print(' Training Set Size : {}'.format(len(image_info[0][0][0])))
    print('   Valid. Set Size : {}'.format(len(image_info[1][0][0])))
    print('       Black level : {}'.format(config.black_level))
    print('            Jitter : {}'.format(config.jitter == 1))
    print('           Shuffle : {}'.format(config.shuffle == 1))
    print('              Skip : {}'.format(config.skip == 1))
    print('          Residual : {}'.format(config.residual == 1))
    print('           Quality : {}'.format(config.quality))
    print('')

    checkpoint(poolpath, population, graveyard, statistics, config)

    # Evolve the genepool

    import Modules.genomics as genomics

    # Repeat until program terminated.

    best_fitness = 0

    while True:

        # Legacy fix to clean up population

        population = [p if isinstance(p, list) else [p, 0.0, 0] for p in population]
        population = [p if len(p) == 3 else p + [10] for p in population]

        # While there are some genomes with less than EPOCHS epochs of fitting,
        # evolve them 1 epoch and remove the worst performer.

        # All sequences that end in a checkpoint are protected by dummy try/except
        # blocks, to ensure that a user-break doesn't cause an incorrect state to
        # be checkpointed

        least_evolved = min([p[2] for p in population])
        while least_evolved < EPOCHS:
            print('Processing Epoch', least_evolved + 1)
            for i, organism in enumerate(population):
                if organism[2] == least_evolved:
                    genome = organism[0]
                    config.paths['model'] = os.path.join(config.paths['genebank'], genome + '.h5')
                    config.paths['state'] = os.path.join(config.paths['genebank'], genome + '.json')
                    config.model_type = genome
                    config.config['model_type'] = config.model_type
                    config.config['paths'] = config.paths
                    config.epochs = 0
                    config.run_epochs = 1

                    try:
                        population[i] = [genome, genomics.train(genome, config, epochs=1), least_evolved + 1]
                    except:
                        raise
                    else:
                        print('checkpoint reached')
                        checkpoint(poolpath, population, graveyard, statistics, config)


            try:
                population.sort(key=lambda o: o[1])

                # Remove a genome -- grab stats on it

                if len(population) > MAX_POPULATION - least_evolved:
                    print('Removing {} @ {}'.format(population[-1][0], population[-1][1]))
                    graveyard.append(population[-1][0])
                    statistics = genomics.ligate(statistics, population[-1][0], population[-1][1])
                    del population[-1]
            except:
                raise
            else:
                checkpoint(poolpath, population, graveyard, statistics, config)

            # What organisms need handling in the next iteration?

            least_evolved = min([p[2] for p in population])

        # Cull excess population

        if len(population) > MIN_POPULATION:
            try:
                # Gather statistics on the genomes that are about to be culled

                for organism in population[MIN_POPULATION:]:
                    statistics = genomics.ligate(statistics, organism[0], organism[1])
                    graveyard.append(organism[0])
                    
                # Cull the population

                population = population[:MIN_POPULATION]
            except:
                raise
            else:
                checkpoint(poolpath, population, graveyard, statistics, config)

        # Expand the population to the maximum size.

        try:
            parents, children = [p[0] for p in population], []

            printlog('Creating new children...')

            while len(children) < (MAX_POPULATION - len(parents)):
                parent, conjugate = [p for p in random.sample(parents, 2)]
                child = '-'.join(genomics.mutate(parent, conjugate,
                                                 best_fitness=best_fitness, statistics=statistics))
                if child not in parents and child not in children and child not in graveyard:
                    children.append([child, 0.0, 0])
                else:
                    printlog('Duplicate genome rejected...')

            population.extend(children)
        except:
            raise
        else:
            checkpoint(poolpath, population, graveyard, statistics, config)


if __name__ == '__main__':

    # The command-line options for the tool

    OPCODES = {
        'width': ('base_tile_width', int, lambda x: x <= 0, 'Tile width invalid ({})'),
        'height': ('base_tile_height', int, lambda x: x <= 0, 'Tile height invalid ({})'),
        'border': ('border', int, lambda x: x <= 0, 'Tile border invalid ({})'),
        'black': ('black_level', float, lambda x: False, 'Black level invalid ({})'),
        'lr': ('learning_rate', float, lambda x: x <= 0.0 or x > 0.01, 'Learning rate should be 0 > and <= 0.01 ({})'),
        'quality': ('quality', float, lambda x: x <= 0.0 or x > 1.0, 'Quality should be 0 > and <= 1.0 ({})'),
        'trimleft': ('trim_left', int, lambda x: x <= 0, 'Left trim value invalid ({})'),
        'trimright': ('trim_right', int, lambda x: x <= 0, 'Right trim value invalid ({})'),
        'trimtop': ('trim_top', int, lambda x: x <= 0, 'Top trim value invalid ({})'),
        'trimbottom': ('trim_bottom', int, lambda x: x <= 0, 'Bottom trim value invalid ({})'),
        'left': ('trim_left', int, lambda x: x <= 0, 'Left trim value invalid ({})'),
        'right': ('trim_right', int, lambda x: x <= 0, 'Right trim value invalid ({})'),
        'top': ('trim_top', int, lambda x: x <= 0, 'Top trim value invalid ({})'),
        'bottom': ('trim_bottom', int, lambda x: x <= 0, 'Bottom trim value invalid ({})'),
        'residual': ('residual', bool, lambda x: not isinstance(x, bool), 'Residual value invalid ({}). Must be 0, 1, T, F.'),
        'jitter': ('jitter', bool, lambda x: not isinstance(x, bool), 'Jitter value invalid ({}). Must be 0, 1, T, F.'),
        'skip': ('skip', bool, lambda x: not isinstance(x, bool), 'Skip value invalid ({}). Must be 0, 1, T, F.'),
        'shuffle': ('shuffle', bool, lambda x: not isinstance(x, bool), 'Shuffle value invalid ({}). Must be 0, 1, T, F.'),
        'data': ('data_path', str, lambda x: False, ''),
        'training': ('training_path', str, lambda x: False, ''),
        'validation': ('validation_path', str, lambda x: False, ''),
        'genepool': ('genepool_path', str, lambda x: False, '')
    }

    OPTIONS = parse_options(OPCODES)
    CONFIG, GENEPOOL, IMAGE_INFO = setup(OPTIONS)
    evolve(CONFIG, GENEPOOL, IMAGE_INFO)
