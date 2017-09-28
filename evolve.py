"""
Usage: evolve.py [option(s)] ...

    Evolves (hopefully) improved models. In each generation, the 5 best models
    are retained, and then 15 new models are evolved from them.

Options are:

    genepool=path       path to genepool file, default is {Data}/genepool.json
    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    lr=.nnn             set initial learning rate. Should be 0.001 or less. Default = 0.001
    quality=.nnn        fraction of the "best" tiles used in training (but not validation). Default is 1.0 (use all)
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

    Notes:

        Creates a "Darwinian-{options}-h5" temp file in {Data}/models. It is deleted before every model fit
        but then automatically created again by Keras.

        There are some hard-coded parameters at present, defined as constants at the start of evolve.py.

        See Modules/genomics.py for details on the structure of genes, codons and other genetic elements.

"""

from Modules.misc import oops, validate, terminate, set_docstring, printlog

import numpy as np
import sys
import os
import json
import random
import itertools

set_docstring(__doc__)

# Evolutionary constants (not saved in the JSON file)

MAX_POPULATION = 20         # Maximum size of population
MIN_POPULATION = 5          # Minimum size of population
EPOCHS = 10                 # Number of epochs to train
best_fitness = 0            # the best fitness we have found so far

# Fitness heuristic function, returns True if training should be aborted
# early.


def grim_reaper(fitness, current_epoch, max_epochs, last_fitness=None):

    if best_fitness >= 0:
        return False

    # if last_fitness is a value, our current fitness must be an improvement

    if last_fitness != None and fitness >= last_fitness:
        return True

    fitness = fitness / best_fitness

    # First epoch must be at least 75% as good, halfway we need about 87.5%, and
    # so on.

    requirement = 1.0 - 0.25 * (max_epochs - current_epoch) / (max_epochs - 1)

    return fitness < requirement

# Checkpoint state to genepool.json file


def checkpoint(path, population, graveyard, statistics, io):

    state = {'population': population,
             'graveyard': graveyard,
             'statistics': statistics,
             'io': io.asdict()
             }

    with open(path, 'w') as f:
        json.dump(state, f, indent=4)


if __name__ == '__main__':

    # Initialize defaults. Note that we trim 240 pixels off right and left, this is
    # because our default use case is 1440x1080 upconverted SD in a 1920x1080 box

    tile_width, tile_height, tile_border, epochs = 60, 60, 2, 10
    trim_left, trim_right, trim_top, trim_bottom = 240, 240, 0, 0
    black_level, quality = -1.0, 0.5
    jitter, shuffle, skip = 1, 1, 1
    lr = 0.001
    paths = {}

    # Order of options in this list can be important; if one option is a substring
    # of the other, the smaller one must come first.

    options = sorted(['genepool', 'width', 'height', 'border', 'training',
                      'validation', 'data', 'black',
                      'jitter', 'shuffle', 'skip', 'lr', 'quality',
                      'trimleft', 'trimright', 'trimtop', 'trimbottom',
                      'left', 'right', 'top', 'bottom'])

    # Parse options

    errors = False

    for option in sys.argv[1:]:

        opvalue = option.split('=', maxsplit=1)

        if len(opvalue) == 1:
            errors = oops(errors, True, 'Invalid option ({})', option)
            continue

        op, value = [s.lower() for s in opvalue]
        _, valuecase = opvalue

        # convert boolean arguments to integer

        value = '1' if 'true'.startswith(value) else value
        value = '0' if 'false'.startswith(value) else value

        # convert value to integer and float with default -1

        try:
            fnum = float(value)
        except ValueError:
            fnum = -1.0
        vnum = int(fnum)

        # Match option, make sure it isn't ambiguous.

        opmatch = [s for s in options if s.startswith(op)]

        if len(opmatch) == 0 or len(opmatch) > 1 and opmatch[0] != op:
            errors = oops(errors, True, '{} option ({})',
                          ('Unknown' if len(opmatch) == 0 else 'Ambiguous', op))
            continue

        # PU: refactored more simply. Used continues (above) to reduce nesting depth, then a
        # validation passthrough routine to collapse contents of each if block to a single
        # statement.

        op = opmatch[0]

        if op == 'width':
            tile_width, errors = validate(
                errors, vnum, vnum <= 0, 'Tile width invalid ({})', option)
        elif op == 'height':
            tile_height, errors = validate(
                errors, vnum, vnum <= 0, 'Tile height invalid ({})', option)
        elif op == 'border':
            tile_border, errors = validate(
                errors, vnum, vnum <= 0, 'Tile border invalid ({})', option)
        elif op == 'black' and value != 'auto':
            black_level, errors = validate(
                errors, fnum, fnum <= 0, 'Black level invalid ({})', option)
        elif op == 'lr':
            lr, errors = validate(
                errors, fnum, errors, fnum <= 0.0 or fnum > 0.01,
                'Learning rate should be 0 > and <= 0.01 ({})', option)
        elif op == 'quality':
            quality, errors = validate(
                errors, fnum, errors, fnum <= 0.0 or fnum > 1.0,
                'Quality should be 0 > and <= 1.0 ({})', option)
        elif op == 'trimleft' or op == 'left':
            trim_left, errors = validate(
                errors, vnum, vnum <= 0, 'Left trim value invalid ({})', option)
        elif op == 'trimright' or op == 'right':
            trim_right, errors = validate(
                errors, vnum, vnum <= 0, 'Right trim value invalid ({})', option)
        elif op == 'trimtop' or op == 'top':
            trim_top, errors = validate(
                errors, vnum, vnum <= 0, 'Top trim value invalid ({})', option)
        elif op == 'trimbottom' or op == 'bottom':
            trim_bottom, errors = validate(
                errors, vnum, vnum <= 0, 'Bottom trim value invalid ({})', option)
        elif op == 'jitter':
            jitter, errors = validate(
                errors, vnum, vnum != 0 and vnum != 1,
                'Jitter value invalid ({}). Must be 0, 1, T, F.', option)
        elif op == 'skip':
            skip, errors = validate(
                errors, vnum, vnum != 0 and vnum != 1,
                'Skip value invalid ({}). Must be 0, 1, T, F.', option)
        elif op == 'shuffle':
            shuffle, errors = validate(
                errors, vnum, vnum != 0 and vnum != 1,
                'Shuffle value invalid ({}). Must be 0, 1, T, F.', option)
        elif op in ['data', 'training', 'validation', 'genepool']:
            paths[op] = value

    terminate(errors)

    # Set remaining defaults

    if 'data' not in paths:
        paths['data'] = 'Data'

    dpath = paths['data']

    if 'genepool' not in paths:
        paths['genepool'] = os.path.join(dpath, 'genepool.json')

    if 'training' not in paths:
        paths['training'] = os.path.join(dpath, 'train_images', 'training')

    if 'validation' not in paths:
        paths['validation'] = os.path.join(dpath, 'train_images', 'validation')

    # set up our initial state

    genepool = {}
    poolpath = paths['genepool']

    if os.path.exists(poolpath):
        if os.path.isfile(poolpath):
            print('Loading existing genepool')
            try:
                with open(poolpath, 'r') as f:
                    genepool = json.load(f)
            except:
                print('Could not load and parse json. Did you edit "population" and forget to delete the trailing comma?')
                errors = True
        else:
            errors = oops(
                errors, True, 'Genepool path is not a reference to a file ({})', poolpath)
    else:
        errors = oops(errors, not os.access(os.path.dirname(
            poolpath), os.W_OK), 'Genepool folder is not writeable ({})', poolpath)

    terminate(errors, False)

    # At this point, we may have loaded a genepool that overrides our paths

    if 'io' in genepool:
        for path in ['data', 'training', 'validation']:
            if path + '_path' in genepool['io']:
                paths[path] = genepool['io'][path + '_path']

    # Validation and error checking

    import Modules.frameops as frameops

    image_paths = ['training', 'validation']
    sub_folders = ['Alpha', 'Beta']
    image_info = [[None, None], [None, None]]

    for fc, f in enumerate(image_paths):
        for sc, s in enumerate(sub_folders):
            image_info[fc][sc] = frameops.image_files(
                os.path.join(paths[f], sub_folders[sc]), True)

    for f in [0, 1]:
        for s in [0, 1]:
            errors = oops(
                errors, image_info[f][s] == None, '{} images folder does not exist', image_paths[f] + '/' + sub_folders[s])

    terminate(errors, False)

    for f in [0, 1]:
        for s in [0, 1]:
            errors = oops(errors, len(
                image_info[f][s]) == 0, '{} images folder does not contain any images', image_paths[f] + '/' + sub_folders[s])
            errors = oops(errors, len(
                image_info[f][s]) > 1, '{} images folder contains more than one type of image', image_paths[f] + '/' + sub_folders[s])

    terminate(errors, False)

    for f in [0, 1]:
        errors = oops(errors, len(image_info[f][0][0]) != len(
            image_info[f][1][0]), '{} images folders have different numbers of images', image_paths[f])

    terminate(errors, False)

    for f in [0, 1]:
        for f1, f2 in zip(image_info[f][0][0], image_info[f][1][0]):
            f1, f2 = os.path.basename(f1), os.path.basename(f2)
            errors = oops(
                errors, f1 != f2, '{} images folders do not have identical image filenames ({} vs {})', (image_paths[f], f1, f2))
            terminate(errors, False)

    # Check sizes, even tiling here.

    test_files = [[image_info[f][g][0][0] for g in [0, 1]] for f in [0, 1]]
    test_images = [[frameops.imread(image_info[f][g][0][0])
                    for g in [0, 1]] for f in [0, 1]]

    # What kind of file is it? Do I win an award for the most brackets?

    img_suffix = os.path.splitext(image_info[0][0][0][0])[1][1:]

    # Check that the Beta tiles are the same size.

    s1, s2 = np.shape(test_images[0][1]), np.shape(test_images[1][1])
    errors = oops(errors, s1 != s2, 'Beta training and evaluation images do not have identical size ({} vs {})',
                  (s1, s2))

    # Warn if we do have some differences between Alpha and Beta sizes

    for f in [0, 1]:
        s1, s2 = np.shape(test_images[f][0]), np.shape(test_images[f][1])
        if s1 != s2:
            print('Warning: {} Alpha and Beta images are not the same size ({} vs {}). Will attempt to scale Alpha images.'.format(
                image_paths[f].title(), s1, s2))

    terminate(errors, False)

    # Only check the size of the Beta output for proper configuration, since Alpha tiles will
    # be scaled as needed.

    errors = oops(errors, len(s1) !=
                  3 or s2[2] != 3, 'Images have improper shape ({0})', str(s1))

    terminate(errors, False)

    image_width, image_height = s2[1], s2[0]
    trimmed_width = image_width - (trim_left + trim_right)
    trimmed_height = image_height - (trim_top + trim_bottom)

    errors = oops(errors, trimmed_width <= 0,
                  'Trimmed images have invalid width ({} - ({} + {}) <= 0)', (s1[0], trim_left, trim_right))
    errors = oops(errors, trimmed_width <= 0,
                  'Trimmed images have invalid height ({} - ({} + {}) <= 0)', (s1[1], trim_top, trim_bottom))

    terminate(errors, False)

    errors = oops(errors, (trimmed_width % tile_width) != 0,
                  'Trimmed images do not evenly tile horizontally ({} % {} != 0)', (trimmed_width, tile_width))
    errors = oops(errors, (trimmed_height % tile_height) != 0,
                  'Trimmed images do not evenly tile vertically ({} % {} != 0)', (trimmed_height, tile_height))

    terminate(errors, False)

    # Attempt to automatically figure out the border color black level, by finding the minimum pixel value in one of our
    # sample images. This will definitely work if we are processing 1440x1080 4:3 embedded in 1920x1080 16:19 images

    if black_level < 0:
        black_level = np.min(test_images[0][0])

    terminate(errors, False)

    # Initialize missing values in genepool

    from Modules.modelio import ModelIO

    if 'population' in genepool:
        population = genepool['population']
    else:
        print('Initializing population...')
        population = ["conv_f64_k9_elu-conv_f32_k1_elu-out_k5_elu",
                      "conv_f64_k9_elu-conv_f32_k1_elu-avg_f32_k135_d3_elu-out_k5_elu",
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

    # Over-ride defaults/options with contents of genepool.json, if any...

        tile_height = io['base_tile_height']
        border = io['border']
        border_mode = io['border_mode']
        black_level = io['black_level']
        trim_top = io['trim_top']
        trim_bottom = io['trim_bottom']
        trim_left = io['trim_left']
        trim_right = io['trim_right']
        jitter = io['jitter']
        shuffle = io['shuffle']
        skip = io['skip']
        quality = io['quality']
        epochs = io['epochs']
        lr = io['lr']
        paths = io['paths']

    # Initialize ModelIO structure

    io = ModelIO(model_type='Darwinian',
                 image_width=image_width, image_height=image_height,
                 base_tile_width=tile_width, base_tile_height=tile_height,
                 channels=3,
                 border=tile_border,
                 border_mode='edge',
                 batch_size=16,
                 black_level=black_level,
                 trim_top=trim_top, trim_bottom=trim_bottom,
                 trim_left=trim_left, trim_right=trim_right,
                 jitter=jitter, shuffle=shuffle, skip=skip,
                 quality=quality,
                 img_suffix=img_suffix,
                 paths=paths,
                 epochs=EPOCHS,
                 lr=lr
                 )

    # Remind user what we're about to do.

    print('          Genepool : {}'.format(paths['genepool']))
    print('        Tile Width : {}'.format(tile_width))
    print('       Tile Height : {}'.format(tile_height))
    print('       Tile Border : {}'.format(tile_border))
    print('    Min Population : {}'.format(MIN_POPULATION))
    print('    Max Population : {}'.format(MAX_POPULATION))
    print('   Epochs to train : {}'.format(EPOCHS))
    print('    Data root path : {}'.format(paths['data']))
    print('   Training Images : {}'.format(paths['training']))
    print(' Validation Images : {}'.format(paths['validation']))
    print('  Input Image Size : {} x {}'.format(s1[1], s1[0]))
    print('          Trimming : Top={}, Bottom={}, Left={}, Right={}'.format(
        trim_top, trim_bottom, trim_left, trim_right))
    print(' Output Image Size : {} x {}'.format(trimmed_width, trimmed_height))
    print(' Training Set Size : {}'.format(len(image_info[0][0][0])))
    print('   Valid. Set Size : {}'.format(len(image_info[1][0][0])))
    print('       Black level : {}'.format(black_level))
    print('            Jitter : {}'.format(jitter == 1))
    print('           Shuffle : {}'.format(shuffle == 1))
    print('              Skip : {}'.format(skip == 1))
    print('           Quality : {}'.format(quality))
    print('')

    checkpoint(poolpath, population, graveyard, statistics, io)

    # Evolve the genepool

    import Modules.models as models
    import Modules.genomics as genomics

    # Repeat until program terminated.

    while True:
        # Get the best fitness value so far to use as a training goal, and
        # the worst fitness so we can trigger a repopulation if we get a
        # good genome during training.

        best_fitness = population[0][1] if type(population[0]) is list else 0.0
        worst_fitness = max([p[1] if type(p) is list else best_fitness for p in population])

        # Train all the untrained genomes in the population

        for i, genome in enumerate(population):
            if type(genome) is not list:
                # Delete the model and state files (if any) so we start with
                # a fresh slate
                for p in (io.model_path, io.state_path):
                    if os.path.isfile(p):
                        os.remove(p)

                # Build a model for the organism, train the model, and record the results

                population[i] = [ genome,
                                  genomics.fitness(genome, io, apoptosis=grim_reaper)
                                ]

                # Generate all sorts of statistics on various genome combinations. Later we
                # may use them to optimize evolution a it.

                statistics = genomics.ligate(statistics, population[i][0], population[i][1])

                checkpoint(poolpath, population, graveyard, statistics, io)

                # If the model we just trained has better fitness than the worst fitness of
                # the previously trained genomes, exit early (which will generate a new
                # population using a "better" genepool)

                if population[i][1] < worst_fitness:
                    break

        # Remove untrained populations

        population = [p for p in population if type(p) is list]

        # If our population has expanded past the minimum limit, cut back to the best.

        if len(population) > MIN_POPULATION:
            printlog('Trimming population to {}...'.format(MIN_POPULATION))
            population.sort(key=lambda org: org[1])
            graveyard.extend([p[0] for p in population[MIN_POPULATION:]])
            population = population[:MIN_POPULATION]
            checkpoint(poolpath, population, graveyard, statistics, io)
            graveyard.sort()

        # Expand the population to the maximum size.

        parents, children = [p[0] for p in population], []

        printlog('Creating new children...')

        while len(children) < (MAX_POPULATION - len(parents)):
            parent, conjugate = [p for p in random.sample(parents, 2)]
            child = '-'.join(genomics.mutate(parent, conjugate, best_fitness=best_fitness, statistics=statistics))
            if child not in parents and child not in children and child not in graveyard:
                children.append(child)
            else:
                printlog('Duplicate genome rejected...')

        population.extend(children)

        checkpoint(poolpath, population, graveyard, statistics, io)
