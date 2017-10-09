# pylint: disable=C0301
# Line too long
"""
Usage: train.py [option(s)] ...

    Trains a model. See Modules/models.py for sample model types.

Options are:

    type=model          model type, default is BasicSR. Model type can also be a genome (see below)
    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    epochs=nnn          max epoch count, default=100. See below for more details
    epochs+=nnn         run epoch count, default=None. Overrides epochs=nnn
    lr=.nnn             set initial learning rate, default = use model's current learning rate. Should be 0.001 or less
    quality=.nnn        fraction of the "best" tiles used in training. Default is 1.0 (use all tiles)
    residual=1|0|T|F    have the model train using residual images. Default is false.
    black=auto|nnn      black level (0..1) for image border pixels, default=auto (use blackest pixel in first image)
    trimleft=nnn        pixels to trim on image left edge, default = 240
    trimright=nnn       pixels to trim on image right edge, default = 240
    trimtop=nnn         pixels to trim on image top edge, default = 0
    trimbottom=nnn      pixels to trim on image bottom edge, default = 0
    jitter=1|0|T|F      include jittered tiles (offset by half a tile across&down) when training; default=False
    skip=1|0|T|F        randomly skip 0-3 tiles between tiles when training; default=False
    shuffle=1|0|T|F     shuffle tiles into random order when training; default=True
    data=path           path to the main data folder, default = Data
    training=path       path to training folder, default = {Data}/train_images/training
    validation=path     path to validation folder, default = {Data}/train_images/validation
    model=path          path to trained model file, default = {Data}/models/{model}-{width}-{height}-{border}-{img_type}.h5
    state=path          path to state file, default = {Data}/models/{model}-{width}-{height}-{border}-{img_type}_state.json
    verbose=1|0|T|F     display verbose training output, default = True
    bargraph=1|0|T|F    display bargraph of training progress, default = True

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK).

    You can terminate a training session with ^C, and then resume training by reissuing the same command. The model's state
    is completely stored in the .h5 file, and the training state is in the _state.json file.

    The epochs value is the maximum number of epochs that will be trained **over multiple sessions**. So if you have
    previously trained a model for 50 epochs, epochs=75 would mean the model trains for 25 additional epochs. Alternately,
    you could specify epochs+=25 to limit the current training run to 25 epochs.

    If the type is not one of the standard defined models, then it is checked to see if it is a valid genome.
    If so, the model and state files (if not set) are searched for in {Data}/models and {Data}/models/genes
"""

import os
import json
import numpy as np

from Modules.misc import oops, terminate, set_docstring, parse_options
from Modules.modelio import ModelIO
import Modules.genomics as genomics
import Modules.models as models
import Modules.frameops as frameops


set_docstring(__doc__)


def setup(options):
    """ Create model configuration """

    # set up our initial state

    errors = False
    config = ModelIO(options)
    dpath = config.paths['data']

    # Fix up model and state paths if genetic model and not explicitly specified in options

    if config.model_type not in models.MODELS:
        errors = oops(errors,
                      genomics.build_model(config.model_type)[0] is None,
                      '{} is an invalid model',
                      config.model_type)
        terminate(errors, False)

        if 'model_path' not in options:
            mpath = os.path.join(dpath, 'models', config.model_type + '.h5')
            if os.path.exists(mpath):
                config.paths['model'] = mpath
            else:
                mpath = os.path.join(dpath, 'models', 'genes', config.model_type + '.h5')
                if os.path.exists(mpath):
                    config.paths['model'] = mpath

        if 'state_path' not in options:
            spath = os.path.join(dpath, 'models', config.model_type + '.json')
            if os.path.exists(spath):
                config.paths['state'] = spath
            else:
                spath = os.path.join(dpath, 'models', 'genes', config.model_type + '.json')
                if os.path.exists(spath):
                    config.paths['state'] = mpath

    # Validation and error checking

    image_paths = ['training', 'validation']
    sub_folders = ['Alpha', 'Beta']
    image_info = [[[], []], [[], []]]

    for fcnt, fpath in enumerate(image_paths):
        for scnt, spath in enumerate(sub_folders):
            image_info[fcnt][scnt] = frameops.image_files(os.path.join(config.paths[fpath], spath), True)

    for fcnt in [0, 1]:
        for scnt in [0, 1]:
            errors = oops(errors,
                          image_info[fcnt][scnt] is None,
                          '{} images folder does not exist',
                          image_paths[fcnt] + '/' + sub_folders[scnt])

    terminate(errors, False)

    for fcnt in [0, 1]:
        for scnt in [0, 1]:
            errors = oops(errors,
                          not image_info[fcnt][scnt],
                          '{} images folder does not contain any images',
                          image_paths[fcnt] + '/' + sub_folders[scnt])
            errors = oops(errors,
                          len(image_info[fcnt][scnt]) > 1,
                          '{} images folder contains more than one type of image',
                          image_paths[fcnt] + '/' + sub_folders[scnt])

    terminate(errors, False)

    for fcnt in [0, 1]:
        errors = oops(errors,
                      len(image_info[fcnt][0][0]) != len(image_info[fcnt][1][0]),
                      '{} images folders have different numbers of images',
                      image_paths[fcnt])

    terminate(errors, False)

    for fcnt in [0, 1]:
        for fpath1, fpath2 in zip(image_info[fcnt][0][0], image_info[fcnt][1][0]):
            fpath1, fpath2 = os.path.basename(fpath1), os.path.basename(fpath2)
            errors = oops(errors,
                          fpath1 != fpath2,
                          '{} images folders do not have identical image filenames ({} vs {})',
                          (image_paths[fcnt], fpath1, fpath2))
            terminate(errors, False)

    # Check sizes, even tiling here.

    #test_files = [[image_info[f][g][0][0] for g in [0, 1]] for f in [0, 1]]

    test_images = [[frameops.imread(image_info[f][g][0][0])
                    for g in [0, 1]] for f in [0, 1]]

    # Check that the Beta tiles are the same size.

    size1, size2 = np.shape(test_images[0][1]), np.shape(test_images[1][1])
    errors = oops(errors,
                  size1 != size2,
                  'Beta training and evaluation images do not have identical size ({} vs {})',
                  (size1, size2))

    # Warn if we do have some differences between Alpha and Beta sizes

    for fcnt in [0, 1]:
        size1, size2 = np.shape(test_images[fcnt][0]), np.shape(test_images[fcnt][1])
        if size1 != size2:
            print('Warning: {} Alpha and Beta images are not the same size.'.format(image_paths[fcnt].title()))

    terminate(errors, False)

    # Only check the size of the Beta output for proper configuration, since Alpha tiles will
    # be scaled as needed.

    errors = oops(errors,
                  len(size2) != 3 or size2[2] != 3,
                  'Images have improper shape ({})',
                  str(size2))

    terminate(errors, False)

    image_width, image_height = size2[1], size2[0]
    trimmed_width = image_width - (config.trim_left + config.trim_right)
    trimmed_height = image_height - (config.trim_top + config.trim_bottom)

    errors = oops(errors,
                  trimmed_width <= 0,
                  'Trimmed images have invalid width ({} - ({} + {}) <= 0)',
                  (size1[0], config.trim_left, config.trim_right))
    errors = oops(errors,
                  trimmed_height <= 0,
                  'Trimmed images have invalid height ({} - ({} + {}) <= 0)',
                  (size1[1], config.trim_top, config.trim_bottom))

    terminate(errors, False)

    errors = oops(errors,
                  (trimmed_width % config.base_tile_width) != 0,
                  'Trimmed images do not evenly tile horizontally ({} % {} != 0)',
                  (trimmed_width, config.tile_width))
    errors = oops(errors,
                  (trimmed_height % config.base_tile_height) != 0,
                  'Trimmed images do not evenly tile vertically ({} % {} != 0)',
                  (trimmed_height, config.tile_height))

    terminate(errors, False)

    # Attempt to automatically figure out the border color black level, by finding the minimum pixel value in one of our
    # sample images. This will definitely work if we are processing 1440x1080 4:3 embedded in 1920x1080 16:19 images.
    # Write back any change into config.

    if config.black_level < 0:
        config.black_level = np.min(test_images[0][0])
        config.config['black_level'] = config.black_level

    # Since we've gone to the trouble of reading in all the path data, let's make it available to our models for reuse

    for fcnt, fpath in enumerate(image_paths):
        for scnt, spath in enumerate(sub_folders):
            config.paths[fpath + '.' + spath] = image_info[fcnt][scnt]

    # Only at this point can we set default model and state filenames because that depends on image type.
    # If we are training a genetic model, then these will already have been set.

    if 'model' not in config.paths:
        name = '{}{}-{}-{}-{}-{}.h5'.format(
            config.model_type,
            '-R' if config.residual else '',
            config.tile_width,
            config.tile_height,
            config.border,
            config.img_suffix)
        config.paths['model'] = os.path.abspath(os.path.join(config.paths['data'], 'models', name))

    if 'state' not in config.paths:
        config.paths['state'] = config.paths['model'][:-3] + '.json'

    tpath = os.path.dirname(config.paths['state'])
    errors = oops(errors,
                  not os.path.exists(tpath),
                  'Model state path ({}) does not exist',
                  tpath)

    tpath = os.path.dirname(config.paths['model'])
    errors = oops(errors,
                  not os.path.exists(tpath),
                  'Model path ({}) does not exist',
                  tpath)

    # If we do have an existing json state, load it and override

    statepath = config.paths['state']
    print(statepath)
    if os.path.exists(statepath):
        print('exists')
        if os.path.isfile(statepath):
            print('Loading existing Model state')
            try:
                with open(statepath, 'r') as jsonfile:
                    state = json.load(jsonfile)

                    # PU: Temp hack to change 'io' key to 'config'

                    if 'io' in state:
                        state['config'] = state['io']
                        del state['io']

            except json.decoder.JSONDecodeError:
                print('Could not parse json. Did you forget to delete a trailing comma?')
                errors = True
        else:
            errors = oops(errors,
                          True,
                          'Model state path is not a reference to a file ({})',
                          statepath)

        for setting in state['config']:
            if setting not in config.config or config.config[setting] != state['config'][setting]:
                config.config[setting] = state['config'][setting]

        # There are a couple of options that override saved configurations.

        config.config['run_epochs'] = options['run_epochs'] if 'run_epochs' in options else config.config['run_epochs']
        config.config['learning_rate'] = options['learning_rate'] if 'learning_rate' in options else config.config['learning_rate']

        # Reload config with possibly changed settings

        config = ModelIO(config.config)

    terminate(errors, False)

    # Remind user what we're about to do.

    print('             Model : {}'.format(config.model_type))
    print('        Tile Width : {}'.format(config.base_tile_width))
    print('       Tile Height : {}'.format(config.base_tile_height))
    print('       Tile Border : {}'.format(config.border))
    print('        Max Epochs : {}'.format(config.epochs))
    print('        Run Epochs : {}'.format(config.run_epochs))
    print('    Data root path : {}'.format(config.paths['data']))
    print('   Training Images : {}'.format(config.paths['training']))
    print(' Validation Images : {}'.format(config.paths['validation']))
    print('        Model File : {}'.format(config.paths['model']))
    print('  Model State File : {}'.format(config.paths['state']))
    print('  Image dimensions : {} x {}'.format(config.image_width, config.image_height))
    print('          Trimming : Top={}, Bottom={}, Left={}, Right={}'.format(
        config.trim_top, config.trim_bottom, config.trim_left, config.trim_right))
    print('Trimmed dimensions : {} x {}'.format(config.trimmed_width, config.trimmed_height))
    print('       Black level : {}'.format(config.black_level))
    print('            Jitter : {}'.format(config.jitter == 1))
    print('           Shuffle : {}'.format(config.shuffle == 1))
    print('              Skip : {}'.format(config.skip == 1))
    print('          Residual : {}'.format(config.residual == 1))
    print('     Learning Rate : {}'.format(config.learning_rate))
    print('           Quality : {}'.format(config.quality))
    print('')

    return config

def train(config, options):
    """ Train the model """



    if config.model_type.lower() == 'all':
        model_list = models.MODELS
    elif config.model_type.lower() == 'test':
        config.model_list = models.TESTMODELS
    else:
        model_list = [config.model_type]

    for model in model_list:

        config.model_type = model

        # Put proper model name in the model and state paths (if not a genetic
        # model), then create the model

        if model in models.MODELS:
            for entry in ['model', 'state']:
                path = config.paths[entry]
                folder_name, file_name = os.path.split(path)
                file_name_parts = file_name.split('-')
                file_name_parts[0] = model
                file_name = '-'.join(file_name_parts)
                path = os.path.join(folder_name, file_name)
                config.paths[entry] = path

            cur_model = models.MODELS[model](config)
        else:
            cur_model = models.BaseSRCNNModel(model, config)
            cur_model.model = genomics.build_model(model)

        # Create and fit model (best model state will be automatically saved)

        cur_config = cur_model.get_config()
        print('Model configuration:')
        for key in cur_config:
            print('{:>18s} : {}'.format(key, cur_config[key]))

        # If learning rate explicitly specified in options, reset it.

        if 'learning_rate' in options:
            print('Learning Rate reset to {}'.format(options['learning_rate']))
            cur_model.set_lr(options['learning_rate'])

        # PU: Cannot adjust ending epoch number until we load the model state,
        # which does not happen until we fit(). So we have to pass both
        # the max epoch and the run # of epochs.

        cur_model.fit(max_epochs=config.epochs, run_epochs=config.run_epochs)

    print('')
    print('Training completed...')
    exit(0)

if __name__ == '__main__':

    # The command-line options for the tool

    OPCODES = {
        'type': ('model_type', str, lambda x: False, ''),
        'width': ('base_tile_width', int, lambda x: x <= 0, 'Tile width invalid ({})'),
        'height': ('base_tile_height', int, lambda x: x <= 0, 'Tile height invalid ({})'),
        'border': ('border', int, lambda x: x <= 0, 'Tile border invalid ({})'),
        'black': ('black_level', float, lambda x: False, 'Black level invalid ({})'),
        'lr': ('learning_rate', float, lambda x: x <= 0.0 or x > 0.01, 'Learning rate should be 0 > and <= 0.01 ({})'),
        'quality': ('quality', float, lambda x: x <= 0.0 or x > 1.0, 'Quality should be 0 > and <= 1.0 ({})'),
        'epochs': ('epochs', int, lambda x: x <= 0, 'Max epoch count invalid ({})'),
        'epochs+': ('run_epochs', int, lambda x: x <= 0, 'Run epoch count invalid ({})'),
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
        'verbose': ('verbose', bool, lambda x: not isinstance(x, bool), 'Verbose value invalid ({}). Must be 0, 1, T, F.'),
        'bargraph': ('bargraph', bool, lambda x: not isinstance(x, bool), 'Bargraph value invalid ({}). Must be 0, 1, T, F.'),
        'data': ('data_path', str, lambda x: False, ''),
        'training': ('training_path', str, lambda x: False, ''),
        'validation': ('validation_path', str, lambda x: False, ''),
        'model': ('model_path', str, lambda x: False, ''),
        'state': ('state_path', str, lambda x: False, ''),
    }

    OPTIONS = parse_options(OPCODES)
    CONFIG = setup(OPTIONS)
    train(CONFIG, OPTIONS)
