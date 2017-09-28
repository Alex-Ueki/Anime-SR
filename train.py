# pylint: disable=C0301
# Line too long
"""
Usage: train.py [option(s)] ...

    Trains a model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR
        PUPSR
        GPUSR
        TestModels : Currently ELU versions of all models

Options are:

    type=model          model type, default is BasicSR
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
    jitter=1|0|T|F      include jittered tiles (offset by half a tile across&down) when training; default=True
    skip=1|0|T|F        randomly skip 0-3 tiles between tiles when training; default=True
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
"""

import sys
import os

import numpy as np

from Modules.misc import oops, validate, terminate, set_docstring

set_docstring(__doc__)

if __name__ == '__main__':

    # Initialize defaults. Note that we trim 240 pixels off right and left, this is
    # because our default use case is 1440x1080 upconverted SD in a 1920x1080 box

    model_type = 'BasicSR'
    tile_width, tile_height, tile_border, epochs, run_epochs = 60, 60, 2, 100, -1
    trim_left, trim_right, trim_top, trim_bottom = 240, 240, 0, 0
    black_level, quality = -1.0, 1.0
    jitter, shuffle, skip, residual = 1, 1, 1, 0
    initial_lr = -1.0 # PU: if no learning rate manually specified, model default will be used
    verbose, bargraph = True, True
    paths = {}

    # Parse options. Order of options in the list is important if one option is a substring
    # of the other; the smaller one must come first.

    options = sorted(['type', 'width', 'height', 'border', 'training',
                      'validation', 'model', 'data', 'state', 'black',
                      'jitter', 'shuffle', 'skip', 'lr', 'quality',
                      'trimleft', 'trimright', 'trimtop', 'trimbottom',
                      'epochs', 'epochs+', 'verbose', 'bargraph', 'residual'])

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

        if len(opmatch) ==0 or len(opmatch) > 1 and opmatch[0] != op:
            errors = oops(errors, True, '{} option ({})',
                          ('Unknown' if len(opmatch) == 0 else 'Ambiguous', op))
            continue

        op = opmatch[0]

        if op == 'type':
            model_type = valuecase
        elif op == 'width':
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
            initial_lr, errors = validate(
                errors, fnum, errors, initial_lr <= 0.0 or initial_lr > 0.01,
                'Learning rate should be 0 > lr <= 0.01 ({})', option)
        elif op == 'quality':
            quality, errors = validate(
                errors, fnum, errors, quality <= 0.0 or quality > 1.0,
                'Quality should be 0 > q <= 1.0 ({})', option)
        elif op == 'epochs':
            epochs, errors = validate(
                errors, vnum, vnum <= 0, 'Max epoch count invalid ({})', option)
        elif op == 'epochs+':
            run_epochs, errors = validate(
                errors, vnum, vnum <= 0, 'Run epoch count invalid ({})', option)
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
        elif op == 'residual':
            residual, errors = validate(
                errors, vnum, vnum != 0 and vnum != 1,
                'Residual value invalid ({}). Must be 0, 1, T, F.', option)
        elif op == 'verbose':
            verbose = vnum
            errors = oops(errors, vnum != 0 and vnum != 1,
                          'Verbose value invalid ({}). Must be 0, 1, T, F.', option)
        elif op == 'bargraph':
            bargraph = vnum
            errors = oops(errors, vnum != 0 and vnum != 1,
                          'Bargraph value invalid ({}). Must be 0, 1, T, F.', option)
        elif op in ['data', 'training', 'validation', 'model', 'state']:
            paths[op] = value


    terminate(errors)

    # Set remaining defaults

    if 'data' not in paths:
        paths['data'] = 'Data'

    dpath = paths['data']

    if 'training' not in paths:
        paths['training'] = os.path.join(dpath, 'train_images', 'training')

    if 'validation' not in paths:
        paths['validation'] = os.path.join(dpath, 'train_images', 'validation')

    # Remind user what we're about to do.

    print('             Model : {}'.format(model_type))
    print('        Tile Width : {}'.format(tile_width))
    print('       Tile Height : {}'.format(tile_height))
    print('       Tile Border : {}'.format(tile_border))
    print('        Max Epochs : {}'.format(epochs))
    print('        Run Epochs : {}'.format(run_epochs))
    print('    Data root path : {}'.format(paths['data']))
    print('   Training Images : {}'.format(paths['training']))
    print(' Validation Images : {}'.format(paths['validation']))

    # Validation and error checking

    import Modules.frameops as frameops

    image_paths = ['training', 'validation']
    sub_folders = ['Alpha', 'Beta']
    image_info = [[[], []], [[], []]]

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
            print('Warning: {} Alpha and Beta images are not the same size ({} vs {}). Will attempt to scale Alpha images.'.format(image_paths[f].title(),s1,s2))

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

    # Since we've gone to the trouble of reading in all the path data, let's make it available to our models for reuse

    for fc, f in enumerate(image_paths):
        for sc, s in enumerate(sub_folders):
            paths[f + '.' + s] = image_info[fc][sc]

    # Only at this point can we set default model and state filenames because that depends on image type

    if 'model' not in paths:
        name = '{}-{}-{}-{}-{}.h5'.format(model_type, tile_width, tile_height, tile_border, img_suffix)
        paths['model'] = os.path.abspath(os.path.join(dpath, 'models', name))

    if 'state' not in paths:
        name = '{}-{}-{}-{}-{}_state.json'.format(model_type, tile_width, tile_height, tile_border, img_suffix)
        paths['state'] = os.path.abspath(os.path.join(dpath, 'models', name))

    tpath = os.path.dirname(paths['state'])
    errors = oops(errors, not os.path.exists(tpath),
                  'Model state path ({}) does not exist'.format(tpath))

    tpath = os.path.dirname(paths['model'])
    errors = oops(errors, not os.path.exists(tpath),
                  'Model path ({}) does not exist'.format(tpath))

    terminate(errors, False)

    print('        Model File : {}'.format(paths['model']))
    print('  Model State File : {}'.format(paths['state']))
    print('  Image dimensions : {} x {}'.format(s1[1], s1[0]))
    print('          Trimming : Top={}, Bottom={}, Left={}, Right={}'.format(
        trim_top, trim_bottom, trim_left, trim_right))
    print('Trimmed dimensions : {} x {}'.format(trimmed_width, trimmed_height))
    print('       Black level : {}'.format(black_level))
    print('            Jitter : {}'.format(jitter == 1))
    print('           Shuffle : {}'.format(shuffle == 1))
    print('              Skip : {}'.format(skip == 1))
    print('          Residual : {}'.format(residual == 1))
    print('           Quality : {}'.format(quality))
    print('')

    # Train the model

    from Modules.modelio import ModelIO
    import Modules.models as models

    if model_type.lower() == 'all':
        model_list = models.MODELS
    elif model_type.lower() == 'test':
        model_list = models.TESTMODELS
    else:
        model_list = [model_type]

    for model in model_list:

        # Put proper model name in the model and state paths

        for entry in ['model', 'state']:
            path = paths[entry]
            folder_name, file_name = os.path.split(path)
            file_name_parts = file_name.split('-')
            file_name_parts[0] = model
            file_name = '-'.join(file_name_parts)
            path = os.path.join(folder_name, file_name)
            paths[entry] = path

        # Configure model IO

        io = ModelIO(model_type=model,
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
                     residual=residual,
                     quality=quality,
                     img_suffix=img_suffix,
                     paths=paths)

        # Create and fit model (best model state will be automatically saved)

        sr = models.MODELS[model](io, verbose=verbose, bargraph=bargraph)

        # If no initial_lr is specified, the default of 0.0 means that the
        # model's default learning rate will be used

        if initial_lr > 0.0:
            print('Learning Rate reset to {}'.format(initial_lr))
            sr.set_lr(initial_lr)

        config = sr.get_config()
        print('Model configuration:')
        for key in config:
            print('{:>18s} : {}'.format(key, config[key]))

        # PU: Cannot adjust ending epoch number until we load the model state,
        # which does not happen until we fit(). So we have to pass both
        # the max epoch and the run # of epochs.

        sr.fit(max_epochs=epochs, run_epochs=run_epochs)

    print('')
    print('Training completed...')
