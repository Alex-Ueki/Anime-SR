"""

Usage: train.py [option(s)] ...

    Trains a model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    type=model          model type, default is BasicSR
    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    epochs=nnn          epoch size, default=9999
    lr=.nnn             set initial learning rate, default = use model's current learning rate. Should be 0.001 or less.
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

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

"""

from Modules.modelio import ModelIO
import Modules.models as models
import Modules.frameops as frameops

import numpy as np
import sys
import os

# If is_error is true, display message and optionally end the run.
# return updated error_state


def oops(error_state, is_error, msg, value=0, end_run=False):

    if is_error:
        # Have to handle the single/multiple argument case.
        # if we pass format a simple string using *value it
        # gets treated as a list of individual characters.
        if type(value) in (list, tuple):
            print('Error: ' + msg.format(*value))
        else:
            print('Error: ' + msg.format(value))
        if end_run:
            terminate(True)

    return error_state or is_error

# Terminate run if oops errors have been encountered.
# I have already done penance for this pun.


def terminate(sarah_connor, verbose=True):
    if sarah_connor:
        if verbose:
            print("""
Usage: train.py [option(s)] ...

    Trains a model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    type=model          model type, default is BasicSR
    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    epochs=nnn          epoch size, default=100
    lr=.nnn             set initial learning rate, default = use model's current learning rate. Should be 0.001 or less.
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
    state=path          path to state file, default = {Data}/models/{model}-{width}-{height}-{border}-{img_type}_state.txt

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)
""")
        sys.exit(1)


if __name__ == '__main__':

    # Initialize defaults. Note that we trim 240 pixels off right and left, this is
    # because our default use case is 1440x1080 upconverted SD in a 1920x1080 box

    model_type = 'BasicSR'
    tile_width, tile_height, tile_border, epochs = 60, 60, 2, 100
    #GPU: Changed default epoc to 100 from 9999, default should be naturally usable
    trim_left, trim_right, trim_top, trim_bottom = 240, 240, 0, 0
    black_level = -1.0
    jitter, shuffle, skip = 1, 1, 1
    initial_lr = 0.001 # why is this 0.0?
    paths = {}

    # Parse options

    errors = False

    for option in sys.argv[1:]:

        opvalue = option.split('=', maxsplit=1)

        if len(opvalue) == 1:
            errors = oops(errors, True, 'Invalid option ({})', option)
        else:
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

            opmatch = [s for s in ['type', 'width', 'height', 'border', 'epochs', 'training',
                                   'validation', 'model', 'data', 'state', 'black',
                                   'jitter', 'shuffle', 'skip', 'lr',
                                   'trimleft', 'trimright', 'trimtop', 'trimbottom'] if s.startswith(op)]

            if len(opmatch) == 0:
                errors = oops(errors, True, 'Unknown option ({})', op)
            elif len(opmatch) > 1:
                errors = oops(errors, True, 'Ambiguous option ({})', op)
            else:
                op = opmatch[0]
                if op == 'type':
                    model_type = valuecase
                    errors = oops(errors, value != 'all' and valuecase not in models.models,
                                  'Unknown model type ({})', valuecase)
                if op == 'width':
                    tile_width = vnum
                    errors = oops(errors, vnum <= 0,
                                  'Tile width invalid ({})', option)
                elif op == 'height':
                    tile_height = vnum
                    errors = oops(errors, vnum <= 0,
                                  'Tile height invalid ({})', option)
                elif op == 'border':
                    tile_border = vnum
                    errors = oops(errors, vnum <= 0,
                                  'Tile border invalid ({})', option)
                elif op == 'black':
                    if value != 'auto':
                        black_level = fnum
                        errors = oops(errors, fnum <= 0,
                                      'Black level invalid ({})', option)
                elif op == 'lr':
                        initial_lr = fnum
                        errors = oops(errors, initial_lr <= 0.0 or initial_lr > 0.01,
                                      'Learning rate out of range ({})', option)
                elif op == 'epochs':
                    epochs = vnum
                    errors = oops(errors, vnum <= 0,
                                  'Epochs invalid ({})', option)
                elif op == 'trimleft':
                    trim_left = vnum
                    errors = oops(errors, vnum <= 0,
                                  'Left trim value invalid ({})', option)
                elif op == 'trimright':
                    trim_right = vnum
                    errors = oops(errors, vnum <= 0,
                                  'Right trim value invalid ({})', option)
                elif op == 'trimtop':
                    trim_top = vnum
                    errors = oops(errors, vnum <= 0,
                                  'Top trim value invalid ({})', option)
                elif op == 'trimbottom':
                    trim_bottom = vnum
                    errors = oops(errors, vnum <= 0,
                                  'Bottom trim value invalid ({})', option)
                elif op == 'jitter':
                    jitter = vnum
                    errors = oops(errors, vnum != 0 and vnum != 1,
                                  'Jitter value invalid ({}). Must be 0, 1, T, F.', option)
                elif op == 'skip':
                    jitter = vnum
                    errors = oops(errors, vnum != 0 and vnum != 1,
                                  'Skip value invalid ({}). Must be 0, 1, T, F.', option)
                elif op == 'shuffle':
                    jitter = vnum
                    errors = oops(errors, vnum != 0 and vnum != 1,
                                  'Shuffle value invalid ({}). Must be 0, 1, T, F.', option)
                elif op == 'data':
                    paths['data'] = os.path.abspath(value)
                elif op == 'training':
                    paths['training'] = os.path.abspath(value)
                elif op == 'validation':
                    paths['validation'] = os.path.abspath(value)
                elif op == 'model':
                    paths['model'] = os.path.abspath(value)
                elif op == 'state':
                    paths['state'] = os.path.abspath(value)

    terminate(errors)

    # Set remaining defaults

    if 'data' not in paths:
        paths['data'] = 'Data'

    dpath = paths['data']

    if 'training' not in paths:
        paths['training'] = os.path.abspath(
            os.path.join(dpath, 'train_images', 'training'))

    if 'validation' not in paths:
        paths['validation'] = os.path.abspath(
            os.path.join(dpath, 'train_images', 'validation'))

    # Remind user what we're about to do.

    print('             Model : {}'.format(model_type))
    print('        Tile Width : {}'.format(tile_width))
    print('       Tile Height : {}'.format(tile_height))
    print('       Tile Border : {}'.format(tile_border))
    print('            Epochs : {}'.format(epochs))
    print('    Data root path : {}'.format(paths['data']))
    print('   Training Images : {}'.format(paths['training']))
    print(' Validation Images : {}'.format(paths['validation']))

    # Validation and error checking

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

    # Check Alpha tiles vs. Alpha and Beta vs. Beta, but allow Alpha and Beta to
    # be different sizes

    for f in [0, 1]:
        s1, s2 = np.shape(test_images[0][f]), np.shape(test_images[1][f])
        errors = oops(errors, s1 != s2, '{} and {} images do not have identical size ({} vs {})',
                      (sub_folders[0], sub_folders[1], s1, s2))

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
        paths['model'] = os.path.abspath(os.path.join(
            dpath, 'models', '{}-{}-{}-{}-{}.h5'.format(model_type, tile_width, tile_height, tile_border, img_suffix)))

    if 'state' not in paths:
        paths['state'] = os.path.abspath(os.path.join(
            dpath, 'models', '{}-{}-{}-{}-{}_state.json'.format(model_type, tile_width, tile_height, tile_border, img_suffix)))

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
    print('')

    # Train the model

    model_list = models.models if model_type.lower() == 'all' else [model_type]
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
                     batch_size=16,
                     black_level=black_level,
                     trim_top=trim_top, trim_bottom=trim_bottom,
                     trim_left=trim_left, trim_right=trim_left,
                     jitter=jitter, shuffle=shuffle, skip=skip,
                     img_suffix=img_suffix,
                     paths=paths)

        # Create and fit model (best model state will be automatically saved)

        sr = models.models[model](io)

        if initial_lr >= 0.0:
            sr.set_lr(initial_lr)

        config = sr.get_config()
        print('Initial model config: {}'.format(config))

        sr.fit(epochs=epochs)

    print('')
    print('Training completed...')
