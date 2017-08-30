"""

Usage: train.py [model] [option(s)] ...

    Trains a model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    epochs=nnn          epoch size, default=255
    data=path           path to the main data folder, default = Data
    training=path       path to training folder, default = {Data}/train_images/training
    validation=path     path to validation folder, default = {Data}/train_images/validation
    model=path          path to model file, default = Weights/{model}-{width}-{height}-{border}.h5
    history=path        path to checkpoint file, default = Weights/{model}-{width}-{height}-{border}_history.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

"""

import Modules.basemodel as basemodel
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
        if type(value) in (list,tuple):
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
Usage: train.py [model] [option(s)] ...

    Trains a model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    width=nnn           tile width, default=60
    height=nnn          tile height, default=60
    border=nnn          border size, default=2
    epochs=nnn          epoch size, default=255
    data=path           path to the main data folder, default = Data
    training=path       path to training folder, default = {Data}/train_images/training
    validation=path     path to validation folder, default = {Data}/train_images/validation
    model=path          path to model file, default = Weights/{model}-{width}-{height}-{border}.h5
    history=path        path to checkpoint file, default = Weights/{model}-{width}-{height}-{border}_history.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)
""")
        sys.exit(1)

if __name__ == '__main__':

    errors = oops(False, len(sys.argv) == 1, 'Model type not specified', len(sys.argv)-1, True)

    model_type = sys.argv[1]

    # Initialize defaults

    (tile_width, tile_height, tile_border, epochs) = (60, 60, 2, 255)
    paths = {}

    # Parse options

    for option in sys.argv[2:]:
        opvalue = option.split('=', maxsplit=1)
        if len(opvalue) == 1:
            errors = oops(errors, True, 'Invalid option ({})', option)
        else:
            op, value = opvalue
            vnum = int(value) if value.isdigit() else -1

            opmatch = [s for s in ['width', 'height', 'border', 'epochs', 'training', 'validation', 'model', 'data', 'history'] if s.startswith(op)]

            if len(opmatch) == 0:
                errors = oops(errors, True, 'Unknown option ({})', op)
            elif len(opmatch) > 1:
                errors = oops(errors, True, 'Ambiguous option ({})', op)
            else:
                op = opmatch[0]
                if op == 'width':
                    tile_width = vnum
                    errors = oops(errors, vnum <= 0, 'Tile width invalid ({})', option)
                elif op == 'height':
                    tile_height = vnum
                    errors = oops(errors, vnum <= 0, 'Tile height invalid ({})', option)
                elif op == 'border':
                    tile_border = vnum
                    errors = oops(errors, vnum <= 0, 'Tile border invalid ({})', option)
                elif op == 'epochs':
                    epochs = vnum
                    errors = oops(errors, vnum <= 0, 'Epochs invalid ({})', option)
                elif op == 'data':
                    paths['data'] = os.path.abspath(value)
                elif op == 'training':
                    paths['training'] = os.path.abspath(value)
                elif op == 'validation':
                    paths['validation'] = os.path.abspath(value)
                elif op == 'model':
                    paths['weights'] = os.path.abspath(value)
                elif op == 'history':
                    paths['history'] = os.path.abspath(value)

    terminate(errors)

    # Set remaining defaults

    if 'data' not in paths:
        paths['data'] = 'Data'

    dpath = paths['data']

    if 'training' not in paths:
        paths['training'] = os.path.abspath(os.path.join(dpath, 'train_images', 'training'))

    if 'validation' not in paths:
        paths['validation'] = os.path.abspath(os.path.join(dpath, 'train_images', 'validation'))

    if 'weights' not in paths:
        paths['weights'] = os.path.abspath(os.path.join(dpath, 'weights', '%s-%d-%d-%d.h5' % (model_type, tile_width, tile_height, tile_border) ))

    # Remind user what we're about to do.

    print('            Model : {}'.format(model_type))
    print('       Tile Width : {}'.format(tile_width))
    print('      Tile Height : {}'.format(tile_height))
    print('      Tile Border : {}'.format(tile_border))
    print('           Epochs : {}'.format(epochs))
    print('   Data root path : {}'.format(paths['data']))
    print('  Training Images : {}'.format(paths['training']))
    print('Validation Images : {}'.format(paths['validation']))
    print('       Model File : {}'.format(paths['weights']))

    # Validation and error checking

    image_paths = ['training', 'validation']
    sub_folders = ['Alpha', 'Beta']
    image_info = [ [None, None], [None, None]]

    for fc, f in enumerate(image_paths):
        for sc, s in enumerate(sub_folders):
            image_info[fc][sc] = frameops.image_files(os.path.join(paths[f], sub_folders[sc]), True)

    for f in [0, 1]:
        for s in [0, 1]:
            errors = oops(errors, image_info[f][s] == None, '{} images folder does not exist', image_paths[f] + '/' + sub_folders[s])

    terminate(errors, False)

    for f in [0, 1]:
        for s in [0, 1]:
            errors = oops(errors, len(image_info[f][s]) == 0, '{} images folder does not contain any images', image_paths[f] + '/' + sub_folders[s])
            errors = oops(errors, len(image_info[f][s]) > 1, '{} images folder contains more than one type of image', image_paths[f] + '/' + sub_folders[s])

    terminate(errors, False)

    for f in [0, 1]:
        errors = oops(errors, len(image_info[f][0][0]) != len(image_info[f][1][0]), '{} images folders have different numbers of images', image_paths[f])

    terminate(errors, False)

    for f in [0, 1]:
        for f1, f2 in zip(image_info[f][0][0], image_info[f][1][0]):
            f1, f2 = os.path.basename(f1), os.path.basename(f2)
            errors = oops(errors, f1 != f2, '{} images folders do not have identical image filenames ({} vs {})', (image_paths[f], f1, f2))
            terminate(errors, False)

    # Check sizes, even tiling here.

    test_files = [ [image_info[f][g][0][0] for g in [0, 1]] for f in [0, 1]]
    test_images = [ [frameops.imread(image_info[f][g][0][0]) for g in [0, 1]] for f in [0, 1]]


    for f in [0, 1]:
        s1, s2 = np.shape(test_images[f][0]), np.shape(test_images[f][1])
        print(s1, s2)
        errors = oops(errors, s1 != s2, '{} {} and {} images do not have identical size ({} vs {})', (image_paths[f], sub_folders[0], sub_folders[1], s1, s2))

    s1, s2 = np.shape(test_images[0][0]), np.shape(test_images[1][0])
    errors = oops(errors, s1 != s2, '{} and {} images do not have identical size ({1} vs {2})', (image_paths[0], image_paths[1], s1, s2))

    terminate(errors, False)

    errors = oops(errors, len(s1) != 3 or s1[2] != 3, 'Images have improper shape ({0})', str(s1))

    terminate(errors, False)

    errors = oops(errors, (s1[0] % tile_width) != 0, 'Images do not evenly tile horizontally ({} % {} != 0)', (s1[0], tile_width))
    errors = oops(errors, (s1[1] % tile_height) != 0, 'Images do not evenly tile vertically ({} % {} != 0)', (s1[1], tile_height))

    terminate(errors, False)

    print(' Image dimensions : {} x {}'.format(s1[0], s1[1]))
    print('')

    img = frameops.imread('Temp/Test.png')
    frameops.imsave('Temp/Test-nomung.png',img)

    tiles = [t for t in frameops.tesselate('Temp/Test.png',tile_width,tile_height,tile_border,trim_left=240,trim_right=240)]

    img = frameops.grout(tiles, tile_border, 24, pad_left=240, pad_right=240)
    print(np.shape(img))

    frameops.imsave('Temp/Test-Out.png',img)

    terminate(True,False)

    # Train the model

    if model_type in models.models:
        sr = models.models[model_type](base_tile_width=tile_width, base_tile_height=tile_height, border=tile_border, paths=paths)
        sr.create_model()
        sr.fit(nb_epochs=epochs)
        sr.save()
        print('Training completed...')
    else:
        errors = oops(errors, True, 'Unknown model type ({})', model_type, True)
