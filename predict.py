# pylint: disable=C0301
# (line_too_long disabled)

"""
Usage: predict.py [option(s)] ...

    Predicts images by applying model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR
        PUPSR (Testing)
        GPUSR (Testing)

Options are:

    data=path           path to the main data folder, default = Data
    images=path         path to images folder, default = {Data}/predict_images
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. The .h5 extension may be omitted.
                        Default = BasicSR-60-60-2-dpx.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching _state.json file for the model
    (ie: BasicSR-60-60-2-dpx_state.json) that contains all the tiling/trimming information
"""


import sys
import os
import json

import numpy as np

from Modules.misc import oops, terminate, set_docstring, opcheck
import Modules.frameops as frameops
from Modules.modelio import ModelIO
import Modules.models as models


set_docstring(__doc__)

# Turn debug code on and off

DEBUG = True

def parse_options():
    """ Parse the command line options """

    # Fetch the path options, if any

    options = {}

    # For each parameter option, a tuple of the config entry it affects, its type, an invalid
    # function (returns true if the value is out of bounds), and a format() string for reporting
    # the error.

    opcodes = {
        'data': ('data', str, lambda x: False, ''),
        'images': ('predict', str, lambda x: False, ''),
        'model': ('model', str, lambda x: False, ''),
    }

    # Order of options in this list can be important; if one option is a substring
    # of the other, the smaller one must come first.

    option_names = sorted(list(opcodes.keys()))

    # Parse options

    errors = False

    for param in sys.argv[1:]:

        opvalue = param.split('=', maxsplit=1)

        if len(opvalue) == 1:
            errors = oops(errors, True, 'Invalid option ({})', param)
            continue

        option, value = opvalue[0].lower(), opvalue[1]

        # Match option, make sure it isn't ambiguous.

        opmatch = [s for s in option_names if s.startswith(option)]

        if not opmatch or len(opmatch) > 1 and opmatch[0] != option:
            errors = oops(errors, True, '{} option ({})',
                          ('Ambiguous' if opmatch else 'Unknown', option))
            continue

        opcode = opcodes[opmatch[0]]
        opname = opcode[0]

        if opname not in options:
            options[opname] = None

        errors, options[opname] = opcheck(opcode, options[opname], value, errors)

    terminate(errors)

    return options

def setup(basepaths):
    """Set up configuration """

    # Set remaining basepaths

    basepaths.setdefault('data', 'Data')
    dpath = basepaths['data']

    basepaths.setdefault('model', os.path.join(dpath, 'models', 'BasicSR-60-60-2-dpx.h5'))
    basepaths.setdefault('predict', os.path.join(dpath, 'predict_images'))

    model_split = os.path.splitext(basepaths['model'])
    if model_split[1] == '':
        basepaths['model'] = basepaths['model'] + '.h5'
    if os.path.dirname(basepaths['model']) == '':
        basepaths['model'] = os.path.join(dpath, 'models', basepaths['model'])

    basepaths['state'] = os.path.splitext(basepaths['model'])[0] + '_state.json'

    model_type = os.path.basename(basepaths['model']).split('-')[0]

    # Remind user what we're about to do.

    print('             Data : {}'.format(basepaths['data']))
    print('   Predict Images : {}'.format(basepaths['predict']))
    print('            Model : {}'.format(basepaths['model']))
    print('            State : {}'.format(basepaths['state']))
    print('       Model Type : {}'.format(model_type))
    print('')

    # Validation and error checking

    for path in basepaths:
        errors = oops(False, not os.path.exists(
            basepaths[path]), 'Path to {} is not valid ({})'.format(path, basepaths[path]))

    terminate(errors, False)

    # Load the actual model state

    with open(basepaths['state'], 'r') as jsonfile:
        state = json.load(jsonfile)

    # Grab the config data (backwards compatible)

    config = state['config' if 'config' in state else 'io']

    # Create real config with configurable parameters. In particular we disable options like
    # jitter, shuffle, skip and quality.

    config['paths'] = basepaths
    config['jitter'] = False
    config['shuffle'] = False
    config['skip'] = False
    config['quality'] = 1.0
    config['model_type'] = model_type

    config = ModelIO(config)

    # Check image files -- we do not explore subfolders. Note we have already checked
    # path validity above

    image_info = frameops.image_files(os.path.join(config.paths['predict'], config.alpha), False)

    errors = oops(False, len(image_info) == 0, 'Input folder does not contain any images')
    errors = oops(errors, len(image_info) > 1, 'Images folder contains more than one type of image')

    terminate(errors, False)

    # Get the list of files and check the filetype is correct

    image_info = image_info[0]

    image_ext = os.path.splitext(image_info[0])[1][1:].lower()

    errors = oops(errors, image_ext != config.img_suffix.lower(),
                  'Image files are of type [{}] but model was trained on [{}]'.format(image_ext, config.img_suffix.lower()))

    terminate(errors, False)

    return (config, image_info)

def predict(config, image_info):
    """ Run predictions using the configuration and list of files """

    # Create model. Since the model file contains the complete model info, not just the
    # weights, we can instantiate it using the base class. So no matter what changes we
    # make to the definition of the models in models.py, old model files will still
    # work.

    sr_model = models.BaseSRCNNModel(name=config.model_type, io=config, verbose=False, bargraph=False)

    # Need to use unjittered tiles_per_imag

    tiles_per_img = config.tiles_across * config.tiles_down

    # Process the images

    if DEBUG:
        image_info = [image_info[-1]]  # just do one image

    # There is no point in caching tiles since we never revisit them.

    frameops.reset_cache(enabled=False)

    for img_path in image_info:
        print('Predicting', os.path.basename(img_path))

        # Generate the tiles for the image. Note that tiles is a generator

        tiles = frameops.tesselate(img_path, config)

        # Create a batch with all the tiles

        tile_batch = np.empty((tiles_per_img, ) + config.image_shape)
        for idx, tile in enumerate(tiles):
            tile_batch[idx] = tile


        if DEBUG:
            fname = os.path.basename(img_path)
            for i in range(0, min(30, tiles_per_img)):
                frameops.imsave(os.path.join(
                    'Temp', 'PNG', fname[:-4] + '-' + str(i) + '-IN.png'), tile_batch[i])
            input_image = frameops.grout(tile_batch, config)
            frameops.imsave(os.path.join('Temp', 'PNG', os.path.basename(
                img_path)[:-4] + '-IN.png'), input_image)

        # Predict the new tiles

        predicted_tiles = sr_model.model.predict(tile_batch, tiles_per_img)

        # GPU : Just using this to debug, uncomment to print section to see results for yourself
        # debugging: if residual, then the np.mean of tile_batch should be a
        # near zero numbers. Testing supports a mean around 0.0003
        # Without residual, mean is usually higher
        # print('Debug: Residual {} Mean {}'.format(io.residual==1, np.mean(predicted_tiles)))

        if config.residual:
            predicted_tiles += tile_batch

        # Merge the tiles back into a single image

        predicted_image = frameops.grout(predicted_tiles, config)

        # Save the image

        frameops.imsave(os.path.join(
            config.paths['predict'], config.beta, os.path.basename(img_path)), predicted_image)

        # Debug code to confirm what we are doing

        if DEBUG:
            fname = os.path.basename(img_path)
            for i in range(0, min(30, tiles_per_img)):
                frameops.imsave(os.path.join(
                    'Temp', 'PNG', fname[0:-4] + '-' + str(i) + '-OUT.png'), predicted_tiles[i])

            frameops.imsave(os.path.join('Temp', 'PNG', os.path.basename(
                img_path)[:-4] + '-OUT.png'), predicted_image)

    print('')
    print('Predictions completed...')


if __name__ == '__main__':
    OPTIONS = parse_options()
    CONFIG, IMAGE_INFO = setup(OPTIONS)
    predict(CONFIG, IMAGE_INFO)
