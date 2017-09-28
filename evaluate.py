"""
Usage: evaluate.py [option(s)] ...

    Evaluate models. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR
        PUPSR (Testing)
        GPUSR (Testing)

Options are:

    data=path           path to the main data folder, default = Data
    images=path         path to images folder, default = {Data}/eval_images
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. The .h5 extension may be omitted.
                        Default = BasicSR-60-60-2-dpx.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching _state.json file for the model
    (ie: BasicSR-60-60-2-dpx_state.json) that contains all the tiling/trimming information
"""

from Modules.misc import oops, validate, terminate, set_docstring

import numpy as np
import sys
import os
import json

set_docstring(__doc__)

# Turn debug code on and off

DEBUG = False

# TODO Implement ALL functionality for evaluation

if __name__ == '__main__':

    # Initialize defaults. Note that we trim 240 pixels off right and left, this is
    # because our default use case is 1440x1080 upconverted SD in a 1920x1080 box

    paths = {}

    # Parse options

    errors = False

    options = sorted(['type', 'data', 'images', 'model'])

    for option in sys.argv[1:]:

        opvalue = option.split('=', maxsplit=1)

        if len(opvalue) == 1:
            errors = oops(errors, True, 'Invalid option ({})', option)
            continue

        op, value = opvalue

        opmatch = [s for s in options if s.startswith(op)]

        if len(opmatch) == 0 or len(opmatch) > 1 and opmatch[0] != op:
            errors = oops(errors, True, '{} option ({})',
                          ('Unknown' if len(opmatch) == 0 else 'Ambiguous', op))
            continue

        op = opmatch[0]
        paths[op] = value

    terminate(errors)

    # Set remaining defaults

    if 'data' not in paths:
        paths['data'] = 'Data'

    dpath = paths['data']

    if 'images' not in paths:
        paths['images'] = os.path.join(dpath, 'eval_images')

    paths['Alpha'] = os.path.join(paths['images'], 'Alpha')
    paths['Beta'] = os.path.join(paths['images'], 'Beta')

    if 'model' not in paths:
        paths['model'] = os.path.join(
            dpath, 'models', 'BasicSR-60-60-2-dpx.h5')
    else:
        model_split = os.path.splitext(paths['model'])
        if model_split[1] == '':
            paths['model'] = paths['model'] + '.h5'
        if os.path.dirname(paths['model']) == '':
            paths['model'] = os.path.join(dpath, 'models', paths['model'])

    state_split = os.path.splitext(paths['model'])
    paths['state'] = state_split[0] + '_state.json'

    model_type = os.path.basename(paths['model']).split('-')[0]

    # Remind user what we're about to do.

    print('             Data : {}'.format(paths['data']))
    print('           Images : {}'.format(paths['images']))
    print('            Model : {}'.format(paths['model']))
    print('            State : {}'.format(paths['state']))
    print('       Model Type : {}'.format(model_type))
    print('')

    # Validation and error checking

    for p in paths:
        errors = oops(errors, not os.path.exists(
            paths[p]), 'Path to {} is not valid ({})'.format(p.title(), paths[p]))

    terminate(errors, False)

    # Load state

    with open(paths['state'], 'r') as f:
        state = json.load(f)

    # We only look at the io info

    iostate = state['io']

    # Check image files -- we do not explore subfolders. Note we have already checked
    # path validity above

    import Modules.frameops as frameops

    image_info = frameops.image_files(paths['Alpha'], False)

    for name in ['Alpha', 'Beta']:
        image_info = frameops.image_files(paths[name], False)
        errors = oops(errors, len(
            image_info) == 0, name + ' folder does not contain any images')
        errors = oops(errors, len(
            image_info) > 1, 'Images folder contains more than one type of image')

    terminate(errors, False)

    # Get the list of files and check the filetype is correct

    image_info = image_info[0]

    image_ext = os.path.splitext(image_info[0])[1][1:].lower()

    errors = oops(errors, image_ext != iostate['img_suffix'].lower(),
                  'Image files are of type [{}] but model was trained on [{}]'.format(image_ext, iostate['img_suffix'].lower()))

    terminate(errors, False)

    # Configure model IO

    from Modules.modelio import ModelIO
    import Modules.models as models

    error = oops(errors, model_type not in models.models,
                 'Unknown model type ({})'.format(model_type))

    terminate(errors, False)

    io = ModelIO(model_type=model_type,
                 image_width=iostate['image_width'],
                 image_height=iostate['image_height'],
                 base_tile_width=iostate['base_tile_width'],
                 base_tile_height=iostate['base_tile_height'],
                 channels=iostate['channels'],
                 border=iostate['border'],
                 border_mode=iostate['border_mode'],
                 batch_size=iostate['batch_size'],
                 black_level=iostate['black_level'],
                 trim_top=iostate['trim_top'],
                 trim_bottom=iostate['trim_bottom'],
                 trim_left=iostate['trim_left'],
                 trim_right=iostate['trim_left'],
                 jitter=False,
                 shuffle=False,
                 skip=False,
                 quality=1.0,
                 residual=iostate['residual'],
                 img_suffix=iostate['img_suffix'],
                 paths={})

    # Create model. Since the model file contains the complete model info, not just the
    # weights, we can instantiate it using the base class. So no matter what changes we
    # make to the definition of the models in models.py, old model files will still
    # work.

    sr = models.BaseSRCNNModel(
        name=model_type, io=io, verbose=False, bargraph=False)

    # Compute some handy information

    row_width = (io.image_width - io.trim_left -
                 io.trim_right) // io.base_tile_width
    row_height = (io.image_height - io.trim_top -
                  io.trim_bottom) // io.base_tile_height
    tiles_per_img = row_width * row_height
    batches_per_image = (tiles_per_img + (io.batch_size - 1)) // io.batch_size
    tile_width = io.base_tile_width + 2 * io.border
    tile_height = io.base_tile_height + 2 * io.border

    # There is no point in caching tiles

    frameops.reset_cache(enabled=False)

    # Process the images

    sr.evaluate()

    print('')
    print('Evaluation completed...')
