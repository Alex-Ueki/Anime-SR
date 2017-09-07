"""
Usage: predict.py [option(s)] ...

    Predicts images by applying model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    data=path           path to the main data folder, default = Data
    images=path         path to images folder, default = {Data}/predict_images
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. Default = BasicSR-60-60-2-dpx.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching _state.json file for the model
    (ie: BasicSR-60-60-2-dpx_state.json) that contains all the tiling/trimming information

"""

from Modules.modelio import ModelIO
import Modules.models as models
import Modules.frameops as frameops
from keras import backend as K

import numpy as np
import sys
import os
import json

# Turn debug code on and off

DEBUG = False

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
Usage: predict.py [option(s)] ...

    Predicts images by applying model. The available models are:

        BasicSR
        ExpansionSR
        DeepDenoiseSR
        VDSR

Options are:

    data=path           path to the main data folder, default = Data
    input=path          path to images folder, default = {Data}/predict_images/Alpha
    output=path         path to the output images folder, default = {Data}/predict_images/Beta
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. Default = BasicSR-60-60-2-dpx.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching _state.json file for the model
    (ie: BasicSR-60-60-2-dpx_state.json) that contains all the tiling/trimming information
""")
        sys.exit(1)


if __name__ == '__main__':

    # Initialize defaults. Note that we trim 240 pixels off right and left, this is
    # because our default use case is 1440x1080 upconverted SD in a 1920x1080 box

    paths = {}

    # Parse options

    errors = False

    for option in sys.argv[1:]:

        opvalue = option.split('=', maxsplit=1)

        if len(opvalue) == 1:
            errors = oops(errors, True, 'Invalid option ({})', option)
        else:
            op, value = opvalue

            opmatch = [s for s in ['data', 'input', 'output', 'model'] if s.startswith(op)]

            if len(opmatch) == 0:
                errors = oops(errors, True, 'Unknown option ({})', op)
            elif len(opmatch) > 1:
                errors = oops(errors, True, 'Ambiguous option ({})', op)
            else:
                op = opmatch[0]
                paths[op] = value

    terminate(errors)

    # Set remaining defaults

    if 'data' not in paths:
        paths['data'] = 'Data'

    dpath = paths['data']

    if 'input' not in paths:
        paths['input'] = os.path.join(dpath, 'predict_images', 'Alpha')

    if 'output' not in paths:
        paths['output'] = os.path.join(dpath, 'predict_images', 'Beta')

    if 'model' not in paths:
        paths['model'] = os.path.join(dpath, 'models', 'BasicSR-60-60-2-dpx.h5')
    else:
        if os.path.dirname(paths['model']) == '':
            paths['model'] = os.path.join(dpath, 'models', paths['model'])

    state_split = os.path.splitext(paths['model'])
    paths['state'] = state_split[0] + '_state.json'

    model_type = os.path.basename(paths['model']).split('-')[0]

    # Remind user what we're about to do.

    print('             Data : {}'.format(paths['data']))
    print('            Input : {}'.format(paths['input']))
    print('           Output : {}'.format(paths['output']))
    print('            Model : {}'.format(paths['model']))
    print('            State : {}'.format(paths['state']))
    print('       Model Type : {}'.format(model_type))
    print('')


    # Validation and error checking

    error = oops(errors,model_type not in models.models,'Unknown model type ({})'.format(model_type))

    for p in paths:
        errors = oops(errors, not os.path.exists(paths[p]),'Path to {} is not valid ({})'.format(p.title(),paths[p]))

    terminate(errors, False)

    # Load state

    with open(paths['state'], 'r') as f:
        state = json.load(f)

    # We only look at the io info

    iostate = state['io']

    # Check image files -- we do not explore subfolders. Note we have already checked
    # path validity above

    image_info = frameops.image_files(paths['input'], False)

    errors = oops(errors, len(
        image_info) == 0, 'Input folder does not contain any images')
    errors = oops(errors, len(
        image_info) > 1, 'Images folder contains more than one type of image')

    terminate(errors, False)

    # Get the list of files and check the filetype is correct

    image_info = image_info[0]

    image_ext = os.path.splitext(image_info[0])[1][1:].lower()

    errors = oops(errors, image_ext != iostate['img_suffix'].lower(),
        'Image files are of type [{}] but model was trained on [{}]'.format(image_ext,iostate['img_suffix'].lower()))

    terminate(errors,False)

    # There is no point in caching tiles

    frameops.reset_cache(enabled=False)

    # Configure model IO

    io = ModelIO(model_type=model_type,
                 image_width=iostate['image_width'],
                 image_height=iostate['image_height'],
                 base_tile_width=iostate['base_tile_width'],
                 base_tile_height=iostate['base_tile_height'],
                 channels=iostate['channels'],
                 border=iostate['border'],
                 batch_size=iostate['batch_size'],
                 black_level=iostate['black_level'],
                 trim_top=iostate['trim_top'],
                 trim_bottom=iostate['trim_bottom'],
                 trim_left=iostate['trim_left'],
                 trim_right=iostate['trim_left'],
                 jitter=False,
                 shuffle=False,
                 skip=False,
                 img_suffix=iostate['img_suffix'],
                 paths={})

    # Create model. Since the model file contains the complete model info, not just the
    # weights, we can instantiate it using the base class. So no matter what changes we
    # make to the definition of the models in models.py, old model files will still
    # work.

    sr = models.BaseSRCNNModel(name=model_type,
                               io=io)

    # Compute some handy information

    row_width = (io.image_width - io.trim_left - io.trim_right) // io.base_tile_width
    row_height = (io.image_height - io.trim_top - io.trim_bottom) // io.base_tile_height
    tiles_per_img = row_width * row_height
    batches_per_image = (tiles_per_img + (io.batch_size - 1)) // io.batch_size
    tile_width = io.base_tile_width + 2 * io.border
    tile_height = io.base_tile_height + 2 * io.border

    print('tiles per image',tiles_per_img)
    print('batches per image',batches_per_image)

    if K.image_dim_ordering() == 'th':
        image_shape = (
            io.channels, tile_width, tile_height)
    else:
        image_shape = (
            tile_width, tile_height, io.channels)

    # Process the images

    first_path = image_info[0]

    for img_path in image_info:
        img_filename = os.path.basename(img_path)
        print('Predicting',img_filename)
        img = frameops.imread(img_path)

        # Generate the tiles for the image
        # Note that tiles is a generator

        tiles = frameops.tesselate(file_paths=img_path,
                                   tile_width=io.base_tile_width,
                                   tile_height=io.base_tile_height,
                                   border=io.border,
                                   black_level=io.black_level,
                                   trim_top=io.trim_top,
                                   trim_bottom=io.trim_bottom,
                                   trim_left=io.trim_left,
                                   trim_right=io.trim_right,
                                   shuffle=False,
                                   jitter=False,
                                   skip=False)

        # Create a batch with all the tiles

        tile_batch = np.empty((tiles_per_img, ) + image_shape)
        for i,tile in enumerate(tiles):
            if K.image_dim_ordering() == 'th':
                tile = tile.transpose((2, 0, 1))
            tile_batch[i] = tile

        # Debug code to confirm what we are doing

        if DEBUG and img_path == first_path:
            fname = os.path.basename(img_path)
            for i in range(0,tiles_per_img):
                ipath = os.path.join('Temp', 'PNG', 'IN', fname[0:-4] + '-' + str(i) + '.png')
                frameops.imsave(ipath,tile_batch[i])

        # predicted_tiles = sr.predict_tiles(tile_generator=tile_batches, batches=batches_per_image)

        predicted_tiles = sr.model.predict(tile_batch, tiles_per_img)

        # Debug code to confirm what we are doing

        if DEBUG and img_path == first_path:
            fname = os.path.basename(img_path)
            for i in range(0,tiles_per_img):
                ipath = os.path.join('Temp', 'PNG', 'OUT', fname[0:-4] + '-' + str(i) + '.png')
                frameops.imsave(ipath,predicted_tiles[i])

        # Merge the tiles back into a single image

        predicted_image = frameops.grout(predicted_tiles,
                                         border=io.border,
                                         row_width=row_width,
                                         black_level=io.black_level,
                                         pad_top=io.trim_top,
                                         pad_bottom=io.trim_bottom,
                                         pad_left=io.trim_left,
                                         pad_right=io.trim_right)

        # Save the image

        frameops.imsave(os.path.join(paths['output'],img_filename), predicted_image)

    print('')
    print('Predictions completed...')
