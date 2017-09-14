"""
Usage: test.py

Misc test scratchpad

"""

from Modules.modelio import ModelIO
import Modules.models as models
import Modules.frameops as frameops
import Modules.dpx as dpx
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
            print(__doc__)
        sys.exit(1)

# Tile quality score. For now it just returns index

def quality(tile, idx):

    return -np.sum(np.absolute(tile[:, 1:, :] - tile[:, :-1, :]))

if __name__ == '__main__':

    # Initialize defaults. Note that we trim 240 pixels off right and left, this is
    # because our default use case is 1440x1080 upconverted SD in a 1920x1080 box

    paths = {}
    errors = False

    # Set remaining defaults - boilerplate just so we have stuff we can easily refer to


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
        model_split = os.path.splitext(paths['model'])
        if model_split[1] == '':
            paths['model'] = paths['model'] + '.h5'
        if os.path.dirname(paths['model']) == '':
            paths['model'] = os.path.join(dpath, 'models', paths['model'])

    if 'training' not in paths:
        paths['training'] = os.path.abspath(
            os.path.join(dpath, 'train_images', 'training'))

    if 'validation' not in paths:
        paths['validation'] = os.path.abspath(
            os.path.join(dpath, 'train_images', 'validation'))

    state_split = os.path.splitext(paths['model'])
    paths['state'] = state_split[0] + '_state.json'

    model_type = os.path.basename(paths['model']).split('-')[0]

    # Remind user what we're about to do.

    print('             Data : {}'.format(paths['data']))
    print('            Input : {}'.format(paths['input']))
    print('           Output : {}'.format(paths['output']))
    print('         Training : {}'.format(paths['training']))
    print('       Validation : {}'.format(paths['validation']))
    print('            Model : {}'.format(paths['model']))
    print('            State : {}'.format(paths['state']))
    print('       Model Type : {}'.format(model_type))


    image_info = frameops.image_files(os.path.join(paths['training'],'Beta'), False)

    errors = oops(errors, len(
        image_info) == 0, 'Training Beta folder does not contain any images')
    errors = oops(errors, len(
        image_info) > 1, 'Training Beta folder folder contains more than one type of image')

    terminate(errors, False)

    # Get the list of files and check the filetype is correct

    image_info = image_info[0]

    image_path = image_info[len(image_info) // 24]
    image_ext = os.path.splitext(image_path)[1][1:].lower()

    alpha_path = image_path.replace('Beta','Alpha')

    print('       Image Path : {}'.format(image_path))

    # There is no point in caching tiles

    frameops.reset_cache(enabled=False)

    # Read in the image and save it into temp folder as a reference

    img = frameops.imread(image_path)
    print(dpx.dpx_meta)

    frameops.imsave(os.path.join('Temp','PNG','Image-Beta.png'),img)

    # OK, read in image tiles

    tiles = frameops.extract_tiles(image_path, tile_width=60, tile_height=60, border=2, black_level=0.0, border_mode='edge',
                                    trim_top=0, trim_bottom=0, trim_left=240, trim_right=240, jitter=False,
                                    expected_width=1920, expected_height=1080)

    img = frameops.grout(tiles, border=2, row_width=24, black_level=0.0, pad_top=0, pad_bottom=0, pad_left=240, pad_right=240)
    frameops.imsave(os.path.join('Temp','PNG','Image-Beta-Grouted.png'),img)

    img2 = frameops.imread(alpha_path)
    print(dpx.dpx_meta)
    frameops.imsave(os.path.join('Temp','PNG','Image-Alpha.png'),img2)

    # OK, read in image tiles

    tiles2 = frameops.extract_tiles(alpha_path, tile_width=60, tile_height=60, border=2, black_level=0.0, border_mode='edge',
                                    trim_top=0, trim_bottom=0, trim_left=240, trim_right=240, jitter=False,
                                    expected_width=1920, expected_height=1080)

    img2 = frameops.grout(tiles2, border=2, row_width=24, black_level=0.0, pad_top=0, pad_bottom=0, pad_left=240, pad_right=240)
    frameops.imsave(os.path.join('Temp','PNG','Image-Alpha-Grouted.png'),img2)

    print(len(tiles))
    frameops.reset_cache()

    tiles = [t for t in frameops.tesselate_pair(alpha_path, image_path, tile_width=60, tile_height=60, border=2, black_level=0.0, border_mode='edge',
                                        trim_top=0, trim_bottom=0, trim_left=240, trim_right=240, shuffle=False, jitter=False, skip=False,
                                        expected_width=1920, expected_height=1080, quality=0.25)]

    print(len(tiles))

    image_path = os.path.join('Temp','PNG','Tile-{}-{}.png')

    for i in range(len(tiles)):
        for j in [0,1]:
            frameops.imsave(image_path.format(i,j),tiles[i][j])
