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

from Modules.misc import oops, terminate, set_docstring
import Modules.frameops as frameops
from Modules.modelio import ModelIO
import Modules.models as models


set_docstring(__doc__)

# Turn debug code on and off

DEBUG = True


def predict():
    """ Run predictor on a folder of images """

    # Initialize defaults. Note that we trim 240 pixels off right and left, this is
    # because our default use case is 1440x1080 upconverted SD in a 1920x1080 box.
    # These variables will be set by parse_options() and configure()

    options = sorted(['type', 'data', 'images', 'model'])

    def parse_options():
        """ Parses options; fills in paths. Returns paths and model_type
            If errors detected, exits
        """
        errors = False
        paths = {}

        # Parse options

        for option in sys.argv[1:]:

            opvalue = option.split('=', maxsplit=1)

            if len(opvalue) == 1:
                errors = oops(errors, True, 'Invalid option ({})', option)
                continue

            option, value = opvalue

            opmatch = [s for s in options if s.startswith(option)]

            if not opmatch or len(opmatch) > 1 and opmatch[0] != option:
                errors = oops(errors, True, '{} option ({})',
                              ('Ambiguous' if opmatch else 'Unknown', option))
                continue

            option = opmatch[0]
            paths[option] = value

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
            paths['model'] = os.path.join(
                dpath, 'models', 'BasicSR-60-60-2-dpx.h5')
        else:
            model_split = os.path.splitext(paths['model'])
            if model_split[1] == '':
                paths['model'] = paths['model'] + '.h5'
            if os.path.dirname(paths['model']) == '':
                paths['model'] = os.path.join(dpath, 'models', paths['model'])

        paths['state'] = os.path.splitext(paths['model'])[0] + '_state.json'

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

        for path in paths:
            errors = oops(errors, not os.path.exists(
                paths[path]), 'Path to {} is not valid ({})'.format(path, paths[path]))

        terminate(errors, False)

        return (paths, model_type)

    def configure(paths, model_type):
        """ Configure the model and returns image_info and ModelIO instance """

        # Load state

        with open(paths['state'], 'r') as jsonfile:
            state = json.load(jsonfile)

        # We only look at the io info

        iostate = state['io']

        # Check image files -- we do not explore subfolders. Note we have already checked
        # path validity above

        image_info = frameops.image_files(paths['input'], False)

        errors = oops(False, len(image_info) == 0,
                      'Input folder does not contain any images')
        errors = oops(errors, len(image_info) > 1,
                      'Images folder contains more than one type of image')

        terminate(errors, False)

        # Get the list of files and check the filetype is correct

        image_info = image_info[0]

        image_ext = os.path.splitext(image_info[0])[1][1:].lower()

        errors = oops(errors, image_ext != iostate['img_suffix'].lower(),
                      'Image files are of type [{}] but model was trained on [{}]'.format(image_ext, iostate['img_suffix'].lower()))

        terminate(errors, False)

        # Configure model IO

        errors = oops(errors, model_type not in models.MODELS,
                      'Unknown model type ({})'.format(model_type))

        terminate(errors, False)

        io_config = ModelIO(model_type=model_type,
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

        return (image_info, io_config)

    paths, model_type = parse_options()
    image_info, io_config = configure(paths, model_type)

    # Create model. Since the model file contains the complete model info, not just the
    # weights, we can instantiate it using the base class. So no matter what changes we
    # make to the definition of the models in models.py, old model files will still
    # work.

    sr_model = models.BaseSRCNNModel(
        name=model_type, io=io_config, verbose=False, bargraph=False)

    # Compute some handy information

    row_width = (io_config.image_width - io_config.trim_left -
                 io_config.trim_right) // io_config.base_tile_width
    row_height = (io_config.image_height - io_config.trim_top -
                  io_config.trim_bottom) // io_config.base_tile_height
    tiles_per_img = row_width * row_height

    print('tiles per image', tiles_per_img)

    # Process the images

    if DEBUG:
        image_info = [image_info[0]]  # just do one image

    # There is no point in caching tiles since we never revisit them.

    frameops.reset_cache(enabled=False)

    for img_path in image_info:
        print('Predicting', os.path.basename(img_path))

        # Generate the tiles for the image. Note that tiles is a generator
        # Also, we must specify quality=1.0 to make sure we get all
        # the tiles in the default order.

        tiles = frameops.tesselate(file_paths=img_path,
                                   tile_width=io_config.base_tile_width,
                                   tile_height=io_config.base_tile_height,
                                   border=io_config.border,
                                   black_level=io_config.black_level,
                                   trim_top=io_config.trim_top,
                                   trim_bottom=io_config.trim_bottom,
                                   trim_left=io_config.trim_left,
                                   trim_right=io_config.trim_right,
                                   shuffle=False,
                                   jitter=False,
                                   skip=False,
                                   quality=1.0,
                                   theano=io_config.theano)

        # Create a batch with all the tiles

        tile_batch = np.empty((tiles_per_img, ) + io_config.image_shape)
        for idx, tile in enumerate(tiles):
            tile_batch[idx] = tile


        if DEBUG:
            fname = os.path.basename(img_path)
            for i in range(0, min(30, tiles_per_img)):
                frameops.imsave(os.path.join(
                    'Temp', 'PNG', fname[:-4] + '-' + str(i) + '-IN.png'), tile_batch[i])
            input_image = frameops.grout(tile_batch,
                                         border=io_config.border,
                                         row_width=row_width,
                                         black_level=io_config.black_level,
                                         pad_top=io_config.trim_top,
                                         pad_bottom=io_config.trim_bottom,
                                         pad_left=io_config.trim_left,
                                         pad_right=io_config.trim_right,
                                         theano=io_config.theano)
            frameops.imsave(os.path.join('Temp', 'PNG', os.path.basename(
                img_path)[:-4] + '-IN.png'), input_image)

        # Predict the new tiles

        predicted_tiles = sr_model.model.predict(tile_batch, tiles_per_img)

        # GPU : Just using this to debug, uncomment to print section to see results for yourself
        # debugging: if residual, then the np.mean of tile_batch should be a
        # near zero numbers. Testing supports a mean around 0.0003
        # Without residual, mean is usually higher
        # print('Debug: Residual {} Mean {}'.format(io.residual==1, np.mean(predicted_tiles)))

        if io_config.residual:
            predicted_tiles += tile_batch

        # Merge the tiles back into a single image

        predicted_image = frameops.grout(predicted_tiles,
                                         border=io_config.border,
                                         row_width=row_width,
                                         black_level=io_config.black_level,
                                         pad_top=io_config.trim_top,
                                         pad_bottom=io_config.trim_bottom,
                                         pad_left=io_config.trim_left,
                                         pad_right=io_config.trim_right,
                                         theano=io_config.theano)

        # Save the image

        frameops.imsave(os.path.join(
            paths['output'], os.path.basename(img_path)), predicted_image)

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
    predict()
