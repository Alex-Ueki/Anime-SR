# pylint: disable=C0301
# (line_too_long disabled)

"""
Usage: predict.py [option(s)] ...

    Predicts images using a model.

Options are:

    data=path           path to the main data folder, default = Data
    images=path         path to images folder, default = {Data}/predict_images
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. The .h5 extension may be omitted.
                        Default = BasicSR-60-60-2-dpx.h5
    test=1|0|T|F        If true, only predict first, last and middle image. Default=False
    png=1|0|T|F         If true, force output images to be png. Default=False
    diff=1|0|T|F        If true, generate input/output difference images. Default=False
                        Generates regular (-diff) and normalized (-ndiff) images.

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching .json file for the model
    (ie: BasicSR-60-60-2-dpx.json) that contains all the tiling/trimming information
"""


import os
import json

import numpy as np

from Modules.misc import oops, terminate, set_docstring, parse_options, printlog
import Modules.frameops as frameops
from Modules.modelio import ModelIO
import Modules.models as models


set_docstring(__doc__)

# Turn debug code on and off

DEBUG = True

def setup(options):
    """Set up configuration """

    # Set remaining options

    options.setdefault('test', False)
    options.setdefault('png', False)
    options.setdefault('diff', False)

    options.setdefault('data', 'Data')
    dpath = options['data']

    options.setdefault('model', os.path.join(dpath, 'models', 'BasicSR-60-60-2-dpx.h5'))
    options.setdefault('predict', os.path.join(dpath, 'predict_images'))

    if not options['model'].endswith('.h5'):
        options['model'] = options['model'] + '.h5'

    if not os.path.exists(options['model']):
        mpath = os.path.join(dpath, 'models', options['model'])
        print('Model not found, looking for', mpath)
        if not os.path.exists(mpath):
            mpath = os.path.join(dpath, 'models', 'genes', options['model'])
            print('Model not found, looking for', mpath)
        if not os.path.exists(mpath):
            mpath = os.path.join(dpath, 'models', 'submodels', options['model'])
            print('Model not found, looking for', mpath)
        options['model'] = mpath

    options['state'] = os.path.splitext(options['model'])[0] + '.json'

    model_type = os.path.basename(options['model']).split('-')[0]

    # Remind user what we're about to do.

    print('             Data : {}'.format(options['data']))
    print('   Predict Images : {}'.format(options['predict']))
    print('            Model : {}'.format(options['model']))
    print('            State : {}'.format(options['state']))
    print('       Model Type : {}'.format(model_type))
    print('        Test Mode : {}'.format(options['test']))
    print('        Force PNG : {}'.format(options['png']))
    print(' Make Differences : {}'.format(options['diff']))
    print('')

    # Validation and error checking

    errors = False
    for path in ['predict', 'state', 'model', 'data']:
        errors = oops(errors,
                      not os.path.exists(options[path]),
                      'Path to {} is not valid ({})',
                      (path, options[path]))

    terminate(errors, False)

    # Load the actual model state

    with open(options['state'], 'r') as jsonfile:
        state = json.load(jsonfile)

    # Grab the config data (backwards compatible)

    config = state['config' if 'config' in state else 'io']

    # Create real config with configurable parameters. In particular we disable options like
    # jitter, shuffle, skip and quality, but we set edges to true because we most definitely
    # want to process edge tiles

    config['paths'] = options
    config['jitter'] = False
    config['shuffle'] = False
    config['skip'] = False
    config['quality'] = 1.0
    config['edges'] = True
    config['model_type'] = model_type

    for option in ['test', 'png', 'diff']:
        config[option] = options[option]

    config = ModelIO(config)

    # Check image files -- we do not explore subfolders. Note we have already checked
    # path validity above

    image_info = frameops.image_files(os.path.join(config.paths['predict'], config.alpha), False)

    errors = oops(False,
                  not image_info,
                  'Input folder does not contain any images')
    errors = oops(errors,
                  len(image_info) > 1,
                  'Images folder contains more than one type of image')

    terminate(errors, False)

    # Get the list of files and check the filetype is correct

    image_info = image_info[0]

    image_ext = os.path.splitext(image_info[0])[1][1:].lower()

    errors = oops(errors,
                  image_ext != config.img_suffix.lower(),
                  'Image files are of type {} but model was trained on {}',
                  (image_ext, config.img_suffix.lower()))

    terminate(errors, True)

    return (config, image_info)

def predict(config, image_info):
    """ Run predictions using the configuration and list of files """

    # Create model. Since the model file contains the complete model info, not just the
    # weights, we can instantiate it using the base class. So no matter what changes we
    # make to the definition of the models in models.py, old model files will still
    # work.

    sr_model = models.BaseSRCNNModel(name=config.model_type, config=config)

    # Need to use unjittered tiles_per_imag

    tiles_per_img = config.tiles_across * config.tiles_down

    # Process the images

    if config.config['test']:
        image_info = [image_info[0], image_info[len(image_info) // 2], image_info[-1]]  # just do a couple of images

    # There is no point in caching tiles since we never revisit them.

    frameops.reset_cache(enabled=False)

    for img_path in image_info:
        printlog('Predicting', os.path.basename(img_path))

        # Generate the tiles for the image. Note that tiles is a generator

        tiles = frameops.tesselate(img_path, config)

        # Create a batch with all the tiles

        tile_batch = np.empty((tiles_per_img, ) + config.image_shape)
        for idx, tile in enumerate(tiles):
            tile_batch[idx] = tile

        """
        if DEBUG:
            fname = os.path.basename(img_path)
            for i in range(0, min(30, tiles_per_img)):
                fpath = os.path.join('Temp', 'PNG', fname[:-4] + '-' + str(i) + '-IN.png')
                frameops.imsave(fpath, tile_batch[i])
            input_image = frameops.grout(tile_batch, config)
            fpath = os.path.join('Temp', 'PNG', os.path.basename(img_path)[:-4] + '-IN.png')
            frameops.imsave(fpath, input_image)
        """

        # Predict the new tiles in relatively small chunks so the GPU doesn't get clogged

        predicted_tiles = sr_model.model.predict(tile_batch, min(config.tiles_across, config.tiles_down))

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

        basename = os.path.basename(img_path)
        fname, ext = os.path.splitext(basename)
        ext = '.png' if config.config['png'] else ext
        basename = fname + ext

        fpath = os.path.join(config.paths['predict'], config.beta, basename)
        frameops.imsave(fpath, predicted_image)

        if config.config['diff']:
            difference_tiles = np.absolute(predicted_tiles - tile_batch)
            difference_image = frameops.grout(difference_tiles, config)
            basename = fname + '-diff' + ext
            fpath = os.path.join(config.paths['predict'], config.beta, basename)
            frameops.imsave(fpath, difference_image)

            # Also generate normalized difference image (easier to see)

            axes = (1, 2) if config.theano else (0, 1)

            maxdiff = np.amax(difference_image)
            maxchan = np.amax(difference_image, axes) * 100.0
            avgchan = np.average(difference_image, axes) * 100.0

            difference_image /= maxdiff
            basename = fname + '-ndiff' + ext
            fpath = os.path.join(config.paths['predict'], config.beta, basename)
            frameops.imsave(fpath, difference_image)

            printlog('  Max channel color error: {:6.2f}%, {:6.2f}%, {:6.2f}%'.format(*maxchan))
            printlog('  Avg channel color error: {:6.2f}%, {:6.2f}%, {:6.2f}%'.format(*avgchan))

        """
        # Debug code to confirm what we are doing

        if DEBUG:
            fname = os.path.basename(img_path)
            for i in range(0, min(30, tiles_per_img)):
                fpath = os.path.join('Temp', 'PNG', fname[0:-4] + '-' + str(i) + '-OUT.png')
                frameops.imsave(fpath, predicted_tiles[i])

            fpath = os.path.join('Temp', 'PNG', os.path.basename(img_path)[:-4] + '-OUT.png')
            frameops.imsave(fpath, predicted_image)
        """

    printlog('Predictions completed...')


if __name__ == '__main__':
    OPCODES = {
        'test': ('test', bool, lambda x: not isinstance(x, bool), 'Test value invalid ({}). Must be 0, 1, T, F.'),
        'png': ('png', bool, lambda x: not isinstance(x, bool), 'Png value invalid ({}). Must be 0, 1, T, F.'),
        'diff': ('diff', bool, lambda x: not isinstance(x, bool), 'Diff value invalid ({}). Must be 0, 1, T, F.'),
        'data': ('data', str, lambda x: False, ''),
        'images': ('predict', str, lambda x: False, ''),
        'model': ('model', str, lambda x: False, '')}

    OPTIONS = parse_options(OPCODES)
    CONFIG, IMAGE_INFO = setup(OPTIONS)
    predict(CONFIG, IMAGE_INFO)
