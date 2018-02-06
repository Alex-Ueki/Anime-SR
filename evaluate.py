# pylint: disable=C0301
# (line_too_long disabled)
"""
Usage: evaluate.py [option(s)] ...

    Evaluates quality of a model.

Options are:

    data=path           path to the main data folder, default = Data
    images=path         path to images folder, default = {Data}/eval_images
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. The .h5 extension may be omitted.
                        Default = BasicSR-60-60-2-dpx.h5

    Option names may be any unambiguous prefix of the option (ie: w=60, wid=60 and width=60 are all OK)

    Expects that there will be a matching .json file for the model
    (ie: BasicSR-60-60-2-dpx.json) that contains all the tiling/trimming information
"""

import os
import json

from Modules.misc import oops, terminate, set_docstring, parse_options
import Modules.frameops as frameops
from Modules.modelio import ModelIO
import Modules.models as models

set_docstring(__doc__)

# Turn debug code on and off

DEBUG = True


def setup(options):
    """Set up configuration """

    # Set remaining options

    options.setdefault('data', 'Data')
    dpath = options['data']

    options.setdefault('model', os.path.join(dpath, 'models', 'BasicSR-R-60-60-2-dpx.h5'))
    options.setdefault('evaluation', os.path.join(dpath, 'eval_images'))

    if not options['model'].endswith('.h5'):
        options['model'] = options['model'] + '.h5'

    if os.path.dirname(options['model']) == '':
        options['model'] = os.path.join(dpath, 'models', options['model'])

    options['state'] = os.path.splitext(options['model'])[0] + '.json'

    model_type = os.path.basename(options['model']).split('-')[0]

    # Remind user what we're about to do.

    print('             Data : {}'.format(options['data']))
    print('Evaluation Images : {}'.format(options['evaluation']))
    print('            Model : {}'.format(options['model']))
    print('            State : {}'.format(options['state']))
    print('       Model Type : {}'.format(model_type))
    print('')

    # Validation and error checking

    errors = False
    for path in ['evaluation', 'state', 'model', 'data']:
        errors = oops(False,
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
    # jitter, shuffle, skip and quality.

    config['paths'] = options
    config['jitter'] = False
    config['shuffle'] = False
    config['skip'] = False
    config['edges'] = True
    config['quality'] = 1.0
    config['model_type'] = model_type

    config = ModelIO(config)

    # Check image files -- we do not explore subfolders. Note we have already checked
    # path validity above

    image_info = frameops.image_files(os.path.join(config.paths['evaluation'], config.alpha), False)

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

    terminate(errors, False)

    return (config, image_info)

def evaluate(config):
    """ Evaluate using the model configuration """

    # Create model. Since the model file contains the complete model info, not just the
    # weights, we can instantiate it using the base class. So no matter what changes we
    # make to the definition of the models in models.py, old model files will still
    # work.

    sr_model = models.BaseSRCNNModel(name=config.model_type, config=config)

    # There is no point in caching tiles

    frameops.reset_cache(enabled=False)

    # Process the images

    sr_model.evaluate()

    print('')
    print('Evaluation completed...')

if __name__ == '__main__':
    OPCODES = {
        'data': ('data', str, lambda x: False, ''),
        'images': ('evaluation', str, lambda x: False, ''),
        'model': ('model', str, lambda x: False, '')}

    OPTIONS = parse_options(OPCODES)
    CONFIG, _ = setup(OPTIONS)
    evaluate(CONFIG)
