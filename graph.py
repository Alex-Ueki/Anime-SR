# pylint: disable=C0301
# (line_too_long disabled)
"""
Usage: graph.py [option(s)] ...

    Generates nice graph of a model.

Options are:

    data=path           path to the main data folder, default = Data
    model=filename|path model filename or absolute path. If just a filename, then the
                        path will be {Data}/models/{model}. The .h5 extension may be omitted.
                        Default = BasicSR-60-60-2-dpx.h5
    graph=path          output file path. Default = {Data}/models/graphs/{model}.png

    Expects that there will be a matching .json file for the model
    (ie: BasicSR-60-60-2-dpx.json) that contains all the tiling/trimming information
"""

import os
import json

from Modules.misc import oops, terminate, set_docstring, parse_options
from Modules.modelio import ModelIO
import Modules.models as models
from keras.utils import plot_model

set_docstring(__doc__)

# Adjust basepath to include graphing software.

_PATH_TO_GRAPHVIZ = 'C:/Program Files (x86)/Graphviz2.38/bin/'

os.environ["PATH"] += os.pathsep + _PATH_TO_GRAPHVIZ

# Turn debug code on and off

DEBUG = True


def setup(options):
    """Set up configuration """

    # Set remaining options

    options.setdefault('data', 'Data')
    dpath = options['data']

    options.setdefault('model', os.path.join(dpath, 'models', 'BasicSR-60-60-2-dpx.h5'))

    basename = os.path.basename(options['model'])
    graphname = os.path.splitext(basename)[0] + '.png'
    options.setdefault('graph', os.path.join(dpath, 'models', 'graphs', graphname))

    model_split = os.path.splitext(options['model'])
    if model_split[1] == '':
        options['model'] = options['model'] + '.h5'
    if os.path.dirname(options['model']) == '':
        options['model'] = os.path.join(dpath, 'models', options['model'])

    options['state'] = os.path.splitext(options['model'])[0] + '.json'

    model_type = basename.split('-')[0]

    # Remind user what we're about to do.

    print('             Data : {}'.format(options['data']))
    print('            Model : {}'.format(options['model']))
    print('            State : {}'.format(options['state']))
    print('            Graph : {}'.format(options['graph']))
    print('       Model Type : {}'.format(model_type))
    print('')

    # Validation and error checking

    errors = False
    for path in ['graph', 'state', 'model', 'data']:
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
    config['model_type'] = model_type

    config = ModelIO(config)

    return (config, None)

def graph(config):
    """ Evaluate using the model configuration """

    # Create model. Since the model file contains the complete model info, not just the
    # weights, we can instantiate it using the base class. So no matter what changes we
    # make to the definition of the models in models.py, old model files will still
    # work.

    sr_model = models.BaseSRCNNModel(name=config.model_type, config=config)

    plot_model(sr_model.model, show_shapes=True, to_file=config.paths['graph'])

    print('')
    print('Graphing completed...')

if __name__ == '__main__':
    OPCODES = {
        'data': ('data', str, lambda x: False, ''),
        'graph': ('graph', str, lambda x: False, ''),
        'model': ('model', str, lambda x: False, '')}

    OPTIONS = parse_options(OPCODES)
    CONFIG, _ = setup(OPTIONS)
    graph(CONFIG)
