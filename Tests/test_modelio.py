""" Tests for Modules/misc.py """

from copy import deepcopy
import itertools
import Modules.modelio as modelio

PARAMETERS = ['model_type',
              'paths',
              'image_width',
              'image_height',
              'base_tile_width',
              'base_tile_height',
              'border',
              'border_mode',
              'trim_left',
              'trim_right',
              'trim_top'
              'trim_bottom',
              'channels',
              'black_level',
              'batch_size',
              'img_suffix',
              'jitter',
              'shuffle',
              'skip',
              'residual',
              'quality',
              'epochs',
              'run_epochs',
              'learning_rate',
              'verbose',
              'bargraph']

def check_modelio(obj, config):
    """ Check the object (a ModelIO) for consistency with a config dictionary """

    # copy the object's variables and remove config.

    objvars = deepcopy(obj.__dict__)
    objconfig = objvars.pop('config')

    # confirm created constants are there, then delete them
    # from objvars

    assert obj.beta == 'Beta'
    assert obj.alpha == 'Alpha'

    objvars.pop('alpha')
    objvars.pop('beta')

    # confirm that everything in template config got created in
    # the object as a variable and is in the object config

    for key in config:
        assert key in objvars
        assert key in objconfig
        assert objvars[key] == config[key]
        assert objvars[key] == objconfig[key]

    # check that everything in the full object config (which is a
    # superset of the template config) got created as a variable.

    for key in objconfig:
        val = objvars.pop(key, None)
        assert val is not None
        assert val == objconfig[key]

    # Check that there is nothing left over

    assert not objvars

def test_modelio_init():
    """ Test initialization of a ModelIO class """

    # init all defaults

    obj = modelio.ModelIO()
    objconfig = obj.config
    check_modelio(obj, objconfig)

    # for each item in our default config, twiddle it, create a new
    # object, and check that it is consistent. Do only for the
    # parameters that will be passed into an instantiation.

    # the 'path' list is a special case handled later

    for key in objconfig:
        if key in PARAMETERS:
            val = objconfig[key]
            if isinstance(val, bool):
                val = not val
            elif isinstance(val, int):
                val = val + 1 if val == 0 else val * 2
            elif isinstance(val, float):
                val = val + 0.1 if val == 0 else val / 2
            elif isinstance(val, (list, tuple, dict)):
                pass
            elif val == 'BasicSR':
                val = 'ExpansionSR'
            elif val == 'constant':
                val = 'valid'
            elif val == 'dpx':
                val = 'png'
            else:
                assert False, 'Whoops, got [{}]'.format(val)
            #print('Testing', key, ':', objconfig[key], '->', val)
            config = {key: val}
            obj = modelio.ModelIO(config)
            assert obj.config[key] == val
            check_modelio(obj, config)

def test_modelio_paths():
    """ Test default path initialization """

    # init all defaults

    obj = modelio.ModelIO()
    default_paths = obj.paths

    # Check pathname default setup

    path_names = ['data',
                  'training',
                  'validation',
                  'evaluation',
                  'predict',
                  'genepool',
                  'genebank',
                  'model',
                  'state']

    for test_path in path_names:
        test_dict = {test_path: 'Tests123'}
        expected_paths = deepcopy(default_paths)
        if test_path == 'data':
            expected_paths = {key: 'Tests123' + expected_paths[key][4:] for key in expected_paths}
        else:
            expected_paths[test_path] = 'Tests123'
        config = {'paths': test_dict}
        obj = modelio.ModelIO(config)
        assert obj.config['paths'] == expected_paths
        assert obj.paths == expected_paths

def test_modelio_computed():
    """ Test modelio computed information """

    obj = modelio.ModelIO()

    parameters = ['base_tile_width',
                  'base_tile_height',
                  'border',
                  'image_width',
                  'trim_left',
                  'trim_right',
                  'image_height',
                  'trim_top',
                  'trim_bottom',
                  'jitter']

    alt_values = {'base_tile_width': 30,
                  'base_tile_height': 15,
                  'border': 10,
                  'image_width': 3840,
                  'trim_left': 0,
                  'trim_right': 0,
                  'image_height': 2160,
                  'trim_top': 240,
                  'trim_bottom': 240,
                  'jitter': False}

    # Try all possible input permutations, just to be pedantic

    for taken in range(len(parameters)):
        for combination in itertools.combinations(parameters, taken):
            config = deepcopy(obj.config)
            for item in combination:
                config[item] = alt_values[item]
            nobj = modelio.ModelIO(config)

            assert nobj.config['tile_width'] == nobj.tile_width
            assert nobj.tile_width == nobj.border * 2 + nobj.base_tile_width

            assert nobj.config['tile_height'] == nobj.tile_height
            assert nobj.tile_height == nobj.border * 2 + nobj.base_tile_height

            assert nobj.config['trimmed_width'] == nobj.trimmed_width
            assert nobj.trimmed_width == nobj.image_width - (nobj.trim_left + nobj.trim_right)

            assert nobj.config['trimmed_height'] == nobj.trimmed_height
            assert nobj.trimmed_height == nobj.image_height - (nobj.trim_top + nobj.trim_bottom)

            assert nobj.config['tiles_across'] == nobj.tiles_across
            assert nobj.tiles_across == nobj.trimmed_width // nobj.base_tile_width

            assert nobj.config['tiles_down'] == nobj.tiles_down
            assert nobj.tiles_down == nobj.trimmed_height // nobj.base_tile_height

            assert nobj.config['tiles_per_image'] == nobj.tiles_per_image
            assert nobj.tiles_per_image == nobj.tiles_across * nobj.tiles_down + \
                ((nobj.tiles_across - 1) * (nobj.tiles_down - 1) if nobj.jitter else 0)

    # Test theano

    assert obj.config['image_shape'] == obj.image_shape
    assert obj.config['theano'] == obj.theano
    assert obj.image_shape[0 if obj.theano else 2] == 3
