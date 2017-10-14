""" Tests for Modules/dpx.py """

import os
import numpy as np
import Modules.frameops as frameops
import Modules.modelio as modelio

_ROOT = os.path.dirname(os.path.abspath(__file__))

_DATAPATH = os.path.join(_ROOT, 'Data')
_TMPPATH = os.path.join(_DATAPATH, 'Temp')
_IMGPATH = os.path.join(_DATAPATH, 'Images')
_DPXPATH = os.path.join(_IMGPATH, 'DPX')
_PNGPATH = os.path.join(_IMGPATH, 'PNG')
_EMPTYPATH = os.path.join(_IMGPATH, 'Empty')

# 2 DPX and 1 PNG image

_IMG = 'frame-002992'
_IMG0 = 'frame-000000'
_DPX = os.path.join(_DPXPATH, _IMG + '.dpx')
_DPX0 = os.path.join(_DPXPATH, _IMG0 + '.dpx')
_TMP_DPX = os.path.join(_TMPPATH, _IMG + '.dpx')
_PNG = os.path.join(_PNGPATH, _IMG + '.png')

_FAKE = os.path.join(_TMPPATH, 'fake.jpg')

def test_reset_cache():
    """ Test reset_cache() """

    frameops.reset_cache()

    assert frameops.CACHED_TILES == {}
    assert frameops.CACHED_QUALITY == {}
    assert frameops.CACHING

    frameops.reset_cache(False)

    assert frameops.CACHED_TILES == {}
    assert frameops.CACHED_QUALITY == {}
    assert not frameops.CACHING

def test_image_file():
    """ Test image_files() fetching """

    # non-deep fetching, null result

    result = frameops.image_files(_IMGPATH)

    assert result == []

    # non-deep fetching, positive result (also tests .jpg finding)

    open(_FAKE, 'a').close()
    result = frameops.image_files(_TMPPATH)
    os.remove(_FAKE)

    assert len(result) == 1
    expected = [[os.path.join(_TMPPATH, 'fake.jpg')]]
    assert result == expected

    # deep fetching, positive result

    result = frameops.image_files(_IMGPATH, deep=True)

    assert len(result) == 2

    expected = [[os.path.join(_PNGPATH, _IMG0 + '.png'), os.path.join(_PNGPATH, _IMG + '.png')],
                [os.path.join(_DPXPATH, _IMG0 + '.dpx'), os.path.join(_DPXPATH, _IMG + '.dpx')]]
    assert result == expected

def test_imread():
    """ Test frameops.imread() """

    # file does not exist

    result = frameops.imread(_FAKE)

    assert result is None

    # DPX file (dpx tests will do finer result)

    result = frameops.imread(_DPX)

    assert result is not None
    assert np.shape(result) == (480, 720, 3)

    # PNG file

    result = frameops.imread(_PNG)

    assert result is not None
    assert np.shape(result) == (1080, 1920, 3)


def test_imsave():
    """ Test frameops.imsave() is covered by dpx.save and misc.imsave """
    pass

def test_extract_tiles():
    """ Test frameops.extract_tiles() """

    # do not want caching to happen because of all the testing we are doing.

    frameops.reset_cache(False)

    # set up a default model configuration

    config = modelio.ModelIO({'shuffle': False, 'jitter': False, 'skip': False})

    # extract from a 1920x1080 png with 4x3 clipping

    tiles = frameops.extract_tiles(_PNG, config)

    assert len(tiles) == 432

    for tile in tiles:
        assert np.shape(tile) == (64, 64, 3)

    # extract from a 720x480 dpx (with implicit scaling)

    tiles = frameops.extract_tiles(_DPX, config)

    assert len(tiles) == 432

    for tile in tiles:
        assert np.shape(tile) == (64, 64, 3)

    # extract from a 720x480 dpx with jittering

    config = modelio.ModelIO({'shuffle': False, 'jitter': True})

    tiles = frameops.extract_tiles(_DPX, config)

    assert len(tiles) == 823

    for tile in tiles:
        assert np.shape(tile) == (64, 64, 3)

def test_tesselate():
    """ Test frameops.tesselate() """

    # do not want caching to happen because of all the testing we are doing.

    frameops.reset_cache(False)

    # set up a default model configuration

    config = modelio.ModelIO({'shuffle': False, 'jitter': False, 'skip': False})

    # get actual tiles to check against

    extracted = frameops.extract_tiles(_DPX, config)

    # tesselate() is a generator (will listify _DPX btw...)

    tiles = [t for t in frameops.tesselate(_DPX, config)]

    assert len(tiles) == len(extracted)
    assert all([np.array_equal(a, b) for a, b in zip(tiles, extracted)])

    # test jitter

    config = modelio.ModelIO({'shuffle': False, 'jitter': True, 'skip': False})
    extracted = frameops.extract_tiles(_DPX, config)
    tiles = [t for t in frameops.tesselate(_DPX, config)]

    assert len(tiles) == len(extracted)
    assert all([np.array_equal(a, b) for a, b in zip(tiles, extracted)])

    # test skip

    config = modelio.ModelIO({'shuffle': False, 'jitter': False, 'skip': True})

    extracted = frameops.extract_tiles(_DPX, config)
    tiles = [t for t in frameops.tesselate(_DPX, config)]

    # Checking for matches is a little tricky, because x in y where y is a
    # list of numpy arrays is ambiguous.

    assert len(tiles) < len(extracted)
    for tile in tiles:
        matches = [np.array_equal(t, tile) for t in extracted]
        assert any(matches)
        del extracted[matches.index(True)]

    # test shuffle

    config = modelio.ModelIO({'shuffle': True, 'jitter': False, 'skip': False})

    extracted = frameops.extract_tiles(_DPX, config)
    tiles = [t for t in frameops.tesselate(_DPX, config)]

    assert len(tiles) == len(extracted)
    for tile in tiles:
        matches = [np.array_equal(t, tile) for t in extracted]
        assert any(matches)
        del extracted[matches.index(True)]

    # test shuffle + jitter

    config = modelio.ModelIO({'shuffle': True, 'jitter': True, 'skip': False})

    extracted = frameops.extract_tiles(_DPX, config)
    tiles = [t for t in frameops.tesselate(_DPX, config)]

    assert len(tiles) == len(extracted)
    for tile in tiles:
        matches = [np.array_equal(t, tile) for t in extracted]
        assert any(matches)
        del extracted[matches.index(True)]

    # test shuffle + jitter + skip

    config = modelio.ModelIO({'shuffle': True, 'jitter': True, 'skip': True})

    extracted = frameops.extract_tiles(_DPX, config)
    tiles = [t for t in frameops.tesselate(_DPX, config)]

    assert len(tiles) < len(extracted)
    for tile in tiles:
        matches = [np.array_equal(t, tile) for t in extracted]
        assert any(matches)
        del extracted[matches.index(True)]

def test_caching():
    """ Test that tile caching is working correctly """

    # cache off

    frameops.reset_cache(False)
    config = modelio.ModelIO({'shuffle': False, 'jitter': False, 'skip': False})
    extracted = frameops.extract_tiles(_DPX, config)
    tiles = [t for t in frameops.tesselate(_DPX, config)]

    assert not frameops.CACHED_TILES
    assert not frameops.CACHED_QUALITY

    # cache on, quality 100%

    frameops.reset_cache(True)
    config = modelio.ModelIO({'shuffle': False, 'jitter': False, 'skip': False})
    extracted = frameops.extract_tiles(_DPX, config)
    tiles = [t for t in frameops.tesselate(_DPX, config)]

    assert frameops.CACHED_TILES
    assert not frameops.CACHED_QUALITY
    assert _DPX in frameops.CACHED_TILES

    tiles = frameops.CACHED_TILES[_DPX]

    assert len(tiles) == len(extracted)
    assert all([np.array_equal(a, b) for a, b in zip(tiles, extracted)])

    # cache on, quality 50%

    frameops.reset_cache(True)
    config = modelio.ModelIO({'shuffle': False, 'jitter': False, 'skip': False, 'quality': 0.50})
    extracted = frameops.extract_tiles(_DPX, config)
    tiles = [t for t in frameops.tesselate(_DPX, config)]

    assert frameops.CACHED_TILES
    assert frameops.CACHED_QUALITY
    assert _DPX in frameops.CACHED_TILES
    assert _DPX in frameops.CACHED_QUALITY

    tiles = frameops.CACHED_TILES[_DPX]

    assert len(tiles) == len(extracted) // 2
    for tile in tiles:
        matches = [np.array_equal(t, tile) for t in extracted]
        assert any(matches)
        del extracted[matches.index(True)]


def test_tesselate_pair():
    """ Test frameops.tesselate_pair(). Also tests update_cache_quality() """

    # For ease of checking, we will test using the same image on both sides
    # of the pair.

    frameops.reset_cache(False)

    # Check all options off

    config = modelio.ModelIO({'shuffle': False,
                              'jitter': False,
                              'skip': False,
                              'quality': 1.0,
                              'residual': False})
    tiles = [tp for tp in frameops.tesselate_pair(_DPX, _DPX, config)]

    assert len(tiles) == 432
    assert all([np.array_equal(a, b) for a, b in tiles])

    # confirm that residual generation works. Since both images are
    # identical, the residuals will all be zero.

    config = modelio.ModelIO({'shuffle': False,
                              'jitter': False,
                              'skip': False,
                              'quality': 1.0,
                              'residual': True})
    tiles = [tp for tp in frameops.tesselate_pair(_DPX, _DPX, config)]

    assert len(tiles) == 432
    assert not any([np.any(b) for _, b in tiles])

    # check that shuffles are matched

    config = modelio.ModelIO({'shuffle': True,
                              'jitter': False,
                              'skip': False,
                              'quality': 1.0,
                              'residual': False})
    tiles = [tp for tp in frameops.tesselate_pair(_DPX, _DPX, config)]

    assert len(tiles) == 432
    assert all([np.array_equal(a, b) for a, b in tiles])

    # check that jitters are matched

    config = modelio.ModelIO({'shuffle': False,
                              'jitter': True,
                              'skip': False,
                              'quality': 1.0,
                              'residual': False})
    tiles = [tp for tp in frameops.tesselate_pair(_DPX, _DPX, config)]

    assert len(tiles) == 823
    assert all([np.array_equal(a, b) for a, b in tiles])

    # check that skips are matched

    config = modelio.ModelIO({'shuffle': False,
                              'jitter': False,
                              'skip': True,
                              'quality': 1.0,
                              'residual': False})
    tiles = [tp for tp in frameops.tesselate_pair(_DPX, _DPX, config)]

    assert len(tiles) < 432
    assert all([np.array_equal(a, b) for a, b in tiles])

    # check that quality reductions are matched.

    config = modelio.ModelIO({'shuffle': False,
                              'jitter': False,
                              'skip': False,
                              'quality': 0.5,
                              'residual': False})
    tiles = [tp for tp in frameops.tesselate_pair(_DPX, _DPX, config)]

    assert len(tiles) == 432 // 2
    assert all([np.array_equal(a, b) for a, b in tiles])

    # turn everything on

    config = modelio.ModelIO({'shuffle': True,
                              'jitter': True,
                              'skip': True,
                              'quality': 0.5,
                              'residual': False})
    tiles = [tp for tp in frameops.tesselate_pair(_DPX, _DPX, config)]

    assert len(tiles) < 823
    assert all([np.array_equal(a, b) for a, b in tiles])

    assert not frameops.CACHED_TILES
    assert not frameops.CACHED_QUALITY

def test_grout():
    """ Test frameops.grout() """

    frameops.reset_cache(False)

    # the actual image we will be testing. Use PNG because it's 1920x1080
    # so it won't be autoscaled

    img = frameops.imread(_PNG)

    # get the tiles

    config = modelio.ModelIO({'shuffle': False,
                              'jitter': False,
                              'skip': False,
                              'residual': False})

    tiles = frameops.extract_tiles(_PNG, config)
    assert len(tiles) == 432

    # grout the tiles back together

    grouted = frameops.grout(tiles, config)

    assert np.shape(img) == np.shape(grouted)
    assert np.array_equal(img, grouted)

    # repeat check with no default borders

    config = modelio.ModelIO({'shuffle': False,
                              'jitter': False,
                              'skip': False,
                              'residual': False,
                              'trim_left': 0,
                              'trim_right': 0})

    tiles = frameops.extract_tiles(_PNG, config)
    grouted = frameops.grout(tiles, config)

    assert np.shape(img) == np.shape(grouted)
    assert np.array_equal(img, grouted)
