""" Tests for Modules/dpx.py """

import os
import numpy as np
import Modules.dpx as dpx

_ROOT = os.path.dirname(os.path.abspath(__file__))

_DATAPATH = os.path.join(_ROOT, 'Data')
_TMPPATH = os.path.join(_DATAPATH, 'Temp')
_IMGPATH = os.path.join(_DATAPATH, 'Images')
_DPXPATH = os.path.join(_IMGPATH, 'DPX')
_PNGPATH = os.path.join(_IMGPATH, 'PNG')
_EMPTYPATH = os.path.join(_IMGPATH, 'Empty')

_IMG = 'frame-002992'
_DPX = os.path.join(_DPXPATH, _IMG + '.dpx')
_TMP_DPX = os.path.join(_TMPPATH, _IMG + '.dpx')
_PNG = os.path.join(_PNGPATH, _IMG + '.png')

_FAKE = os.path.join(_TMPPATH, 'fake.jpg')

def test_read():
    """ Test dpx.read """

    dpxfile = open(_DPX, 'rb')
    result = dpx.read(dpxfile)

    assert result is not None
    assert np.shape(result) == (480, 720, 3)
    assert dpx.DPX_META == {'width': 720,
                            'descriptor': 50,
                            'packing': 1,
                            'file_size': 1390592,
                            'x_originalsize': 720,
                            'encoding': 0,
                            'height': 480,
                            'depth': 10,
                            'endianness': '<',
                            'offset': 8192,
                            'y_originalsize': 480}

def test_save():
    """ Test dpx.save """

    dpxfile = open(_DPX, 'rb')
    img = dpx.read(dpxfile)
    img_meta = dpx.DPX_META

    dpxfile = open(_TMP_DPX, 'wb')
    dpx.save(dpxfile, img)

    dpxfile = open(_TMP_DPX, 'rb')
    img2 = dpx.read(dpxfile)
    dpxfile.close()

    os.remove(_TMP_DPX)

    assert np.array_equal(img, img2)
    assert img_meta == dpx.DPX_META
