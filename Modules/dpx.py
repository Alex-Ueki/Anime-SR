# pylint: disable=C0301, W0603
# (line_too_long, used_globals disabled)

"""
Based on:

https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc

Read Metadata and Image data from 10-bit DPX files in Python 3

Copyright (c) 2016 Jack Doerner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the 'Software'), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Note: this only works for simple, uncompressed DPX files.

import struct
import numpy as np

# Cache of DPX header information that we set whenever we read a file;
# used to write a file in the same format. Warning: Mutable globals!

DPX_META = None
DPX_HEADER = None

# More extensive propertymap can be found in dpxderex.py

SIMPLE_PROPERTYMAP = [

    # These elements are the ones that we have to mung.
    # They MUST appear in this position and in this order,
    # as dpx_save() depends on this correspondence.

    ('height', 776, 4, 'I'),
    ('width', 772, 4, 'I'),
    ('y_originalsize', 1428, 4, 'I'),
    ('x_originalsize', 1424, 4, 'I'),
    ('file_size', 16, 4, 'I'),

    # Other elements that we reference, but do not alter.

    ('offset', 4, 4, 'I'),

    ('descriptor', 800, 1, 'B'),
    ('depth', 803, 1, 'B'),
    ('packing', 804, 2, 'H'),
    ('encoding', 806, 2, 'H'),

]


def read(dpxfile):

    """ Read a DPX file and extract image. Stash the header and meta information in a global for use when writing. """

    global DPX_META
    global DPX_HEADER

    # Open file if passed a path instead of a file descriptor

    if isinstance(dpxfile, str):
        dpxfile = open(dpxfile, 'rb')
        
    # Figure out the byte order of the file

    dpxfile.seek(0)
    magic = dpxfile.read(4)
    magic = magic.decode(encoding='UTF-8')
    if magic != 'SDPX' and magic != 'XPDS':
        return None
    dpx_endian = '>' if magic == 'SDPX' else '<'

    # Read the header values we need

    DPX_META = {'endianness': dpx_endian}

    for prop in SIMPLE_PROPERTYMAP:
        dpxfile.seek(prop[1])
        rawbytes = dpxfile.read(prop[2])
        if prop[3] == 'utf8':
            DPX_META[prop[0]] = rawbytes.decode(encoding='UTF-8')
        elif prop[3] == 'B':
            DPX_META[prop[0]] = struct.unpack(dpx_endian + 'B', rawbytes)[0]
        elif prop[3] == 'H':
            DPX_META[prop[0]] = struct.unpack(dpx_endian + 'H', rawbytes)[0]
        elif prop[3] == 'I':
            DPX_META[prop[0]] = struct.unpack(dpx_endian + 'I', rawbytes)[0]
        elif prop[3] == 'f':
            DPX_META[prop[0]] = struct.unpack(dpx_endian + 'f', rawbytes)[0]

    # Grab a copy of the whole header

    dpxfile.seek(0)
    DPX_HEADER = dpxfile.read(DPX_META['offset'])

    # If the format is not what we expect, don't proceed

    if DPX_META['depth'] != 10 or DPX_META['packing'] != 1 or DPX_META['encoding'] != 0 or DPX_META['descriptor'] != 50:
        return None

    # Read and decode the image

    width = DPX_META['width']
    height = DPX_META['height']
    image = np.empty((height, width, 3), dtype=float)

    dpxfile.seek(DPX_META['offset'])
    raw = np.fromfile(dpxfile, dtype=np.dtype(np.int32),
                      count=width * height, sep='')

    raw = raw.reshape((height, width))

    # GPU : Some of the .zip images didn't transfer correctly
    # I put this try-block to catch these bad images and warn me if I lost one
    # If you get an error like the following, uncomment this
    #    raw = raw.reshape((height, width))
    #    ValueError: cannot reshape array of size 0 into shape (1080,1920)
    """
    try:
        raw = raw.reshape((height, width))
    except:
        [_, img_name, _] = str.split(str(f), "'")
        print("Reshaping image failed for %s" % img_name)
        print("You may want to check this file's size")
        sys.exit(1)
    """

    if dpx_endian == '>':
        raw = raw.byteswap()

    # extract and normalize color channel values to 0..1 inclusive.

    image[:, :, 0] = ((raw >> 22) & 0x000003FF) / 1023.0
    image[:, :, 1] = ((raw >> 12) & 0x000003FF) / 1023.0
    image[:, :, 2] = ((raw >> 2) & 0x000003FF) / 1023.0

    return image


def save(dpxfile, image, meta=None, header=None):
    """ Use cached header and meta information if not explicitly specified. While we can
        rewrite the header to handle image size changes, there may be situations where
        cloning the header/meta information may be more appropriate.
    """

    header = DPX_HEADER if header is None else header
    meta = DPX_META if meta is None else meta

    # Write the cached header

    dpxfile.seek(0)
    dpxfile.write(header)

    # Now we have to get a little clever. If the image size does not match the size
    # recorded in our cached dpx header, we need to tweak those values. This will
    # happen if we are reading 720x480 or 720x486 images in predict.py, but are
    # generating 1920x1080.

    shape = np.shape(image)
    dpx_endian = meta['endianness']

    if shape[0] != meta['height'] or shape[1] != meta['width']:

        # Write updated header records. Depends on fields in SIMPLE_PROPERTYMAP
        # being in the correct order.

        for idx, val in enumerate([shape[0], shape[1], shape[0], shape[1], meta['offset'] + (shape[0] * shape[1] * 4)]):
            prop = SIMPLE_PROPERTYMAP[idx]
            dpxfile.seek(prop[1])
            rawbytes = struct.pack(dpx_endian + prop[3], val)
            dpxfile.write(rawbytes)

    # Write the image data

    raw = ((((image[:, :, 0] * 1023.0).astype(np.dtype(np.int32)) & 0x000003FF) << 22)
           | (((image[:, :, 1] * 1023.0).astype(np.dtype(np.int32)) & 0x000003FF) << 12)
           | (((image[:, :, 2] * 1023.0).astype(np.dtype(np.int32)) & 0x000003FF) << 2)
          )

    if dpx_endian == '>':
        raw = raw.byteswap()

    dpxfile.seek(meta['offset'])
    raw.tofile(dpxfile, sep='')
