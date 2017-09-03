"""
Based on:

https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc

Read Metadata and Image data from 10-bit DPX files in Python 3

Copyright (c) 2016 Jack Doerner

Tweaked to also work in Python 2 (fixed normalization code to ensure floating
point division) -- RJW 08/19/17

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

import struct
import numpy as np

orientations = {
    0: 'Left to Right, Top to Bottom',
    1: 'Right to Left, Top to Bottom',
    2: 'Left to Right, Bottom to Top',
    3: 'Right to Left, Bottom to Top',
    4: 'Top to Bottom, Left to Right',
    5: 'Top to Bottom, Right to Left',
    6: 'Bottom to Top, Left to Right',
    7: 'Bottom to Top, Right to Left'
}

descriptors = {
    1: 'Red',
    2: 'Green',
    3: 'Blue',
    4: 'Alpha',
    6: 'Luma (Y)',
    7: 'Color Difference',
    8: 'Depth (Z)',
    9: 'Composite Video',
    50: 'RGB',
    51: 'RGBA',
    52: 'ABGR',
    100: 'Cb, Y, Cr, Y (4:2:2)',
    102: 'Cb, Y, Cr (4:4:4)',
    103: 'Cb, Y, Cr, A (4:4:4:4)'
}

packings = {
    0: 'Packed into 32-bit words',
    1: 'Filled to 32-bit words, Padding First',
    2: 'Filled to 32-bit words, Padding Last'
}

encodings = {
    0: 'No encoding',
    1: 'Run Length Encoding'
}

transfers = {
    1: 'Printing Density',
    2: 'Linear',
    3: 'Logarithmic',
    4: 'Unspecified Video',
    5: 'SMPTE 274M',
    6: 'ITU-R 709-4',
    7: 'ITU-R 601-5 system B or G',
    8: 'ITU-R 601-5 system M',
    9: 'Composite Video (NTSC)',
    10: 'Composite Video (PAL)',
    11: 'Z (Linear Depth)',
    12: 'Z (Homogenous Depth)'
}

colorimetries = {
    1: 'Printing Density',
    4: 'Unspecified Video',
    5: 'SMPTE 274M',
    6: 'ITU-R 709-4',
    7: 'ITU-R 601-5 system B or G',
    8: 'ITU-R 601-5 system M',
    9: 'Composite Video (NTSC)',
    10: 'Composite Video (PAL)'
}

propertymap = [
    #(field name, offset, length, type)

    ('magic', 0, 4, 'magic'),
    ('offset', 4, 4, 'I'),
    ('dpx_version', 8, 8, 'utf8'),
    ('file_size', 16, 4, 'I'),
    ('ditto', 20, 4, 'I'),
    ('filename', 36, 100, 'utf8'),
    ('timestamp', 136, 24, 'utf8'),
    ('creator', 160, 100, 'utf8'),
    ('project_name', 260, 200, 'utf8'),
    ('copyright', 460, 200, 'utf8'),
    ('encryption_key', 660, 4, 'I'),

    ('orientation', 768, 2, 'H'),
    ('image_element_count', 770, 2, 'H'),
    ('width', 772, 4, 'I'),
    ('height', 776, 4, 'I'),

    ('data_sign', 780, 4, 'I'),
    ('descriptor', 800, 1, 'B'),
    ('transfer_characteristic', 801, 1, 'B'),
    ('colorimetry', 802, 1, 'B'),
    ('depth', 803, 1, 'B'),
    ('packing', 804, 2, 'H'),
    ('encoding', 806, 2, 'H'),
    ('line_padding', 812, 4, 'I'),
    ('image_padding', 816, 4, 'I'),
    ('image_element_description', 820, 32, 'utf8'),

    ('input_device_name', 1556, 32, 'utf8'),
    ('input_device_sn', 1588, 32, 'utf8'),

    ('gamma', 1948, 4, 'f'),
    ('black_level', 1952, 4, 'f'),
    ('black_gain', 1956, 4, 'f'),
    ('break_point', 1960, 4, 'f'),
    ('white_level', 1964, 4, 'f')

]


def readDPXMetaData(f):
    f.seek(0)
    bytes = f.read(4)
    magic = bytes.decode(encoding='UTF-8')
    if magic != 'SDPX' and magic != 'XPDS':
        return None
    endianness = '>' if magic == 'SDPX' else '<'

    meta = {}

    for p in propertymap:
        f.seek(p[1])
        bytes = f.read(p[2])
        if p[3] == 'magic':
            meta[p[0]] = bytes.decode(encoding='UTF-8')
            meta['endianness'] = 'be' if magic == 'SDPX' else 'le'
        elif p[3] == 'utf8':
            meta[p[0]] = bytes.decode(encoding='UTF-8')
        elif p[3] == 'B':
            meta[p[0]] = struct.unpack(endianness + 'B', bytes)[0]
        elif p[3] == 'H':
            meta[p[0]] = struct.unpack(endianness + 'H', bytes)[0]
        elif p[3] == 'I':
            meta[p[0]] = struct.unpack(endianness + 'I', bytes)[0]
        elif p[3] == 'f':
            meta[p[0]] = struct.unpack(endianness + 'f', bytes)[0]

    return meta


def readDPXImageData(f, meta):
    if meta['depth'] != 10 or meta['packing'] != 1 or meta['encoding'] != 0 or meta['descriptor'] != 50:
        return None

    width = meta['width']
    height = meta['height']
    image = np.empty((height, width, 3), dtype=float)

    f.seek(meta['offset'])
    raw = np.fromfile(f, dtype=np.dtype(np.int32),
                      count=width * height, sep='')
    raw = raw.reshape((height, width))

    if meta['endianness'] == 'be':
        raw = raw.byteswap()

    # extract and normalize color channel values to 0..1 inclusive.

    image[:, :, 0] = ((raw >> 22) & 0x000003FF) / 1023.0
    image[:, :, 1] = ((raw >> 12) & 0x000003FF) / 1023.0
    image[:, :, 2] = ((raw >> 2) & 0x000003FF) / 1023.0

    return image


def writeDPX(f, image, meta):
    endianness = '>' if meta['endianness'] == 'be' else '<'
    for p in propertymap:
        if p[0] in meta:
            f.seek(p[1])
            if p[3] == 'magic':
                bytes = ('SDPX' if meta['endianness'] ==
                         'be' else 'XPDS').encode(encoding='UTF-8')
            elif p[3] == 'utf8':
                bytes = meta[p[0]].encode(encoding='UTF-8')
            else:
                bytes = struct.pack(endianness + p[3], meta[p[0]])
            f.write(bytes)

    raw = ((((image[:, :, 0] * 1023.0).astype(np.dtype(np.int32)) & 0x000003FF) << 22)
           | (((image[:, :, 1] * 1023.0).astype(np.dtype(np.int32)) & 0x000003FF) << 12)
           | (((image[:, :, 2] * 1023.0).astype(np.dtype(np.int32)) & 0x000003FF) << 2)
           )

    if meta['endianness'] == 'be':
        raw = raw.byteswap()

    f.seek(meta['offset'])
    raw.tofile(f, sep='')
