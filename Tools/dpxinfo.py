# pylint: disable=C0301
# (line_too_long disabled)
"""
python3 dpxinfo.py {source file}

Displays information about DPX file headers

Based on:

https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc

Read Metadata and Image data from 10-bit DPX files in Python 3

Copyright (c) 2016 Jack Doerner

Tweaked to also work in Python 2 (fixed normalization code to ensure floating
point division) -- RJW 08/19/17

Hacked to brute-force copy the header information -- RJW 09/04/17

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys
import os
import struct
import numpy as np

# Cache of DPX header information that we set when we read a file;
# used to write a file in the same format.

DPX_HEADER = None
DPX_ENDIAN = None
DPX_META = None
DPX_OFFSET = -1

ORIENTATIONS = {
    0: "Left to Right, Top to Bottom",
    1: "Right to Left, Top to Bottom",
    2: "Left to Right, Bottom to Top",
    3: "Right to Left, Bottom to Top",
    4: "Top to Bottom, Left to Right",
    5: "Top to Bottom, Right to Left",
    6: "Bottom to Top, Left to Right",
    7: "Bottom to Top, Right to Left"
}

DESCRIPTORS = {
    1: "Red",
    2: "Green",
    3: "Blue",
    4: "Alpha",
    6: "Luma (Y)",
    7: "Color Difference",
    8: "Depth (Z)",
    9: "Composite Video",
    50: "RGB",
    51: "RGBA",
    52: "ABGR",
    100: "Cb, Y, Cr, Y (4:2:2)",
    102: "Cb, Y, Cr (4:4:4)",
    103: "Cb, Y, Cr, A (4:4:4:4)"
}

PACKINGS = {
    0: "Packed into 32-bit words",
    1: "Filled to 32-bit words, Padding First",
    2: "Filled to 32-bit words, Padding Last"
}

ENCODINGS = {
    0: "No encoding",
    1: "Run Length Encoding"
}

TRANSFERS = {
    1: "Printing Density",
    2: "Linear",
    3: "Logarithmic",
    4: "Unspecified Video",
    5: "SMPTE 274M",
    6: "ITU-R 709-4",
    7: "ITU-R 601-5 system B or G",
    8: "ITU-R 601-5 system M",
    9: "Composite Video (NTSC)",
    10: "Composite Video (PAL)",
    11: "Z (Linear Depth)",
    12: "Z (Homogenous Depth)"
}

COLORIMETRIES = {
    1: "Printing Density",
    4: "Unspecified Video",
    5: "SMPTE 274M",
    6: "ITU-R 709-4",
    7: "ITU-R 601-5 system B or G",
    8: "ITU-R 601-5 system M",
    9: "Composite Video (NTSC)",
    10: "Composite Video (PAL)"
}

# (field name, offset, length, type)

PROPERTYMAP = [

    # Generic Header

    ('magic', 0, 4, 'magic'),
    ('offset', 4, 4, 'I'),
    ('dpx_version', 8, 8, 'utf8'),
    ('file_size', 16, 4, 'I'),
    ('ditto', 20, 4, 'I'),
    ('generic_size', 24, 4, 'I'),
    ('industry_size', 28, 4, 'I'),
    ('user_size', 32, 4, 'I'),
    ('filename', 36, 100, 'utf8'),
    ('timestamp', 136, 24, 'utf8'),
    ('creator', 160, 100, 'utf8'),
    ('project_name', 260, 200, 'utf8'),
    ('copyright', 460, 200, 'utf8'),
    ('encryption_key', 660, 4, 'I'),
    ('generic_reserved', 664, 104, 's'),

    # Image Header

    ('orientation', 768, 2, 'H'),
    ('image_element_count', 770, 2, 'H'),
    ('width', 772, 4, 'I'),
    ('height', 776, 4, 'I'),

    # Only first image element  decoded

    ('data_sign', 780, 4, 'I'),
    ('low_data', 784, 4, 'I'),
    ('low_quantity', 788, 4, 'f'),
    ('high_data', 792, 4, 'I'),
    ('high_quantity', 796, 4, 'f'),
    ('descriptor', 800, 1, 'B'),
    ('transfer_characteristic', 801, 1, 'B'),
    ('colorimetry', 802, 1, 'B'),
    ('depth', 803, 1, 'B'),
    ('packing', 804, 2, 'H'),
    ('encoding', 806, 2, 'H'),
    ('line_padding', 812, 4, 'I'),
    ('image_padding', 816, 4, 'I'),
    ('image_element_description', 820, 32, 'utf8'),
    ('image_reserved', 852, 556, 's'),

    # Orientation header

    ('x_offset', 1408, 4, 'I'),
    ('y_offset', 1412, 4, 'I'),
    ('x_center', 1416, 4, 'f'),
    ('y_center', 1420, 4, 'f'),
    ('x_originalsize', 1424, 4, 'I'),
    ('y_originalsize', 1428, 4, 'I'),
    ('source_filename', 1432, 100, 'utf8'),
    ('source_timestamp', 1532, 24, 'utf8'),
    ('input_device_name', 1556, 32, 'utf8'),
    ('input_device_sn', 1588, 32, 'utf8'),
    ('border_xl', 1620, 2, 'H'),
    ('border_xr', 1622, 2, 'H'),
    ('border_yt', 1624, 2, 'H'),
    ('border_yb', 1626, 2, 'H'),
    ('aspect_h', 1628, 4, 'I'),
    ('aspect_v', 1632, 4, 'I'),
    ('orientation_reserved', 1636, 28, 's'),

    # Film industry info

    ('film_industry_header', 1664, 256, 's'),

    # Television industry info

    ('timecode', 1920, 4, 'I'),
    ('user_bits', 1924, 4, 'I'),
    ('interlace', 1928, 1, 'B'),
    ('field_number', 1929, 1, 'B'),
    ('video_signal', 1930, 1, 'B'),
    ('tv_padding', 1931, 1, 'B'),
    ('h_sample_rate', 1932, 4, 'f'),
    ('v_sample_rate', 1936, 4, 'f'),
    ('frame_rate', 1940, 4, 'f'),
    ('time_offset', 1944, 4, 'f'),
    ('gamma', 1948, 4, 'f'),
    ('black_level', 1952, 4, 'f'),
    ('black_gain', 1956, 4, 'f'),
    ('break_point', 1960, 4, 'f'),
    ('white_level', 1964, 4, 'f'),
    ('integration_times', 1968, 4, 'f'),
    ('tv_reserved', 1972, 76, 's')

]

def read_dpx_metadata(dpxfile):
    """ Read dpx file metadata """

    global DPX_HEADER
    global DPX_ENDIAN
    global DPX_OFFSET
    global DPX_META

    dpxfile.seek(0)
    rawbytes = dpxfile.read(4)
    magic = rawbytes.decode(encoding='UTF-8')
    if magic != "SDPX" and magic != "XPDS":
        return None
    endianness = ">" if magic == "SDPX" else "<"

    metadata = {}

    for prop in PROPERTYMAP:
        dpxfile.seek(prop[1])
        rawbytes = dpxfile.read(prop[2])
        if prop[0] in metadata:
            print('Duplicate map field', prop[0])
        if prop[3] == 'magic':
            metadata[prop[0]] = rawbytes.decode(encoding='UTF-8')
            metadata['endianness'] = "be" if magic == "SDPX" else "le"
        elif prop[3] == 'utf8':
            metadata[prop[0]] = rawbytes.decode(encoding='UTF-8')
        elif prop[3] == 'B':
            metadata[prop[0]] = struct.unpack(endianness + 'B', rawbytes)[0]
        elif prop[3] == 'H':
            metadata[prop[0]] = struct.unpack(endianness + 'H', rawbytes)[0]
        elif prop[3] == 'I':
            metadata[prop[0]] = struct.unpack(endianness + 'I', rawbytes)[0]
        elif prop[3] == 'f':
            metadata[prop[0]] = struct.unpack(endianness + 'f', rawbytes)[0]
        elif prop[3] == 's':
            metadata[prop[0]] = struct.unpack(endianness + str(prop[2]) + 's', rawbytes)[0]

    # Save header values

    DPX_ENDIAN = endianness
    DPX_OFFSET = metadata['offset']
    dpxfile.seek(0)
    DPX_HEADER = dpxfile.read(DPX_OFFSET)
    DPX_META = metadata

    return metadata

def read_dpx_imagedata(fname, metadata):
    """ Read image data from DPX """

    if metadata['depth'] != 10 or metadata['packing'] != 1 or metadata['encoding'] != 0 or metadata['descriptor'] != 50:
        return None

    width = metadata['width']
    height = metadata['height']
    image = np.empty((height, width, 3), dtype=float)

    fname.seek(metadata['offset'])
    raw = np.fromfile(fname, dtype=np.dtype(np.int32), count=width*height, sep="")
    raw = raw.reshape((height, width))

    if metadata['endianness'] == 'be':
        raw = raw.byteswap()

    # extract and normalize color channel values to 0..1 inclusive.

    image[:, :, 0] = ((raw >> 22) & 0x000003FF) / 1023.0
    image[:, :, 1] = ((raw >> 12) & 0x000003FF) / 1023.0
    image[:, :, 2] = ((raw >> 2) & 0x000003FF) / 1023.0

    return image

# Assumes a file has already been read, so DPX_HEADER has been initialized

def write_dpx_image(fname, image):
    """ write dpx image to file """

    global DPX_HEADER
    global DPX_ENDIAN
    global DPX_OFFSET

    fname.seek(0)
    fname.write(DPX_HEADER)

    raw = ((((image[:, :, 0] * 1023.0).astype(np.dtype(np.int32)) & 0x000003FF) << 22) |
           (((image[:, :, 1] * 1023.0).astype(np.dtype(np.int32)) & 0x000003FF) << 12) |
           (((image[:, :, 2] * 1023.0).astype(np.dtype(np.int32)) & 0x000003FF) << 2))

    if DPX_ENDIAN == 'be':
        raw = raw.byteswap()

    fname.seek(DPX_OFFSET)
    raw.tofile(fname, sep="")

def info():
    """ Display info about DPX images """

    filepath = sys.argv[1]

    if not os.path.exists(filepath):
        print("File not found:", filepath)
    else:
        print("Examining:", filepath)
        with open(filepath, "rb") as fname:
            metadata = read_dpx_metadata(fname)
            if metadata is None:
                print("Invalid File")
            else:
                import binascii
                print("\nFILE INFORMATION HEADER")

                print("Endianness:", "Big Endian" if metadata['endianness'] == ">" else "Little Endian")
                print("Image Offset (Bytes):", metadata['offset'])
                print("DPX Version:", metadata['dpx_version'])
                print("File Size (Bytes):", metadata['file_size'])
                print("Ditto Flag:", "New Frame" if metadata['ditto'] else "Same as Previous Frame")
                print("Image Filename:", metadata['filename'])
                print("Creation Timestamp:", metadata['timestamp'])
                print("Creator:", metadata['creator'])
                print("Project Name:", metadata['project_name'])
                print("Copyright:", metadata['copyright'])
                print("Encryption Key:", "Unencrypted" if metadata['encryption_key'] == 0xFFFFFFFF else binascii.hexlify(bin(metadata['encryption_key'])))


                print("\nIMAGE INFORMATION HEADER")
                print("Orientation:", ORIENTATIONS[metadata['orientation']] if metadata['orientation'] in ORIENTATIONS else "unknown")
                print("Image Element Count:", metadata['image_element_count'])
                print("Width:", metadata['width'])
                print("Height:", metadata['height'])

                print("\nIMAGE ELEMENT 1")
                print("Data Sign:", "signed" if metadata['data_sign'] == 1 else "unsigned")
                print("Descriptor:", DESCRIPTORS[metadata['descriptor']] if metadata['descriptor'] in DESCRIPTORS else "unknown")
                print("Transfer:", TRANSFERS[metadata['transfer_characteristic']] if metadata['transfer_characteristic'] in TRANSFERS else "unknown")
                print("Colorimetry:", COLORIMETRIES[metadata['colorimetry']] if metadata['colorimetry'] in COLORIMETRIES else "unknown")
                print("Bit Depth:", metadata['depth'])
                print("Packing:", PACKINGS[metadata['packing']] if metadata['packing'] in PACKINGS else "unknown")
                print("Encoding:", ENCODINGS[metadata['encoding']] if metadata['encoding'] in ENCODINGS else "unknown")
                print("End of Line Padding:", metadata['line_padding'])
                print("End of Image Padding:", metadata['image_padding'])
                print("Image Element Description:", metadata['image_element_description'])

                print("\nIMAGE SOURCE INFORMATION HEADER")
                print("Input Device Name:", metadata['input_device_name'])
                print("Input Device Serial Number:", metadata['input_device_sn'])

                print("\n")

if __name__ == '__main__':
    info()
