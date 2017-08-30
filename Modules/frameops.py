"""
frameops.py

Read, write and manipulate frame image files.

DPX code from https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc (Jack Doerner)

"""

import numpy as np
import scipy.misc as misc
import os
import copy

import Modules.dpx as dpx

IMAGETYPES = [ '.jpg', '.png', '.dpx']

# Look in folder_path for all the files that are of one of the IMAGETYPES,
# and return a sorted list of lists containing the absolute paths to those files. So if
# there are only png files, the result will be a list containing a single list, but if
# there are jpg and dpx files, there will be two sublists, one for each type.
#
# If deep is True, look at all the subdirectories as well.

def image_files(folder_path, deep=False):

    file_paths = []
    file_lists = []

    for (root, dirs, files) in os.walk(folder_path):
        file_paths.extend([os.path.join(root, f) for f in files])
        if not deep:
            break

    if len(file_paths) == 0:
        return file_lists

    for ext in IMAGETYPES:
        ext_list = [f for f in file_paths if os.path.splitext(f)[1] == ext]
        if len(ext_list) > 0:
            file_lists.append(sorted(ext_list))

    return file_lists

# Keep a copy of the last meta information read

last_meta = None

# Read an image file and return numpy array of pixel values. Extends scipy.misc.imread
# with support for 10-bit DPX files. Always returns RGB images, with pixel values
# normalized to 0..1 range (inclusive).
#
# Note that the dpx library routines already handle (de-)normalization, so we only
# have to handle that for operations we hand off to scipy.misc.

def imread(file_path):

    global last_meta

    file_type = os.path.splitext(file_path)[1]

    if file_type == '.dpx':
        with open(file_path, 'rb') as f:
            meta = dpx.readDPXMetaData(f)
            if meta is None:
                img = None
            else:
                img = dpx.readDPXImageData(f, meta)
                last_meta = copy.deepcopy(meta)
    else:
        img = misc.imread(file_path, mode='RGB')
        img = img.astype('float32') / 255.0

    return img

# Write a numpy array to an image file. Extends scipy.misc.imsave
# with support for 10-bit DPX files. Expects the input array to be
# normalized to 0..1 range (inclusive). If DPX is being saved, then
# if meta information is None, the meta information of the Last
# image loaded will be used.

def imsave(file_path, img, meta=None):

    global last_meta

    file_type = os.path.splitext(file_path)[1]

    if file_type == '.dpx':
        with open(file_path, 'wb') as f:
            dpx.writeDPX(f, img, last_meta if meta == None else meta)
    else:
        img = (img * 255.0).astype(np.uint8)
        misc.imsave(file_path, img)

# Generator for image tiles. Trims each image file (useful for handling 4:3 images in 16:9 HD files),
# then adds a border of zeros before generating the tiles. Each tile will be of shape
# (tile_height+border*2, tile_width+border*2, 3)
#
#   file_paths      list of image files to tesselate
#   tile_width      width of tile in image files
#   tile_height     height of tile in image files
#   border          number of pixels to add to tile borders
#   border_color    black color to use when adding borders
#   trim_...        pixels to trim from images before tesselating

def tesselate(file_paths, tile_width, tile_height, border, border_color = 0, trim_top=0, trim_bottom=0, trim_left=0, trim_right=0):

    # Convert non-list to list

    file_paths = file_paths if type(file_paths) in (list,tuple) else [ file_paths ]

    for f in file_paths:
        img = imread(f)
        # Trim the image
        if trim_top + trim_bottom + trim_left + trim_right > 0:
            shape = np.shape(img)
            print(shape)
            img = img[trim_top:shape[0]-trim_bottom, trim_left:shape[1]-trim_right, :]
        shape = np.shape(img)
        print(shape)
        rows = shape[0] // tile_height
        cols = shape[1] // tile_width
        if len(shape) != 3 or (shape[0] > rows * tile_height) or (shape[1] > cols * tile_width):
            print('Error: file {} has incorrect shape {}'.format(f,str(np.shape.img)))
        else:
            # Pad the image - the pixels added have value (border_color, border_color, border_color)
            img = np.pad(img, ((border, border), (border, border), (0, 0)), mode='constant', constant_values=border_color)
            shape = np.shape(img)
            print(shape)
            # Generate tiles
            across, down = tile_width+(2 * border), tile_height+(2 * border)
            for row in range(0, rows):
                rpos = row * tile_width
                for col in range(0, cols):
                    cpos = col * tile_height
                    tile = img[rpos:rpos + down, cpos:cpos + across,:]
                    print('Tile r={},c={} starts at {},{} with shape {}'.format(row, col, rpos, cpos, str(np.shape(tile))))
                    yield tile

# Quasi-inverse to tesselate; glue together tiles to form a final image; takes list of numpy tile arrays,
# trims off the borders, stitches them together, and pads as needed.
#
#   tiles           1-d list of numpy image tiles
#   border          number of pixels to remove from the border
#   row_width       number of tiles per row (number of rows is thus implicit)
#   border_color    black color to use when padding
#   pad_...         amount of padding to create on each edge

def grout(tiles, border, row_width, border_color=0, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0):

    # Figure out the size of the final image and allocate it

    tile_shape = np.shape(tiles[0])
    print(tile_shape)
    tile_width, tile_height = tile_shape[1] - border * 2, tile_shape[0] - border * 2
    print(tile_width, tile_height)

    row_count = (len(tiles) // row_width)
    img_width, img_height = tile_width * row_width, tile_height * row_count
    print(img_width, img_height)

    img = np.empty((img_height, img_width, 3), dtype='float32')

    # Tile clipping range

    first_col, last_col = border, tile_shape[1] - border
    first_row, last_row = border, tile_shape[0] - border

    # Grout the tiles

    cur_tile = 0
    for row in range(row_count):
        img_row = row * tile_height
        for col in range(row_width):
            img_col = col * tile_width
            img[img_row:img_row+tile_height, img_col:img_col+tile_width] = tiles[cur_tile][first_row:last_row, first_col:last_col]
            cur_tile += 1

    # Pad the tiles

    if pad_top + pad_bottom + pad_left + pad_right > 0:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=border_color)

    return img

# Test code

def test_list(folder_path, deep=False):

    files = image_files(folder_path, deep)

    print('Listing ' + folder_path)

    for img_type in files:
        if len(img_type) > 0:
            fname = img_type[len(img_type)//4]
            fname_info = os.path.splitext(os.path.basename(fname))
            print('Type  : ' + fname_info[1])
            print('Number: ' + str(len(img_type)))
            img = imread(fname)
            shape = np.shape(img)
            print('File  : ' + fname)
            print('Shape : ' + str(shape))
            print('Center: ' + str(img[shape[0]//2][shape[1]//2]))
            print('')
            tfilename = 'X-'+''.join(fname_info)
            imsave(tfilename, img)
            img2 = imread(tfilename)
            print('save/load OK!' if np.array_equal(img, img2) else 'save/load bad!')
