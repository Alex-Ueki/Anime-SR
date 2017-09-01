"""
frameops.py

Read, write and manipulate frame image files.

DPX code from https://gist.github.com/jackdoerner/1c9c48956a1e00a29dbc (Jack Doerner)

"""

import numpy as np
import scipy.misc as misc
import os
import copy
import random

import Modules.dpx as dpx

IMAGETYPES = ['.jpg', '.png', '.dpx']

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

    if os.path.isfile(file_path):
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
    else:
        img = None

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
#   black_level     black color to use when adding borders
#   trim_...        pixels to trim from images before tesselating
#   shuffle         Do a random permutation of files
#   jitter          Jitter the tile position randomly
#   skip            Randomly skip 0 to 4 tiles between returned tiles


def tesselate(file_paths, tile_width, tile_height, border, black_level=0.0,
              trim_top=0, trim_bottom=0, trim_left=0, trim_right=0, shuffle=True, jitter=False, skip=False):

    # Convert non-list to list

    file_paths = file_paths if type(file_paths) in (
        list, tuple) else [file_paths]

    # Shuffle the list

    if shuffle:
        random.shuffle(file_paths)

    # Process image files

    for f in file_paths:
        img = imread(f)

        if img is None:
            print('Tesselation error: could not read image {}'.format(f))
        else:
            # Trim the image

            if trim_top + trim_bottom + trim_left + trim_right > 0:
                shape = np.shape(img)
                img = img[trim_top:shape[0] - trim_bottom,
                          trim_left:shape[1] - trim_right, :]

            # Tessellate image

            shape = np.shape(img)
            rows = shape[0] // tile_height
            cols = shape[1] // tile_width

            if len(shape) != 3 or (shape[0] > rows * tile_height) or (shape[1] > cols * tile_width):
                print('Tesselation Error: file {} has incorrect shape {}'.format(
                    f, str(shape)))
            else:
                # Pad the image - the pixels added have value (black_level, black_level, black_level)

                img = np.pad(img, ((border, border), (border, border),
                                   (0, 0)), mode='constant', constant_values=black_level)

                # Actual tile width and height

                across, down = tile_width + \
                    (2 * border), tile_height + (2 * border)

                # Jitter tiles; if we are jittering, then the number of tiles along an axis is reduced by 1

                jitter_x = random.randint(0, tile_width - 1) if jitter else 0
                jitter_y = random.randint(0, tile_width - 1) if jitter else 0
                rows = rows if jitter_y == 0 else rows - 1
                cols = cols if jitter_x == 0 else cols - 1

                # Shuffle tiles

                row_list = list(range(rows))
                col_list = list(range(cols))
                if shuffle:
                    random.shuffle(row_list)
                    random.shuffle(col_list)

                # Generate tiles

                skip_count = random.randint(0, 4) if skip else 0

                for row in row_list:
                    rpos = (row * tile_height) + jitter_y
                    for col in col_list:
                        if skip_count <= 0:
                            skip_count = random.randint(0, 4) if skip else 0
                            cpos = (col * tile_width) + jitter_x
                            tile = img[rpos:rpos + down, cpos:cpos + across, :]
                            yield tile
                        else:
                            skip_count -= 1

# This version tesellates matched pairs of images, with identical shuffling behavior. Used for model training


def tesselate_pair(alpha_paths, beta_paths, tile_width, tile_height, border, black_level=0.0,
                   trim_top=0, trim_bottom=0, trim_left=0, trim_right=0, shuffle=True, jitter=False, skip=False):

    # Convert non-lists to lists

    alpha_paths = alpha_paths if type(alpha_paths) in (
        list, tuple) else [alpha_paths]
    beta_paths = beta_paths if type(beta_paths) in (
        list, tuple) else [beta_paths]

    all_paths = list(zip(alpha_paths, beta_paths))

    # Shuffle the lists

    if shuffle:
        random.shuffle(all_paths)

    # Process the image file pairs

    for alpha_path, beta_path in all_paths:
        alpha_img = imread(alpha_path)
        beta_img = imread(beta_path)

        if alpha_img is None:
            print('Tesselation error: could not read image {}'.format(alpha_path))
        elif beta_img is None:
            print('Tesselation error: could not read image {}'.format(beta_path))
        else:
            # Trim the images

            if trim_top + trim_bottom + trim_left + trim_right > 0:
                shape = np.shape(alpha_img)
                alpha_img = alpha_img[trim_top:shape[0] -
                                      trim_bottom, trim_left:shape[1] - trim_right, :]
                beta_img = beta_img[trim_top:shape[0] -
                                    trim_bottom, trim_left:shape[1] - trim_right, :]

            # Tessellate image pairs

            alpha_shape = np.shape(alpha_img)
            beta_shape = np.shape(beta_img)
            rows = alpha_shape[0] // tile_height
            cols = alpha_shape[1] // tile_width

            if alpha_shape != beta_shape:
                print('Tesselation error: file pairs {} and {} have different shapes {} and {}'.format(
                    alpha_path, beta_path, str(alpha_shape), str(beta_shape)))
            elif len(alpha_shape) != 3 or (alpha_shape[0] > rows * tile_height) or (alpha_shape[1] > cols * tile_width):
                print('Tesselation error: file pairs {} and {} have incorrect shape {}'.format(
                    alpha_path, beta_path, str(alpha_shape)))
            else:
                # Pad the images - the pixels added have value (black_level, black_level, black_level)

                alpha_img = np.pad(alpha_img, ((border, border), (border, border),
                                               (0, 0)), mode='constant', constant_values=black_level)
                beta_img = np.pad(beta_img, ((border, border), (border, border),
                                             (0, 0)), mode='constant', constant_values=black_level)

                # Actual tile width and height

                across, down = tile_width + 2 * border, tile_height + 2 * border

                # Jitter tiles; if we are jittering, then the number of tiles along an axis is reduced by 1

                jitter_x = random.randint(0, tile_width - 1) if jitter else 0
                jitter_y = random.randint(0, tile_width - 1) if jitter else 0
                rows = rows if jitter_y == 0 else rows - 1
                cols = cols if jitter_x == 0 else cols - 1

                # Shuffle tiles

                row_list = list(range(rows))
                col_list = list(range(cols))
                if shuffle:
                    random.shuffle(row_list)
                    random.shuffle(col_list)

                # Generate tiles

                skip_count = random.randint(0, 4) if skip else 0

                for row in row_list:
                    rpos = (row * tile_height) + jitter_y
                    for col in col_list:
                        if skip_count <= 0:
                            skip_count = random.randint(0, 4) if skip else 0
                            cpos = (col * tile_width) + jitter_x
                            alpha_tile = alpha_img[rpos:rpos +
                                                   down, cpos:cpos + across, :]
                            beta_tile = beta_img[rpos:rpos +
                                                 down, cpos:cpos + across, :]
                            # Debug code, will be removed next push
                            if np.shape(alpha_tile) != np.shape(beta_tile) or np.shape(alpha_tile) != (64,64,3):
                                print('')
                                print('')
                                print(alpha_path)
                                print(np.shape(alpha_img))
                                print(beta_path)
                                print(np.shape(beta_img))
                                print(np.shape(alpha_tile))
                                print(np.shape(beta_tile))
                                print('row {} tile {} jitter {} rpos {} down {} rpos+down {}'.format(row,tile_height,jitter_y,rpos,down,rpos+down))
                                print('col {} tile {} jitter {} cpos {} across {} cpos+across {}'.format(col,tile_width,jitter_x,cpos,across,cpos+across))
                                print('')
                                print('')
                            yield (alpha_tile, beta_tile)
                        else:
                            skip_count -= 1

# Quasi-inverse to tesselate; glue together tiles to form a final image; takes list of numpy tile arrays,
# trims off the borders, stitches them together, and pads as needed.
#
#   tiles           1-d list of numpy image tiles
#   border          number of pixels to remove from the border
#   row_width       number of tiles per row (number of rows is thus implicit)
#   black_level    black color to use when padding
#   pad_...         amount of padding to create on each edge


def grout(tiles, border, row_width, black_level=0.0, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0):

    # Figure out the size of the final image and allocate it

    tile_shape = np.shape(tiles[0])
    print(tile_shape)
    tile_width, tile_height = tile_shape[1] - \
        border * 2, tile_shape[0] - border * 2
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
            img[img_row:img_row + tile_height, img_col:img_col +
                tile_width] = tiles[cur_tile][first_row:last_row, first_col:last_col]
            cur_tile += 1

    # Pad the tiles

    if pad_top + pad_bottom + pad_left + pad_right > 0:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right),
                           (0, 0)), mode='constant', constant_values=black_level)

    return img

# Compute the size of the tile grid for an image


def tile_grid(file_path, tile_width=60, tile_height=60, border=2, trim_top=0, trim_bottom=0, trim_left=0, trim_right=0):

    # We might get passed a list, unpack until we get a string (filepath)

    while type(file_path) is list:
        file_path = file_path[0]

    img = imread(file_path)
    shape = np.shape(img)

    rows, cols = shape[0] - (trim_top + trim_bottom) // tile_height, shape[1] - \
        (trim_left + trim_right) // tile_width

    return (rows, cols)
